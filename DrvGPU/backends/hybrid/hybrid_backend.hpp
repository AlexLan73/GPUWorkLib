#pragma once

/**
 * @file hybrid_backend.hpp
 * @brief HybridBackend — гибридный OpenCL + ROCm бэкенд для одного GPU
 *
 * HybridBackend содержит оба sub-backend (OpenCL и ROCm) и позволяет:
 * - Выполнять OpenCL операции (clFFT, существующие kernels)
 * - Выполнять HIP/ROCm операции (hipFFT, rocPRIM, hiprtc kernels)
 * - Обмениваться данными через ZeroCopyBridge (без копирования CPU)
 *
 * Архитектура: Вариант A (обёртка) — HybridBackend : IBackend
 * хранит OpenCLBackend + ROCmBackend, делегирует по контексту.
 *
 * Выбор primary backend:
 * - По умолчанию: OpenCL (основной для legacy пайплайна)
 * - ROCm: для модулей, использующих hipFFT / rocPRIM / hiprtc
 * - Модули сами решают, какой backend использовать через GetOpenCL() / GetROCm()
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../../interface/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"
#include "../../memory/memory_manager.hpp"
#include "../../logger/logger.hpp"

#include "../opencl/opencl_backend.hpp"
#include "../rocm/rocm_backend.hpp"
#include "../rocm/zero_copy_bridge.hpp"

#include <memory>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: HybridBackend — OpenCL + ROCm на одном GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class HybridBackend
 * @brief Гибридный бэкенд, объединяющий OpenCL и ROCm для одного GPU
 *
 * По умолчанию операции памяти (Allocate/Free/Memcpy) делегируются OpenCL.
 * Доступ к обоим sub-backend через GetOpenCL() / GetROCm().
 *
 * @code
 * DrvGPU gpu(BackendType::OPENCLandROCm, 0);
 * gpu.Initialize();
 *
 * auto& hybrid = static_cast<HybridBackend&>(gpu.GetBackend());
 *
 * // OpenCL операции
 * auto* cl_backend = hybrid.GetOpenCL();
 * void* cl_buf = cl_backend->Allocate(1024);
 *
 * // ROCm операции
 * auto* rocm_backend = hybrid.GetROCm();
 * void* hip_buf = rocm_backend->Allocate(1024);
 *
 * // ZeroCopy: OpenCL → ROCm
 * auto bridge = hybrid.CreateZeroCopyBridge(
 *     static_cast<cl_mem>(cl_buf), 1024);
 * void* hip_ptr = bridge->GetHipPtr();  // тот же буфер!
 * @endcode
 */
class HybridBackend : public IBackend {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  HybridBackend();
  ~HybridBackend() override;

  // Запрет копирования
  HybridBackend(const HybridBackend&) = delete;
  HybridBackend& operator=(const HybridBackend&) = delete;

  // Перемещение
  HybridBackend(HybridBackend&& other) noexcept;
  HybridBackend& operator=(HybridBackend&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Инициализация
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Инициализировать оба sub-backend для одного GPU
   *
   * Порядок:
   * 1. OpenCLBackend::Initialize(device_index)
   * 2. ROCmBackend::Initialize(device_index)
   * 3. Проверка ZeroCopy capabilities
   *
   * @param device_index Индекс GPU устройства
   */
  void Initialize(int device_index) override;
  bool IsInitialized() const override { return initialized_; }
  void Cleanup() override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Владение ресурсами
  // ═══════════════════════════════════════════════════════════════

  void SetOwnsResources(bool owns) override { owns_resources_ = owns; }
  bool OwnsResources() const override { return owns_resources_; }

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Информация
  // ═══════════════════════════════════════════════════════════════

  BackendType GetType() const override { return BackendType::OPENCLandROCm; }
  GPUDeviceInfo GetDeviceInfo() const override;
  int GetDeviceIndex() const override { return device_index_; }
  std::string GetDeviceName() const override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Нативные хэндлы (делегируем OpenCL)
  // ═══════════════════════════════════════════════════════════════

  void* GetNativeContext() const override;   // → OpenCL cl_context
  void* GetNativeDevice() const override;    // → OpenCL cl_device_id
  void* GetNativeQueue() const override;     // → OpenCL cl_command_queue

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Память (делегируем OpenCL по умолчанию)
  // ═══════════════════════════════════════════════════════════════

  void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
  void Free(void* ptr) override;

  void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Синхронизация (оба backend)
  // ═══════════════════════════════════════════════════════════════

  void Synchronize() override;
  void Flush() override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: Capabilities (от OpenCL backend)
  // ═══════════════════════════════════════════════════════════════

  bool SupportsSVM() const override;
  bool SupportsDoublePrecision() const override;
  size_t GetMaxWorkGroupSize() const override;
  size_t GetGlobalMemorySize() const override;
  size_t GetFreeMemorySize() const override;
  size_t GetLocalMemorySize() const override;

  // ═══════════════════════════════════════════════════════════════
  // IBackend: MemoryManager
  // ═══════════════════════════════════════════════════════════════

  MemoryManager* GetMemoryManager() override;
  const MemoryManager* GetMemoryManager() const override;

  // ═══════════════════════════════════════════════════════════════
  // HybridBackend-specific: доступ к sub-backends
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Получить OpenCL sub-backend
   * @return Указатель на OpenCLBackend (nullptr если не инициализирован)
   */
  OpenCLBackend* GetOpenCL() { return opencl_.get(); }
  const OpenCLBackend* GetOpenCL() const { return opencl_.get(); }

  /**
   * @brief Получить ROCm sub-backend
   * @return Указатель на ROCmBackend (nullptr если не инициализирован)
   */
  ROCmBackend* GetROCm() { return rocm_.get(); }
  const ROCmBackend* GetROCm() const { return rocm_.get(); }

  // ═══════════════════════════════════════════════════════════════
  // HybridBackend-specific: ZeroCopy
  // ═══════════════════════════════════════════════════════════════

  /**
   * @brief Создать ZeroCopy мост для cl_mem буфера
   *
   * Автоматически определяет лучший метод (AMD GPU VA → DMA-BUF → SVM).
   *
   * @param cl_buffer OpenCL буфер для импорта в HIP
   * @param buffer_size Размер буфера в байтах
   * @return unique_ptr на ZeroCopyBridge (владеет ресурсами)
   * @throws std::runtime_error если ZeroCopy не поддерживается
   */
  std::unique_ptr<ZeroCopyBridge> CreateZeroCopyBridge(
      cl_mem cl_buffer, size_t buffer_size);

  /**
   * @brief Определить лучший метод ZeroCopy для данного GPU
   * @return ZeroCopyMethod (AMD_GPU_VA, DMA_BUF, SVM или NONE)
   */
  ZeroCopyMethod GetBestZeroCopyMethod() const;

  /**
   * @brief Синхронизировать перед ZeroCopy передачей
   *
   * Вызывает clFinish на OpenCL queue, чтобы гарантировать,
   * что все данные записаны в VRAM перед доступом из HIP.
   */
  void SyncBeforeZeroCopy();

  /**
   * @brief Синхронизировать после ZeroCopy передачи
   *
   * Вызывает hipStreamSynchronize, чтобы гарантировать,
   * что HIP завершил работу с данными перед доступом из OpenCL.
   */
  void SyncAfterZeroCopy();

private:
  int device_index_;
  bool initialized_;
  bool owns_resources_;

  std::unique_ptr<OpenCLBackend> opencl_;
  std::unique_ptr<ROCmBackend> rocm_;

  mutable std::mutex mutex_;
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM — Windows stub

#include "../../interface/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"
#include "../../memory/memory_manager.hpp"
#include "../rocm/zero_copy_bridge.hpp"

#include <memory>
#include <stdexcept>

namespace drv_gpu_lib {

/**
 * @class HybridBackend
 * @brief Windows stub — HybridBackend не доступен без ROCm
 *
 * Все методы бросают std::runtime_error.
 * Компилируется, но не работает — для Windows используйте
 * BackendType::OPENCL.
 */
class HybridBackend : public IBackend {
public:
  HybridBackend() = default;
  ~HybridBackend() override = default;

  HybridBackend(const HybridBackend&) = delete;
  HybridBackend& operator=(const HybridBackend&) = delete;
  HybridBackend(HybridBackend&&) noexcept = default;
  HybridBackend& operator=(HybridBackend&&) noexcept = default;

  void Initialize(int) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  bool IsInitialized() const override { return false; }
  void Cleanup() override {}

  void SetOwnsResources(bool) override {}
  bool OwnsResources() const override { return false; }

  BackendType GetType() const override { return BackendType::OPENCLandROCm; }
  GPUDeviceInfo GetDeviceInfo() const override { return {}; }
  int GetDeviceIndex() const override { return -1; }
  std::string GetDeviceName() const override { return "HybridBackend (stub)"; }

  void* GetNativeContext() const override { return nullptr; }
  void* GetNativeDevice() const override { return nullptr; }
  void* GetNativeQueue() const override { return nullptr; }

  void* Allocate(size_t, unsigned int) override {
    throw std::runtime_error("HybridBackend::Allocate: not available (ENABLE_ROCM=OFF)");
  }
  void Free(void*) override {}
  void MemcpyHostToDevice(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  void MemcpyDeviceToHost(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }
  void MemcpyDeviceToDevice(void*, const void*, size_t) override {
    throw std::runtime_error("HybridBackend: not available (ENABLE_ROCM=OFF)");
  }

  void Synchronize() override {}
  void Flush() override {}

  bool SupportsSVM() const override { return false; }
  bool SupportsDoublePrecision() const override { return false; }
  size_t GetMaxWorkGroupSize() const override { return 0; }
  size_t GetGlobalMemorySize() const override { return 0; }
  size_t GetFreeMemorySize() const override { return 0; }
  size_t GetLocalMemorySize() const override { return 0; }
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
