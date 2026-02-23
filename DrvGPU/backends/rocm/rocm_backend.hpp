#pragma once

/**
 * @file rocm_backend.hpp
 * @brief Реализация IBackend для ROCm/HIP
 *
 * ROCmBackend - полная реализация бэкенда на базе HIP API.
 *
 * MULTI-GPU (v2.0):
 * Каждый экземпляр ROCmBackend владеет СВОИМ ROCmCore,
 * что позволяет работать с разными AMD GPU параллельно.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../../interface/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"
#include "../../logger/logger.hpp"
#include "../../memory/memory_manager.hpp"

#include "rocm_core.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ROCmBackend - Реализация бэкенда для HIP/ROCm
// ════════════════════════════════════════════════════════════════════════════

class ROCmBackend : public IBackend {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  ROCmBackend();
  ~ROCmBackend() override;

  // ═══════════════════════════════════════════════════════════════
  // Запрет копирования, разрешение перемещения
  // ═══════════════════════════════════════════════════════════════
  ROCmBackend(const ROCmBackend&) = delete;
  ROCmBackend& operator=(const ROCmBackend&) = delete;
  ROCmBackend(ROCmBackend&& other) noexcept;
  ROCmBackend& operator=(ROCmBackend&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Инициализация
  // ═══════════════════════════════════════════════════════════════

  void Initialize(int device_index) override;
  bool IsInitialized() const override { return initialized_; }
  void Cleanup() override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Управление владением ресурсами
  // ═══════════════════════════════════════════════════════════════

  void SetOwnsResources(bool owns) override { owns_resources_ = owns; }
  bool OwnsResources() const override { return owns_resources_; }

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Информация об устройстве
  // ═══════════════════════════════════════════════════════════════

  BackendType GetType() const override { return BackendType::ROCm; }
  GPUDeviceInfo GetDeviceInfo() const override;
  int GetDeviceIndex() const override { return device_index_; }
  std::string GetDeviceName() const override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Нативные хэндлы
  // ═══════════════════════════════════════════════════════════════

  void* GetNativeContext() const override;
  void* GetNativeDevice() const override;
  void* GetNativeQueue() const override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Управление памятью
  // ═══════════════════════════════════════════════════════════════

  void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
  void Free(void* ptr) override;

  void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
  void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Синхронизация
  // ═══════════════════════════════════════════════════════════════

  void Synchronize() override;
  void Flush() override;

  // ═══════════════════════════════════════════════════════════════
  // Реализация IBackend: Возможности устройства
  // ═══════════════════════════════════════════════════════════════

  bool SupportsSVM() const override { return false; }
  bool SupportsDoublePrecision() const override;
  size_t GetMaxWorkGroupSize() const override;
  size_t GetGlobalMemorySize() const override;
  size_t GetFreeMemorySize() const override;
  size_t GetLocalMemorySize() const override;

  // ═══════════════════════════════════════════════════════════════
  // Специфичные для ROCm методы
  // ═══════════════════════════════════════════════════════════════

  ROCmCore& GetCore();
  const ROCmCore& GetCore() const;

  MemoryManager* GetMemoryManager() override;
  const MemoryManager* GetMemoryManager() const override;

protected:
  int device_index_;
  bool initialized_;
  bool owns_resources_;

  std::unique_ptr<ROCmCore> core_;
  std::unique_ptr<MemoryManager> memory_manager_;

  // Cached HIP handles
  hipDevice_t device_;
  hipStream_t stream_;

  mutable std::mutex mutex_;

private:
  GPUDeviceInfo QueryDeviceInfo() const;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
