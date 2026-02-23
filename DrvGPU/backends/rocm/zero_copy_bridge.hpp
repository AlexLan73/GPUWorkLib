#pragma once

/**
 * @file zero_copy_bridge.hpp
 * @brief ZeroCopy мост между OpenCL и ROCm/HIP
 *
 * Позволяет импортировать OpenCL cl_mem буфер в HIP address space
 * без копирования через CPU. Поддерживает три метода:
 * - DMA-BUF (cl_khr_external_memory_dma_buf → hipImportExternalMemory)
 * - AMD GPU VA (прямой GPU virtual address, zero overhead)
 * - SVM (Shared Virtual Memory)
 *
 * ВАЖНО: Linux only! На Windows — стаб, бросающий исключение.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../opencl/opencl_export.hpp"

#include <hip/hip_runtime.h>
#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: ZeroCopyBridge — мост OpenCL ↔ ROCm
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ZeroCopyBridge
 * @brief Импорт OpenCL cl_mem в HIP address space без копирования
 *
 * Последовательность использования:
 * 1. Создать ZeroCopyBridge
 * 2. Вызвать ImportFromOpenCl(fd, size) или ImportFromGpuVA(va, size)
 * 3. Получить HIP указатель через GetHipPtr()
 * 4. Использовать указатель в HIP kernels
 * 5. Деструктор освободит ресурсы
 *
 * @code
 * // Экспорт cl_mem → dma-buf fd
 * int fd = ExportClBufferToFd(cl_buffer);
 *
 * // Импорт в HIP
 * ZeroCopyBridge bridge;
 * bridge.ImportFromDmaBuf(fd, buffer_size);
 *
 * // Использовать в HIP kernel
 * void* hip_ptr = bridge.GetHipPtr();
 * my_kernel<<<grid, block>>>((float*)hip_ptr, N);
 * @endcode
 */
class ZeroCopyBridge {
public:
  ZeroCopyBridge();
  ~ZeroCopyBridge();

  // Запрет копирования
  ZeroCopyBridge(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge& operator=(const ZeroCopyBridge&) = delete;

  // Перемещение
  ZeroCopyBridge(ZeroCopyBridge&& other) noexcept;
  ZeroCopyBridge& operator=(ZeroCopyBridge&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════════════
  // Методы импорта
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Импорт через dma-buf file descriptor (метод A)
   *
   * Использует hipImportExternalMemory для маппинга dma-buf fd
   * в HIP address space. Minimal overhead (~микросекунды).
   *
   * @param dma_buf_fd File descriptor от ExportClBufferToFd()
   * @param buffer_size Размер буфера в байтах
   * @return hipSuccess при успехе
   * @throws std::runtime_error при ошибке HIP
   */
  hipError_t ImportFromDmaBuf(int dma_buf_fd, size_t buffer_size);

  /**
   * @brief Импорт через GPU virtual address (метод B, AMD-only)
   *
   * Если OpenCL и HIP работают на одном AMD GPU с unified address space,
   * GPU VA из cl_mem можно использовать в HIP напрямую.
   * Zero overhead — просто сохраняем указатель.
   *
   * @param gpu_va GPU virtual address от ExportClBufferToGpuVA()
   * @param buffer_size Размер буфера в байтах
   * @throws std::runtime_error если gpu_va == nullptr
   */
  void ImportFromGpuVA(void* gpu_va, size_t buffer_size);

  /**
   * @brief Универсальный импорт — автоопределение метода
   *
   * Пробует методы в порядке приоритета:
   * 1. AMD GPU VA (если OpenCL device передан и поддерживает)
   * 2. DMA-BUF
   * 3. Ошибка
   *
   * @param cl_buffer OpenCL буфер для импорта
   * @param buffer_size Размер буфера в байтах
   * @param cl_device OpenCL device (для проверки capabilities)
   * @throws std::runtime_error если ни один метод не сработал
   */
  void ImportFromOpenCl(cl_mem cl_buffer, size_t buffer_size, cl_device_id cl_device);

  // ═══════════════════════════════════════════════════════════════════════
  // Доступ к данным
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Получить HIP указатель на данные
   * @return void* — указатель, пригодный для HIP kernels
   */
  void* GetHipPtr() const { return hip_ptr_; }

  /**
   * @brief Размер буфера в байтах
   */
  size_t GetSize() const { return size_; }

  /**
   * @brief Активен ли мост (импорт выполнен)
   */
  bool IsActive() const { return hip_ptr_ != nullptr; }

  /**
   * @brief Какой метод ZeroCopy использован
   */
  ZeroCopyMethod GetMethod() const { return method_; }

  /**
   * @brief Освободить ресурсы (вызывается автоматически в деструкторе)
   */
  void Release();

private:
  hipExternalMemory_t ext_mem_;   ///< HIP external memory handle (для DMA-BUF)
  void* hip_ptr_;                  ///< HIP device pointer
  size_t size_;                    ///< Размер буфера
  ZeroCopyMethod method_;          ///< Используемый метод
  bool owns_memory_;               ///< true если ext_mem_ нужно освободить
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM — Windows stub

#include "../opencl/opencl_export.hpp"
#include <CL/cl.h>
#include <cstddef>
#include <stdexcept>

namespace drv_gpu_lib {

/**
 * @class ZeroCopyBridge
 * @brief Windows stub — ZeroCopy не поддерживается
 */
class ZeroCopyBridge {
public:
  ZeroCopyBridge() = default;
  ~ZeroCopyBridge() = default;

  ZeroCopyBridge(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge& operator=(const ZeroCopyBridge&) = delete;
  ZeroCopyBridge(ZeroCopyBridge&&) noexcept = default;
  ZeroCopyBridge& operator=(ZeroCopyBridge&&) noexcept = default;

  void ImportFromOpenCl(cl_mem, size_t, cl_device_id) {
    throw std::runtime_error("ZeroCopyBridge: not available (ENABLE_ROCM=OFF, Linux required)");
  }

  void* GetHipPtr() const { return nullptr; }
  size_t GetSize() const { return 0; }
  bool IsActive() const { return false; }
  ZeroCopyMethod GetMethod() const { return ZeroCopyMethod::NONE; }
  void Release() {}
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
