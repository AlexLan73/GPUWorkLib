#pragma once

/**
 * @file opencl_export.hpp
 * @brief Экспорт OpenCL cl_mem буфера в dma-buf file descriptor (Linux only)
 *
 * Используется для ZeroCopy передачи данных между OpenCL и ROCm/HIP
 * без копирования через CPU. Требуется расширение cl_khr_external_memory_dma_buf.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#include <CL/cl.h>
#include <string>
#include <stdexcept>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// OpenCL Export Utilities — экспорт cl_mem в dma-buf fd
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверить поддержку dma-buf экспорта на устройстве
 *
 * Проверяет наличие расширений:
 * - cl_khr_external_memory
 * - cl_khr_external_memory_dma_buf
 *
 * @param device OpenCL device для проверки
 * @return true если dma-buf экспорт поддерживается
 */
inline bool SupportsDmaBufExport(cl_device_id device) {
#ifdef __linux__
  if (!device) return false;

  size_t ext_size = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
  if (err != CL_SUCCESS || ext_size == 0) return false;

  std::string extensions(ext_size, '\0');
  err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, &extensions[0], nullptr);
  if (err != CL_SUCCESS) return false;

  return extensions.find("cl_khr_external_memory_dma_buf") != std::string::npos;
#else
  (void)device;
  return false;
#endif
}

/**
 * @brief Проверить поддержку AMD GPU Virtual Address экспорта
 *
 * Fallback вариант: использование CL_MEM_AMD_GPU_VA для получения
 * прямого GPU virtual address из cl_mem (AMD-only).
 *
 * @param device OpenCL device
 * @return true если AMD GPU VA поддерживается
 */
inline bool SupportsAmdGpuVA(cl_device_id device) {
#ifdef __linux__
  if (!device) return false;

  size_t ext_size = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
  if (err != CL_SUCCESS || ext_size == 0) return false;

  std::string extensions(ext_size, '\0');
  err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, &extensions[0], nullptr);
  if (err != CL_SUCCESS) return false;

  // AMD-specific: SVM fine grain buffer implies unified address space
  return extensions.find("cl_amd_svm") != std::string::npos ||
         extensions.find("cl_khr_svm") != std::string::npos;
#else
  (void)device;
  return false;
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// CL_MEM_LINUX_DMA_BUF_FD_KHR — может не быть в старых заголовках
// ════════════════════════════════════════════════════════════════════════════

#ifndef CL_MEM_LINUX_DMA_BUF_FD_KHR
#define CL_MEM_LINUX_DMA_BUF_FD_KHR 0x10011
#endif

// AMD-specific: GPU virtual address property
#ifndef CL_MEM_AMD_GPU_VA
#define CL_MEM_AMD_GPU_VA 0x4100
#endif

/**
 * @brief Экспортировать cl_mem буфер в dma-buf file descriptor
 *
 * Метод A (рекомендуемый): Использует cl_khr_external_memory_dma_buf
 * для получения dma-buf fd, который затем можно импортировать в HIP.
 *
 * @param buffer OpenCL cl_mem буфер для экспорта
 * @return dma-buf file descriptor (>= 0), или -1 при ошибке
 *
 * @note Linux only! На Windows всегда возвращает -1.
 * @note Буфер должен быть создан с совместимыми флагами.
 *
 * @code
 * cl_mem gpu_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
 * int fd = ExportClBufferToFd(gpu_buf);
 * if (fd >= 0) {
 *     // Можно передать fd в ZeroCopyBridge::ImportFromOpenCl()
 *     bridge.ImportFromOpenCl(fd, size);
 * }
 * @endcode
 */
inline int ExportClBufferToFd(cl_mem buffer) {
#ifdef __linux__
  if (!buffer) return -1;

  int dma_buf_fd = -1;
  cl_int err = clGetMemObjectInfo(
      buffer,
      CL_MEM_LINUX_DMA_BUF_FD_KHR,
      sizeof(int),
      &dma_buf_fd,
      nullptr
  );

  if (err != CL_SUCCESS) {
    return -1;
  }

  return dma_buf_fd;
#else
  (void)buffer;
  return -1;  // Windows: not supported
#endif
}

/**
 * @brief Получить GPU virtual address из cl_mem (AMD-only fallback)
 *
 * Метод B (AMD-specific): Использует CL_MEM_AMD_GPU_VA для получения
 * прямого GPU VA, который HIP может использовать напрямую
 * (тот же unified address space на AMD GPU).
 *
 * @param buffer OpenCL cl_mem буфер
 * @return GPU virtual address, или nullptr при ошибке
 *
 * @note Работает только на AMD GPU с поддержкой unified address space.
 * @note HIP может использовать этот адрес напрямую через cast.
 */
inline void* ExportClBufferToGpuVA(cl_mem buffer) {
#ifdef __linux__
  if (!buffer) return nullptr;

  void* gpu_va = nullptr;
  cl_int err = clGetMemObjectInfo(
      buffer,
      CL_MEM_AMD_GPU_VA,
      sizeof(void*),
      &gpu_va,
      nullptr
  );

  if (err != CL_SUCCESS) {
    return nullptr;
  }

  return gpu_va;
#else
  (void)buffer;
  return nullptr;  // Windows: not supported
#endif
}

/**
 * @brief Описание поддерживаемых методов ZeroCopy
 */
enum class ZeroCopyMethod {
  NONE,         ///< ZeroCopy не поддерживается
  DMA_BUF,      ///< cl_khr_external_memory_dma_buf (рекомендуемый)
  AMD_GPU_VA,   ///< CL_MEM_AMD_GPU_VA (AMD-only, zero overhead)
  SVM           ///< Shared Virtual Memory (OpenCL 2.0+)
};

/**
 * @brief Определить лучший метод ZeroCopy для данного устройства
 *
 * Приоритет:
 * 1. AMD_GPU_VA — нулевой overhead (тот же адрес)
 * 2. DMA_BUF — официальный API, минимальный overhead
 * 3. SVM — требует OpenCL 2.0+
 * 4. NONE — ZeroCopy не поддерживается
 *
 * @param device OpenCL device
 * @return Лучший доступный метод ZeroCopy
 */
inline ZeroCopyMethod DetectBestZeroCopyMethod(cl_device_id device) {
#ifdef __linux__
  if (SupportsAmdGpuVA(device))       return ZeroCopyMethod::AMD_GPU_VA;
  if (SupportsDmaBufExport(device))   return ZeroCopyMethod::DMA_BUF;

  // Check SVM support
  cl_device_svm_capabilities svm_caps = 0;
  cl_int err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                                sizeof(svm_caps), &svm_caps, nullptr);
  if (err == CL_SUCCESS && (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
    return ZeroCopyMethod::SVM;
  }

  return ZeroCopyMethod::NONE;
#else
  (void)device;
  return ZeroCopyMethod::NONE;
#endif
}

/**
 * @brief Получить строковое описание метода ZeroCopy
 */
inline const char* ZeroCopyMethodToString(ZeroCopyMethod method) {
  switch (method) {
    case ZeroCopyMethod::DMA_BUF:    return "DMA-BUF (cl_khr_external_memory_dma_buf)";
    case ZeroCopyMethod::AMD_GPU_VA: return "AMD GPU VA (CL_MEM_AMD_GPU_VA)";
    case ZeroCopyMethod::SVM:        return "SVM (Shared Virtual Memory)";
    case ZeroCopyMethod::NONE:       return "None (ZeroCopy not supported)";
    default:                         return "Unknown";
  }
}

}  // namespace drv_gpu_lib
