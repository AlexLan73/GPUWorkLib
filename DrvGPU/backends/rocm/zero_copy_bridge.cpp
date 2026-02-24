/**
 * @file zero_copy_bridge.cpp
 * @brief Реализация ZeroCopyBridge (OpenCL ↔ ROCm)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "zero_copy_bridge.hpp"
#include "../../logger/logger.hpp"

#include <hip/hip_runtime.h>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Helper: проверка hipError
// ════════════════════════════════════════════════════════════════════════════

static void CheckHip(hipError_t err, const char* operation) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("ZeroCopyBridge: HIP error in ") + operation +
        ": " + hipGetErrorString(err) +
        " (code " + std::to_string(static_cast<int>(err)) + ")"
    );
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

ZeroCopyBridge::ZeroCopyBridge()
    : ext_mem_(nullptr)
    , hip_ptr_(nullptr)
    , size_(0)
    , method_(ZeroCopyMethod::NONE)
    , owns_memory_(false) {
}

ZeroCopyBridge::~ZeroCopyBridge() {
  Release();
}

ZeroCopyBridge::ZeroCopyBridge(ZeroCopyBridge&& other) noexcept
    : ext_mem_(other.ext_mem_)
    , hip_ptr_(other.hip_ptr_)
    , size_(other.size_)
    , method_(other.method_)
    , owns_memory_(other.owns_memory_) {
  other.ext_mem_ = nullptr;
  other.hip_ptr_ = nullptr;
  other.size_ = 0;
  other.method_ = ZeroCopyMethod::NONE;
  other.owns_memory_ = false;
}

ZeroCopyBridge& ZeroCopyBridge::operator=(ZeroCopyBridge&& other) noexcept {
  if (this != &other) {
    Release();
    ext_mem_ = other.ext_mem_;
    hip_ptr_ = other.hip_ptr_;
    size_ = other.size_;
    method_ = other.method_;
    owns_memory_ = other.owns_memory_;
    other.ext_mem_ = nullptr;
    other.hip_ptr_ = nullptr;
    other.size_ = 0;
    other.method_ = ZeroCopyMethod::NONE;
    other.owns_memory_ = false;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Import: DMA-BUF (метод A)
// ════════════════════════════════════════════════════════════════════════════

hipError_t ZeroCopyBridge::ImportFromDmaBuf(int dma_buf_fd, size_t buffer_size) {
  if (IsActive()) {
    Release();
  }

  if (dma_buf_fd < 0) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromDmaBuf: invalid fd (" +
                             std::to_string(dma_buf_fd) + ")");
  }

  if (buffer_size == 0) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromDmaBuf: buffer_size must be > 0");
  }

  size_ = buffer_size;

  // 1. Описание внешней памяти — dma-buf fd
  hipExternalMemoryHandleDesc ext_mem_desc = {};
  ext_mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
  ext_mem_desc.handle.fd = dma_buf_fd;
  ext_mem_desc.size = buffer_size;
  ext_mem_desc.flags = 0;

  // 2. Импорт в HIP
  hipError_t err = hipImportExternalMemory(&ext_mem_, &ext_mem_desc);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR("ZeroCopyBridge", "hipImportExternalMemory failed: " +
                     std::string(hipGetErrorString(err)));
    return err;
  }

  // 3. Маппинг в HIP address space
  hipExternalMemoryBufferDesc buf_desc = {};
  buf_desc.offset = 0;
  buf_desc.size = buffer_size;
  buf_desc.flags = 0;

  err = hipExternalMemoryGetMappedBuffer(&hip_ptr_, ext_mem_, &buf_desc);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR("ZeroCopyBridge", "hipExternalMemoryGetMappedBuffer failed: " +
                     std::string(hipGetErrorString(err)));
    (void)hipDestroyExternalMemory(ext_mem_);
    ext_mem_ = nullptr;
    return err;
  }

  method_ = ZeroCopyMethod::DMA_BUF;
  owns_memory_ = true;

  DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via DMA-BUF (fd=" +
                  std::to_string(dma_buf_fd) + ", size=" +
                  std::to_string(buffer_size) + " bytes)");

  return hipSuccess;
}

// ════════════════════════════════════════════════════════════════════════════
// Import: GPU VA (метод B, AMD-only)
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromGpuVA(void* gpu_va, size_t buffer_size) {
  if (IsActive()) {
    Release();
  }

  if (!gpu_va) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromGpuVA: gpu_va is nullptr");
  }

  if (buffer_size == 0) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromGpuVA: buffer_size must be > 0");
  }

  // AMD unified address space: тот же GPU VA доступен из HIP
  hip_ptr_ = gpu_va;
  size_ = buffer_size;
  method_ = ZeroCopyMethod::AMD_GPU_VA;
  owns_memory_ = false;  // Память принадлежит OpenCL, не освобождаем

  DRVGPU_LOG_INFO("ZeroCopyBridge", "Imported via AMD GPU VA (ptr=" +
                  std::to_string(reinterpret_cast<uintptr_t>(gpu_va)) +
                  ", size=" + std::to_string(buffer_size) + " bytes)");
}

// ════════════════════════════════════════════════════════════════════════════
// Universal Import: автоопределение метода
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::ImportFromOpenCl(cl_mem cl_buffer, size_t buffer_size,
                                       cl_device_id cl_device) {
  if (!cl_buffer) {
    throw std::runtime_error("ZeroCopyBridge::ImportFromOpenCl: cl_buffer is nullptr");
  }

  // Метод B (приоритет): AMD GPU VA — zero overhead
  if (SupportsAmdGpuVA(cl_device)) {
    void* gpu_va = ExportClBufferToGpuVA(cl_buffer);
    if (gpu_va) {
      ImportFromGpuVA(gpu_va, buffer_size);
      DRVGPU_LOG_INFO("ZeroCopyBridge", "Using AMD GPU VA method (zero overhead)");
      return;
    }
    DRVGPU_LOG_WARNING("ZeroCopyBridge", "AMD GPU VA export failed, trying DMA-BUF...");
  }

  // Метод A (fallback): DMA-BUF
  if (SupportsDmaBufExport(cl_device)) {
    int fd = ExportClBufferToFd(cl_buffer);
    if (fd >= 0) {
      hipError_t err = ImportFromDmaBuf(fd, buffer_size);
      CheckHip(err, "ImportFromDmaBuf");
      DRVGPU_LOG_INFO("ZeroCopyBridge", "Using DMA-BUF method");
      return;
    }
    DRVGPU_LOG_WARNING("ZeroCopyBridge", "DMA-BUF export failed (fd=-1)");
  }

  throw std::runtime_error(
      "ZeroCopyBridge::ImportFromOpenCl: no ZeroCopy method available. "
      "Device does not support cl_khr_external_memory_dma_buf or CL_MEM_AMD_GPU_VA. "
      "Method: " + std::string(ZeroCopyMethodToString(
          DetectBestZeroCopyMethod(cl_device)))
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Release
// ════════════════════════════════════════════════════════════════════════════

void ZeroCopyBridge::Release() {
  if (owns_memory_ && ext_mem_) {
    (void)hipDestroyExternalMemory(ext_mem_);
    DRVGPU_LOG_INFO("ZeroCopyBridge", "Released external memory");
  }

  ext_mem_ = nullptr;
  hip_ptr_ = nullptr;
  size_ = 0;
  method_ = ZeroCopyMethod::NONE;
  owns_memory_ = false;
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
