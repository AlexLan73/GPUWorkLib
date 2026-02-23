#if ENABLE_ROCM

#include "rocm_backend.hpp"
#include "../../config/gpu_config.hpp"

#include <sstream>
#include <iomanip>
#include <vector>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

ROCmBackend::ROCmBackend()
    : device_index_(-1),
      initialized_(false),
      owns_resources_(true),
      core_(nullptr),
      device_(0),
      stream_(nullptr) {
}

ROCmBackend::~ROCmBackend() {
  Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

ROCmBackend::ROCmBackend(ROCmBackend&& other) noexcept
    : device_index_(other.device_index_),
      initialized_(other.initialized_),
      owns_resources_(other.owns_resources_),
      core_(std::move(other.core_)),
      memory_manager_(std::move(other.memory_manager_)),
      device_(other.device_),
      stream_(other.stream_) {
  other.device_index_ = -1;
  other.initialized_ = false;
  other.owns_resources_ = false;
  other.device_ = 0;
  other.stream_ = nullptr;
}

ROCmBackend& ROCmBackend::operator=(ROCmBackend&& other) noexcept {
  if (this != &other) {
    Cleanup();
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    owns_resources_ = other.owns_resources_;
    core_ = std::move(other.core_);
    memory_manager_ = std::move(other.memory_manager_);
    device_ = other.device_;
    stream_ = other.stream_;

    other.device_index_ = -1;
    other.initialized_ = false;
    other.owns_resources_ = false;
    other.device_ = 0;
    other.stream_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

void ROCmBackend::Initialize(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    Cleanup();
  }

  device_index_ = device_index;
  owns_resources_ = true;

  DRVGPU_LOG_INFO_GPU(device_index, "ROCmBackend",
                      "Creating ROCmCore for device " + std::to_string(device_index));

  // Создаём СОБСТВЕННЫЙ ROCmCore для этого устройства
  core_ = std::make_unique<ROCmCore>(device_index);
  core_->Initialize();

  // Кешируем нативные хэндлы
  device_ = core_->GetDevice();
  stream_ = core_->GetStream();

  // Создаём MemoryManager
  memory_manager_ = std::make_unique<MemoryManager>(this);

  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmBackend",
                      "Initialized for device " + std::to_string(device_index) +
                      " (" + core_->GetDeviceName() + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка
// ════════════════════════════════════════════════════════════════════════════

void ROCmBackend::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return;
  }

  int gpu_id_for_log = device_index_;
  DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "ROCmBackend",
                      "Cleanup started for device " + std::to_string(device_index_) +
                      " (owns_resources = " + std::string(owns_resources_ ? "true" : "false") + ")");

  // Освобождаем MemoryManager
  memory_manager_.reset();

  if (owns_resources_) {
    // Очищаем ROCmCore (уничтожает stream и пр.)
    core_.reset();
  } else {
    // Non-owning: просто обнуляем
    core_.reset();
  }

  device_ = 0;
  stream_ = nullptr;
  device_index_ = -1;
  initialized_ = false;

  DRVGPU_LOG_INFO_GPU(gpu_id_for_log, "ROCmBackend", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo ROCmBackend::GetDeviceInfo() const {
  return QueryDeviceInfo();
}

std::string ROCmBackend::GetDeviceName() const {
  if (!core_ || !core_->IsInitialized()) {
    return "Unknown";
  }
  return core_->GetDeviceName();
}

// ════════════════════════════════════════════════════════════════════════════
// Нативные хэндлы
// ════════════════════════════════════════════════════════════════════════════

void* ROCmBackend::GetNativeContext() const {
  // HIP не имеет отдельного context как OpenCL
  // Возвращаем nullptr — context управляется неявно через hipSetDevice
  return nullptr;
}

void* ROCmBackend::GetNativeDevice() const {
  return reinterpret_cast<void*>(static_cast<intptr_t>(device_));
}

void* ROCmBackend::GetNativeQueue() const {
  return static_cast<void*>(stream_);
}

// ════════════════════════════════════════════════════════════════════════════
// Управление памятью
// ════════════════════════════════════════════════════════════════════════════

void* ROCmBackend::Allocate(size_t size_bytes, unsigned int /*flags*/) {
  if (!initialized_) {
    return nullptr;
  }

  void* device_ptr = nullptr;
  hipError_t err = hipMalloc(&device_ptr, size_bytes);

  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "hipMalloc failed: " + std::string(hipGetErrorString(err)) +
                         " (requested " + std::to_string(size_bytes) + " bytes)");
    return nullptr;
  }

  return device_ptr;
}

void ROCmBackend::Free(void* ptr) {
  if (ptr) {
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) {
      DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                           "hipFree failed: " + std::string(hipGetErrorString(err)));
    }
  }
}

void ROCmBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyHostToDevice - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyHtoDAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyHostToDevice failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  // Синхронизируем для совместимости с синхронным API OpenCL backend
  hipStreamSynchronize(stream_);
}

void ROCmBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToHost - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyDtoHAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToHost failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  hipStreamSynchronize(stream_);
}

void ROCmBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
  if (!dst || !src || !initialized_) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToDevice - Invalid parameters");
    return;
  }

  hipError_t err = hipMemcpyDtoDAsync(dst, const_cast<void*>(src), size_bytes, stream_);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                         "MemcpyDeviceToDevice failed: " + std::string(hipGetErrorString(err)));
    return;
  }

  hipStreamSynchronize(stream_);
}

// ════════════════════════════════════════════════════════════════════════════
// Синхронизация
// ════════════════════════════════════════════════════════════════════════════

void ROCmBackend::Synchronize() {
  if (stream_) {
    hipError_t err = hipStreamSynchronize(stream_);
    if (err != hipSuccess) {
      DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
                           "Synchronize failed: " + std::string(hipGetErrorString(err)));
    }
  }
}

void ROCmBackend::Flush() {
  if (stream_) {
    // HIP: hipStreamQuery возвращает hipSuccess если все операции завершены,
    // или hipErrorNotReady если ещё в процессе — non-blocking check
    hipStreamQuery(stream_);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Возможности устройства
// ════════════════════════════════════════════════════════════════════════════

bool ROCmBackend::SupportsDoublePrecision() const {
  if (!core_ || !core_->IsInitialized()) return false;
  return core_->SupportsDoublePrecision();
}

size_t ROCmBackend::GetMaxWorkGroupSize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetMaxWorkGroupSize();
}

size_t ROCmBackend::GetGlobalMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetGlobalMemorySize();
}

size_t ROCmBackend::GetFreeMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetFreeMemorySize();
}

size_t ROCmBackend::GetLocalMemorySize() const {
  if (!core_ || !core_->IsInitialized()) return 0;
  return core_->GetLocalMemorySize();
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для ROCm методы
// ════════════════════════════════════════════════════════════════════════════

ROCmCore& ROCmBackend::GetCore() {
  if (!core_) {
    throw std::runtime_error("ROCmBackend::GetCore - Core not initialized");
  }
  return *core_;
}

const ROCmCore& ROCmBackend::GetCore() const {
  if (!core_) {
    throw std::runtime_error("ROCmBackend::GetCore - Core not initialized");
  }
  return *core_;
}

MemoryManager* ROCmBackend::GetMemoryManager() {
  return memory_manager_.get();
}

const MemoryManager* ROCmBackend::GetMemoryManager() const {
  return memory_manager_.get();
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo ROCmBackend::QueryDeviceInfo() const {
  GPUDeviceInfo info{};

  if (!core_ || !core_->IsInitialized()) {
    return info;
  }

  info.name = core_->GetDeviceName();
  info.vendor = core_->GetVendor();
  info.driver_version = core_->GetArchName();
  info.opencl_version = "N/A (ROCm/HIP)";
  info.device_index = device_index_;
  info.global_memory_size = core_->GetGlobalMemorySize();
  info.local_memory_size = core_->GetLocalMemorySize();
  info.max_mem_alloc_size = core_->GetGlobalMemorySize();  // HIP: no separate limit
  info.max_compute_units = static_cast<size_t>(core_->GetComputeUnits());
  info.max_work_group_size = core_->GetMaxWorkGroupSize();
  info.max_clock_frequency = core_->GetMaxClockFrequency();
  info.supports_svm = false;
  info.supports_double = core_->SupportsDoublePrecision();
  info.supports_half = false;
  info.supports_unified_memory = false;

  return info;
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
