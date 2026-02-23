/**
 * @file hybrid_backend.cpp
 * @brief Реализация HybridBackend (OpenCL + ROCm)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "hybrid_backend.hpp"

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

HybridBackend::HybridBackend()
    : device_index_(-1)
    , initialized_(false)
    , owns_resources_(true)
    , opencl_(nullptr)
    , rocm_(nullptr) {
}

HybridBackend::~HybridBackend() {
  Cleanup();
}

HybridBackend::HybridBackend(HybridBackend&& other) noexcept
    : device_index_(other.device_index_)
    , initialized_(other.initialized_)
    , owns_resources_(other.owns_resources_)
    , opencl_(std::move(other.opencl_))
    , rocm_(std::move(other.rocm_)) {
  other.initialized_ = false;
  other.device_index_ = -1;
}

HybridBackend& HybridBackend::operator=(HybridBackend&& other) noexcept {
  if (this != &other) {
    Cleanup();
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    owns_resources_ = other.owns_resources_;
    opencl_ = std::move(other.opencl_);
    rocm_ = std::move(other.rocm_);
    other.initialized_ = false;
    other.device_index_ = -1;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Initialize / Cleanup
// ════════════════════════════════════════════════════════════════════════════

void HybridBackend::Initialize(int device_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    DRVGPU_LOG_WARNING("HybridBackend", "Already initialized");
    return;
  }

  device_index_ = device_index;

  // 1. Инициализация OpenCL sub-backend
  DRVGPU_LOG_INFO("HybridBackend", "Initializing OpenCL sub-backend (device " +
                  std::to_string(device_index) + ")...");
  opencl_ = std::make_unique<OpenCLBackend>();
  opencl_->Initialize(device_index);

  // 2. Инициализация ROCm sub-backend
  DRVGPU_LOG_INFO("HybridBackend", "Initializing ROCm sub-backend (device " +
                  std::to_string(device_index) + ")...");
  rocm_ = std::make_unique<ROCmBackend>();
  rocm_->Initialize(device_index);

  // 3. Логируем доступные методы ZeroCopy
  auto zc_method = GetBestZeroCopyMethod();
  DRVGPU_LOG_INFO("HybridBackend", "ZeroCopy method: " +
                  std::string(ZeroCopyMethodToString(zc_method)));

  initialized_ = true;
  DRVGPU_LOG_INFO("HybridBackend", "Initialized successfully (OpenCL + ROCm, device " +
                  std::to_string(device_index) + ")");
}

void HybridBackend::Cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (rocm_) {
    rocm_->Cleanup();
    rocm_.reset();
  }

  if (opencl_) {
    opencl_->Cleanup();
    opencl_.reset();
  }

  initialized_ = false;
  DRVGPU_LOG_INFO("HybridBackend", "Cleaned up");
}

// ════════════════════════════════════════════════════════════════════════════
// Device Info (от OpenCL)
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo HybridBackend::GetDeviceInfo() const {
  if (opencl_ && opencl_->IsInitialized()) {
    return opencl_->GetDeviceInfo();
  }
  return {};
}

std::string HybridBackend::GetDeviceName() const {
  if (opencl_ && opencl_->IsInitialized()) {
    return opencl_->GetDeviceName() + " [Hybrid: OpenCL+ROCm]";
  }
  return "HybridBackend (not initialized)";
}

// ════════════════════════════════════════════════════════════════════════════
// Native Handles (OpenCL primary)
// ════════════════════════════════════════════════════════════════════════════

void* HybridBackend::GetNativeContext() const {
  if (opencl_) return opencl_->GetNativeContext();
  return nullptr;
}

void* HybridBackend::GetNativeDevice() const {
  if (opencl_) return opencl_->GetNativeDevice();
  return nullptr;
}

void* HybridBackend::GetNativeQueue() const {
  if (opencl_) return opencl_->GetNativeQueue();
  return nullptr;
}

// ════════════════════════════════════════════════════════════════════════════
// Memory (OpenCL primary)
// ════════════════════════════════════════════════════════════════════════════

void* HybridBackend::Allocate(size_t size_bytes, unsigned int flags) {
  if (opencl_) return opencl_->Allocate(size_bytes, flags);
  throw std::runtime_error("HybridBackend::Allocate: OpenCL not initialized");
}

void HybridBackend::Free(void* ptr) {
  if (opencl_) {
    opencl_->Free(ptr);
  }
}

void HybridBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
  if (opencl_) {
    opencl_->MemcpyHostToDevice(dst, src, size_bytes);
  } else {
    throw std::runtime_error("HybridBackend::MemcpyHostToDevice: OpenCL not initialized");
  }
}

void HybridBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
  if (opencl_) {
    opencl_->MemcpyDeviceToHost(dst, src, size_bytes);
  } else {
    throw std::runtime_error("HybridBackend::MemcpyDeviceToHost: OpenCL not initialized");
  }
}

void HybridBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
  if (opencl_) {
    opencl_->MemcpyDeviceToDevice(dst, src, size_bytes);
  } else {
    throw std::runtime_error("HybridBackend::MemcpyDeviceToDevice: OpenCL not initialized");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Synchronization (оба backend)
// ════════════════════════════════════════════════════════════════════════════

void HybridBackend::Synchronize() {
  if (opencl_ && opencl_->IsInitialized()) {
    opencl_->Synchronize();
  }
  if (rocm_ && rocm_->IsInitialized()) {
    rocm_->Synchronize();
  }
}

void HybridBackend::Flush() {
  if (opencl_ && opencl_->IsInitialized()) {
    opencl_->Flush();
  }
  if (rocm_ && rocm_->IsInitialized()) {
    rocm_->Flush();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Capabilities (от OpenCL)
// ════════════════════════════════════════════════════════════════════════════

bool HybridBackend::SupportsSVM() const {
  if (opencl_) return opencl_->SupportsSVM();
  return false;
}

bool HybridBackend::SupportsDoublePrecision() const {
  if (opencl_) return opencl_->SupportsDoublePrecision();
  return false;
}

size_t HybridBackend::GetMaxWorkGroupSize() const {
  if (opencl_) return opencl_->GetMaxWorkGroupSize();
  return 0;
}

size_t HybridBackend::GetGlobalMemorySize() const {
  if (opencl_) return opencl_->GetGlobalMemorySize();
  return 0;
}

size_t HybridBackend::GetFreeMemorySize() const {
  if (opencl_) return opencl_->GetFreeMemorySize();
  return 0;
}

size_t HybridBackend::GetLocalMemorySize() const {
  if (opencl_) return opencl_->GetLocalMemorySize();
  return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// MemoryManager (от OpenCL)
// ════════════════════════════════════════════════════════════════════════════

MemoryManager* HybridBackend::GetMemoryManager() {
  if (opencl_) return opencl_->GetMemoryManager();
  return nullptr;
}

const MemoryManager* HybridBackend::GetMemoryManager() const {
  if (opencl_) return opencl_->GetMemoryManager();
  return nullptr;
}

// ════════════════════════════════════════════════════════════════════════════
// ZeroCopy
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<ZeroCopyBridge> HybridBackend::CreateZeroCopyBridge(
    cl_mem cl_buffer, size_t buffer_size) {

  if (!opencl_ || !opencl_->IsInitialized()) {
    throw std::runtime_error("HybridBackend::CreateZeroCopyBridge: OpenCL not initialized");
  }
  if (!rocm_ || !rocm_->IsInitialized()) {
    throw std::runtime_error("HybridBackend::CreateZeroCopyBridge: ROCm not initialized");
  }

  auto bridge = std::make_unique<ZeroCopyBridge>();

  // Получаем cl_device_id для проверки capabilities
  cl_device_id cl_device = static_cast<cl_device_id>(opencl_->GetNativeDevice());

  // Автоматический выбор метода
  bridge->ImportFromOpenCl(cl_buffer, buffer_size, cl_device);

  DRVGPU_LOG_INFO("HybridBackend", "ZeroCopy bridge created: " +
                  std::string(ZeroCopyMethodToString(bridge->GetMethod())) +
                  ", size=" + std::to_string(buffer_size) + " bytes");

  return bridge;
}

ZeroCopyMethod HybridBackend::GetBestZeroCopyMethod() const {
  if (!opencl_ || !opencl_->IsInitialized()) {
    return ZeroCopyMethod::NONE;
  }

  cl_device_id cl_device = static_cast<cl_device_id>(opencl_->GetNativeDevice());
  return DetectBestZeroCopyMethod(cl_device);
}

void HybridBackend::SyncBeforeZeroCopy() {
  // OpenCL должен завершить запись перед тем, как HIP прочитает
  if (opencl_ && opencl_->IsInitialized()) {
    opencl_->Synchronize();
  }
}

void HybridBackend::SyncAfterZeroCopy() {
  // HIP должен завершить работу перед тем, как OpenCL прочитает
  if (rocm_ && rocm_->IsInitialized()) {
    rocm_->Synchronize();
  }
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
