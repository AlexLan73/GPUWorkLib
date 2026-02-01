#include "drv_gpu.hpp"
#include "memory/memory_manager.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_core.hpp"
#include <iostream>
#include <sstream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// MemoryManager Implementation
// ════════════════════════════════════════════════════════════════════════════

MemoryManager::MemoryManager(IBackend* backend)
    : backend_(backend),
      total_allocations_(0),
      total_frees_(0),
      current_allocations_(0),
      total_bytes_allocated_(0),
      peak_bytes_allocated_(0) {
}

MemoryManager::~MemoryManager() {
    Cleanup();
}

MemoryManager::MemoryManager(MemoryManager&& other) noexcept
    : backend_(other.backend_),
      total_allocations_(other.total_allocations_),
      total_frees_(other.total_frees_),
      current_allocations_(other.current_allocations_),
      total_bytes_allocated_(other.total_bytes_allocated_),
      peak_bytes_allocated_(other.peak_bytes_allocated_) {
    other.backend_ = nullptr;
}

MemoryManager& MemoryManager::operator=(MemoryManager&& other) noexcept {
    if (this != &other) {
        backend_ = other.backend_;
        total_allocations_ = other.total_allocations_;
        total_frees_ = other.total_frees_;
        current_allocations_ = other.current_allocations_;
        total_bytes_allocated_ = other.total_bytes_allocated_;
        peak_bytes_allocated_ = other.peak_bytes_allocated_;
        other.backend_ = nullptr;
    }
    return *this;
}

void* MemoryManager::Allocate(size_t size_bytes, unsigned int flags) {
    if (!backend_) {
        throw std::runtime_error("MemoryManager: backend is null");
    }
    void* ptr = backend_->Allocate(size_bytes, flags);
    if (ptr) {
        TrackAllocation(size_bytes);
    }
    return ptr;
}

void MemoryManager::Free(void* ptr) {
    [[maybe_unused]] void* p = ptr;
    // Note: actual deallocation happens via backend
    // We track the deallocation when buffer is destroyed
}

size_t MemoryManager::GetAllocationCount() const {
    return current_allocations_;
}

size_t MemoryManager::GetTotalAllocatedBytes() const {
    return total_bytes_allocated_;
}

void MemoryManager::PrintStatistics() const {
    const char separator = static_cast<char>(205);  // ═
    std::cout << "\n" << std::string(50, separator) << "\n";
    std::cout << "Memory Statistics\n";
    std::cout << std::string(50, separator) << "\n";
    std::cout << "Total Allocations: " << total_allocations_ << "\n";
    std::cout << "Total Frees: " << total_frees_ << "\n";
    std::cout << "Current Allocations: " << current_allocations_ << "\n";
    std::cout << "Total Bytes Allocated: " << total_bytes_allocated_ << "\n";
    std::cout << "Peak Bytes Allocated: " << peak_bytes_allocated_ << "\n";
    std::cout << std::string(50, separator) << "\n\n";
}

std::string MemoryManager::GetStatistics() const {
    std::ostringstream oss;
    oss << "Memory Statistics:\n";
    oss << "  Total Allocations: " << total_allocations_ << "\n";
    oss << "  Current Allocations: " << current_allocations_ << "\n";
    oss << "  Total Bytes: " << total_bytes_allocated_ << "\n";
    oss << "  Peak Bytes: " << peak_bytes_allocated_ << "\n";
    return oss.str();
}

void MemoryManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_allocations_ = 0;
    total_frees_ = 0;
    current_allocations_ = 0;
    total_bytes_allocated_ = 0;
    peak_bytes_allocated_ = 0;
}

void MemoryManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_allocations_ = 0;
    total_bytes_allocated_ = 0;
}

void MemoryManager::TrackAllocation(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_allocations_++;
    current_allocations_++;
    total_bytes_allocated_ += size_bytes;
    if (current_allocations_ > peak_bytes_allocated_) {
        peak_bytes_allocated_ = current_allocations_;
    }
}

void MemoryManager::TrackFree([[maybe_unused]] size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_frees_++;
    if (current_allocations_ > 0) {
        current_allocations_--;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DrvGPU Implementation
// ════════════════════════════════════════════════════════════════════════════

DrvGPU::DrvGPU(BackendType backend_type, int device_index)
    : backend_type_(backend_type),
      device_index_(device_index),
      initialized_(false),
      backend_(nullptr),
      memory_manager_(nullptr),
      module_registry_(nullptr) {
    CreateBackend();
    InitializeSubsystems();
}

DrvGPU::~DrvGPU() {
    Cleanup();
}

DrvGPU::DrvGPU(DrvGPU&& other) noexcept
    : backend_type_(other.backend_type_),
      device_index_(other.device_index_),
      initialized_(other.initialized_),
      backend_(std::move(other.backend_)),
      memory_manager_(std::move(other.memory_manager_)),
      module_registry_(std::move(other.module_registry_)) {
    other.initialized_ = false;
}

DrvGPU& DrvGPU::operator=(DrvGPU&& other) noexcept {
    if (this != &other) {
        Cleanup();
        backend_type_ = other.backend_type_;
        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        backend_ = std::move(other.backend_);
        memory_manager_ = std::move(other.memory_manager_);
        module_registry_ = std::move(other.module_registry_);
        other.initialized_ = false;
    }
    return *this;
}

void DrvGPU::CreateBackend() {
    switch (backend_type_) {
        case BackendType::OPENCL:
            backend_ = std::make_unique<OpenCLBackend>();
            break;
        case BackendType::ROCm:
            // CUDA backend would be implemented here
            throw std::runtime_error("ROCm backend not yet implemented");
        case BackendType::OPENCLandROCm:
            // Vulkan backend would be implemented here
            throw std::runtime_error("OPENCLandROCm backend not yet implemented");
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

void DrvGPU::InitializeSubsystems() {
    memory_manager_ = std::make_unique<MemoryManager>(backend_.get());
    module_registry_ = std::make_unique<ModuleRegistry>();
}

void DrvGPU::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        std::cout << "[WARNING] DrvGPU already initialized\n";
        return;
    }
    
    if (!backend_) {
        throw std::runtime_error("DrvGPU: backend is null");
    }
    
    backend_->Initialize(device_index_);
    initialized_ = true;
    
    std::cout << "[OK] DrvGPU initialized successfully\n";
}

void DrvGPU::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (memory_manager_) {
        memory_manager_->Cleanup();
    }
    
    if (module_registry_) {
        module_registry_->Clear();
    }
    
    if (backend_) {
        backend_->Cleanup();
    }
    
    memory_manager_.reset();
    module_registry_.reset();
    backend_.reset();
    
    initialized_ = false;
    std::cout << "[OK] DrvGPU cleaned up\n";
}

GPUDeviceInfo DrvGPU::GetDeviceInfo() const {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    return backend_->GetDeviceInfo();
}

std::string DrvGPU::GetDeviceName() const {
    if (!initialized_ || !backend_) {
        return "Unknown";
    }
    auto info = backend_->GetDeviceInfo();
    return info.name;
}

void DrvGPU::PrintDeviceInfo() const {
    if (!initialized_ || !backend_) {
        std::cout << "Device not initialized\n";
        return;
    }
    auto info = backend_->GetDeviceInfo();
    std::cout << "Device Info:\n";
    std::cout << "  Name: " << info.name << "\n";
    std::cout << "  Vendor: " << info.vendor << "\n";
    std::cout << "  Global Memory: " << info.GetGlobalMemoryGB() << " GB\n";
}

MemoryManager& DrvGPU::GetMemoryManager() {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

const MemoryManager& DrvGPU::GetMemoryManager() const {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

ModuleRegistry& DrvGPU::GetModuleRegistry() {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

const ModuleRegistry& DrvGPU::GetModuleRegistry() const {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

IBackend& DrvGPU::GetBackend() {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

const IBackend& DrvGPU::GetBackend() const {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

void DrvGPU::Synchronize() {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    backend_->Synchronize();
}

void DrvGPU::Flush() {
    if (!initialized_ || !backend_) {
        return;
    }
    backend_->Flush();
}

void DrvGPU::PrintStatistics() const {
    const char separator = static_cast<char>(205);  // ═
    std::cout << "\n" << std::string(50, separator) << "\n";
    std::cout << "DrvGPU Statistics\n";
    std::cout << std::string(50, separator) << "\n";
    std::cout << "Device Index: " << device_index_ << "\n";
    std::cout << "Backend Type: " << static_cast<int>(backend_type_) << "\n";
    std::cout << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    
    if (memory_manager_) {
        memory_manager_->PrintStatistics();
    }
    
    std::cout << std::string(50, separator) << "\n\n";
}

std::string DrvGPU::GetStatistics() const {
    std::ostringstream oss;
    oss << "DrvGPU Statistics:\n";
    oss << "  Device Index: " << device_index_ << "\n";
    oss << "  Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    if (memory_manager_) {
        oss << memory_manager_->GetStatistics();
    }
    return oss.str();
}

void DrvGPU::ResetStatistics() {
    if (memory_manager_) {
        memory_manager_->ResetStatistics();
    }
}

} // namespace drv_gpu_lib
