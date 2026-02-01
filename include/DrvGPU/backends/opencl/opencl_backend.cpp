#include "opencl_backend.hpp"
#include <iostream>
#include <cstring>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

OpenCLBackend::OpenCLBackend() 
    : device_index_(-1)
    , initialized_(false)
    , context_(nullptr)
    , device_(nullptr)
    , queue_(nullptr) {
}

OpenCLBackend::~OpenCLBackend() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор
// ════════════════════════════════════════════════════════════════════════════

OpenCLBackend::OpenCLBackend(OpenCLBackend&& other) noexcept
    : device_index_(other.device_index_)
    , initialized_(other.initialized_)
    , memory_manager_(std::move(other.memory_manager_))
    , svm_capabilities_(std::move(other.svm_capabilities_))
    , context_(other.context_)
    , device_(other.device_)
    , queue_(other.queue_) {
    
    other.device_index_ = -1;
    other.initialized_ = false;
    other.context_ = nullptr;
    other.device_ = nullptr;
    other.queue_ = nullptr;
}

OpenCLBackend& OpenCLBackend::operator=(OpenCLBackend&& other) noexcept {
    if (this != &other) {
        Cleanup();
        
        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        memory_manager_ = std::move(other.memory_manager_);
        svm_capabilities_ = std::move(other.svm_capabilities_);
        context_ = other.context_;
        device_ = other.device_;
        queue_ = other.queue_;
        
        other.device_index_ = -1;
        other.initialized_ = false;
        other.context_ = nullptr;
        other.device_ = nullptr;
        other.queue_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Инициализация
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackend::Initialize(int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        Cleanup();
    }
    
    device_index_ = device_index;
    
    // Инициализируем OpenCLCore (Singleton)
    DeviceType dev_type = (device_index == 0) ? DeviceType::GPU : DeviceType::CPU;
    OpenCLCore::Initialize(dev_type);
    
    // Получаем ссылку на Singleton
    OpenCLCore& core = OpenCLCore::GetInstance();
    
    // Получаем нативные хэндлы
    context_ = core.GetContext();
    device_ = core.GetDevice();
    
    // Инициализируем SVM capabilities
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );
    
    // Инициализируем MemoryManager
    memory_manager_ = std::make_unique<MemoryManager>(this);
    
    initialized_ = true;
    
    std::cout << "[OpenCLBackend] Initialized for device index: " << device_index << "\n";
}

void OpenCLBackend::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Освобождаем ресурсы в обратном порядке
    svm_capabilities_.reset();
    memory_manager_.reset();
    
    // OpenCLCore - Singleton, очищается статическим методом
    OpenCLCore::Cleanup();
    
    context_ = nullptr;
    device_ = nullptr;
    queue_ = nullptr;
    device_index_ = -1;
    initialized_ = false;
    
    std::cout << "[OpenCLBackend] Cleaned up\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo OpenCLBackend::GetDeviceInfo() const {
    return QueryDeviceInfo();
}

std::string OpenCLBackend::GetDeviceName() const {
    if (!OpenCLCore::IsInitialized()) {
        return "Unknown";
    }
    return OpenCLCore::GetInstance().GetDeviceName();
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Нативные хэндлы
// ════════════════════════════════════════════════════════════════════════════

void* OpenCLBackend::GetNativeContext() const {
    return static_cast<void*>(context_);
}

void* OpenCLBackend::GetNativeDevice() const {
    return static_cast<void*>(device_);
}

void* OpenCLBackend::GetNativeQueue() const {
    return static_cast<void*>(queue_);
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Управление памятью
// ════════════════════════════════════════════════════════════════════════════

void* OpenCLBackend::Allocate(size_t size_bytes, unsigned int flags) {
    if (!context_) {
        return nullptr;
    }
    
    cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
    if (flags & 1) mem_flags |= CL_MEM_HOST_READ_ONLY;
    if (flags & 2) mem_flags |= CL_MEM_HOST_WRITE_ONLY;
    if (flags & 4) mem_flags |= CL_MEM_HOST_NO_ACCESS;
    
    cl_mem mem = clCreateBuffer(context_, mem_flags, size_bytes, nullptr, nullptr);
    if (!mem) {
        return nullptr;
    }
    
    return static_cast<void*>(mem);
}

void OpenCLBackend::Free(void* ptr) {
    if (ptr) {
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }
}

void OpenCLBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // Используем clEnqueueWriteBuffer
    cl_int err = clEnqueueWriteBuffer(queue_, static_cast<cl_mem>(dst), 
                                       CL_TRUE, 0, size_bytes, src, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCLBackend] MemcpyHostToDevice error: " << err << "\n";
    }
}

void OpenCLBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // Используем clEnqueueReadBuffer
    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    cl_int err = clEnqueueReadBuffer(queue_, src_mem, CL_TRUE, 0, size_bytes, dst, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCLBackend] MemcpyDeviceToHost error: " << err << "\n";
    }
}

void OpenCLBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // Используем clEnqueueCopyBuffer
    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    cl_mem dst_mem = static_cast<cl_mem>(dst);
    cl_int err = clEnqueueCopyBuffer(queue_, src_mem, dst_mem, 0, 0, size_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCLBackend] MemcpyDeviceToDevice error: " << err << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Синхронизация
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackend::Synchronize() {
    if (queue_) {
        clFinish(queue_);
    }
}

void OpenCLBackend::Flush() {
    if (queue_) {
        clFlush(queue_);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Возможности устройства
// ════════════════════════════════════════════════════════════════════════════

bool OpenCLBackend::SupportsSVM() const {
    return svm_capabilities_ && svm_capabilities_->HasAnySVM();
}

bool OpenCLBackend::SupportsDoublePrecision() const {
    if (!OpenCLCore::IsInitialized()) {
        return false;
    }
    // TODO: проверить расширения для double precision
    return false;
}

size_t OpenCLBackend::GetMaxWorkGroupSize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetMaxWorkGroupSize();
}

size_t OpenCLBackend::GetGlobalMemorySize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetGlobalMemorySize();
}

size_t OpenCLBackend::GetLocalMemorySize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetLocalMemorySize();
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для OpenCL методы
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore& OpenCLBackend::GetCore() {
    return OpenCLCore::GetInstance();
}

const OpenCLCore& OpenCLBackend::GetCore() const {
    return OpenCLCore::GetInstance();
}

MemoryManager& OpenCLBackend::GetMemoryManager() {
    return *memory_manager_;
}

const MemoryManager& OpenCLBackend::GetMemoryManager() const {
    return *memory_manager_;
}

const SVMCapabilities& OpenCLBackend::GetSVMCapabilities() const {
    static SVMCapabilities empty_caps;
    return svm_capabilities_ ? *svm_capabilities_ : empty_caps;
}

void OpenCLBackend::InitializeCommandQueuePool(size_t num_queues) {
    (void)num_queues;
    // TODO: реализовать если есть CommandQueuePool
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

void OpenCLBackend::InitializeOpenCLCore() {
    // OpenCLCore инициализируется через Singleton паттерн в Initialize()
}

void OpenCLBackend::InitializeMemoryManager() {
    // MemoryManager инициализируется в Initialize()
}

void OpenCLBackend::InitializeSVMCapabilities() {
    if (!device_) {
        svm_capabilities_ = std::make_unique<SVMCapabilities>();
        return;
    }
    
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );
}

GPUDeviceInfo OpenCLBackend::QueryDeviceInfo() const {
    GPUDeviceInfo info;
    
    if (!OpenCLCore::IsInitialized()) {
        return info;
    }
    
    const OpenCLCore& core = OpenCLCore::GetInstance();
    
    info.name = core.GetDeviceName();
    info.vendor = core.GetVendor();
    info.driver_version = core.GetDriverVersion();
    info.opencl_version = std::to_string(core.GetOpenCLVersionMajor()) + "." + 
                          std::to_string(core.GetOpenCLVersionMinor());
    info.device_index = device_index_;
    info.global_memory_size = core.GetGlobalMemorySize();
    info.local_memory_size = core.GetLocalMemorySize();
    info.max_compute_units = core.GetComputeUnits();
    info.max_work_group_size = core.GetMaxWorkGroupSize();
    info.supports_svm = core.IsSVMSupported();
    info.supports_double = SupportsDoublePrecision();
    info.supports_half = false;
    info.supports_unified_memory = SupportsSVM();
    
    return info;
}

} // namespace drv_gpu_lib
