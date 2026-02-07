#include "opencl_backend.hpp"
#include "../../common/logger.hpp"

#include <sstream>
#include <iomanip>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать OpenCLBackend (без инициализации)
 */
OpenCLBackend::OpenCLBackend()
    : device_index_(-1)
    , initialized_(false)
    , owns_resources_(true)
    , core_(nullptr)  // ✅ MULTI-GPU: Per-device core
    , context_(nullptr)
    , device_(nullptr)
    , queue_(nullptr) {
}

/**
 * @brief Деструктор - автоматическая очистка ресурсов
 */
OpenCLBackend::~OpenCLBackend() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор
// ════════════════════════════════════════════════════════════════════════════

OpenCLBackend::OpenCLBackend(OpenCLBackend&& other) noexcept
    : device_index_(other.device_index_)
    , initialized_(other.initialized_)
    , owns_resources_(other.owns_resources_)
    , core_(std::move(other.core_))  // ✅ MULTI-GPU: Move core
    , memory_manager_(std::move(other.memory_manager_))
    , svm_capabilities_(std::move(other.svm_capabilities_))
    , context_(other.context_)
    , device_(other.device_)
    , queue_(other.queue_) {

    // Обнуляем источник
    other.device_index_ = -1;
    other.initialized_ = false;
    other.owns_resources_ = false;
    other.context_ = nullptr;
    other.device_ = nullptr;
    other.queue_ = nullptr;
}

OpenCLBackend& OpenCLBackend::operator=(OpenCLBackend&& other) noexcept {
    if (this != &other) {
        Cleanup();

        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        owns_resources_ = other.owns_resources_;
        core_ = std::move(other.core_);  // ✅ MULTI-GPU: Move core
        memory_manager_ = std::move(other.memory_manager_);
        svm_capabilities_ = std::move(other.svm_capabilities_);
        context_ = other.context_;
        device_ = other.device_;
        queue_ = other.queue_;

        other.device_index_ = -1;
        other.initialized_ = false;
        other.owns_resources_ = false;
        other.context_ = nullptr;
        other.device_ = nullptr;
        other.queue_ = nullptr;
    }

    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Инициализация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать бэкенд для конкретного устройства
 *
 * ✅ MULTI-GPU: Теперь каждый backend имеет СВОЙ OpenCLCore!
 *
 * Процесс:
 * 1. Создаём OpenCLCore для device_index
 * 2. Инициализируем OpenCLCore (выбор устройства по индексу)
 * 3. Получаем context/device из OpenCLCore
 * 4. Создаём command queue для ЭТОГО устройства
 * 5. Инициализируем SVM capabilities и MemoryManager
 */
void OpenCLBackend::Initialize(int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        Cleanup();
    }

    device_index_ = device_index;
    owns_resources_ = true;

    // ═══════════════════════════════════════════════════════════════════════
    // ✅ MULTI-GPU: Создаём СОБСТВЕННЫЙ OpenCLCore для этого устройства
    // ═══════════════════════════════════════════════════════════════════════

    DRVGPU_LOG_INFO("OpenCLBackend", "Creating OpenCLCore for device " + std::to_string(device_index));

    core_ = std::make_unique<OpenCLCore>(device_index, DeviceType::GPU);
    core_->Initialize();

    // ═══════════════════════════════════════════════════════════════════════
    // Получаем нативные хэндлы из НАШЕГО OpenCLCore
    // ═══════════════════════════════════════════════════════════════════════

    context_ = core_->GetContext();
    device_ = core_->GetDevice();

    DRVGPU_LOG_INFO("OpenCLBackend", "Got context and device from OpenCLCore");

    // ═══════════════════════════════════════════════════════════════════════
    // Создаём COMMAND QUEUE для этого устройства
    // ═══════════════════════════════════════════════════════════════════════

    cl_int err;

    #ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    #else
        queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    #endif

    if (err != CL_SUCCESS || !queue_) {
        core_.reset();  // Очищаем OpenCLCore при ошибке
        throw std::runtime_error(
            "OpenCLBackend::Initialize - Failed to create command queue for device " +
            std::to_string(device_index) + ". Error code: " + std::to_string(err)
        );
    }

    DRVGPU_LOG_INFO("OpenCLBackend", "Command queue created for device " + std::to_string(device_index));

    // ═══════════════════════════════════════════════════════════════════════
    // SVM capabilities и MemoryManager
    // ═══════════════════════════════════════════════════════════════════════

    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );

    memory_manager_ = std::make_unique<MemoryManager>(this);

    initialized_ = true;

    DRVGPU_LOG_INFO("OpenCLBackend",
        "Initialized for device " + std::to_string(device_index) +
        " (" + core_->GetDeviceName() + ")");
}

/**
 * @brief Очистить все ресурсы бэкенда
 *
 * ✅ MULTI-GPU: Очищает СОБСТВЕННЫЙ OpenCLCore
 */
void OpenCLBackend::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    DRVGPU_LOG_INFO("OpenCLBackend",
        "Cleanup started for device " + std::to_string(device_index_) +
        " (owns_resources = " + std::string(owns_resources_ ? "true" : "false") + ")");

    // Освобождаем MemoryManager и SVM capabilities
    svm_capabilities_.reset();
    memory_manager_.reset();

    if (owns_resources_) {
        // ═══════════════════════════════════════════════════════════════════
        // OWNING MODE: Освобождаем ресурсы
        // ═══════════════════════════════════════════════════════════════════

        if (queue_) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
            DRVGPU_LOG_DEBUG("OpenCLBackend", "Command queue released");
        }

        // ✅ MULTI-GPU: Очищаем СВОЙ OpenCLCore
        core_.reset();

        context_ = nullptr;
        device_ = nullptr;

    } else {
        // NON-OWNING MODE: Просто обнуляем указатели
        DRVGPU_LOG_DEBUG("OpenCLBackend", "Non-owning mode: NOT releasing resources");

        queue_ = nullptr;
        context_ = nullptr;
        device_ = nullptr;
        core_.reset();
    }

    device_index_ = -1;
    initialized_ = false;

    DRVGPU_LOG_INFO("OpenCLBackend", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

GPUDeviceInfo OpenCLBackend::GetDeviceInfo() const {
    return QueryDeviceInfo();
}

std::string OpenCLBackend::GetDeviceName() const {
    if (!core_ || !core_->IsInitialized()) {
        return "Unknown";
    }
    return core_->GetDeviceName();
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
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyHostToDevice - Invalid parameters");
        return;
    }

    cl_int err = clEnqueueWriteBuffer(
        queue_,
        static_cast<cl_mem>(dst),
        CL_TRUE,
        0,
        size_bytes,
        src,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyHostToDevice error: " + std::to_string(err));
    }
}

void OpenCLBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }

    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));

    cl_int err = clEnqueueReadBuffer(
        queue_,
        src_mem,
        CL_TRUE,
        0,
        size_bytes,
        dst,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyDeviceToHost error: " + std::to_string(err));
    }
}

void OpenCLBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }

    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    cl_mem dst_mem = static_cast<cl_mem>(dst);

    cl_int err = clEnqueueCopyBuffer(
        queue_,
        src_mem,
        dst_mem,
        0,
        0,
        size_bytes,
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyDeviceToDevice error: " + std::to_string(err));
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
    if (!core_ || !core_->IsInitialized()) {
        return false;
    }
    return false;  // TODO: проверить cl_khr_fp64
}

size_t OpenCLBackend::GetMaxWorkGroupSize() const {
    if (!core_ || !core_->IsInitialized()) {
        return 0;
    }
    return core_->GetMaxWorkGroupSize();
}

size_t OpenCLBackend::GetGlobalMemorySize() const {
    if (!core_ || !core_->IsInitialized()) {
        return 0;
    }
    return core_->GetGlobalMemorySize();
}

size_t OpenCLBackend::GetLocalMemorySize() const {
    if (!core_ || !core_->IsInitialized()) {
        return 0;
    }
    return core_->GetLocalMemorySize();
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для OpenCL методы
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore& OpenCLBackend::GetCore() {
    if (!core_) {
        throw std::runtime_error("OpenCLBackend::GetCore - Core not initialized");
    }
    return *core_;
}

const OpenCLCore& OpenCLBackend::GetCore() const {
    if (!core_) {
        throw std::runtime_error("OpenCLBackend::GetCore - Core not initialized");
    }
    return *core_;
}

MemoryManager* OpenCLBackend::GetMemoryManager() {
    return memory_manager_.get();
}

const MemoryManager* OpenCLBackend::GetMemoryManager() const {
    return memory_manager_.get();
}

MemoryManager& OpenCLBackend::GetMemoryManagerRef() {
    return *memory_manager_;
}

const MemoryManager& OpenCLBackend::GetMemoryManagerRef() const {
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
    // Теперь инициализация в Initialize()
}

void OpenCLBackend::InitializeMemoryManager() {
    // Инициализация в Initialize()
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

    if (!core_ || !core_->IsInitialized()) {
        return info;
    }

    info.name = core_->GetDeviceName();
    info.vendor = core_->GetVendor();
    info.driver_version = core_->GetDriverVersion();
    info.opencl_version = std::to_string(core_->GetOpenCLVersionMajor()) + "." +
                          std::to_string(core_->GetOpenCLVersionMinor());
    info.device_index = device_index_;
    info.global_memory_size = core_->GetGlobalMemorySize();
    info.local_memory_size = core_->GetLocalMemorySize();
    info.max_mem_alloc_size = core_->GetGlobalMemorySize();
    info.max_compute_units = core_->GetComputeUnits();
    info.max_work_group_size = core_->GetMaxWorkGroupSize();
    info.supports_svm = core_->IsSVMSupported();
    info.supports_double = SupportsDoublePrecision();
    info.supports_half = false;
    info.supports_unified_memory = SupportsSVM();

    return info;
}

} // namespace drv_gpu_lib
