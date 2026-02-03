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
 * 
 * Инициализирует члены класса значениями по умолчанию:
 * - device_index_ = -1 (устройство ещё не выбрано)
 * - initialized_ = false (не инициализирован)
 * - owns_resources_ = true (по умолчанию владеем ресурсами)
 * - context_/device_/queue_ = nullptr (дескрипторы OpenCL)
 */
OpenCLBackend::OpenCLBackend()
    : device_index_(-1)
    , initialized_(false)
    , owns_resources_(true)  // ✅ НОВОЕ: По умолчанию владеем ресурсами
    , context_(nullptr)
    , device_(nullptr)
    , queue_(nullptr) {
}

/**
 * @brief Деструктор - автоматическая очистка ресурсов
 * 
 * Вызывает Cleanup() для освобождения всех ресурсов.
 * Это обеспечивает RAII - ресурсы освобождаются даже при исключениях.
 */
OpenCLBackend::~OpenCLBackend() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Move конструктор и оператор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Move конструктор
 * 
 * Переносит ресурсы из другого объекта, оставляя тот в валидном но пустом состоянии.
 */
OpenCLBackend::OpenCLBackend(OpenCLBackend&& other) noexcept
    : device_index_(other.device_index_)
    , initialized_(other.initialized_)
    , owns_resources_(other.owns_resources_)  // ✅ НОВОЕ: Переносим флаг владения
    , memory_manager_(std::move(other.memory_manager_))
    , svm_capabilities_(std::move(other.svm_capabilities_))
    , context_(other.context_)
    , device_(other.device_)
    , queue_(other.queue_) {
    
    // Обнуляем источник
    other.device_index_ = -1;
    other.initialized_ = false;
    other.owns_resources_ = false;  // ✅ НОВОЕ: Источник больше не владеет
    other.context_ = nullptr;
    other.device_ = nullptr;
    other.queue_ = nullptr;
}

/**
 * @brief Move оператор присваивания
 * 
 * Очищает текущие ресурсы и переносит из other.
 */
OpenCLBackend& OpenCLBackend::operator=(OpenCLBackend&& other) noexcept {
    if (this != &other) {
        // Очищаем текущие ресурсы
        Cleanup();
        
        // Переносим данные
        device_index_ = other.device_index_;
        initialized_ = other.initialized_;
        owns_resources_ = other.owns_resources_;  // ✅ НОВОЕ: Переносим флаг
        memory_manager_ = std::move(other.memory_manager_);
        svm_capabilities_ = std::move(other.svm_capabilities_);
        context_ = other.context_;
        device_ = other.device_;
        queue_ = other.queue_;
        
        // Обнуляем источник
        other.device_index_ = -1;
        other.initialized_ = false;
        other.owns_resources_ = false;  // ✅ НОВОЕ
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
 * Процесс инициализации:
 * 1. Захватываем мьютекс для thread-safety
 * 2. Если уже инициализирован - очищаем старые ресурсы
 * 3. Сохраняем индекс устройства
 * 4. ✅ Устанавливаем owns_resources_ = true (создаём ресурсы сами)
 * 5. Инициализируем OpenCLCore (Singleton)
 * 6. Получаем нативные хэндлы из OpenCLCore
 * 7. СОЗДАЁМ COMMAND QUEUE
 * 8. Инициализируем SVM capabilities
 * 9. Создаём MemoryManager
 * 10. Устанавливаем флаг initialized_
 * 
 * @param device_index Индекс GPU устройства (0 = первая GPU)
 * 
 * @throws std::runtime_error если OpenCLCore не может быть инициализирован
 */
void OpenCLBackend::Initialize(int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Если уже инициализирован - переинициализируем
    if (initialized_) {
        Cleanup();
    }
    
    // Сохраняем индекс устройства
    device_index_ = device_index;
    
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ КРИТИЧЕСКИ ВАЖНО: Устанавливаем режим владения
    // ═══════════════════════════════════════════════════════════════════════
    // Мы создаём контекст сами → мы должны его освободить
    owns_resources_ = true;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 1: Инициализируем OpenCLCore (Singleton)
    // ═══════════════════════════════════════════════════════════════════════
    DeviceType dev_type = (device_index == 0) ? DeviceType::GPU : DeviceType::CPU;
    OpenCLCore::Initialize(dev_type);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 2: Получаем нативные хэндлы из OpenCLCore
    // ═══════════════════════════════════════════════════════════════════════
    OpenCLCore& core = OpenCLCore::GetInstance();
    context_ = core.GetContext();
    device_ = core.GetDevice();
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 3: СОЗДАЁМ COMMAND QUEUE
    // ═══════════════════════════════════════════════════════════════════════
    cl_int err;
    
    #ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    #else
        queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    #endif
    
    if (err != CL_SUCCESS || !queue_) {
        throw std::runtime_error(
            "OpenCLBackend::Initialize - Failed to create command queue. Error code: " + 
            std::to_string(err)
        );
    }
    
    DRVGPU_LOG_INFO("OpenCLBackend", "Command queue created (owning mode) ✅");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 4-5: Инициализируем SVM capabilities и MemoryManager
    // ═══════════════════════════════════════════════════════════════════════
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );
    
    memory_manager_ = std::make_unique<MemoryManager>(this);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Завершение
    // ═══════════════════════════════════════════════════════════════════════
    initialized_ = true;
    DRVGPU_LOG_INFO("OpenCLBackend", 
        "Initialized for device " + std::to_string(device_index) + " (owns_resources = true)");
}

/**
 * @brief Очистить все ресурсы бэкенда
 * 
 * ✅ КРИТИЧЕСКАЯ ЛОГИКА ВЛАДЕНИЯ:
 * - Если owns_resources_ = true → освобождает OpenCL ресурсы
 * - Если owns_resources_ = false → только обнуляет указатели
 * 
 * Порядок освобождения (обратный созданию):
 * 1. Сбрасываем unique_ptr'ы (MemoryManager, SVMCapabilities)
 * 2. Проверяем owns_resources_
 * 3. Если владеем → освобождаем queue, очищаем OpenCLCore
 * 4. Если не владеем → только обнуляем указатели
 * 5. Сбрасываем флаг initialized_
 */
void OpenCLBackend::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    DRVGPU_LOG_INFO("OpenCLBackend", 
        "Cleanup started (owns_resources = " + std::string(owns_resources_ ? "true" : "false") + ")");
    
    // ═══════════════════════════════════════════════════════════════════════
    // Освобождаем MemoryManager и SVM capabilities (всегда)
    // ═══════════════════════════════════════════════════════════════════════
    svm_capabilities_.reset();
    memory_manager_.reset();
    
    // ═══════════════════════════════════════════════════════════════════════
    // ✅ КЛЮЧЕВАЯ ЛОГИКА: Освобождаем OpenCL ресурсы ТОЛЬКО если владеем
    // ═══════════════════════════════════════════════════════════════════════
    
    if (owns_resources_) {
        // ───────────────────────────────────────────────────────────────────
        // OWNING MODE: Мы создали ресурсы → МЫ освобождаем
        // ───────────────────────────────────────────────────────────────────
        
        DRVGPU_LOG_INFO("OpenCLBackend", "Owning mode: releasing OpenCL resources");
        
        // Освобождаем command queue
        if (queue_) {
            clReleaseCommandQueue(queue_);
            queue_ = nullptr;
            DRVGPU_LOG_INFO("OpenCLBackend", "Command queue released ✅");
        }
        
        // Очищаем OpenCLCore (Singleton)
        OpenCLCore::Cleanup();
        
        context_ = nullptr;
        device_ = nullptr;
        
    } else {
        // ───────────────────────────────────────────────────────────────────
        // NON-OWNING MODE: Ресурсы пришли извне → НЕ освобождаем
        // ───────────────────────────────────────────────────────────────────
        
        DRVGPU_LOG_INFO("OpenCLBackend", 
            "Non-owning mode: NOT releasing external OpenCL resources ⚠️");
        
        // Просто обнуляем указатели
        queue_ = nullptr;
        context_ = nullptr;
        device_ = nullptr;
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
    if (!OpenCLCore::IsInitialized()) {
        return false;
    }
    return false;  // TODO: проверить cl_khr_fp64
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
    info.max_mem_alloc_size = core.GetGlobalMemorySize();
    info.max_compute_units = core.GetComputeUnits();
    info.max_work_group_size = core.GetMaxWorkGroupSize();
    info.supports_svm = core.IsSVMSupported();
    info.supports_double = SupportsDoublePrecision();
    info.supports_half = false;
    info.supports_unified_memory = SupportsSVM();
    
    return info;
}

} // namespace drv_gpu_lib
