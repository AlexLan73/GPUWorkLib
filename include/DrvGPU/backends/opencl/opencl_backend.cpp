#include "opencl_backend.hpp"
#include "../../common/logger.hpp"
#include <iostream>
#include <cstring>

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
 * - context_/device_/queue_ = nullptr (дескрипторы OpenCL)
 */
OpenCLBackend::OpenCLBackend() 
    : device_index_(-1)
    , initialized_(false)
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
    , memory_manager_(std::move(other.memory_manager_))
    , svm_capabilities_(std::move(other.svm_capabilities_))
    , context_(other.context_)
    , device_(other.device_)
    , queue_(other.queue_) {
    
    // Обнуляем источник
    other.device_index_ = -1;
    other.initialized_ = false;
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
        memory_manager_ = std::move(other.memory_manager_);
        svm_capabilities_ = std::move(other.svm_capabilities_);
        context_ = other.context_;
        device_ = other.device_;
        queue_ = other.queue_;
        
        // Обнуляем источник
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

/**
 * @brief Инициализировать бэкенд для конкретного устройства
 * 
 * Процесс инициализации:
 * 1. Захватываем мьютекс для thread-safety
 * 2. Если уже инициализирован - очищаем старые ресурсы
 * 3. Сохраняем индекс устройства
 * 4. Инициализируем OpenCLCore (Singleton)
 * 5. Получаем нативные хэндлы из OpenCLCore
 * 6. Инициализируем SVM capabilities
 * 7. Создаём MemoryManager
 * 8. Устанавливаем флаг initialized_
 * 
 * @param device_index Индекс GPU устройства (0 = первая GPU)
 * 
 * @code
 * // Пример использования:
 * OpenCLBackend backend;
 * 
 * // Инициализация для первой GPU
 * backend.Initialize(0);
 * 
 * // Теперь можно использовать
 * auto info = backend.GetDeviceInfo();
 * @endcode
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
    // Шаг 1: Инициализируем OpenCLCore (Singleton)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Выбираем тип устройства: device 0 = GPU, остальные = CPU
    // TODO: добавить поддержку нескольких GPU устройств
    DeviceType dev_type = (device_index == 0) ? DeviceType::GPU : DeviceType::CPU;
    OpenCLCore::Initialize(dev_type);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 2: Получаем нативные хэндлы из OpenCLCore
    // ═══════════════════════════════════════════════════════════════════════
    
    // OpenCLCore - Singleton, содержит контекст и устройство
    OpenCLCore& core = OpenCLCore::GetInstance();
    
    // Получаем нативные дескрипторы OpenCL
    context_ = core.GetContext();
    device_ = core.GetDevice();
    // queue_ остаётся nullptr - используем DefaultQueue из OpenCLCore если нужен
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 3: Инициализируем SVM capabilities
    // ═══════════════════════════════════════════════════════════════════════
    
    // SVMCapabilities определяет, какие типы SVM поддерживаются
    // SVM (Shared Virtual Memory) - общая память между CPU и GPU
    svm_capabilities_ = std::make_unique<SVMCapabilities>(
        SVMCapabilities::Query(device_)
    );
    
    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 4: Инициализируем MemoryManager
    // ═══════════════════════════════════════════════════════════════════════
    
    // MemoryManager управляет выделением и освобождением GPU памяти
    memory_manager_ = std::make_unique<MemoryManager>(this);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Завершение
    // ═══════════════════════════════════════════════════════════════════════
    
    initialized_ = true;
    
    DRVGPU_LOG_INFO("OpenCLBackend", "Initialized for device index: " + std::to_string(device_index));
}

/**
 * @brief Очистить все ресурсы бэкенда
 * 
 * Порядок освобождения (обратный созданию):
 * 1. Сбрасываем unique_ptr'ы (освобождают MemoryManager, SVMCapabilities)
 * 2. Очищаем OpenCLCore (Singleton)
 * 3. Обнуляем дескрипторы
 * 4. Сбрасываем флаг initialized_
 */
void OpenCLBackend::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Если не инициализирован - ничего не делаем
    if (!initialized_) {
        return;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Освобождаем ресурсы в обратном порядке
    // ═══════════════════════════════════════════════════════════════════════
    
    // Сначала: MemoryManager и SVM capabilities
    svm_capabilities_.reset();
    memory_manager_.reset();
    
    // Затем: OpenCLCore (Singleton)
    // Примечание: очищаем только если это последний пользователь
    // TODO: добавить счётчик ссылок на OpenCLCore
    OpenCLCore::Cleanup();
    
    // Обнуляем дескрипторы
    context_ = nullptr;
    device_ = nullptr;
    queue_ = nullptr;
    device_index_ = -1;
    initialized_ = false;
    
    DRVGPU_LOG_INFO("OpenCLBackend", "Cleaned up");
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Информация об устройстве
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить информацию об устройстве
 * @return GPUDeviceInfo с информацией об устройстве
 * 
 * @code
 * auto info = backend.GetDeviceInfo();
 * std::cout << "Device: " << info.name << std::endl;
 * std::cout << "Vendor: " << info.vendor << std::endl;
 * std::cout << "Memory: " << info.GetGlobalMemoryGB() << " GB" << std::endl;
 * @endcode
 */
GPUDeviceInfo OpenCLBackend::GetDeviceInfo() const {
    return QueryDeviceInfo();
}

/**
 * @brief Получить название устройства
 * @return Строка с названием или "Unknown" если не инициализирован
 */
std::string OpenCLBackend::GetDeviceName() const {
    if (!OpenCLCore::IsInitialized()) {
        return "Unknown";
    }
    return OpenCLCore::GetInstance().GetDeviceName();
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Нативные хэндлы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить нативный контекст OpenCL
 * @return void* на cl_context (можно кастить к cl_context)
 */
void* OpenCLBackend::GetNativeContext() const {
    return static_cast<void*>(context_);
}

/**
 * @brief Получить нативное устройство OpenCL
 * @return void* на cl_device_id (можно кастить к cl_device_id)
 */
void* OpenCLBackend::GetNativeDevice() const {
    return static_cast<void*>(device_);
}

/**
 * @brief Получить нативную очередь команд
 * @return void* на cl_command_queue (можно кастить к cl_command_queue)
 */
void* OpenCLBackend::GetNativeQueue() const {
    return static_cast<void*>(queue_);
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Управление памятью
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделить память на GPU
 * 
 * Создаёт буфер в глобальной памяти устройства.
 * 
 * @param size_bytes Размер в байтах
 * @param flags Флаги выделения (битовая маска):
 *              0 = CL_MEM_READ_WRITE (по умолчанию)
 *              1 = CL_MEM_HOST_READ_ONLY
 *              2 = CL_MEM_HOST_WRITE_ONLY
 *              4 = CL_MEM_HOST_NO_ACCESS
 * @return Указатель на cl_mem или nullptr при ошибке
 * 
 * @code
 * // Простое выделение (READ_WRITE)
 * void* buf = backend->Allocate(1024 * sizeof(float));
 * 
 * // Выделение с флагами
 * void* input = backend->Allocate(size, 1);   // HOST_READ_ONLY
 * void* output = backend->Allocate(size, 2);  // HOST_WRITE_ONLY
 * @endcode
 */
void* OpenCLBackend::Allocate(size_t size_bytes, unsigned int flags) {
    if (!context_) {
        return nullptr;
    }
    
    // Формируем флаги cl_mem
    cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
    
    // Битовые флаги для часто используемых режимов доступа с хоста
    if (flags & 1) mem_flags |= CL_MEM_HOST_READ_ONLY;
    if (flags & 2) mem_flags |= CL_MEM_HOST_WRITE_ONLY;
    if (flags & 4) mem_flags |= CL_MEM_HOST_NO_ACCESS;
    
    // Создаём буфер
    // clCreateBuffer - создаёт буфер в глобальной памяти
    // Параметры:
    //   context - контекст OpenCL
    //   flags - флаги режима доступа
    //   size - размер буфера в байтах
    //   host_ptr - указатель на хост-память (nullptr = выделить новую)
    //   errcode_ret - код ошибки
    cl_mem mem = clCreateBuffer(context_, mem_flags, size_bytes, nullptr, nullptr);
    
    if (!mem) {
        return nullptr;
    }
    
    // Возвращаем как void* (внешний код кастит к cl_mem)
    return static_cast<void*>(mem);
}

/**
 * @brief Освободить память GPU
 * @param ptr Указатель на cl_mem (полученный из Allocate)
 */
void OpenCLBackend::Free(void* ptr) {
    if (ptr) {
        // clReleaseMemObject - освобождает буфер памяти
        // Уменьшает счётчик ссылок, удаляет если 0
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }
}

/**
 * @brief Копировать данные Host -> Device
 * 
 * Синхронное копирование (блокирующее).
 * Копирует данные из памяти хоста в буфер GPU.
 * 
 * @param dst Указатель на буфер GPU (cl_mem)
 * @param src Указатель на данные хоста
 * @param size_bytes Размер данных в байтах
 * 
 * @code
 * std::vector<float> data(1024, 1.0f);
 * void* gpu_buffer = backend->Allocate(sizeof(data));
 * backend->MemcpyHostToDevice(gpu_buffer, data.data(), sizeof(data));
 * @endcode
 */
void OpenCLBackend::MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) {
    // Проверяем валидность параметров
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // clEnqueueWriteBuffer - записывает данные в буфер
    // Параметры:
    //   command_queue - очередь команд
    //   buffer - буфер назначения
    //   blocking_write - CL_TRUE = синхронно (ждём завершения)
    //   offset - смещение в буфере (0 = начало)
    //   size - размер данных
    //   ptr - указатель на данные хоста
    //   num_events_in_wait_list = 0 (без зависимостей)
    //   event_wait_list = nullptr
    //   event = nullptr (не возвращаем event)
    cl_int err = clEnqueueWriteBuffer(queue_, static_cast<cl_mem>(dst), 
                                       CL_TRUE, 0, size_bytes, src, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyHostToDevice error: " + std::to_string(err));
    }
}

/**
 * @brief Копировать данные Device -> Host
 * 
 * Синхронное копирование (блокирующее).
 * Копирует данные из буфера GPU в память хоста.
 * 
 * @param dst Указатель на буфер хоста
 * @param src Указатель на буфер GPU (cl_mem)
 * @param size_bytes Размер данных в байтах
 */
void OpenCLBackend::MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // Приводим const void* к cl_mem (OpenCL требует не-const для source)
    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    
    // clEnqueueReadBuffer - читает данные из буфера
    cl_int err = clEnqueueReadBuffer(queue_, src_mem, CL_TRUE, 0, size_bytes, dst, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyDeviceToHost error: " + std::to_string(err));
    }
}

/**
 * @brief Копировать данные Device -> Device
 * 
 * Копирует данные между двумя буферами GPU.
 * Это наиболее эффективный способ копирования (не затрагивает память хоста).
 * 
 * @param dst Буфер назначения (GPU)
 * @param src Буфер источник (GPU)
 * @param size_bytes Размер данных
 */
void OpenCLBackend::MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) {
    if (!context_ || !queue_ || !dst || !src) {
        return;
    }
    
    // Приводим типы для OpenCL API
    cl_mem src_mem = static_cast<cl_mem>(const_cast<void*>(src));
    cl_mem dst_mem = static_cast<cl_mem>(dst);
    
    // clEnqueueCopyBuffer - копирует между буферами
    // Быстрее чем Read+Write, т.к. данные не покидают GPU
    cl_int err = clEnqueueCopyBuffer(queue_, src_mem, dst_mem, 0, 0, size_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        DRVGPU_LOG_ERROR("OpenCLBackend", "MemcpyDeviceToDevice error: " + std::to_string(err));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Синхронизация
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Синхронизировать (ждать завершения всех операций)
 * 
 * Блокирует до тех пор, пока все команды в очереди не выполнятся.
 * 
 * @code
 * // Асинхронная отправка команды
 * clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
 * 
 * // Ожидание завершения
 * backend->Synchronize();
 * 
 * // Теперь результаты гарантированно готовы
 * @endcode
 */
void OpenCLBackend::Synchronize() {
    if (queue_) {
        // clFinish - ожидает завершения всех команд в очереди
        clFinish(queue_);
    }
}

/**
 * @brief Flush команд (без ожидания)
 * 
 * Отправляет команды в очередь, но не ждёт их выполнения.
 * Полезно для асинхронной работы.
 * 
 * @note В отличие от Synchronize(), не блокирует выполнение
 */
void OpenCLBackend::Flush() {
    if (queue_) {
        // clFlush - отправляет команды в очередь
        // Возвращает немедленно, не дожидаясь выполнения
        clFlush(queue_);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация IBackend: Возможности устройства
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверить поддержку SVM
 */
bool OpenCLBackend::SupportsSVM() const {
    return svm_capabilities_ && svm_capabilities_->HasAnySVM();
}

/**
 * @brief Проверить поддержку double precision
 */
bool OpenCLBackend::SupportsDoublePrecision() const {
    if (!OpenCLCore::IsInitialized()) {
        return false;
    }
    // TODO: проверить расширения для double precision (cl_khr_fp64)
    return false;
}

/**
 * @brief Получить максимальный размер work group
 */
size_t OpenCLBackend::GetMaxWorkGroupSize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetMaxWorkGroupSize();
}

/**
 * @brief Получить размер глобальной памяти
 */
size_t OpenCLBackend::GetGlobalMemorySize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetGlobalMemorySize();
}

/**
 * @brief Получить размер локальной памяти
 */
size_t OpenCLBackend::GetLocalMemorySize() const {
    if (!OpenCLCore::IsInitialized()) {
        return 0;
    }
    return OpenCLCore::GetInstance().GetLocalMemorySize();
}

// ════════════════════════════════════════════════════════════════════════════
// Специфичные для OpenCL методы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить ссылку на OpenCLCore
 */
OpenCLCore& OpenCLBackend::GetCore() {
    return OpenCLCore::GetInstance();
}

const OpenCLCore& OpenCLBackend::GetCore() const {
    return OpenCLCore::GetInstance();
}

/**
 * @brief Получить ссылку на MemoryManager
 */
MemoryManager& OpenCLBackend::GetMemoryManager() {
    return *memory_manager_;
}

const MemoryManager& OpenCLBackend::GetMemoryManager() const {
    return *memory_manager_;
}

/**
 * @brief Получить SVM capabilities
 */
const SVMCapabilities& OpenCLBackend::GetSVMCapabilities() const {
    static SVMCapabilities empty_caps;
    return svm_capabilities_ ? *svm_capabilities_ : empty_caps;
}

/**
 * @brief Инициализировать CommandQueuePool
 */
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

/**
 * @brief Запросить информацию об устройстве из OpenCLCore
 * 
 * Собирает всю информацию из OpenCLCore и упаковывает в GPUDeviceInfo.
 */
GPUDeviceInfo OpenCLBackend::QueryDeviceInfo() const {
    GPUDeviceInfo info;
    
    // Проверяем инициализацию
    if (!OpenCLCore::IsInitialized()) {
        return info;
    }
    
    // Получаем ссылку на OpenCLCore
    const OpenCLCore& core = OpenCLCore::GetInstance();
    
    // ═══════════════════════════════════════════════════════════════════════
    // Заполняем информацию
    // ═══════════════════════════════════════════════════════════════════════
    
    // Основная информация
    info.name = core.GetDeviceName();
    info.vendor = core.GetVendor();
    info.driver_version = core.GetDriverVersion();
    
    // Версия OpenCL
    info.opencl_version = std::to_string(core.GetOpenCLVersionMajor()) + "." + 
                          std::to_string(core.GetOpenCLVersionMinor());
    
    // Индекс устройства
    info.device_index = device_index_;
    
    // Память
    info.global_memory_size = core.GetGlobalMemorySize();
    info.local_memory_size = core.GetLocalMemorySize();
    info.max_mem_alloc_size = core.GetGlobalMemorySize();  // TODO: уточнить
    
    // Вычислительные возможности
    info.max_compute_units = core.GetComputeUnits();
    info.max_work_group_size = core.GetMaxWorkGroupSize();
    
    // Возможности
    info.supports_svm = core.IsSVMSupported();
    info.supports_double = SupportsDoublePrecision();
    info.supports_half = false;  // TODO: добавить проверку cl_khr_fp16
    info.supports_unified_memory = SupportsSVM();
    
    return info;
}

} // namespace drv_gpu_lib
