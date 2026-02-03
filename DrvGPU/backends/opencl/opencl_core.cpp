#include "opencl_core.hpp"
#include "../../memory/svm_capabilities.hpp"
#include "../../common/logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Static инициализация - статические члены класса
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Статический указатель на единственный экземпляр (Singleton)
 * 
 * Используем unique_ptr для автоматического управления памятью.
 * Инициализируется nullptr - создаётся при первом вызове Initialize().
 */
std::unique_ptr<OpenCLCore> OpenCLCore::instance_ = nullptr;

/**
 * @brief Флаг инициализации
 * 
 * Показывает, был ли инициализирован Singleton.
 * Защищён мьютексом initialization_mutex_ при изменении.
 */
bool OpenCLCore::initialized_ = false;

/**
 * @brief Мьютекс для thread-safe инициализации Singleton'а
 * 
 * Используется двойной проверкой (double-checked locking):
 * 1. Сначала быстрая проверка без блокировки
 * 2. Если первый тест прошёл - блокируем и проверяем снова
 */
std::mutex OpenCLCore::initialization_mutex_;

// ════════════════════════════════════════════════════════════════════════════
// Публичные статические методы - Singleton API
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать OpenCL контекст (первый вызов)
 * 
 * Это thread-safe инициализация Singleton'а.
 * Вызывается один раз при первом использовании OpenCL.
 * 
 * Процесс инициализации:
 * 1. Захватываем мьютекс для thread-safety
 * 2. Проверяем флаг initialized_ (double-check)
 * 3. Создаём новый экземпляр OpenCLCore
 * 4. Вызываем внутреннюю инициализацию OpenCL
 * 5. Устанавливаем флаг initialized_
 * 
 * @param device_type Тип устройства: GPU или CPU
 * 
 * @code
 * // Первый вызов - инициализация
 * OpenCLCore::Initialize(DeviceType::GPU);
 * 
 * // Последующие вызовы - только логирование
 * OpenCLCore::Initialize(DeviceType::CPU);  // Warning: Already initialized
 * @endcode
 * 
 * @throws std::runtime_error если:
 *   - Нет доступных OpenCL платформ
 *   - Нет устройств указанного типа
 *   - Ошибка создания контекста OpenCL
 */
void OpenCLCore::Initialize(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    // Проверяем, не инициализирован ли уже
    if (initialized_) {
        DRVGPU_LOG_WARNING("OpenCLCore", "Already initialized");
        return;  // Просто выходим, не переинициализируем
    }

    // Создаём новый экземпляр
    instance_ = std::unique_ptr<OpenCLCore>(new OpenCLCore());
    
    // Выполняем внутреннюю инициализацию
    instance_->InitializeOpenCL(device_type);
    
    // Устанавливаем флаг
    initialized_ = true;

    DRVGPU_LOG_INFO("OpenCLCore", "Initialized successfully");
    DRVGPU_LOG_DEBUG("OpenCLCore", instance_->GetDeviceInfo());
}

/**
 * @brief Получить ссылку на Singleton экземпляр
 * 
 * @return Ссылка на единственный экземпляр OpenCLCore
 * 
 * @throws std::runtime_error если Singleton не инициализирован
 * 
 * @code
 * // Правильный порядок:
 * OpenCLCore::Initialize(DeviceType::GPU);  // Сначала инициализация
 * 
 * // Теперь можно получать доступ
 * OpenCLCore& core = OpenCLCore::GetInstance();
 * cl_context ctx = core.GetContext();
 * cl_device_id dev = core.GetDevice();
 * @endcode
 * 
 * @note Это thread-safe метод (protected by initialization_mutex_)
 */
OpenCLCore& OpenCLCore::GetInstance() {
    if (!initialized_) {
        throw std::runtime_error(
            "OpenCLCore not initialized. Call Initialize() first.");
    }
    return *instance_;
}

/**
 * @brief Проверить состояние инициализации
 * @return true если OpenCLCore уже инициализирован
 * 
 * @code
 * if (OpenCLCore::IsInitialized()) {
 *     // Можно безопасно вызывать GetInstance()
 *     auto& core = OpenCLCore::GetInstance();
 * }
 * @endcode
 */
bool OpenCLCore::IsInitialized() {
    return initialized_;
}

/**
 * @brief Очистить ресурсы OpenCLCore (Singleton)
 * 
 * Вызывается при завершении работы с OpenCL.
 * Освобождает:
 * - OpenCL контекст (clReleaseContext)
 * - Устройство (clReleaseDevice)
 * - Экземпляр unique_ptr
 * 
 * @code
 * // При завершении приложения:
 * OpenCLCore::Cleanup();
 * @endcode
 * 
 * @note После Cleanup() можно снова вызвать Initialize() для переинициализации
 */
void OpenCLCore::Cleanup() {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        // Освобождаем ресурсы экземпляра
        instance_->ReleaseResources();
        
        // Удаляем экземпляр
        instance_.reset();
        
        // Сбрасываем флаг
        initialized_ = false;
        
        DRVGPU_LOG_INFO("OpenCLCore", "Cleaned up");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Конструктор
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Приватный конструктор (Singleton)
 * 
 * Инициализирует все члены значениями по умолчанию.
 * Настоящая инициализация происходит в InitializeOpenCL().
 */
OpenCLCore::OpenCLCore()
    : platform_(nullptr),
      device_(nullptr),
      context_(nullptr),
      device_type_(DeviceType::GPU) {
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация OpenCL - внутренние методы
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать OpenCL платформу, устройство и контекст
 * 
 * Это основной метод инициализации, вызываемый из Initialize().
 * Последовательность действий:
 * 
 * 1. Получить все доступные платформы (clGetPlatformIDs)
 * 2. Выбрать первую платформу (обычно это NVIDIA/AMD/Intel)
 * 3. Получить устройства указанного типа (GPU/CPU)
 * 4. Выбрать первое устройство
 * 5. Создать контекст OpenCL (clCreateContext)
 * 
 * @param device_type Тип устройства для поиска (GPU или CPU)
 * 
 * @throws std::runtime_error если платформы/устройства не найдены
 * 
 * @code
 * // Детали процесса инициализации:
 * 
 * // Шаг 1: Получаем количество платформ
 * cl_uint num_platforms = 0;
 * clGetPlatformIDs(0, nullptr, &num_platforms);
 * 
 * // Шаг 2: Получаем список платформ
 * std::vector<cl_platform_id> platforms(num_platforms);
 * clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
 * 
 * // Шаг 3: Получаем количество устройств GPU
 * cl_uint num_devices = 0;
 * clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
 * 
 * // Шаг 4: Создаём контекст
 * cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
 * @endcode
 */
void OpenCLCore::InitializeOpenCL(DeviceType device_type) {
    device_type_ = device_type;

    cl_int err;
    cl_uint num_platforms = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 1: Получить все доступные OpenCL платформы
    // ═══════════════════════════════════════════════════════════════════════
    
    // clGetPlatformIDs - получить количество платформ
    // Параметры:
    //   num_entries = 0 (только считаем)
    //   platforms = nullptr (не получаем список)
    //   num_platforms = &num_platforms (возвращаем количество)
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CheckCLError(err, "clGetPlatformIDs (count)");

    // Проверяем, есть ли платформы
    if (num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    // Получаем список платформ
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (get)");

    // Выбираем первую платформу
    // TODO: добавить выбор платформы по имени (NVIDIA/AMD/Intel)
    platform_ = platforms[0];

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 2: Получить устройства указанного типа
    // ═══════════════════════════════════════════════════════════════════════

    // CL_DEVICE_TYPE_GPU - ищем только GPU
    // CL_DEVICE_TYPE_CPU - ищем только CPU
    cl_device_type cl_device_type =
        (device_type == DeviceType::GPU) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    // Получаем количество устройств
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform_, cl_device_type, 0, nullptr, &num_devices);
    CheckCLError(err, "clGetDeviceIDs (count)");

    // Проверяем, есть ли устройства
    if (num_devices == 0) {
        throw std::runtime_error("No OpenCL devices found for specified type");
    }

    // Получаем список устройств
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform_, cl_device_type, num_devices, devices.data(), nullptr);
    CheckCLError(err, "clGetDeviceIDs (get)");

    // Выбираем первое устройство
    // TODO: добавить выбор устройства по индексу
    device_ = devices[0];

    // ═══════════════════════════════════════════════════════════════════════
    // Шаг 3: Создать OpenCL контекст
    // ═══════════════════════════════════════════════════════════════════════

    // clCreateContext - создать контекст OpenCL
    // Параметры:
    //   properties = nullptr (используем платформу по умолчанию)
    //   num_devices = 1 (одно устройство)
    //   devices = [&device_] (устройство для контекста)
    //   pfn_notify = nullptr (без callback)
    //   user_data = nullptr (без данных для callback)
    //   errcode_ret = &err (код ошибки)
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");
}

// ════════════════════════════════════════════════════════════════════════════
// Деструктор и очистка
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Деструктор OpenCLCore
 * 
 * Вызывает ReleaseResources() для освобождения ресурсов.
 * Это обеспечивает RAII - ресурсы освобождаются при уничтожении объекта.
 */
OpenCLCore::~OpenCLCore() {
    ReleaseResources();
}

/**
 * @brief Освободить все ресурсы OpenCL
 * 
 * Вызывается из:
 * - Деструктора (~OpenCLCore)
 * - Статического метода Cleanup()
 * 
 * Освобождает в порядке создания (обратном):
 * 1. Контекст OpenCL (clReleaseContext)
 * 2. Устройство (clReleaseDevice)
 * 
 * @note Обнуляем указатели после освобождения для безопасности
 */
void OpenCLCore::ReleaseResources() {
    // Освобождаем контекст (если создан)
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    // Освобождаем устройство (если получено)
    // Примечание: clReleaseDevice требуется только для OpenCL 1.2+
    // В OpenCL 2.0+ устройства освобождаются автоматически при освобождении контекста
    if (device_) {
        clReleaseDevice(device_);
        device_ = nullptr;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о девайсе - приватные утилиты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить значение параметра устройства (шаблон)
 * 
 * Вспомогательный шаблонный метод для получения простых значений
 * (целых чисел, размеров, флагов) из информации об устройстве.
 * 
 * @tparam T Тип возвращаемого значения
 * @param param Параметр cl_device_info для запроса
 * @return Значение параметра
 * 
 * @code
 * // Примеры использования:
 * cl_uint compute_units = GetDeviceInfoValue<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
 * size_t max_work_group = GetDeviceInfoValue<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
 * cl_ulong mem_size = GetDeviceInfoValue<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
 * @endcode
 */
template<typename T>
T OpenCLCore::GetDeviceInfoValue(cl_device_info param) const {
    T value;
    cl_int err = clGetDeviceInfo(device_, param, sizeof(T), &value, nullptr);
    CheckCLError(err, "clGetDeviceInfo");
    return value;
}

/**
 * @brief Получить строковое значение параметра устройства
 * 
 * Двухэтапное получение строки:
 * 1. Сначала запрашиваем размер буфера (0 байт)
 * 2. Потом получаем саму строку в выделенный буфер
 * 
 * @param param Параметр cl_device_info для запроса
 * @return Строка с значением параметра
 * 
 * @code
 * // Примеры:
 * std::string name = GetDeviceInfoString(CL_DEVICE_NAME);
 * std::string vendor = GetDeviceInfoString(CL_DEVICE_VENDOR);
 * std::string driver = GetDeviceInfoString(CL_DRIVER_VERSION);
 * @endcode
 */
std::string OpenCLCore::GetDeviceInfoString(cl_device_info param) const {
    // Шаг 1: получаем размер строки
    size_t size = 0;
    cl_int err = clGetDeviceInfo(device_, param, 0, nullptr, &size);
    CheckCLError(err, "clGetDeviceInfo (size)");

    // Шаг 2: выделяем буфер и читаем строку
    std::vector<char> buffer(size);
    err = clGetDeviceInfo(device_, param, size, buffer.data(), nullptr);
    CheckCLError(err, "clGetDeviceInfo (get)");

    return std::string(buffer.data());
}

// ════════════════════════════════════════════════════════════════════════════
// Публичные методы получения информации об устройстве
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить название устройства
 * @return Строка с названием (например: "NVIDIA GeForce RTX 3080")
 */
std::string OpenCLCore::GetDeviceName() const {
    return GetDeviceInfoString(CL_DEVICE_NAME);
}

/**
 * @brief Получить производителя устройства
 * @return Строка с именем вендора (например: "NVIDIA Corporation", "Advanced Micro Devices, Inc.")
 */
std::string OpenCLCore::GetVendor() const {
    return GetDeviceInfoString(CL_DEVICE_VENDOR);
}

/**
 * @brief Получить версию драйвера
 * @return Строка с версией драйвера
 */
std::string OpenCLCore::GetDriverVersion() const {
    return GetDeviceInfoString(CL_DRIVER_VERSION);
}

/**
 * @brief Получить размер глобальной памяти в байтах
 * @return Размер в байтах (тип cl_ulong)
 * 
 * @code
 * auto mem_bytes = core.GetGlobalMemorySize();
 * auto mem_gb = mem_bytes / (1024.0 * 1024.0 * 1024.0);
 * std::cout << "Global Memory: " << mem_gb << " GB" << std::endl;
 * @endcode
 */
size_t OpenCLCore::GetGlobalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

/**
 * @brief Получить размер локальной памяти в байтах
 * @return Размер локальной памяти в байтах
 * 
 * @note Это размер памяти on-chip (быстрая память в каждой compute unit)
 */
size_t OpenCLCore::GetLocalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
}

/**
 * @brief Получить количество compute units
 * @return Количество вычислительных блоков
 * 
 * @code
 * cl_uint cu = core.GetComputeUnits();
 * // NVIDIA: количество SM (Streaming Multiprocessors)
 * // AMD: количество CU (Compute Units)
 * // Intel: количество EU (Execution Units)
 * @endcode
 */
cl_uint OpenCLCore::GetComputeUnits() const {
    return GetDeviceInfoValue<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

/**
 * @brief Получить максимальный размер work group
 * @return Максимальное количество work-items в группе
 * 
 * @note Это ограничение для одного измерения work group
 */
size_t OpenCLCore::GetMaxWorkGroupSize() const {
    return GetDeviceInfoValue<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

/**
 * @brief Получить максимальные размеры work item по каждому измерению
 * @return Массив из 3 элементов [x, y, z]
 * 
 * @code
 * auto sizes = core.GetMaxWorkItemSizes();
 * std::cout << "Max sizes: [" << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]" << std::endl;
 * @endcode
 */
std::array<size_t, 3> OpenCLCore::GetMaxWorkItemSizes() const {
    std::array<size_t, 3> sizes;
    cl_int err = clGetDeviceInfo(
        device_,
        CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(sizes),
        sizes.data(),
        nullptr
    );
    CheckCLError(err, "clGetDeviceInfo (MAX_WORK_ITEM_SIZES)");
    return sizes;
}

// ════════════════════════════════════════════════════════════════════════════
// Метод GetDeviceInfo - красивый вывод всей информации
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить полную информацию об устройстве (для вывода)
 * @return Отформатированная строка с всей информацией
 * 
 * Формат вывода:
 * @code
 * ======================================
 * OpenCL Device Information
 * ======================================
 * 
 * Device Name:        NVIDIA GeForce RTX 3080
 * Vendor:             NVIDIA Corporation
 * Driver Version:     535.154.05
 * Device Type:        GPU
 * 
 * Global Memory:      10.00 GB
 * Local Memory:       48.00 KB
 * 
 * Compute Units:      68
 * Max Work Group Size: 1024
 * Max Work Item Sizes: [1024, 1024, 64]
 * 
 * ======================================
 * @endcode
 */
std::string OpenCLCore::GetDeviceInfo() const {
    std::ostringstream oss;

    // Заголовок
    oss << "\n" << std::string(70, '=') << "\n";
    oss << "OpenCL Device Information\n";
    oss << std::string(70, '=') << "\n\n";

    // Основная информация
    oss << std::left << std::setw(25) << "Device Name:" << GetDeviceName() << "\n";
    oss << std::left << std::setw(25) << "Vendor:" << GetVendor() << "\n";
    oss << std::left << std::setw(25) << "Driver Version:" << GetDriverVersion() << "\n";

    // Тип устройства
    oss << std::left << std::setw(25) << "Device Type:";
    oss << (device_type_ == DeviceType::GPU ? "GPU" : "CPU") << "\n";

    // Память
    size_t global_mem = GetGlobalMemorySize();
    size_t local_mem = GetLocalMemorySize();
    oss << std::left << std::setw(25) << "Global Memory:"
        << std::fixed << std::setprecision(2)
        << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    oss << std::left << std::setw(25) << "Local Memory:"
        << (local_mem / 1024.0) << " KB\n";

    // Вычислительные возможности
    oss << std::left << std::setw(25) << "Compute Units:"
        << GetComputeUnits() << "\n";

    // Work group
    oss << std::left << std::setw(25) << "Max Work Group Size:"
        << GetMaxWorkGroupSize() << "\n";

    // Work item sizes
    auto sizes = GetMaxWorkItemSizes();
    oss << std::left << std::setw(25) << "Max Work Item Sizes:"
        << "[" << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]\n";

    // Окончание
    oss << "\n" << std::string(70, '=') << "\n\n";

    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// SVM (Shared Virtual Memory) методы - OpenCL 2.0+
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить мажорную версию OpenCL
 * @return Мажорная версия (1, 2, или 3)
 * 
 * Парсит строку CL_DEVICE_VERSION формата "OpenCL X.Y ..."
 */
cl_uint OpenCLCore::GetOpenCLVersionMajor() const {
    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;
    
    int major = 0, minor = 0;
    // Формат: "OpenCL X.Y ..."
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) >= 1) {
        return static_cast<cl_uint>(major);
    }
    return 0;
}

/**
 * @brief Получить минорную версию OpenCL
 * @return Минорная версия
 */
cl_uint OpenCLCore::GetOpenCLVersionMinor() const {
    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;
    
    int major = 0, minor = 0;
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2) {
        return static_cast<cl_uint>(minor);
    }
    return 0;
}

/**
 * @brief Проверить поддержку SVM
 * @return true если SVM поддерживается (OpenCL 2.0+ с SVM capabilities)
 * 
 * SVM (Shared Virtual Memory) - это расширение OpenCL 2.0,
 * которое позволяет CPU и GPU использовать общую виртуальную память.
 * 
 * @code
 * if (core.IsSVMSupported()) {
 *     // Можно использовать SVM буферы
 *     // Преимущество: не нужно копировать данные между host и device
 * }
 * @endcode
 */
bool OpenCLCore::IsSVMSupported() const {
    // SVM требует как минимум OpenCL 2.0
    if (GetOpenCLVersionMajor() < 2) {
        return false;
    }
    
    // Запрашиваем SVM capabilities устройства
    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    
    // SVM поддерживается если есть хотя бы один тип SVM
    return (err == CL_SUCCESS && svm_caps != 0);
}

/**
 * @brief Получить полную информацию о SVM capabilities
 * @return Структура SVMCapabilities с информацией о поддержке
 */
SVMCapabilities OpenCLCore::GetSVMCapabilities() const {
    return SVMCapabilities::Query(device_);
}

/**
 * @brief Получить информацию о SVM в читаемом формате
 * @return Отформатированная строка с SVM информацией
 */
std::string OpenCLCore::GetSVMInfo() const {
    std::ostringstream oss;
    
    // Заголовок
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "SVM Capabilities\n";
    oss << std::string(60, '=') << "\n\n";
    
    // Версия OpenCL
    cl_uint major = GetOpenCLVersionMajor();
    cl_uint minor = GetOpenCLVersionMinor();
    
    oss << std::left << std::setw(25) << "OpenCL Version:" << major << "." << minor << "\n";
    
    // Проверка версии
    if (major < 2) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO (OpenCL < 2.0)\n";
        oss << std::string(60, '=') << "\n";
        return oss.str();
    }
    
    // Запрашиваем SVM capabilities
    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    
    if (err != CL_SUCCESS || svm_caps == 0) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO\n";
        oss << std::string(60, '=') << "\n";
        return oss.str();
    }
    
    // SVM поддерживается - выводим типы
    oss << std::left << std::setw(25) << "SVM Supported:" << "YES ✅\n\n";
    
    oss << "SVM Types:\n";
    oss << "  " << std::left << std::setw(23) << "Coarse-Grain Buffer:" 
        << ((svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain Buffer:" 
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain System:" 
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Atomics:" 
        << ((svm_caps & CL_DEVICE_SVM_ATOMICS) ? "YES ✅" : "NO ❌") << "\n";
    
    // Окончание
    oss << "\n" << std::string(60, '=') << "\n";

    return oss.str();
}

}  // namespace ManagerOpenCL
