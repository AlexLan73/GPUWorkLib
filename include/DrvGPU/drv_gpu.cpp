#include "drv_gpu.hpp"
#include "memory/memory_manager.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/opencl/opencl_core.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <sstream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// MemoryManager Implementation - Менеджер памяти GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор MemoryManager
 * @param backend Указатель на бэкенд для выделения памяти
 * 
 * Инициализирует статистику:
 * - total_allocations_ = 0
 * - total_frees_ = 0
 * - current_allocations_ = 0
 * - total_bytes_allocated_ = 0
 * - peak_bytes_allocated_ = 0
 */
MemoryManager::MemoryManager(IBackend* backend)
    : backend_(backend),
      total_allocations_(0),
      total_frees_(0),
      current_allocations_(0),
      total_bytes_allocated_(0),
      peak_bytes_allocated_(0) {
}

/**
 * @brief Деструктор MemoryManager
 * 
 * Вызывает Cleanup() для освобождения ресурсов.
 */
MemoryManager::~MemoryManager() {
    Cleanup();
}

/**
 * @brief Move конструктор
 * @param other Перемещаемый объект
 * 
 * Переносит все данные из other в новый объект.
 */
MemoryManager::MemoryManager(MemoryManager&& other) noexcept
    : backend_(other.backend_),
      total_allocations_(other.total_allocations_),
      total_frees_(other.total_frees_),
      current_allocations_(other.current_allocations_),
      total_bytes_allocated_(other.total_bytes_allocated_),
      peak_bytes_allocated_(other.peak_bytes_allocated_) {
    other.backend_ = nullptr;
}

/**
 * @brief Move оператор присваивания
 * @param other Перемещаемый объект
 * @return Ссылка на this
 */
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

/**
 * @brief Выделить память на GPU
 * @param size_bytes Размер в байтах
 * @param flags Флаги выделения (backend-specific)
 * @return Указатель на выделенную память или nullptr при ошибке
 * 
 * Делегирует выделение памяти бэкенду и отслеживает статистику.
 * 
 * @throws std::runtime_error если backend_ == nullptr
 */
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

/**
 * @brief Освободить память на GPU
 * @param ptr Указатель на память
 * 
 * @note Фактическое освобождение происходит через бэкенд.
 *       Этот метод предназначен для обратной совместимости.
 */
void MemoryManager::Free(void* ptr) {
    [[maybe_unused]] void* p = ptr;
    // Note: actual deallocation happens via backend
    // We track the deallocation when buffer is destroyed
}

/**
 * @brief Получить количество активных аллокаций
 * @return Количество текущих аллокаций
 */
size_t MemoryManager::GetAllocationCount() const {
    return current_allocations_;
}

/**
 * @brief Получить общее количество выделенных байт
 * @return Общее количество байт с момента создания
 */
size_t MemoryManager::GetTotalAllocatedBytes() const {
    return total_bytes_allocated_;
}

/**
 * @brief Вывести статистику в консоль
 * 
 * Форматированный вывод:
 * ═══════════════════════════════════════════
 * Memory Statistics
 * ═══════════════════════════════════════════
 * Total Allocations: X
 * Total Frees: X
 * Current Allocations: X
 * Total Bytes Allocated: X
 * Peak Bytes Allocated: X
 * ═══════════════════════════════════════════
 */
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

/**
 * @brief Получить статистику в виде строки
 * @return Строка с статистикой
 */
std::string MemoryManager::GetStatistics() const {
    std::ostringstream oss;
    oss << "Memory Statistics:\n";
    oss << "  Total Allocations: " << total_allocations_ << "\n";
    oss << "  Current Allocations: " << current_allocations_ << "\n";
    oss << "  Total Bytes: " << total_bytes_allocated_ << "\n";
    oss << "  Peak Bytes: " << peak_bytes_allocated_ << "\n";
    return oss.str();
}

/**
 * @brief Сбросить статистику на ноль
 * 
 * Обнуляет все счётчики статистики.
 * Thread-safe через mutex.
 */
void MemoryManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_allocations_ = 0;
    total_frees_ = 0;
    current_allocations_ = 0;
    total_bytes_allocated_ = 0;
    peak_bytes_allocated_ = 0;
}

/**
 * @brief Очистить ресурсы MemoryManager
 * 
 * Сбрасывает счётчики current_allocations_ и total_bytes_allocated_.
 */
void MemoryManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_allocations_ = 0;
    total_bytes_allocated_ = 0;
}

/**
 * @brief Отслеживать выделение памяти (внутренний метод)
 * @param size_bytes Размер выделенной памяти
 * 
 * Обновляет:
 * - total_allocations_
 * - current_allocations_
 * - total_bytes_allocated_
 * - peak_bytes_allocated_
 */
void MemoryManager::TrackAllocation(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_allocations_++;
    current_allocations_++;
    total_bytes_allocated_ += size_bytes;
    if (current_allocations_ > peak_bytes_allocated_) {
        peak_bytes_allocated_ = current_allocations_;
    }
}

/**
 * @brief Отслеживать освобождение памяти (внутренний метод)
 * @param size_bytes Размер освобождённой памяти
 */
void MemoryManager::TrackFree([[maybe_unused]] size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_frees_++;
    if (current_allocations_ > 0) {
        current_allocations_--;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DrvGPU Implementation - Главный класс библиотеки
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор DrvGPU
 * @param backend_type Тип бэкенда (OPENCL, ROCm, etc.)
 * @param device_index Индекс GPU устройства (0-based)
 * 
 * Создаёт бэкенд и инициализирует подсистемы:
 * - MemoryManager
 * - ModuleRegistry
 * 
 * @throws std::runtime_error если не удалось создать бэкенд
 */
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

/**
 * @brief Деструктор DrvGPU
 * 
 * Вызывает Cleanup() для освобождения всех ресурсов.
 * RAII гарантирует корректную очистку.
 */
DrvGPU::~DrvGPU() {
    Cleanup();
}

/**
 * @brief Move конструктор
 * @param other Перемещаемый объект
 */
DrvGPU::DrvGPU(DrvGPU&& other) noexcept
    : backend_type_(other.backend_type_),
      device_index_(other.device_index_),
      initialized_(other.initialized_),
      backend_(std::move(other.backend_)),
      memory_manager_(std::move(other.memory_manager_)),
      module_registry_(std::move(other.module_registry_)) {
    other.initialized_ = false;
}

/**
 * @brief Move оператор присваивания
 * @param other Перемещаемый объект
 * @return Ссылка на this
 */
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

/**
 * @brief Создать бэкенд на основе типа (внутренний метод)
 * 
 * Создаёт соответствующий бэкенд:
 * - OPENCL -> OpenCLBackend
 * - ROCm -> (не реализовано, throw)
 * - OPENCLandROCm -> (не реализовано, throw)
 * 
 * @throws std::runtime_error если тип бэкенда не поддерживается
 */
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

/**
 * @brief Инициализировать подсистемы (внутренний метод)
 * 
 * Создаёт:
 * - MemoryManager для управления памятью
 * - ModuleRegistry для регистрации модулей
 */
void DrvGPU::InitializeSubsystems() {
    memory_manager_ = std::make_unique<MemoryManager>(backend_.get());
    module_registry_ = std::make_unique<ModuleRegistry>();
}

/**
 * @brief Инициализировать GPU
 * 
 * Инициализирует бэкенд для указанного устройства.
 * После инициализации DrvGPU готов к работе.
 * 
 * @throws std::runtime_error если backend_ == nullptr или инициализация не удалась
 * 
 * Пример:
 * @code
 * DrvGPU gpu(BackendType::OPENCL, 0);
 * gpu.Initialize();  // Инициализировать GPU
 * @endcode
 */
void DrvGPU::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        DRVGPU_LOG_WARNING("DrvGPU", "Already initialized");
        return;
    }
    
    if (!backend_) {
        throw std::runtime_error("DrvGPU: backend is null");
    }
    
    backend_->Initialize(device_index_);
    initialized_ = true;
    
    DRVGPU_LOG_INFO("DrvGPU", "Initialized successfully");
}

/**
 * @brief Очистить все ресурсы
 * 
 * Освобождает в порядке:
 * 1. MemoryManager
 * 2. ModuleRegistry
 * 3. Backend
 * 
 * Вызывается автоматически в деструкторе.
 */
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
    DRVGPU_LOG_INFO("DrvGPU", "Cleaned up");
}

/**
 * @brief Получить информацию об устройстве
 * @return GPUDeviceInfo с информацией о GPU
 * 
 * @throws std::runtime_error если DrvGPU не инициализирован
 */
GPUDeviceInfo DrvGPU::GetDeviceInfo() const {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    return backend_->GetDeviceInfo();
}

/**
 * @brief Получить название устройства
 * @return Строка с названием или "Unknown"
 */
std::string DrvGPU::GetDeviceName() const {
    if (!initialized_ || !backend_) {
        return "Unknown";
    }
    auto info = backend_->GetDeviceInfo();
    return info.name;
}

/**
 * @brief Вывести информацию об устройстве в лог
 */
void DrvGPU::PrintDeviceInfo() const {
    if (!initialized_ || !backend_) {
        DRVGPU_LOG_WARNING("DrvGPU", "Device not initialized");
        return;
    }
    auto info = backend_->GetDeviceInfo();
    DRVGPU_LOG_INFO("DrvGPU", "Device Info - Name: " + info.name + ", Vendor: " + info.vendor);
}

/**
 * @brief Получить менеджер памяти (не-const версия)
 * @return Ссылка на MemoryManager
 * 
 * @throws std::runtime_error если MemoryManager не инициализирован
 */
MemoryManager& DrvGPU::GetMemoryManager() {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

/**
 * @brief Получить менеджер памяти (const версия)
 * @return Константная ссылка на MemoryManager
 */
const MemoryManager& DrvGPU::GetMemoryManager() const {
    if (!memory_manager_) {
        throw std::runtime_error("MemoryManager not initialized");
    }
    return *memory_manager_;
}

/**
 * @brief Получить регистр модулей (не-const версия)
 * @return Ссылка на ModuleRegistry
 */
ModuleRegistry& DrvGPU::GetModuleRegistry() {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

/**
 * @brief Получить регистр модулей (const версия)
 * @return Константная ссылка на ModuleRegistry
 */
const ModuleRegistry& DrvGPU::GetModuleRegistry() const {
    if (!module_registry_) {
        throw std::runtime_error("ModuleRegistry not initialized");
    }
    return *module_registry_;
}

/**
 * @brief Получить бэкенд (не-const версия)
 * @return Ссылка на IBackend
 * 
 * @warning Используйте только если абстракции недостаточно!
 */
IBackend& DrvGPU::GetBackend() {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

/**
 * @brief Получить бэкенд (const версия)
 * @return Константная ссылка на IBackend
 */
const IBackend& DrvGPU::GetBackend() const {
    if (!backend_) {
        throw std::runtime_error("Backend not initialized");
    }
    return *backend_;
}

/**
 * @brief Синхронизировать (дождаться завершения всех операций)
 * 
 * Блокирует CPU до завершения всех GPU операций.
 * 
 * @throws std::runtime_error если DrvGPU не инициализирован
 */
void DrvGPU::Synchronize() {
    if (!initialized_ || !backend_) {
        throw std::runtime_error("DrvGPU not initialized");
    }
    backend_->Synchronize();
}

/**
 * @brief Flush всех команд (без ожидания)
 * 
 * Отправляет все команды на выполнение без ожидания.
 */
void DrvGPU::Flush() {
    if (!initialized_ || !backend_) {
        return;
    }
    backend_->Flush();
}

/**
 * @brief Вывести статистику в консоль
 */
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

/**
 * @brief Получить статистику в виде строки
 * @return Строка с статистикой
 */
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

/**
 * @brief Сбросить статистику
 */
void DrvGPU::ResetStatistics() {
    if (memory_manager_) {
        memory_manager_->ResetStatistics();
    }
}

} // namespace drv_gpu_lib
