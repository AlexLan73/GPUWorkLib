#pragma once

/**
 * @file gpu_manager.hpp
 * @brief GPUManager - Multi-GPU coordinator
 * 
 * GPUManager отвечает за:
 * - Обнаружение всех доступных GPU
 * - Выбор и инициализацию бэкенда
 * - Создание DrvGPU экземпляров для каждой GPU
 * - Load balancing между GPU
 * 
 * Архитектурный паттерн: Facade + Factory
 * 
 * @example Multi-GPU usage:
 * @code
 * DrvGPU::GPUManager manager;
 * manager.Initialize(DrvGPU::BackendType::OPENCL);
 * 
 * auto gpu_ids = manager.GetAllGPUs();
 * for (auto id : gpu_ids) {
 *     auto drv = manager.CreateDrvGPU(id);
 *     // Work with each GPU independently
 * }
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "drvgpu.hpp"
#include "ibackend.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Struct: GPUInfo - информация о доступной GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct GPUInfo
 * @brief Информация о доступной GPU в системе
 */
struct GPUInfo {
    uint32_t device_id;             ///< ID устройства
    std::string name;               ///< Название устройства
    std::string vendor;             ///< Производитель
    size_t global_memory_mb;        ///< Глобальная память (MB)
    BackendType backend_type;       ///< Тип бэкенда
    bool is_available;              ///< Доступна ли GPU
    
    std::string ToString() const;
};

// ════════════════════════════════════════════════════════════════════════════
// Enum: LoadBalancingStrategy - стратегия распределения нагрузки
// ════════════════════════════════════════════════════════════════════════════

/**
 * @enum LoadBalancingStrategy
 * @brief Стратегия распределения задач между GPU
 */
enum class LoadBalancingStrategy {
    ROUND_ROBIN,        ///< Поочерёдное распределение
    LEAST_LOADED,       ///< Выбор наименее загруженной GPU
    MEMORY_BASED,       ///< На основе доступной памяти
    MANUAL              ///< Ручное управление
};

// ════════════════════════════════════════════════════════════════════════════
// Class: GPUManager - координатор Multi-GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GPUManager
 * @brief Центральный координатор для работы с несколькими GPU
 * 
 * GPUManager управляет:
 * - Инициализацией бэкенда
 * - Обнаружением GPU устройств
 * - Созданием DrvGPU экземпляров
 * - Балансировкой нагрузки (опционально)
 * 
 * Thread-safety: все методы thread-safe
 * 
 * @code
 * // Инициализация
 * GPUManager manager;
 * manager.Initialize(BackendType::OPENCL);
 * 
 * // Получить информацию обо всех GPU
 * auto gpus = manager.GetAvailableGPUs();
 * for (const auto& gpu : gpus) {
 *     std::cout << gpu.ToString() << "\n";
 * }
 * 
 * // Создать DrvGPU для конкретной GPU
 * auto drv0 = manager.CreateDrvGPU(0);
 * auto drv1 = manager.CreateDrvGPU(1);
 * 
 * // Или получить следующую GPU с load balancing
 * auto drv_next = manager.GetNextGPU(LoadBalancingStrategy::LEAST_LOADED);
 * @endcode
 */
class GPUManager {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Конструктор (пустой GPUManager)
     */
    GPUManager();
    
    /**
     * @brief Деструктор (освобождение ресурсов)
     */
    ~GPUManager();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение move
    // ═══════════════════════════════════════════════════════════════
    
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    GPUManager(GPUManager&&) noexcept;
    GPUManager& operator=(GPUManager&&) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать GPUManager с конкретным бэкендом
     * @param backend_type Тип бэкенда (OPENCL, CUDA, VULKAN, AUTO)
     * @throws std::runtime_error если инициализация не удалась
     */
    void Initialize(BackendType backend_type = BackendType::AUTO);
    
    /**
     * @brief Проверить инициализацию
     * @return true если GPUManager инициализирован
     */
    bool IsInitialized() const;
    
    /**
     * @brief Очистить ресурсы (опционально, вызывается в деструкторе)
     */
    void Cleanup();
    
    // ═══════════════════════════════════════════════════════════════
    // Получение информации о GPU
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество доступных GPU
     * @return Количество GPU
     */
    size_t GetGPUCount() const;
    
    /**
     * @brief Получить список ID всех доступных GPU
     * @return Вектор ID устройств
     */
    std::vector<uint32_t> GetAllGPUs() const;
    
    /**
     * @brief Получить информацию обо всех доступных GPU
     * @return Вектор GPUInfo
     */
    std::vector<GPUInfo> GetAvailableGPUs() const;
    
    /**
     * @brief Получить информацию о конкретной GPU
     * @param device_id ID устройства
     * @return GPUInfo
     * @throws std::out_of_range если device_id невалиден
     */
    GPUInfo GetGPUInfo(uint32_t device_id) const;
    
    /**
     * @brief Проверить доступность GPU
     * @param device_id ID устройства
     * @return true если GPU доступна
     */
    bool IsGPUAvailable(uint32_t device_id) const;
    
    // ═══════════════════════════════════════════════════════════════
    // Создание DrvGPU экземпляров
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать DrvGPU для конкретной GPU
     * @param device_id ID устройства
     * @return unique_ptr на DrvGPU
     * @throws std::runtime_error если создание не удалось
     */
    std::unique_ptr<DrvGPU> CreateDrvGPU(uint32_t device_id);
    
    /**
     * @brief Создать DrvGPU для конкретной GPU с конкретным бэкендом
     * @param device_id ID устройства
     * @param backend_type Тип бэкенда (override)
     * @return unique_ptr на DrvGPU
     */
    std::unique_ptr<DrvGPU> CreateDrvGPU(
        uint32_t device_id, 
        BackendType backend_type
    );
    
    /**
     * @brief Создать DrvGPU экземпляры для всех GPU
     * @return Вектор unique_ptr на DrvGPU
     */
    std::vector<std::unique_ptr<DrvGPU>> CreateAllDrvGPUs();
    
    // ═══════════════════════════════════════════════════════════════
    // Load Balancing (опционально)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить ID следующей GPU для использования
     * @param strategy Стратегия выбора
     * @return ID устройства
     */
    uint32_t GetNextGPU(
        LoadBalancingStrategy strategy = LoadBalancingStrategy::ROUND_ROBIN
    );
    
    /**
     * @brief Создать DrvGPU с автоматическим load balancing
     * @param strategy Стратегия выбора
     * @return unique_ptr на DrvGPU
     */
    std::unique_ptr<DrvGPU> GetNextDrvGPU(
        LoadBalancingStrategy strategy = LoadBalancingStrategy::ROUND_ROBIN
    );
    
    /**
     * @brief Установить стратегию балансировки по умолчанию
     */
    void SetDefaultStrategy(LoadBalancingStrategy strategy);
    
    // ═══════════════════════════════════════════════════════════════
    // Информация и диагностика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести информацию обо всех GPU
     */
    void PrintAllGPUs() const;
    
    /**
     * @brief Получить статистику использования
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Получить текущий тип бэкенда
     */
    BackendType GetBackendType() const { return backend_type_; }
    
private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    bool initialized_;                          ///< Флаг инициализации
    BackendType backend_type_;                  ///< Текущий тип бэкенда
    std::vector<GPUInfo> available_gpus_;       ///< Список доступных GPU
    
    // Load balancing
    mutable std::mutex mutex_;                  ///< Мьютекс для thread-safety
    size_t round_robin_index_;                  ///< Индекс для round-robin
    LoadBalancingStrategy default_strategy_;    ///< Стратегия по умолчанию
    std::vector<size_t> gpu_usage_counter_;     ///< Счётчики использования
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Обнаружить все доступные GPU для текущего бэкенда
     */
    void DiscoverGPUs();
    
    /**
     * @brief Создать бэкенд для конкретной GPU
     */
    std::unique_ptr<IBackend> CreateBackend(
        uint32_t device_id, 
        BackendType backend_type
    );
    
    /**
     * @brief Получить ID GPU по стратегии LEAST_LOADED
     */
    uint32_t GetLeastLoadedGPU() const;
    
    /**
     * @brief Получить ID GPU по стратегии MEMORY_BASED
     */
    uint32_t GetMemoryBasedGPU() const;
    
    /**
     * @brief Валидация device_id
     */
    void ValidateDeviceID(uint32_t device_id) const;
};

} // namespace drv_gpu_lib
