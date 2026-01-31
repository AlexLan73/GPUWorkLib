#pragma once

/**
 * @file gpu_manager.hpp
 * @brief GPUManager - центральный координатор для Multi-GPU
 * 
 * КЛЮЧЕВОЙ КОМПОНЕНТ для Multi-GPU сценариев!
 * 
 * GPUManager управляет множественными экземплярами DrvGPU и предоставляет:
 * - Автоматическое обнаружение всех GPU
 * - Load balancing (Round-Robin, Least Loaded, Manual)
 * - Централизованное управление ресурсами
 * - Thread-safe доступ к GPU
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "drv_gpu.hpp"
#include "backend_type.hpp"
#include "load_balancing.hpp"
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: GPUManager - Координатор для Multi-GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class GPUManager
 * @brief Facade для управления множественными GPU
 * 
 * GPUManager - это "Single entry point" для работы с несколькими GPU.
 * Он создаёт и управляет экземплярами DrvGPU для каждого устройства.
 * 
 * Примеры использования:
 * 
 * @code
 * // Инициализация всех GPU
 * GPUManager manager;
 * manager.InitializeAll(BackendType::OPENCL);
 * 
 * // Round-Robin распределение
 * for (int i = 0; i < 100; ++i) {
 *     auto& gpu = manager.GetNextGPU();
 *     gpu.GetMemoryManager().Allocate(...);
 * }
 * 
 * // Явный выбор GPU
 * auto& gpu0 = manager.GetGPU(0);
 * auto& gpu1 = manager.GetGPU(1);
 * 
 * // Load balancing
 * auto& least_loaded = manager.GetLeastLoadedGPU();
 * @endcode
 * 
 * Паттерны:
 * - Facade (упрощение работы с Multi-GPU)
 * - Factory (создание DrvGPU экземпляров)
 * - Strategy (load balancing strategies)
 */
class GPUManager {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать GPUManager (без инициализации GPU)
     */
    GPUManager();
    
    /**
     * @brief Деструктор (освободит все GPU)
     */
    ~GPUManager();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    
    GPUManager(GPUManager&& other) noexcept;
    GPUManager& operator=(GPUManager&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать все доступные GPU
     * @param backend_type Тип бэкенда (OPENCL, CUDA, VULKAN)
     * @throws std::runtime_error если не найдено ни одной GPU
     */
    void InitializeAll(BackendType backend_type);
    
    /**
     * @brief Инициализировать конкретные GPU по индексам
     * @param backend_type Тип бэкенда
     * @param device_indices Список индексов GPU для инициализации
     */
    void InitializeSpecific(BackendType backend_type, 
                           const std::vector<int>& device_indices);
    
    /**
     * @brief Проверить, инициализирован ли менеджер
     */
    bool IsInitialized() const { return !gpus_.empty(); }
    
    /**
     * @brief Очистить все GPU и освободить ресурсы
     */
    void Cleanup();
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к GPU
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить GPU по индексу
     * @param index Индекс GPU (0 до GetGPUCount()-1)
     * @throws std::out_of_range если индекс некорректен
     */
    DrvGPU& GetGPU(size_t index);
    const DrvGPU& GetGPU(size_t index) const;
    
    /**
     * @brief Получить следующую GPU (Round-Robin)
     * Thread-safe, автоматически инкрементирует счётчик
     */
    DrvGPU& GetNextGPU();
    
    /**
     * @brief Получить наименее загруженную GPU
     * Использует метрику: количество активных задач
     */
    DrvGPU& GetLeastLoadedGPU();
    
    /**
     * @brief Получить все GPU
     */
    std::vector<DrvGPU*> GetAllGPUs();
    std::vector<const DrvGPU*> GetAllGPUs() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество доступных GPU
     */
    size_t GetGPUCount() const { return gpus_.size(); }
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetBackendType() const { return backend_type_; }
    
    /**
     * @brief Вывести информацию обо всех GPU
     */
    void PrintAllDevices() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Load Balancing
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Установить стратегию load balancing
     */
    void SetLoadBalancingStrategy(LoadBalancingStrategy strategy);
    
    /**
     * @brief Получить текущую стратегию
     */
    LoadBalancingStrategy GetLoadBalancingStrategy() const { 
        return lb_strategy_; 
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать все GPU (ждать завершения всех операций)
     */
    void SynchronizeAll();
    
    /**
     * @brief Flush всех GPU
     */
    void FlushAll();
    
    // ═══════════════════════════════════════════════════════════════
    // Статистика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести статистику по всем GPU
     */
    void PrintStatistics() const;
    
    /**
     * @brief Получить строку со статистикой
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Сбросить статистику всех GPU
     */
    void ResetStatistics();
    
    // ═══════════════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество доступных GPU в системе (статический метод)
     * @param backend_type Тип бэкенда для запроса
     */
    static int GetAvailableGPUCount(BackendType backend_type);

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    BackendType backend_type_;
    LoadBalancingStrategy lb_strategy_;
    
    // GPU экземпляры (владение через unique_ptr)
    std::vector<std::unique_ptr<DrvGPU>> gpus_;
    
    // Round-Robin счётчик (thread-safe)
    std::atomic<size_t> round_robin_index_;
    
    // Load tracking (простая метрика: количество задач)
    std::vector<std::atomic<size_t>> gpu_task_count_;
    
    // Thread-safety
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Обнаружить все доступные GPU
     */
    int DiscoverGPUs(BackendType backend_type);
    
    /**
     * @brief Инициализировать GPU по индексу
     */
    void InitializeGPU(int device_index);
    
    /**
     * @brief Получить индекс наименее загруженной GPU
     */
    size_t GetLeastLoadedGPUIndex() const;
};

} // namespace DrvGPU
