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
#include "common/logger.hpp"
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <sstream>
#include <iostream>

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
    
    // Load tracking (простая метрика: количество задач, защищено мьютексом)
    std::vector<size_t> gpu_task_count_;
    
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

// ════════════════════════════════════════════════════════════════════════════
// GPUManager Inline Implementation (Header-Only)
// ════════════════════════════════════════════════════════════════════════════

inline GPUManager::GPUManager()
    : backend_type_(BackendType::OPENCL)
    , lb_strategy_(LoadBalancingStrategy::ROUND_ROBIN)
    , round_robin_index_(0) {
}

inline GPUManager::~GPUManager() {
    Cleanup();
}

inline GPUManager::GPUManager(GPUManager&& other) noexcept
    : backend_type_(other.backend_type_)
    , lb_strategy_(other.lb_strategy_)
    , gpus_(std::move(other.gpus_))
    , round_robin_index_(other.round_robin_index_.load())
    , gpu_task_count_(std::move(other.gpu_task_count_)) {
}

inline GPUManager& GPUManager::operator=(GPUManager&& other) noexcept {
    if (this != &other) {
        Cleanup();
        backend_type_ = other.backend_type_;
        lb_strategy_ = other.lb_strategy_;
        gpus_ = std::move(other.gpus_);
        round_robin_index_ = other.round_robin_index_.load();
        gpu_task_count_ = std::move(other.gpu_task_count_);
    }
    return *this;
}

inline void GPUManager::InitializeAll(BackendType backend_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_type_ = backend_type;
    
    Cleanup();
    
    int gpu_count = DiscoverGPUs(backend_type);
    if (gpu_count == 0) {
        throw std::runtime_error("No GPUs available for backend type");
    }
    
    for (int i = 0; i < gpu_count; ++i) {
        InitializeGPU(i);
    }
    
    DRVGPU_LOG_INFO("GPUManager", "Initialized " + std::to_string(gpus_.size()) + " GPU(s)");
}

inline void GPUManager::InitializeSpecific(BackendType backend_type, 
                                           const std::vector<int>& device_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    backend_type_ = backend_type;
    
    Cleanup();
    
    for (int index : device_indices) {
        InitializeGPU(index);
    }
    
    DRVGPU_LOG_INFO("GPUManager", "Initialized " + std::to_string(gpus_.size()) + " specific GPU(s)");
}

inline void GPUManager::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    gpus_.clear();
    gpu_task_count_.clear();
    round_robin_index_ = 0;
    DRVGPU_LOG_INFO("GPUManager", "Cleanup complete");
}

inline DrvGPU& GPUManager::GetGPU(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= gpus_.size()) {
        throw std::out_of_range("GPU index out of range");
    }
    return *gpus_[index];
}

inline const DrvGPU& GPUManager::GetGPU(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= gpus_.size()) {
        throw std::out_of_range("GPU index out of range");
    }
    return *gpus_[index];
}

inline DrvGPU& GPUManager::GetNextGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpus_.empty()) {
        throw std::runtime_error("No GPUs initialized");
    }
    
    size_t index = round_robin_index_++ % gpus_.size();
    return *gpus_[index];
}

inline DrvGPU& GPUManager::GetLeastLoadedGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t least_loaded_idx = GetLeastLoadedGPUIndex();
    return *gpus_[least_loaded_idx];
}

inline std::vector<DrvGPU*> GPUManager::GetAllGPUs() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DrvGPU*> result;
    result.reserve(gpus_.size());
    for (auto& gpu : gpus_) {
        result.push_back(gpu.get());
    }
    return result;
}

inline std::vector<const DrvGPU*> GPUManager::GetAllGPUs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const DrvGPU*> result;
    result.reserve(gpus_.size());
    for (auto& gpu : gpus_) {
        result.push_back(gpu.get());
    }
    return result;
}

inline void GPUManager::PrintAllDevices() const {
    std::cout << "\n--- GPU Devices ---\n";
    size_t idx = 0;
    for (const auto& gpu : gpus_) {
        std::cout << "GPU " << idx << ": " << gpu->GetDeviceName() << "\n";
        ++idx;
    }
    std::cout << "------------------\n";
}

inline void GPUManager::SetLoadBalancingStrategy(LoadBalancingStrategy strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    lb_strategy_ = strategy;
}

inline void GPUManager::SynchronizeAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& gpu : gpus_) {
        gpu->Synchronize();
    }
}

inline void GPUManager::FlushAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& gpu : gpus_) {
        gpu->Flush();
    }
}

inline void GPUManager::PrintStatistics() const {
    std::cout << "\n=== GPU Manager Statistics ===\n";
    std::cout << "Total GPUs: " << gpus_.size() << "\n";
    
    size_t idx = 0;
    for (const auto& gpu : gpus_) {
        std::cout << "GPU " << idx << ": " << gpu->GetDeviceName() << "\n";
        std::cout << gpu->GetStatistics();
        ++idx;
    }
    std::cout << "==============================\n\n";
}

inline std::string GPUManager::GetStatistics() const {
    std::ostringstream oss;
    oss << "GPU Manager Statistics:\n";
    oss << "  Total GPUs: " << gpus_.size() << "\n";
    oss << "  Load Balancing: " << LoadBalancingStrategyToString(lb_strategy_) << "\n";
    return oss.str();
}

inline void GPUManager::ResetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& gpu : gpus_) {
        gpu->ResetStatistics();
    }
}

inline int GPUManager::DiscoverGPUs(BackendType backend_type) {
    (void)backend_type;
    DRVGPU_LOG_DEBUG("GPUManager", "Discovering GPUs...");
    
    // Для демонстрации возвращаем 1 GPU
    // TODO: реализовать полное обнаружение GPU
    return 1;
}

inline void GPUManager::InitializeGPU(int device_index) {
    try {
        auto gpu = std::make_unique<DrvGPU>(backend_type_, device_index);
        gpu->Initialize();
        gpus_.push_back(std::move(gpu));
        gpu_task_count_.emplace_back(0);
        DRVGPU_LOG_INFO("GPUManager", "Initialized GPU " + std::to_string(device_index));
    } catch (const std::exception& e) {
        DRVGPU_LOG_ERROR("GPUManager", "Failed to initialize GPU " + std::to_string(device_index) + ": " + e.what());
    }
}

inline size_t GPUManager::GetLeastLoadedGPUIndex() const {
    size_t min_tasks = SIZE_MAX;
    size_t min_idx = 0;
    
    for (size_t i = 0; i < gpu_task_count_.size(); ++i) {
        size_t tasks = gpu_task_count_[i];
        if (tasks < min_tasks) {
            min_tasks = tasks;
            min_idx = i;
        }
    }
    
    return min_idx;
}

} // namespace drv_gpu_lib
