#pragma once

/**
 * @file drv_gpu.hpp
 * @brief Главный класс DrvGPU - абстракция GPU устройства (Multi-Instance!)
 * 
 * ВАЖНО: DrvGPU НЕ является Singleton!
 * Для Multi-GPU используйте GPUManager (см. gpu_manager.hpp)
 * 
 * Архитектура:
 * - Backend Abstraction через IBackend интерфейс
 * - RAII управление ресурсами
 * - Thread-safe операции
 * - Поддержка OpenCL (расширяемо на CUDA/Vulkan)
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "i_backend.hpp"
#include "backend_type.hpp"
#include "gpu_device_info.hpp"
#include "memory_manager.hpp"
#include "module_registry.hpp"
#include <memory>
#include <string>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: DrvGPU - Главный класс библиотеки (НЕ Singleton!)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class DrvGPU
 * @brief Абстракция GPU устройства с поддержкой разных бэкендов
 * 
 * DrvGPU предоставляет единый интерфейс для работы с GPU через различные
 * бэкенды (OpenCL, CUDA, Vulkan). Класс НЕ является Singleton - вы можете
 * создать экземпляр для каждой GPU.
 * 
 * Для Multi-GPU сценариев используйте GPUManager:
 * @code
 * // Multi-GPU (правильный способ)
 * GPUManager manager;
 * manager.InitializeAll(BackendType::OPENCL);
 * auto gpu0 = manager.GetGPU(0);
 * auto gpu1 = manager.GetGPU(1);
 * 
 * // Single GPU (можно напрямую)
 * DrvGPU gpu(BackendType::OPENCL, 0);
 * @endcode
 * 
 * Основные возможности:
 * - Backend-агностичный интерфейс
 * - Управление памятью (MemoryManager)
 * - Регистр compute модулей (ModuleRegistry)
 * - RAII для автоматической очистки
 * - Thread-safe
 * 
 * Паттерны:
 * - Bridge Pattern (абстракция бэкенда)
 * - Facade Pattern (упрощённый интерфейс)
 * - RAII (автоматическое управление ресурсами)
 */
class DrvGPU {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать DrvGPU для конкретного устройства
     * @param backend_type Тип бэкенда (OPENCL, CUDA, VULKAN)
     * @param device_index Индекс GPU устройства (0-based)
     * @throws std::runtime_error если устройство недоступно
     */
    explicit DrvGPU(BackendType backend_type, int device_index = 0);
    
    /**
     * @brief Деструктор (RAII - автоматическая очистка)
     */
    ~DrvGPU();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    
    DrvGPU(DrvGPU&& other) noexcept;
    DrvGPU& operator=(DrvGPU&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация и очистка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать GPU
     * @throws std::runtime_error при ошибке инициализации
     */
    void Initialize();
    
    /**
     * @brief Проверить, инициализирован ли GPU
     */
    bool IsInitialized() const { return initialized_; }
    
    /**
     * @brief Очистить все ресурсы (вызывается автоматически в деструкторе)
     */
    void Cleanup();
    
    // ═══════════════════════════════════════════════════════════════
    // Информация об устройстве
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить информацию об устройстве
     */
    GPUDeviceInfo GetDeviceInfo() const;
    
    /**
     * @brief Получить индекс устройства
     */
    int GetDeviceIndex() const { return device_index_; }
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetBackendType() const { return backend_type_; }
    
    /**
     * @brief Получить название устройства
     */
    std::string GetDeviceName() const;
    
    /**
     * @brief Вывести информацию об устройстве
     */
    void PrintDeviceInfo() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к подсистемам
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить менеджер памяти
     */
    MemoryManager& GetMemoryManager();
    const MemoryManager& GetMemoryManager() const;
    
    /**
     * @brief Получить регистр модулей
     */
    ModuleRegistry& GetModuleRegistry();
    const ModuleRegistry& GetModuleRegistry() const;
    
    /**
     * @brief Получить бэкенд (для прямого доступа)
     * ВНИМАНИЕ: Используйте только если абстракции недостаточно!
     */
    IBackend& GetBackend();
    const IBackend& GetBackend() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Дождаться завершения всех операций на GPU
     */
    void Synchronize();
    
    /**
     * @brief Flush всех команд (без ожидания)
     */
    void Flush();
    
    // ═══════════════════════════════════════════════════════════════
    // Статистика и отладка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вывести статистику использования
     */
    void PrintStatistics() const;
    
    /**
     * @brief Получить строку со статистикой
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Сбросить статистику
     */
    void ResetStatistics();

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    BackendType backend_type_;
    int device_index_;
    bool initialized_;
    
    // Backend (Bridge Pattern)
    std::unique_ptr<IBackend> backend_;
    
    // Подсистемы
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<ModuleRegistry> module_registry_;
    
    // Thread-safety
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать бэкенд на основе типа
     */
    void CreateBackend();
    
    /**
     * @brief Инициализировать подсистемы
     */
    void InitializeSubsystems();
};

} // namespace DrvGPU
