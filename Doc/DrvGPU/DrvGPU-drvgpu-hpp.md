#pragma once

/**
 * @file drvgpu.hpp
 * @brief Main DrvGPU class - GPU driver abstraction
 * 
 * DrvGPU represents a single GPU instance with backend abstraction.
 * For multi-GPU scenarios, create multiple DrvGPU instances via GPUManager.
 * 
 * @example Basic usage:
 * @code
 * DrvGPU::GPUManager manager;
 * manager.Initialize(DrvGPU::BackendType::OPENCL);
 * 
 * auto drv = manager.CreateDrvGPU(0); // GPU 0
 * auto buffer = drv->CreateBuffer(1024);
 * buffer->Write(data);
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "ibackend.hpp"
#include "memory/buffer_factory.hpp"
#include "memory/i_memory_buffer.hpp"
#include <memory>
#include <string>
#include <cstdint>

namespace drv_gpu_lib {

// Forward declarations
class IBackend;
class BufferFactory;
class GPUManager;

// ════════════════════════════════════════════════════════════════════════════
// Enum: BackendType - тип бэкенда
// ════════════════════════════════════════════════════════════════════════════

/**
 * @enum BackendType
 * @brief Поддерживаемые GPU бэкенды
 */
enum class BackendType {
    OPENCL,     ///< OpenCL backend (ready)
    CUDA,       ///< CUDA backend (planned)
    VULKAN,     ///< Vulkan Compute backend (planned)
    AUTO        ///< Автовыбор лучшего доступного бэкенда
};

/**
 * @brief Преобразовать тип бэкенда в строку
 */
inline std::string BackendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::OPENCL: return "OpenCL";
        case BackendType::CUDA:   return "CUDA";
        case BackendType::VULKAN: return "Vulkan";
        case BackendType::AUTO:   return "Auto";
        default:                  return "Unknown";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Struct: DeviceInfo - информация о GPU устройстве
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct DeviceInfo
 * @brief Информация о GPU устройстве
 */
struct DeviceInfo {
    std::string name;                   ///< Название устройства
    std::string vendor;                 ///< Производитель
    std::string version;                ///< Версия драйвера
    size_t global_memory_mb;            ///< Глобальная память (MB)
    size_t local_memory_kb;             ///< Локальная память (KB)
    uint32_t compute_units;             ///< Количество вычислительных блоков
    uint32_t max_clock_mhz;             ///< Максимальная частота (MHz)
    BackendType backend_type;           ///< Тип бэкенда
    
    /**
     * @brief Вывести информацию в строку
     */
    std::string ToString() const;
};

// ════════════════════════════════════════════════════════════════════════════
// Class: DrvGPU - главный класс для работы с GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class DrvGPU
 * @brief Главный класс для работы с одной GPU
 * 
 * DrvGPU инкапсулирует:
 * - Backend abstraction (OpenCL, CUDA, Vulkan)
 * - Memory management (BufferFactory)
 * - Kernel execution
 * - Device information
 * 
 * Архитектурный паттерн: Facade + Bridge
 * 
 * ВАЖНО: DrvGPU НЕ singleton! Для multi-GPU создавайте экземпляры через GPUManager.
 * 
 * @code
 * // Single GPU
 * auto drv = manager.CreateDrvGPU(0);
 * 
 * // Multi-GPU
 * auto drv0 = manager.CreateDrvGPU(0);
 * auto drv1 = manager.CreateDrvGPU(1);
 * @endcode
 */
class DrvGPU {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать DrvGPU для конкретного устройства
     * @param backend Указатель на бэкенд (ownership передаётся)
     * @param device_id ID устройства
     * @throws std::runtime_error если инициализация не удалась
     */
    DrvGPU(std::unique_ptr<IBackend> backend, uint32_t device_id);
    
    /**
     * @brief Деструктор (RAII cleanup)
     */
    ~DrvGPU();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение move
    // ═══════════════════════════════════════════════════════════════
    
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    DrvGPU(DrvGPU&&) noexcept;
    DrvGPU& operator=(DrvGPU&&) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация о устройстве
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить информацию о устройстве
     */
    DeviceInfo GetDeviceInfo() const;
    
    /**
     * @brief Вывести информацию о устройстве в консоль
     */
    void PrintDeviceInfo() const;
    
    /**
     * @brief Получить ID устройства
     */
    uint32_t GetDeviceID() const { return device_id_; }
    
    /**
     * @brief Получить тип бэкенда
     */
    BackendType GetBackendType() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Создание фабрики буферов
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать BufferFactory с автовыбором стратегии
     * @return unique_ptr на BufferFactory
     */
    std::unique_ptr<BufferFactory> CreateBufferFactory() const;
    
    /**
     * @brief Создать BufferFactory с конкретной конфигурацией
     * @param config Конфигурация буфера
     * @return unique_ptr на BufferFactory
     */
    std::unique_ptr<BufferFactory> CreateBufferFactory(
        const BufferConfig& config
    ) const;
    
    // ═══════════════════════════════════════════════════════════════
    // Быстрое создание буферов (convenience methods)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать буфер с автовыбором стратегии
     * @param num_elements Количество элементов
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) const;
    
    /**
     * @brief Создать буфер с начальными данными
     * @param data Начальные данные
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateBufferWithData(
        const ComplexVector& data,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) const;
    
    // ═══════════════════════════════════════════════════════════════
    // Доступ к бэкенду (для расширенного использования)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить указатель на бэкенд
     * @return Указатель на IBackend (не владеющий)
     */
    IBackend* GetBackend() const { return backend_.get(); }
    
    /**
     * @brief Синхронизация - ждать завершения всех операций
     */
    void Synchronize() const;
    
    // ═══════════════════════════════════════════════════════════════
    // Диагностика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить статистику использования
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Проверить поддержку SVM
     */
    bool SupportsSVM() const;
    
private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    std::unique_ptr<IBackend> backend_;    ///< Backend implementation
    uint32_t device_id_;                   ///< Device ID
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void ValidateBackend() const;
};

} // namespace drv_gpu_lib
