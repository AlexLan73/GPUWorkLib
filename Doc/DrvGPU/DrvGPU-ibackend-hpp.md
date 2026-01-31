#pragma once

/**
 * @file ibackend.hpp
 * @brief Backend abstraction interface
 * 
 * IBackend определяет абстрактный интерфейс для всех GPU бэкендов.
 * Каждый бэкенд (OpenCL, CUDA, Vulkan) реализует этот интерфейс.
 * 
 * Архитектурный паттерн: Bridge Pattern + Strategy Pattern
 * 
 * @example Backend implementation:
 * @code
 * class BackendOpenCL : public IBackend {
 * public:
 *     void Initialize(uint32_t device_id) override;
 *     std::unique_ptr<IMemoryBuffer> CreateBuffer(...) override;
 *     // ... other methods
 * };
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "memory/i_memory_buffer.hpp"
#include "memory/memory_config.hpp"
#include <memory>
#include <string>
#include <cstdint>

namespace drv_gpu_lib {

// Forward declarations
class IMemoryBuffer;
class DeviceInfo;

// ════════════════════════════════════════════════════════════════════════════
// Interface: IBackend - абстрактный интерфейс для GPU бэкендов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class IBackend
 * @brief Абстрактный интерфейс для GPU бэкендов
 * 
 * Все бэкенды (OpenCL, CUDA, Vulkan) должны реализовывать этот интерфейс.
 * Это позволяет DrvGPU работать с разными бэкендами через единый API.
 * 
 * Паттерн: Bridge Pattern
 * 
 * @code
 * // Пример использования через интерфейс
 * std::unique_ptr<IBackend> backend = CreateOpenCLBackend();
 * backend->Initialize(0);
 * auto buffer = backend->CreateBuffer(size);
 * @endcode
 */
class IBackend {
public:
    // ═══════════════════════════════════════════════════════════════
    // Виртуальный деструктор (RAII)
    // ═══════════════════════════════════════════════════════════════
    
    virtual ~IBackend() = default;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация и управление
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать бэкенд для конкретного устройства
     * @param device_id ID GPU устройства
     * @throws std::runtime_error если инициализация не удалась
     */
    virtual void Initialize(uint32_t device_id) = 0;
    
    /**
     * @brief Проверить инициализацию
     * @return true если бэкенд инициализирован
     */
    virtual bool IsInitialized() const = 0;
    
    /**
     * @brief Синхронизация - ждать завершения всех операций
     */
    virtual void Synchronize() = 0;
    
    /**
     * @brief Очистить ресурсы (опционально, вызывается в деструкторе)
     */
    virtual void Cleanup() = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация о устройстве
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить название устройства
     */
    virtual std::string GetDeviceName() const = 0;
    
    /**
     * @brief Получить производителя
     */
    virtual std::string GetVendor() const = 0;
    
    /**
     * @brief Получить версию драйвера
     */
    virtual std::string GetVersion() const = 0;
    
    /**
     * @brief Получить размер глобальной памяти (MB)
     */
    virtual size_t GetGlobalMemoryMB() const = 0;
    
    /**
     * @brief Получить размер локальной памяти (KB)
     */
    virtual size_t GetLocalMemoryKB() const = 0;
    
    /**
     * @brief Получить количество вычислительных блоков
     */
    virtual uint32_t GetComputeUnits() const = 0;
    
    /**
     * @brief Получить максимальную частоту (MHz)
     */
    virtual uint32_t GetMaxClockMHz() const = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Создание буферов
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать буфер с автовыбором стратегии
     * @param num_elements Количество элементов
     * @param mem_type Тип памяти
     * @param hint Подсказка по использованию
     * @return unique_ptr на IMemoryBuffer
     */
    virtual std::unique_ptr<IMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE,
        const BufferUsageHint& hint = BufferUsageHint::Default()
    ) = 0;
    
    /**
     * @brief Создать буфер с конкретной стратегией
     * @param num_elements Количество элементов
     * @param strategy Стратегия памяти
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    virtual std::unique_ptr<IMemoryBuffer> CreateBufferWithStrategy(
        size_t num_elements,
        MemoryStrategy strategy,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) = 0;
    
    /**
     * @brief Создать буфер с начальными данными
     * @param data Начальные данные
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    virtual std::unique_ptr<IMemoryBuffer> CreateBufferWithData(
        const ComplexVector& data,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) = 0;
    
    /**
     * @brief Обернуть внешний буфер (non-owning)
     * @param external_handle Внешний handle (void* для универсальности)
     * @param num_elements Количество элементов
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    virtual std::unique_ptr<IMemoryBuffer> WrapExternalBuffer(
        void* external_handle,
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Возможности устройства
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Проверить поддержку SVM
     */
    virtual bool SupportsSVM() const = 0;
    
    /**
     * @brief Получить лучшую доступную стратегию памяти
     */
    virtual MemoryStrategy GetBestMemoryStrategy() const = 0;
    
    /**
     * @brief Получить рекомендуемую стратегию для размера
     * @param size_bytes Размер буфера в байтах
     * @return Рекомендуемая стратегия
     */
    virtual MemoryStrategy RecommendStrategy(size_t size_bytes) const = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Kernel execution (опционально для расширенного использования)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Компилировать kernel из исходника
     * @param source Исходный код kernel
     * @param kernel_name Имя kernel функции
     * @param options Опции компиляции
     * @return Handle на kernel (void* для универсальности)
     */
    virtual void* CompileKernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& options = ""
    ) = 0;
    
    /**
     * @brief Освободить kernel
     * @param kernel_handle Handle на kernel
     */
    virtual void ReleaseKernel(void* kernel_handle) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Диагностика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить статистику использования
     */
    virtual std::string GetStatistics() const = 0;
    
    /**
     * @brief Вывести информацию о устройстве
     */
    virtual void PrintDeviceInfo() const = 0;
    
protected:
    // Защищённый конструктор (только для наследников)
    IBackend() = default;
    
    // Запрет копирования
    IBackend(const IBackend&) = delete;
    IBackend& operator=(const IBackend&) = delete;
};

// ════════════════════════════════════════════════════════════════════════════
// Фабричные функции для создания бэкендов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать OpenCL бэкенд
 * @return unique_ptr на IBackend
 */
std::unique_ptr<IBackend> CreateOpenCLBackend();

/**
 * @brief Создать CUDA бэкенд (planned)
 * @return unique_ptr на IBackend
 */
std::unique_ptr<IBackend> CreateCUDABackend();

/**
 * @brief Создать Vulkan бэкенд (planned)
 * @return unique_ptr на IBackend
 */
std::unique_ptr<IBackend> CreateVulkanBackend();

/**
 * @brief Создать бэкенд по типу
 * @param type Тип бэкенда
 * @return unique_ptr на IBackend
 */
std::unique_ptr<IBackend> CreateBackend(BackendType type);

} // namespace drv_gpu_lib
