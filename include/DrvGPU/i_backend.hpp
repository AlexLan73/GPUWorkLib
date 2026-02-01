#pragma once

/**
 * @file i_backend.hpp
 * @brief Абстрактный интерфейс для бэкендов (OpenCL, CUDA, Vulkan)
 * 
 * IBackend - ключевая абстракция в DrvGPU, реализующая Bridge Pattern.
 * Позволяет переключаться между бэкендами без изменения клиентского кода.
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "common/backend_type.hpp"
#include "common/gpu_device_info.hpp"
#include <string>
#include <cstddef>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Interface: IBackend - абстракция GPU бэкенда
// ════════════════════════════════════════════════════════════════════════════

/**
 * @interface IBackend
 * @brief Абстрактный интерфейс для всех GPU бэкендов
 * 
 * Каждый бэкенд (OpenCL, CUDA, Vulkan) реализует этот интерфейс,
 * предоставляя единообразный API для DrvGPU.
 * 
 * Паттерн: Bridge (отделяет абстракцию от реализации)
 * 
 * Основные методы:
 * - Initialize/Cleanup - жизненный цикл
 * - GetNativeHandle - доступ к нативным объектам
 * - Allocate/Free - управление памятью
 * - Synchronize/Flush - синхронизация
 * 
 * Реализации:
 * - OpenCLBackend (см. opencl_backend.hpp)
 * - CUDABackend (будущее)
 * - VulkanBackend (будущее)
 */
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация и жизненный цикл
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать бэкенд для конкретного устройства
     * @param device_index Индекс GPU устройства
     * @throws std::runtime_error при ошибке инициализации
     */
    virtual void Initialize(int device_index) = 0;
    
    /**
     * @brief Проверить, инициализирован ли бэкенд
     */
    virtual bool IsInitialized() const = 0;
    
    /**
     * @brief Очистить ресурсы бэкенда
     */
    virtual void Cleanup() = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Информация об устройстве
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить тип бэкенда
     */
    virtual BackendType GetType() const = 0;
    
    /**
     * @brief Получить информацию об устройстве
     */
    virtual GPUDeviceInfo GetDeviceInfo() const = 0;
    
    /**
     * @brief Получить индекс устройства
     */
    virtual int GetDeviceIndex() const = 0;
    
    /**
     * @brief Получить название устройства
     */
    virtual std::string GetDeviceName() const = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Нативные хэндлы (для прямого доступа к API)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить нативный context
     * OpenCL: возвращает cl_context
     * CUDA: возвращает CUcontext
     * Vulkan: возвращает VkDevice
     */
    virtual void* GetNativeContext() const = 0;
    
    /**
     * @brief Получить нативный device
     * OpenCL: возвращает cl_device_id
     * CUDA: возвращает CUdevice
     * Vulkan: возвращает VkPhysicalDevice
     */
    virtual void* GetNativeDevice() const = 0;
    
    /**
     * @brief Получить нативную command queue/stream
     * OpenCL: возвращает cl_command_queue
     * CUDA: возвращает CUstream
     * Vulkan: возвращает VkQueue
     */
    virtual void* GetNativeQueue() const = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Управление памятью (базовые операции)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Выделить память на GPU
     * @param size_bytes Размер в байтах
     * @param flags Флаги (backend-specific)
     * @return Указатель на выделенную память
     */
    virtual void* Allocate(size_t size_bytes, unsigned int flags = 0) = 0;
    
    /**
     * @brief Освободить память на GPU
     * @param ptr Указатель на память
     */
    virtual void Free(void* ptr) = 0;
    
    /**
     * @brief Копировать данные Host -> Device
     */
    virtual void MemcpyHostToDevice(void* dst, const void* src, 
                                    size_t size_bytes) = 0;
    
    /**
     * @brief Копировать данные Device -> Host
     */
    virtual void MemcpyDeviceToHost(void* dst, const void* src, 
                                    size_t size_bytes) = 0;
    
    /**
     * @brief Копировать данные Device -> Device
     */
    virtual void MemcpyDeviceToDevice(void* dst, const void* src, 
                                      size_t size_bytes) = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Синхронизировать (ждать завершения всех операций)
     */
    virtual void Synchronize() = 0;
    
    /**
     * @brief Flush команд (без ожидания)
     */
    virtual void Flush() = 0;
    
    // ═══════════════════════════════════════════════════════════════
    // Возможности устройства
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Поддерживается ли SVM (Shared Virtual Memory)
     */
    virtual bool SupportsSVM() const = 0;
    
    /**
     * @brief Поддерживается ли double precision
     */
    virtual bool SupportsDoublePrecision() const = 0;
    
    /**
     * @brief Максимальный размер work group
     */
    virtual size_t GetMaxWorkGroupSize() const = 0;
    
    /**
     * @brief Глобальная память (bytes)
     */
    virtual size_t GetGlobalMemorySize() const = 0;
    
    /**
     * @brief Локальная память (bytes)
     */
    virtual size_t GetLocalMemorySize() const = 0;
};

} // namespace DrvGPU
