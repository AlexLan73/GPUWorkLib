#pragma once

/**
 * @file opencl_backend.hpp
 * @brief Реализация IBackend для OpenCL
 * 
 * OpenCLBackend - полная реализация бэкенда на базе вашего кода OpenCL.
 * Интегрирует ваши классы: OpenCLCore, CommandQueuePool, GPUMemoryManager.
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "i_backend.hpp"
#include "backend_type.hpp"
#include "gpu_device_info.hpp"

// Включаем ваш OpenCL код
#include "opencl_core.hpp"
#include "command_queue_pool.hpp"
#include "gpu_memory_manager.hpp"
#include "svm_capabilities.hpp"

#include <memory>
#include <string>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: OpenCLBackend - Реализация бэкенда для OpenCL
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLBackend
 * @brief Реализация IBackend интерфейса для OpenCL API
 * 
 * Интегрирует вашу существующую OpenCL библиотеку:
 * - ManagerOpenCL::OpenCLCore - управление OpenCL контекстом
 * - ManagerOpenCL::CommandQueuePool - пул command queues
 * - ManagerOpenCL::GPUMemoryManager - управление памятью
 * - ManagerOpenCL::SVMCapabilities - проверка SVM
 * 
 * Особенности:
 * - НЕ Singleton (каждый экземпляр для своей GPU)
 * - Thread-safe
 * - Полная интеграция с вашим кодом
 * - RAII управление ресурсами
 */
class OpenCLBackend : public IBackend {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать OpenCL бэкенд (без инициализации)
     */
    OpenCLBackend();
    
    /**
     * @brief Деструктор (RAII cleanup)
     */
    ~OpenCLBackend() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    OpenCLBackend(const OpenCLBackend&) = delete;
    OpenCLBackend& operator=(const OpenCLBackend&) = delete;
    
    OpenCLBackend(OpenCLBackend&& other) noexcept;
    OpenCLBackend& operator=(OpenCLBackend&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Инициализация
    // ═══════════════════════════════════════════════════════════════
    
    void Initialize(int device_index) override;
    bool IsInitialized() const override { return initialized_; }
    void Cleanup() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Информация об устройстве
    // ═══════════════════════════════════════════════════════════════
    
    BackendType GetType() const override { 
        return BackendType::OPENCL; 
    }
    
    GPUDeviceInfo GetDeviceInfo() const override;
    int GetDeviceIndex() const override { return device_index_; }
    std::string GetDeviceName() const override;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Нативные хэндлы
    // ═══════════════════════════════════════════════════════════════
    
    void* GetNativeContext() const override;
    void* GetNativeDevice() const override;
    void* GetNativeQueue() const override;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Управление памятью
    // ═══════════════════════════════════════════════════════════════
    
    void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
    void Free(void* ptr) override;
    
    void MemcpyHostToDevice(void* dst, const void* src, 
                           size_t size_bytes) override;
    void MemcpyDeviceToHost(void* dst, const void* src, 
                           size_t size_bytes) override;
    void MemcpyDeviceToDevice(void* dst, const void* src, 
                             size_t size_bytes) override;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Синхронизация
    // ═══════════════════════════════════════════════════════════════
    
    void Synchronize() override;
    void Flush() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend: Возможности устройства
    // ═══════════════════════════════════════════════════════════════
    
    bool SupportsSVM() const override;
    bool SupportsDoublePrecision() const override;
    size_t GetMaxWorkGroupSize() const override;
    size_t GetGlobalMemorySize() const override;
    size_t GetLocalMemorySize() const override;
    
    // ═══════════════════════════════════════════════════════════════
    // Специфичные для OpenCL методы (расширение интерфейса)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить OpenCLCore (ваш класс)
     */
    ManagerOpenCL::OpenCLCore& GetCore();
    const ManagerOpenCL::OpenCLCore& GetCore() const;
    
    /**
     * @brief Получить GPUMemoryManager (ваш класс)
     */
    ManagerOpenCL::GPUMemoryManager& GetMemoryManager();
    const ManagerOpenCL::GPUMemoryManager& GetMemoryManager() const;
    
    /**
     * @brief Получить SVMCapabilities
     */
    const ManagerOpenCL::SVMCapabilities& GetSVMCapabilities() const;
    
    /**
     * @brief Инициализировать CommandQueuePool
     * @param num_queues Количество очередей (0 = auto)
     */
    void InitializeCommandQueuePool(size_t num_queues = 0);

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    int device_index_;
    bool initialized_;
    
    // Интеграция с вашим OpenCL кодом
    std::unique_ptr<ManagerOpenCL::OpenCLCore> opencl_core_;
    std::unique_ptr<ManagerOpenCL::GPUMemoryManager> memory_manager_;
    std::unique_ptr<ManagerOpenCL::SVMCapabilities> svm_capabilities_;
    
    // OpenCL objects (кэшируем для быстрого доступа)
    cl_context context_;
    cl_device_id device_;
    cl_command_queue queue_;
    
    // Thread-safety
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void InitializeOpenCLCore();
    void InitializeMemoryManager();
    void InitializeSVMCapabilities();
    GPUDeviceInfo QueryDeviceInfo() const;
};

} // namespace DrvGPU
