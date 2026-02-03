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

#include "../../common/i_backend.hpp"
#include "../../common/backend_type.hpp"
#include "../../common/gpu_device_info.hpp"

// Включаем ваш OpenCL код (из той же папки backends/opencl)
#include "opencl_core.hpp"
#include "command_queue_pool.hpp"

// Включаем memory модуль (из папки memory)
#include "../../memory/memory_manager.hpp"
#include "../../memory/svm_capabilities.hpp"

#include <CL/cl.h>
#include <memory>
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
 * - drv_gpu_lib::OpenCLCore - управление OpenCL контекстом
 * - drv_gpu_lib::CommandQueuePool - пул command queues
 * - drv_gpu_lib::MemoryManager - управление памятью
 * - drv_gpu_lib::SVMCapabilities - проверка SVM
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
    // ✅ НОВОЕ: Реализация IBackend: Управление владением ресурсами
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Установить режим владения OpenCL ресурсами
     * 
     * @param owns true = backend создал context/queue сам → освободит их
     *             false = context/queue пришли извне → НЕ освободит
     * 
     * Устанавливается автоматически:
     * - Initialize(device_index) → owns_resources_ = true
     * - InitializeFromExternalContext() → owns_resources_ = false
     * 
     * Можно изменить вручную для специальных сценариев.
     */
    void SetOwnsResources(bool owns) override { owns_resources_ = owns; }
    
    /**
     * @brief Проверить, владеет ли backend OpenCL ресурсами
     * @return true если backend освободит context/queue при Cleanup()
     */
    bool OwnsResources() const override { return owns_resources_; }
    
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
    drv_gpu_lib::OpenCLCore& GetCore();
    const drv_gpu_lib::OpenCLCore& GetCore() const;
    
    /**
     * @brief Получить MemoryManager (ваш класс)
     */
    drv_gpu_lib::MemoryManager* GetMemoryManager() override;
    const drv_gpu_lib::MemoryManager* GetMemoryManager() const override;
    
    drv_gpu_lib::MemoryManager& GetMemoryManagerRef();
    const drv_gpu_lib::MemoryManager& GetMemoryManagerRef() const;
    
    /**
     * @brief Получить SVMCapabilities
     */
    const drv_gpu_lib::SVMCapabilities& GetSVMCapabilities() const;
    
    /**
     * @brief Инициализировать CommandQueuePool
     * @param num_queues Количество очередей (0 = auto)
     */
    void InitializeCommandQueuePool(size_t num_queues = 0);

protected:
    // ═══════════════════════════════════════════════════════════════
    // ✅ Protected члены для доступа из OpenCLBackendExternal
    // ═══════════════════════════════════════════════════════════════
    
    int device_index_;
    bool initialized_;
    
    /**
     * ✅ НОВОЕ: Флаг владения OpenCL ресурсами
     * 
     * true (по умолчанию): Backend создал context/queue сам
     *                      → освободит их в Cleanup()
     * 
     * false: Context/queue пришли извне (external context)
     *        → НЕ освободит их в Cleanup()
     * 
     * Автоматически устанавливается:
     * - Initialize() → true
     * - InitializeFromExternalContext() → false (в наследнике)
     */
    bool owns_resources_;
    
    // Интеграция с вашим OpenCL кодом
    std::unique_ptr<drv_gpu_lib::MemoryManager> memory_manager_;
    std::unique_ptr<drv_gpu_lib::SVMCapabilities> svm_capabilities_;
    
    // OpenCL objects (кэшируем для быстрого доступа)
    cl_context context_;
    cl_device_id device_;
    cl_command_queue queue_;
    
    // Thread-safety
    mutable std::mutex mutex_;

private:
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void InitializeOpenCLCore();
    void InitializeMemoryManager();
    void InitializeSVMCapabilities();
    
    GPUDeviceInfo QueryDeviceInfo() const;
    
    // Дружественный класс для работы с внешним контекстом
    friend class OpenCLBackendExternal;
};

} // namespace drv_gpu_lib
