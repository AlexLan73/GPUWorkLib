#pragma once

/**
 * @file backend_opencl.hpp
 * @brief OpenCL backend implementation
 * 
 * BackendOpenCL - реализация IBackend интерфейса для OpenCL.
 * Базируется на проверенной реализацииDrvGPU.
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "DrvGPU/ibackend.hpp"
#include "opencl_core.hpp"
#include "opencl_buffer_factory.hpp"
#include <CL/cl.h>
#include <memory>
#include <string>

namespace drv_gpu_lib {
namespace OpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Class: BackendOpenCL - OpenCL реализация IBackend
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class BackendOpenCL
 * @brief OpenCL реализация backend интерфейса
 * 
 * Использует:
 * - OpenCLCore для управления контекстом и device
 * - OpenCLBufferFactory для создания буферов
 * - SVM capabilities для автовыбора стратегии
 * 
 * @code
 * auto backend = std::make_unique<BackendOpenCL>();
 * backend->Initialize(0); // GPU 0
 * auto buffer = backend->CreateBuffer(size);
 * @endcode
 */
class BackendOpenCL : public IBackend {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Конструктор (пустой backend)
     */
    BackendOpenCL();
    
    /**
     * @brief Деструктор (RAII cleanup)
     */
    ~BackendOpenCL() override;
    
    // Запрет копирования, разрешение move
    BackendOpenCL(const BackendOpenCL&) = delete;
    BackendOpenCL& operator=(const BackendOpenCL&) = delete;
    BackendOpenCL(BackendOpenCL&&) noexcept;
    BackendOpenCL& operator=(BackendOpenCL&&) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IBackend интерфейса
    // ═══════════════════════════════════════════════════════════════
    
    // --- Инициализация ---
    void Initialize(uint32_t device_id) override;
    bool IsInitialized() const override;
    void Synchronize() override;
    void Cleanup() override;
    
    // --- Информация о устройстве ---
    std::string GetDeviceName() const override;
    std::string GetVendor() const override;
    std::string GetVersion() const override;
    size_t GetGlobalMemoryMB() const override;
    size_t GetLocalMemoryKB() const override;
    uint32_t GetComputeUnits() const override;
    uint32_t GetMaxClockMHz() const override;
    
    // --- Создание буферов ---
    std::unique_ptr<IMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE,
        const BufferUsageHint& hint = BufferUsageHint::Default()
    ) override;
    
    std::unique_ptr<IMemoryBuffer> CreateBufferWithStrategy(
        size_t num_elements,
        MemoryStrategy strategy,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) override;
    
    std::unique_ptr<IMemoryBuffer> CreateBufferWithData(
        const ComplexVector& data,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) override;
    
    std::unique_ptr<IMemoryBuffer> WrapExternalBuffer(
        void* external_handle,
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    ) override;
    
    // --- Возможности устройства ---
    bool SupportsSVM() const override;
    MemoryStrategy GetBestMemoryStrategy() const override;
    MemoryStrategy RecommendStrategy(size_t size_bytes) const override;
    
    // --- Kernel execution ---
    void* CompileKernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& options = ""
    ) override;
    
    void ReleaseKernel(void* kernel_handle) override;
    
    // --- Диагностика ---
    std::string GetStatistics() const override;
    void PrintDeviceInfo() const override;
    
    // ═══════════════════════════════════════════════════════════════
    // OpenCL-специфичные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить OpenCL context
     */
    cl_context GetContext() const;
    
    /**
     * @brief Получить OpenCL device
     */
    cl_device_id GetDevice() const;
    
    /**
     * @brief Получить OpenCL command queue
     */
    cl_command_queue GetQueue() const;
    
    /**
     * @brief Получить SVM capabilities
     */
    SVMCapabilities GetSVMCapabilities() const;
    
private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    std::unique_ptr<OpenCLCore> core_;              ///< OpenCL core
    std::unique_ptr<OpenCLBufferFactory> factory_;  ///< Buffer factory
    bool initialized_;                              ///< Флаг инициализации
    uint32_t device_id_;                            ///< ID устройства
    SVMCapabilities svm_caps_;                      ///< SVM capabilities
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void ValidateInitialized() const;
    void QueryDeviceInfo();
};

} // namespace OpenCL
} // namespace drv_gpu_lib
