#pragma once

/**
 * @file opencl_backend_external.hpp
 * @brief OpenCL Backend для интеграции с существующим OpenCL контекстом
 * 
 * OpenCLBackendExternal позволяет использовать DrvGPU с вашим существующим
 * cl_context, cl_device_id, cl_command_queue БЕЗ передачи владения.
 * 
 * КЛЮЧЕВОЕ ОТЛИЧИЕ от OpenCLBackend:
 * - НЕ создаёт новый OpenCL контекст
 * - НЕ освобождает ресурсы при деструкции (owns_resources_ = false)
 * - Использует ваш существующий контекст/очередь
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include "opencl_backend.hpp"
#include <CL/cl.h>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// OpenCLBackendExternal - для интеграции с внешним OpenCL контекстом
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLBackendExternal
 * @brief Backend для работы с существующим OpenCL контекстом
 * 
 * Пример использования:
 * 
 * @code
 * // В вашем коде уже есть OpenCL контекст
 * cl_context your_context = ...;
 * cl_device_id your_device = ...;
 * cl_command_queue your_queue = ...;
 * 
 * // Создать external backend
 * auto backend = std::make_unique<OpenCLBackendExternal>();
 * 
 * // Передать ваш контекст (автоматически устанавливает owns_resources_ = false)
 * backend->InitializeFromExternalContext(your_context, your_device, your_queue);
 * 
 * // Использовать DrvGPU
 * DrvGPU gpu(std::move(backend));
 * gpu.Initialize();
 * 
 * // ... работа с GPU
 * 
 * // DrvGPU НЕ освободит ваш контекст/очередь при деструкции
 * // Вы должны освободить их сами:
 * clReleaseCommandQueue(your_queue);
 * clReleaseContext(your_context);
 * clReleaseDevice(your_device);
 * @endcode
 */
class OpenCLBackendExternal : public OpenCLBackend {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать external backend (без инициализации)
     * 
     * Автоматически устанавливает owns_resources_ = false
     */
    OpenCLBackendExternal();
    
    /**
     * @brief Деструктор
     * 
     * Родительский деструктор вызовет Cleanup().
     * Cleanup() проверит owns_resources_ и НЕ освободит ресурсы.
     */
    ~OpenCLBackendExternal() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования и перемещения
    // ═══════════════════════════════════════════════════════════════
    OpenCLBackendExternal(const OpenCLBackendExternal&) = delete;
    OpenCLBackendExternal& operator=(const OpenCLBackendExternal&) = delete;
    OpenCLBackendExternal(OpenCLBackendExternal&&) = delete;
    OpenCLBackendExternal& operator=(OpenCLBackendExternal&&) = delete;
    
    // ═══════════════════════════════════════════════════════════════
    // Инициализация из внешнего контекста
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать из существующего OpenCL контекста
     * 
     * После вызова этого метода, backend готов к использованию.
     * 
     * ✅ ВАЖНО: Автоматически устанавливает owns_resources_ = false
     * 
     * @param external_context Ваш cl_context
     * @param external_device Ваш cl_device_id
     * @param external_queue Ваш cl_command_queue
     * 
     * @throws std::invalid_argument если любой из параметров nullptr
     * 
     * @code
     * // Пример:
     * cl_context ctx = clCreateContext(...);
     * cl_device_id dev = ...;
     * cl_command_queue queue = clCreateCommandQueue(...);
     * 
     * auto backend = std::make_unique<OpenCLBackendExternal>();
     * backend->InitializeFromExternalContext(ctx, dev, queue);
     * 
     * // Теперь DrvGPU будет использовать ваш контекст
     * DrvGPU gpu(std::move(backend));
     * gpu.Initialize();
     * @endcode
     */
    void InitializeFromExternalContext(
        cl_context external_context,
        cl_device_id external_device,
        cl_command_queue external_queue
    );
    
    // ═══════════════════════════════════════════════════════════════
    // Переопределение Initialize() - блокируем обычную инициализацию
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief ЗАБЛОКИРОВАНО для external backend
     * @throws std::runtime_error всегда
     * 
     * Для external backend используйте InitializeFromExternalContext()
     */
    void Initialize(int device_index) override;
};

} // namespace drv_gpu_lib
