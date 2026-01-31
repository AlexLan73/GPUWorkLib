#pragma once

#include "opencl_core.hpp"
#include "kernel_program.hpp"
#include "command_queue_pool.hpp"
#include "memory_type.hpp"
#include "gpu_memory_buffer.hpp"
#include "svm_capabilities.hpp"
#include "hybrid_buffer.hpp"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <array>
#include <stdexcept>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// MemoryType и GPUMemoryBuffer определены в отдельных файлах:
// - memory_type.hpp
// - gpu_memory_buffer.hpp
// Используем их определения
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// OpenCLComputeEngine - ГЛАВНЫЙ ФАСАД
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLComputeEngine
 * @brief Главный фасад для работы с OpenCL
 *
 * Это единственный класс, который пользователь обычно использует.
 * Он объединяет все компоненты:
 * - OpenCLCore (контекст)
 * - KernelProgram (программы + kernels)
 * - GPUMemoryBuffer (память)
 *
 * Использование:
 * ```cpp
 * // 1. Инициализация
 * ManagerOpenCL::OpenCLComputeEngine::Initialize(ManagerOpenCL::DeviceType::GPU);
 * auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
 *
 * // 2. Создать программу и kernel
 * auto program = engine.LoadProgram(kernel_source);
 * auto kernel = engine.GetKernel(program, "my_kernel");
 *
 * // 3. Создать буферы
 * auto input = engine.CreateBuffer(1024, ManagerOpenCL::MemoryType::GPU_READ_WRITE);
 * auto output = engine.CreateBuffer(1024, ManagerOpenCL::MemoryType::GPU_WRITE_ONLY);
 *
 * // 4. Загрузить данные
 * input->WriteToGPU(my_data);
 *
 * // 5. Выполнить kernel
 * engine.ExecuteKernel(kernel, {input.get(), output.get()},
 *                      {{1024, 1, 1}}, {{256, 1, 1}});
 *
 * // 6. Прочитать результаты
 * auto result = output->ReadFromGPU();
 *
 * // 7. Статистика
 * std::cout << engine.GetStatistics();
 * ```
 */
class OpenCLComputeEngine {
public:
    // ═══════════════════════════════════════════════════════════════
    // Singleton
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать OpenCL (один раз)
     */
    static void Initialize(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить Singleton
     */
    static OpenCLComputeEngine& GetInstance();

    /**
     * @brief Проверить инициализацию
     */
    static bool IsInitialized();

    /**
     * @brief Очистка (опционально)
     */
    static void Cleanup();

    // ═══════════════════════════════════════════════════════════════
    // Программы и kernels
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Загрузить OpenCL программу (компилируется один раз благодаря кэшу)
     * @param source OpenCL C код
     * @return Shared pointer на KernelProgram (не удалять вручную)
     */
    std::shared_ptr<KernelProgram> LoadProgram(const std::string& source);

    /**
     * @brief Получить kernel из программы
     * @param program Программа (из LoadProgram)
     * @param kernel_name Имя kernel в программе
     * @return cl_kernel (управляется программой)
     */
    cl_kernel GetKernel(
        const std::shared_ptr<KernelProgram>& program,
        const std::string& kernel_name
    );

    // ═══════════════════════════════════════════════════════════════
    // Память
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Создать GPU буфер
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief Создать GPU буфер с начальными данными
     */
    std::unique_ptr<GPUMemoryBuffer> CreateBufferWithData(
        const std::vector<std::complex<float>>& data,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    /**
     * @brief Создать GPU буфер с начальными данными для любого POD-типа
     */
    template <typename T>
    std::unique_ptr<GPUMemoryBuffer> CreateTypedBufferWithData(
        const std::vector<T>& data,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    // ═══════════════════════════════════════════════════════════════
    // Выполнение kernels
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Выполнить kernel синхронно (ждёт завершения)
     * @param kernel Kernel для выполнения
     * @param buffers Буферы (в порядке аргументов kernel)
     * @param global_work_size Глобальные размеры работы
     * @param local_work_size Локальные размеры работы
     */
    void ExecuteKernel(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size
    );

    /**
     * @brief Выполнить kernel асинхронно (возвращает event)
     * @return cl_event для синхронизации
     */
    cl_event ExecuteKernelAsync(
        cl_kernel kernel,
        const std::vector<cl_mem>& buffers,
        const std::array<size_t, 3>& global_work_size,
        const std::array<size_t, 3>& local_work_size
    );

    // ═══════════════════════════════════════════════════════════════
    // Синхронизация
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Ждать завершения всех операций в queue
     */
    void Flush();

    /**
     * @brief Ждать полного завершения queue
     */
    void Finish();

    /**
     * @brief Ждать завершения конкретного события
     */
    void WaitForEvent(cl_event event);

    /**
     * @brief Ждать завершения нескольких событий
     */
    void WaitForEvents(const std::vector<cl_event>& events);

    // ═══════════════════════════════════════════════════════════════
    // НОВАЯ СИСТЕМА ПАМЯТИ (SVM/Hybrid)
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Создать фабрику буферов с автовыбором стратегии
     * @param config Конфигурация (опционально)
     * @return unique_ptr на BufferFactory
     * 
     * @code
     * auto factory = engine.CreateBufferFactory();
     * auto buffer = factory->Create(1024);  // Автовыбор SVM/Regular
     * @endcode
     */
    std::unique_ptr<BufferFactory> CreateBufferFactory(
        const BufferConfig& config = BufferConfig::Default()
    );

    /**
     * @brief Создать буфер с автовыбором стратегии (удобный метод)
     * @param num_elements Количество элементов
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateHybridBuffer(
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief Создать буфер с конкретной стратегией
     * @param num_elements Количество элементов
     * @param strategy Стратегия (REGULAR, SVM_COARSE, etc.)
     * @param mem_type Тип памяти
     * @return unique_ptr на IMemoryBuffer
     */
    std::unique_ptr<IMemoryBuffer> CreateBufferWithStrategy(
        size_t num_elements,
        MemoryStrategy strategy,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );

    /**
     * @brief Получить SVM capabilities текущего устройства
     */
    SVMCapabilities GetSVMCapabilities() const;

    /**
     * @brief Проверить поддержку SVM
     */
    bool IsSVMSupported() const;

    /**
     * @brief Получить информацию о SVM
     */
    std::string GetSVMInfo() const;

    // ═══════════════════════════════════════════════════════════════
    // Информация и статистика
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить подробную статистику
     */
    std::string GetStatistics() const;

    /**
     * @brief Получить информацию о девайсе
     */
    std::string GetDeviceInfo() const;

    /**
     * @brief Получить кэш статистику
     */
    std::string GetCacheStatistics() const;

    // ═══════════════════════════════════════════════════════════════
    // Деструктор
    // ═══════════════════════════════════════════════════════════════

    ~OpenCLComputeEngine();

    OpenCLComputeEngine(const OpenCLComputeEngine&) = delete;
    OpenCLComputeEngine& operator=(const OpenCLComputeEngine&) = delete;

private:
    // ═══════════════════════════════════════════════════════════════
    // Singleton реализация
    // ═══════════════════════════════════════════════════════════════

    OpenCLComputeEngine();

    static std::unique_ptr<OpenCLComputeEngine> instance_;
    static bool initialized_;
    static std::mutex initialization_mutex_;

    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    size_t total_allocated_bytes_;
    size_t num_buffers_;
    size_t kernel_executions_;
};

// ==========================
// Inline-реализации шаблонов
// ==========================
template <typename T>
inline std::unique_ptr<GPUMemoryBuffer>
OpenCLComputeEngine::CreateTypedBufferWithData(
    const std::vector<T>& data,
    MemoryType type)
{
    if (data.empty()) {
        throw std::invalid_argument(
            "CreateTypedBufferWithData: data vector is empty"
        );
    }

    auto& core = OpenCLCore::GetInstance();
    cl_command_queue queue = CommandQueuePool::GetNextQueue();

    // Создаём буфер с инициализацией из памяти хоста
    auto buffer = std::make_unique<GPUMemoryBuffer>(
        core.GetContext(),
        queue,
        static_cast<const void*>(data.data()),
        data.size() * sizeof(T),
        data.size(),
        type
    );

    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;

    return buffer;
}

}  // namespace ManagerOpenCL
