#pragma once

/**
 * @file vector_ops_module.hpp
 * @brief Vector Operations Module - примитивные операции с векторами на GPU
 * 
 * Реализует IComputeModule интерфейс для базовых операций:
 * - Добавление/вычитание скаляра
 * - Сложение двух векторов
 * - In-place и out-of-place варианты
 * 
 * @author DrvGPU Team
 * @date 2026-02-03
 */

//#include "inteface/i_compute_module.hpp"
//#include "inteface/i_backend.hpp"
#include "i_compute_module.hpp"
#include "i_backend.hpp"
#include "memory/gpu_buffer.hpp"
#include <CL/cl.h>
#include <string>
#include <memory>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: VectorOpsModule - Модуль операций с векторами
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class VectorOpsModule
 * @brief Compute модуль для примитивных векторных операций
 * 
 * Предоставляет следующие операции:
 * 
 * **Добавление скаляра:**
 * - AddOneOut()     -> C[] = A[] + 1
 * - AddOneInPlace() -> A[] = A[] + 1
 * 
 * **Вычитание скаляра:**
 * - SubOneOut()     -> C[] = A[] - 1
 * - SubOneInPlace() -> A[] = A[] - 1
 * 
 * **Сложение векторов:**
 * - AddVectorsOut()     -> C[] = A[] + B[]
 * - AddVectorsInPlace() -> A[] = A[] + B[]
 * 
 * Использование:
 * @code
 * // Создать и зарегистрировать модуль
 * auto module = std::make_shared<VectorOpsModule>(backend);
 * module->Initialize();
 * registry.RegisterModule("VectorOps", module);
 * 
 * // Выполнить операцию
 * auto input = mem_mgr.CreateBuffer<float>(1024);
 * auto output = mem_mgr.CreateBuffer<float>(1024);
 * module->AddOneOut(input, output, 1024);
 * @endcode
 */
class VectorOpsModule : public IComputeModule {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать VectorOpsModule привязанный к бэкенду
     * @param backend Указатель на IBackend (OpenCL/CUDA/...)
     */
    explicit VectorOpsModule(IBackend* backend);
    
    /**
     * @brief Деструктор (очищает kernels)
     */
    ~VectorOpsModule() override;
    
    // Запрет копирования
    VectorOpsModule(const VectorOpsModule&) = delete;
    VectorOpsModule& operator=(const VectorOpsModule&) = delete;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Реализация IComputeModule: Жизненный цикл
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать модуль (компилировать kernels)
     * @throws std::runtime_error если компиляция не удалась
     */
    void Initialize() override;
    
    /**
     * @brief Проверить инициализацию
     */
    bool IsInitialized() const override { return initialized_; }
    
    /**
     * @brief Очистить ресурсы модуля
     */
    void Cleanup() override;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Реализация IComputeModule: Информация
    // ═══════════════════════════════════════════════════════════════════════
    
    std::string GetName() const override { return "VectorOps"; }
    std::string GetVersion() const override { return "1.0.0"; }
    std::string GetDescription() const override {
        return "Primitive vector operations (add, subtract, scalar operations)";
    }
    
    IBackend* GetBackend() const override { return backend_; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Векторные операции: Добавление скаляра
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Добавить 1 к каждому элементу (out-of-place)
     * 
     * C[i] = A[i] + 1
     * 
     * @param input  Входной буфер A[]
     * @param output Выходной буфер C[]
     * @param size   Размер векторов
     */
    void AddOneOut(
        std::shared_ptr<GPUBuffer<float>> input,
        std::shared_ptr<GPUBuffer<float>> output,
        size_t size);
    
    /**
     * @brief Добавить 1 к каждому элементу (in-place)
     * 
     * A[i] = A[i] + 1
     * 
     * @param data Буфер для модификации A[]
     * @param size Размер вектора
     */
    void AddOneInPlace(
        std::shared_ptr<GPUBuffer<float>> data,
        size_t size);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Векторные операции: Вычитание скаляра
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Вычесть 1 из каждого элемента (out-of-place)
     * 
     * C[i] = A[i] - 1
     * 
     * @param input  Входной буфер A[]
     * @param output Выходной буфер C[]
     * @param size   Размер векторов
     */
    void SubOneOut(
        std::shared_ptr<GPUBuffer<float>> input,
        std::shared_ptr<GPUBuffer<float>> output,
        size_t size);
    
    /**
     * @brief Вычесть 1 из каждого элемента (in-place)
     * 
     * A[i] = A[i] - 1
     * 
     * @param data Буфер для модификации A[]
     * @param size Размер вектора
     */
    void SubOneInPlace(
        std::shared_ptr<GPUBuffer<float>> data,
        size_t size);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Векторные операции: Сложение векторов
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Сложить два вектора (out-of-place)
     * 
     * C[i] = A[i] + B[i]
     * 
     * @param input_a Первый входной буфер A[]
     * @param input_b Второй входной буфер B[]
     * @param output  Выходной буфер C[]
     * @param size    Размер векторов
     */
    void AddVectorsOut(
        std::shared_ptr<GPUBuffer<float>> input_a,
        std::shared_ptr<GPUBuffer<float>> input_b,
        std::shared_ptr<GPUBuffer<float>> output,
        size_t size);
    
    /**
     * @brief Сложить два вектора (in-place)
     * 
     * A[i] = A[i] + B[i]
     * 
     * @param data_a  Первый буфер A[] (будет модифицирован)
     * @param input_b Второй входной буфер B[]
     * @param size    Размер векторов
     */
    void AddVectorsInPlace(
        std::shared_ptr<GPUBuffer<float>> data_a,
        std::shared_ptr<GPUBuffer<float>> input_b,
        size_t size);

private:
    // ═══════════════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════════════
    
    IBackend* backend_;         ///< Указатель на бэкенд (не владеет)
    bool initialized_;          ///< Флаг инициализации
    
    // OpenCL объекты
    cl_program program_;        ///< Скомпилированная программа
    cl_kernel kernel_add_one_out_;        ///< Kernel: add_one (out)
    cl_kernel kernel_add_one_inplace_;    ///< Kernel: add_one (inplace)
    cl_kernel kernel_sub_one_out_;        ///< Kernel: sub_one (out)
    cl_kernel kernel_sub_one_inplace_;    ///< Kernel: sub_one (inplace)
    cl_kernel kernel_add_vectors_out_;    ///< Kernel: add_vectors (out)
    cl_kernel kernel_add_vectors_inplace_;///< Kernel: add_vectors (inplace)
    
    cl_context context_;        ///< Кэш контекста
    cl_device_id device_;       ///< Кэш устройства
    cl_command_queue queue_;    ///< Кэш очереди команд
    
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Загрузить и скомпилировать kernels
     */
    void CompileKernels();
    
    /**
     * @brief Создать kernel объекты из программы
     */
    void CreateKernelObjects();
    
    /**
     * @brief Освободить OpenCL ресурсы
     */
    void ReleaseKernels();
    
    /**
     * @brief Загрузить исходный код kernel из файла
     * @param filename Имя файла (напр., "vector_ops.cl")
     * @return Исходный код kernel
     */
    std::string LoadKernelSource(const std::string& filename);
};

} // namespace drv_gpu_lib
