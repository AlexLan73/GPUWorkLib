#pragma once

/**
 * @file antenna_module.hpp
 * @brief Antenna FFT Module - FFT обработка с поиском максимальных частот
 * 
 * Портирован из AntennaFFTProcMax, адаптирован под DrvGPU архитектуру.
 * Основная функция: ProcessNew() - автоматический выбор стратегии обработки.
 * 
 * Без генерации сигнала (данные передаются через SVM).
 * Без профилирования (убрано согласно требованиям).
 * 
 * @author DrvGPU Team
 * @date 2026-02-03
 */

#include "common/i_compute_module.hpp"
#include "common/i_backend.hpp"
#include "memory/svm_buffer.hpp"
#include "memory/gpu_buffer.hpp"
#include "antenna_params.hpp"
#include "antenna_result.hpp"
#include <CL/cl.h>
#include <clFFT.h>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace drv_gpu_lib {
namespace antenna {

/**
 * @class AntennaModule
 * @brief Модуль FFT обработки для антенной системы
 * 
 * ОСНОВНОЙ МЕТОД: ProcessNew()
 * 
 * Автоматически выбирает стратегию:
 * - Single-batch: если памяти хватает, обрабатывает все лучи за один проход
 * - Multi-batch: если памяти не хватает, разбивает на батчи
 * 
 * PIPELINE:
 * 1. Padding kernel: копирование + zero padding (count_points → nFFT)
 * 2. FFT: clFFT трансформация
 * 3. Post kernel: выбор диапазона + вычисление magnitude/phase
 * 4. Reduction kernel: поиск топ-N максимумов на GPU
 * 
 * ОСОБЕННОСТИ:
 * - Кэширование буферов и FFT планов для переиспользования
 * - Поддержка SVM буферов (zero-copy с CPU)
 * - Batch processing с умным выбором размера батча
 * - Все вычисления на GPU (включая поиск максимумов)
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * ```cpp
 * // 1. Инициализация
 * DrvGPU gpu(BackendType::OPENCL, 0);
 * gpu.Initialize();
 * 
 * // 2. Создание модуля
 * AntennaParams params(5, 1000, 512, 3); // 5 лучей, 1000 точек, 512 FFT out, 3 пика
 * auto module = std::make_shared<AntennaModule>(&gpu.GetBackend(), params);
 * module->Initialize();
 * 
 * // 3. Подготовка данных (SVM буфер)
 * auto signal = mem.CreateSVMBuffer<std::complex<float>>(5 * 1000);
 * // ... заполнить signal данными на CPU ...
 * 
 * // 4. Обработка
 * auto result = module->ProcessNew(signal);
 * 
 * // 5. Результаты
 * for (size_t i = 0; i < result.results.size(); ++i) {
 *     auto& beam = result.results[i];
 *     std::cout << "Beam " << i << ": max freq = " << beam.refined_frequency << " Hz\n";
 * }
 * ```
 */
class AntennaModule : public IComputeModule {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать Antenna модуль
     * @param backend Указатель на IBackend (OpenCL)
     * @param params Параметры обработки FFT
     */
    explicit AntennaModule(IBackend* backend, const AntennaParams& params);
    
    /**
     * @brief Деструктор (очищает kernels, планы FFT, буферы)
     */
    ~AntennaModule() override;
    
    // Запрет копирования, поддержка перемещения
    AntennaModule(const AntennaModule&) = delete;
    AntennaModule& operator=(const AntennaModule&) = delete;
    AntennaModule(AntennaModule&&) noexcept;
    AntennaModule& operator=(AntennaModule&&) noexcept;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Реализация IComputeModule: Жизненный цикл
    // ═══════════════════════════════════════════════════════════════════════
    
    void Initialize() override;
    bool IsInitialized() const override { return initialized_; }
    void Cleanup() override;
    
    std::string GetName() const override { return "Antenna"; }
    std::string GetVersion() const override { return "1.0.0"; }
    std::string GetDescription() const override {
        return "FFT processing with maximum frequency detection for antenna systems";
    }
    
    IBackend* GetBackend() const override { return backend_; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // ГЛАВНЫЙ МЕТОД: ProcessNew()
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Обработать FFT с автоматическим выбором стратегии
     * 
     * Автоматически выбирает между:
     * - SINGLE-BATCH: полная обработка если памяти хватает
     * - MULTI-BATCH: batch processing если памяти не хватает
     * 
     * @param input_signal SVM буфер с входными данными (beam_count * count_points комплексных чисел)
     * @return AntennaFFTResult с найденными максимумами для каждого луча
     */
    AntennaFFTResult ProcessNew(std::shared_ptr<SVMBuffer<std::complex<float>>> input_signal);
    
    /**
     * @brief Обработать FFT с автоматическим выбором стратегии (GPU буфер)
     * 
     * @param input_signal GPU буфер с входными данными (beam_count * count_points комплексных чисел)
     * @return AntennaFFTResult с найденными максимумами для каждого луча
     */
    AntennaFFTResult ProcessNew(std::shared_ptr<GPUBuffer<std::complex<float>>> input_signal);
    
    /**
     * @brief Обработать FFT с автоматическим выбором стратегии (cl_mem напрямую)
     * 
     * @param input_signal OpenCL cl_mem буфер с входными данными
     * @return AntennaFFTResult с найденными максимумами для каждого луча
     */
    AntennaFFTResult ProcessNew(cl_mem input_signal);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Публичные утилиты
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить вычисленный размер nFFT
     */
    size_t GetNFFT() const { return nFFT_; }
    
    /**
     * @brief Обновить параметры (пересоздаст план FFT если нужно)
     */
    void UpdateParams(const AntennaParams& params);
    
    /**
     * @brief Получить конфигурацию batch processing
     */
    BatchConfig& GetBatchConfig() { return batch_config_; }
    const BatchConfig& GetBatchConfig() const { return batch_config_; }

private:
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы: FFT размер
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Вычислить nFFT из count_points
     * Округляет до степени 2, затем умножает на 2
     */
    size_t CalculateNFFT(size_t count_points) const;
    
    bool IsPowerOf2(size_t n) const;
    size_t NextPowerOf2(size_t n) const;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы: Память и стратегия
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Оценить требуемую память для текущих параметров
     * @return Размер в байтах
     */
    size_t EstimateRequiredMemory() const;
    
    /**
     * @brief Проверить достаточно ли доступной памяти
     * @param required_memory Требуемая память в байтах
     * @return true если памяти хватает для полной обработки
     */
    bool CheckAvailableMemory(size_t required_memory) const;
    
    /**
     * @brief Рассчитать размер батча (количество лучей)
     * @param total_beams Общее количество лучей
     * @return Размер батча (минимум 1 луч)
     */
    size_t CalculateBatchSize(size_t total_beams) const;
    
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы: Создание ресурсов
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать или переиспользовать FFT план для полной обработки
     */
    void CreateOrReuseFFTPlan();
    
    /**
     * @brief Создать FFT план для batch processing
     * @param batch_size Количество лучей в батче
     */
    void CreateBatchFFTPlan(size_t batch_size);
    
    /**
     * @brief Создать OpenCL kernels (padding, post, reduction)
     */
    void CreateKernels();
    
    /**
     * @brief Освободить FFT планы
     */
    void ReleaseFFTPlan();
    
    /**
     * @brief Освободить kernels
     */
    void ReleaseKernels();
    
    /**
     * @brief Загрузить исходный код kernel из файла
     */
    std::string LoadKernelSource(const std::string& filename);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы: Обработка
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Полная обработка (single-batch)
     * @param input_signal Входной буфер cl_mem
     * @return Результаты для всех лучей
     */
    AntennaFFTResult ProcessSingleBatch(cl_mem input_signal);
    
    /**
     * @brief Обработка с разбиением на батчи (multi-batch)
     * @param input_signal Входной буфер cl_mem
     * @return Результаты для всех лучей
     */
    AntennaFFTResult ProcessMultiBatch(cl_mem input_signal);
    
    /**
     * @brief Обработать один батч лучей
     * @param input_signal Входной буфер (все лучи)
     * @param start_beam Индекс первого луча в батче
     * @param num_beams Количество лучей в батче
     * @return Результаты для лучей этого батча
     */
    std::vector<BeamFFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams);
    
    /**
     * @brief Выполнить поиск максимумов на GPU
     * @param num_beams Количество лучей
     * @return Результаты для каждого луча
     */
    std::vector<BeamFFTResult> FindMaximaOnGPU(size_t num_beams);
    
    // ═══════════════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════════════
    
    IBackend* backend_;              ///< Указатель на бэкенд (не владеет)
    AntennaParams params_;           ///< Параметры обработки
    BatchConfig batch_config_;       ///< Конфигурация batch processing
    bool initialized_;               ///< Флаг инициализации
    size_t nFFT_;                    ///< Вычисленный размер FFT
    
    // OpenCL ресурсы
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    
    // clFFT планы
    clfftPlanHandle main_plan_handle_;     ///< План для полной обработки
    clfftPlanHandle batch_plan_handle_;    ///< План для batch обработки
    size_t batch_plan_beams_;              ///< Размер батча для текущего плана
    
    // OpenCL kernels
    cl_kernel padding_kernel_;       ///< Kernel для padding данных
    cl_kernel post_kernel_;          ///< Kernel для post-processing (magnitude + select)
    cl_kernel reduction_kernel_;     ///< Kernel для поиска максимумов
    cl_program program_;             ///< Программа с kernels
    
    // Кэшированные буферы для полной обработки
    std::unique_ptr<GPUBuffer<std::complex<float>>> buffer_fft_input_;
    std::unique_ptr<GPUBuffer<std::complex<float>>> buffer_fft_output_;
    
    // Кэшированные буферы для batch обработки
    std::unique_ptr<GPUBuffer<std::complex<float>>> batch_fft_input_;
    std::unique_ptr<GPUBuffer<std::complex<float>>> batch_fft_output_;
    size_t batch_buffers_size_;      ///< Текущий размер буферов (num_beams)
    
    // Буфер для результатов (MaxValue структуры)
    std::unique_ptr<GPUBuffer<float>> buffer_maxima_;
};

} // namespace antenna
} // namespace drv_gpu_lib
