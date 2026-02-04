#pragma once

#include "interface/antenna_fft_params.h"
#include "ManagerOpenCL/opencl_compute_engine.hpp"
#include "ManagerOpenCL/opencl_core.hpp"
#include "ManagerOpenCL/command_queue_pool.hpp"
#include "ManagerOpenCL/gpu_memory_buffer.hpp"
#include <CL/cl.h>
#include <clFFT.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace antenna_fft {

/**
 * @class AntennaFFTProcMax
 * @brief Высокопроизводительный класс для FFT обработки с поиском максимальных амплитуд
 * 
 * Использует clFFT с callback'ами для максимальной производительности.
 * Все операции поиска максимумов выполняются ТОЛЬКО на GPU.
 * 
 * АРХИТЕКТУРА:
 * - Использует OpenCLComputeEngine для доступа к контексту и очередям
 * - Кэширует clFFT план для переиспользования
 * - Pre-callback: подготовка данных (перенос + padding)
 * - Post-callback: fftshift + вычисление magnitude/phase
 * - Reduction kernel: поиск топ-N максимумов на GPU
 * - Детальное профилирование всех операций
 * 
 * ИСПОЛЬЗОВАНИЕ:
 * ```cpp
 * // Инициализация OpenCL (один раз)
 * ManagerOpenCL::OpenCLComputeEngine::Initialize(ManagerOpenCL::DeviceType::GPU);
 * 
 * // Создание процессора
 * AntennaFFTParams params(5, 1000, 512, 3); // 5 лучей, 1000 точек, 512 выходных точек, 3 максимума
 * AntennaFFTProcMax processor(params);
 * 
 * // Обработка данных
 * cl_mem input_signal = ...; // Входной сигнал на GPU
 * AntennaFFTResult result = processor.Process(input_signal);
 * 
 * // Вывод результатов
 * processor.PrintResults(result);
 * processor.SaveResultsToFile(result, "Reports/result.md");
 * ```
 */
class AntennaFFTProcMax {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // Публичные типы для профилирования
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Структура для хранения профилирования одного батча
     */
    struct BatchProfilingData {
        size_t batch_index = 0;
        size_t start_beam = 0;
        size_t num_beams = 0;
        double padding_time_ms = 0.0;   // Время padding kernel
        double fft_time_ms = 0.0;       // Время FFT
        double post_time_ms = 0.0;      // Время post kernel
        double gpu_time_ms = 0.0;       // Общее GPU время (сумма)
    };
    
    /**
     * @brief Конструктор
     * @param params Параметры обработки (beam_count, count_points, out_count_points_fft, max_peaks_count)
     * @throws std::runtime_error если OpenCLComputeEngine не инициализирован или параметры невалидны
     */
    explicit AntennaFFTProcMax(const AntennaFFTParams& params);
    
    /**
     * @brief Деструктор
     * Освобождает clFFT план и буферы
     */
    ~AntennaFFTProcMax();
    
    // DELETE COPY, ALLOW MOVE
    AntennaFFTProcMax(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax& operator=(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax(AntennaFFTProcMax&&) noexcept;
    AntennaFFTProcMax& operator=(AntennaFFTProcMax&&) noexcept;
    
    /**
     * @brief Основной метод обработки FFT
     * @param input_signal GPU буфер с входными комплексными данными (beam_count * count_points элементов)
     * @return AntennaFFTResult с результатами для всех лучей
     * @throws std::runtime_error если обработка не удалась
     */
    AntennaFFTResult Process(cl_mem input_signal);
    
    /**
     * @brief Обработка FFT с входными данными из CPU
     * @param input_data Вектор комплексных чисел (beam_count * count_points элементов)
     * @return AntennaFFTResult с результатами для всех лучей
     */
    AntennaFFTResult Process(const std::vector<std::complex<float>>& input_data);
    
    /**
     * @brief Новый метод обработки FFT с автоматическим выбором стратегии
     * 
     * Автоматически выбирает между:
     * - Полная обработка (если памяти хватает) - вызывает Process()
     * - Batch processing (если памяти не хватает) - разбивает на батчи
     * 
     * @param input_signal GPU буфер с входными комплексными данными
     * @return AntennaFFTResult с результатами для всех лучей
     */
    AntennaFFTResult ProcessNew(cl_mem input_signal);
    
    /**
     * @brief Обработка батчей с ПАРАЛЛЕЛЬНЫМ выполнением (2-3 потока)
     * 
     * Использует несколько command queues и FFT планов для
     * параллельной обработки батчей на GPU. ПУБЛИЧНЫЙ МЕТОД!
     * 
     * @param input_signal Буфер входных данных на GPU
     * @return AntennaFFTResult с результатами для всех лучей
     */
    AntennaFFTResult ProcessWithBatchingNew(cl_mem input_signal);
    
    /**
     * @brief Вывести результаты в консоль (таблица)
     * @param result Результаты обработки
     */
    void PrintResults(const AntennaFFTResult& result) const;
    
    /**
     * @brief Сохранить результаты в файл (таблица + JSON)
     * @param result Результаты обработки
     * @param filepath Путь к файлу (будет создан в Reports/ если путь относительный)
     */
    void SaveResultsToFile(const AntennaFFTResult& result, const std::string& filepath);
    
    /**
     * @brief Получить статистику профилирования последней операции
     * @return Строка с детальной статистикой
     */
    std::string GetProfilingStats() const;
    
    /**
     * @brief Получить вычисленный размер nFFT
     */
    size_t GetNFFT() const { return nFFT_; }
    
    /**
     * @brief Получить вычисленный размер nFFT (алиас для GetNFFT)
     */
    size_t GetNFFTSize() const { return nFFT_; }
    
    /**
     * @brief Получить последние результаты профилирования
     */
    const FFTProfilingResults& GetLastProfilingResults() const;
    
    /**
     * @brief Обновить параметры (пересоздаст план FFT если нужно)
     * @param params Новые параметры
     */
    void UpdateParams(const AntennaFFTParams& params);

private:
    // ═══════════════════════════════════════════════════════════════
    // Внутренние методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Вычислить nFFT из count_points
     * Проверяет кратность 2^n, дополняет до ближайшего большего, умножает на 2
     */
    size_t CalculateNFFT(size_t count_points) const;
    
    /**
     * @brief Проверить является ли число степенью двойки
     */
    bool IsPowerOf2(size_t n) const;
    
    /**
     * @brief Найти ближайшую большую степень двойки
     */
    size_t NextPowerOf2(size_t n) const;
    
    /**
     * @brief Создать или переиспользовать clFFT план
     */
    void CreateOrReuseFFTPlan();
    
    /**
     * @brief Освободить clFFT план
     */
    void ReleaseFFTPlan();

    /**
     * @brief Создать pre-callback строку для clFFT
     */
    std::string GetPreCallbackSource() const;

    /**
     * @brief Создать post-callback строку для clFFT
     */
    std::string GetPostCallbackSource() const;

    /**
     * @brief Создать reduction kernel для поиска максимумов
     */
    void CreateMaxReductionKernel();
    
    // ═══════════════════════════════════════════════════════════════
    // ОТЛАДОЧНЫЕ методы (без callback'ов)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать clFFT план БЕЗ callback'ов
     */
    void CreateFFTPlanNoCallbacks();
    
    /**
     * @brief Создать clFFT план с ТОЛЬКО pre-callback
     */
    void CreateFFTPlanWithPreCallbackOnly();
    
    /**
     * @brief Создать kernel для padding данных
     */
    void CreatePaddingKernel();
    
    /**
     * @brief Создать kernel для post-processing (magnitude + select)
     */
    void CreatePostKernel();
    
    /**
     * @brief Создать N параллельных kernel'ов для многопоточной обработки
     * @param num_streams Количество параллельных потоков
     */
    void CreateParallelKernels(size_t num_streams);
    
    /**
     * @brief Освободить параллельные kernel'ы
     */
    void ReleaseParallelKernels();
    
    /**
     * @brief Найти максимумы из отдельных буферов (отладка)
     */
    std::vector<std::vector<FFTMaxResult>> FindMaximaFromBuffers(
        cl_mem selected_complex, cl_mem selected_magnitude, size_t search_range);
    
    /**
     * @brief Выполнить поиск максимумов на GPU (event-based pipeline)
     * @param wait_event Событие для ожидания перед запуском (может быть nullptr)
     * @param out_reduction_event Выходное событие reduction kernel (может быть nullptr)
     * @param out_read_event Выходное событие чтения результатов (может быть nullptr)
     * @return Вектор FFTMaxResult с найденными максимумами
     */
    std::vector<std::vector<FFTMaxResult>> FindMaximaAllBeamsOnGPU(
        cl_event wait_event = nullptr,
        cl_event* out_reduction_event = nullptr,
        cl_event* out_read_event = nullptr
    );
    
    /**
     * @brief Выполнить поиск максимумов на GPU с заданными буферами
     * @param selected_complex Буфер с комплексными данными (selected points)
     * @param selected_magnitude Буфер с магнитудами (selected points)
     * @param search_range Количество точек для поиска (out_count_points_fft)
     * @param wait_event Событие для ожидания перед запуском (может быть nullptr)
     * @return Вектор FFTMaxResult с найденными максимумами и фазами
     */
    std::vector<std::vector<FFTMaxResult>> FindMaximaAllBeamsOnGPU(
        cl_mem selected_complex, 
        cl_mem selected_magnitude, 
        size_t search_range,
        cl_event wait_event = nullptr
    );
    
    /**
     * @brief Профилировать событие OpenCL
     * @param event Событие OpenCL
     * @param operation_name Название операции
     * @return Время выполнения в миллисекундах
     */
    double ProfileEvent(cl_event event, const std::string& operation_name);
    
    // ═══════════════════════════════════════════════════════════════
    // Batch Processing методы (для ProcessNew)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Оценить требуемую память для текущих параметров
     * @return Размер в байтах
     */
    size_t EstimateRequiredMemory() const;
    
    /**
     * @brief Проверить достаточно ли доступной памяти
     * @param required_memory Требуемая память в байтах
     * @param threshold Порог использования памяти (0.0-1.0, по умолчанию 0.4 = 40%)
     * @return true если памяти хватает для полной обработки
     */
    bool CheckAvailableMemory(size_t required_memory, double threshold = 0.4) const;
    
    /**
     * @brief Рассчитать размер батча (количество лучей)
     * @param total_beams Общее количество лучей
     * @param batch_percent Процент от общего количества (0.0-1.0, по умолчанию 0.2 = 20%)
     * @return Размер батча (минимум 1 луч)
     */
    size_t CalculateBatchSize(size_t total_beams, double batch_percent = 0.2) const;
    
    /**
     * @brief Обработать один батч лучей
     * @param input_signal Входной буфер (полный, все лучи)
     * @param start_beam Индекс первого луча в батче
     * @param num_beams Количество лучей в батче
     * @param batch_queue Command queue для этого батча
     * @param completion_event Выходное событие завершения (для синхронизации)
     * @return Результаты для лучей этого батча
     */
    std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        cl_command_queue batch_queue,
        cl_event* completion_event,
        BatchProfilingData* out_profiling = nullptr);  // Детальное профилирование
    
    /**
     * @brief Обработать все лучи с использованием batch processing
     * @param input_signal GPU буфер с входными данными
     * @return AntennaFFTResult с результатами для всех лучей
     */
    AntennaFFTResult ProcessWithBatching(cl_mem input_signal);
    
    /**
     * @brief Инициализировать ресурсы для параллельной обработки
     * @param max_beams_per_stream Максимальное количество лучей на поток
     * @param num_streams Количество параллельных потоков (default = batch_config_.num_parallel_streams)
     */
    void InitializeParallelResources(size_t max_beams_per_stream, size_t num_streams = 0);
    
    /**
     * @brief Освободить ресурсы параллельной обработки
     */
    void ReleaseParallelResources();
    
    /**
     * @brief Обработать один батч в указанном потоке
     * @param input_signal Входные данные
     * @param start_beam Начальный луч
     * @param num_beams Количество лучей
     * @param stream_idx Индекс потока (0, 1, 2...)
     * @param completion_event Выходное событие завершения
     * @return Результаты для этого батча
     */
    std::vector<FFTResult> ProcessBatchParallel(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        size_t stream_idx,
        cl_event* completion_event);
    
    /**
     * @brief Запустить батч асинхронно БЕЗ ожидания
     */
    std::vector<FFTResult> ProcessBatchParallelNoWait(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        size_t stream_idx,
        cl_event* completion_event);
    
    /**
     * @brief Прочитать результаты батча после завершения GPU
     */
    std::vector<FFTResult> ReadBatchResults(
        size_t stream_idx,
        size_t num_beams,
        size_t start_beam);
    
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    AntennaFFTParams params_;              // Параметры обработки
    size_t nFFT_;                          // Вычисленный размер FFT
    
    // OpenCL ресурсы
    ManagerOpenCL::OpenCLComputeEngine* engine_;     // Указатель на engine (не владеем)
    cl_context context_;                   // OpenCL контекст
    cl_command_queue queue_;               // Command queue
    cl_device_id device_;                 // OpenCL устройство
    
    // clFFT ресурсы
    clfftPlanHandle plan_handle_;         // Handle плана FFT
    bool plan_created_;                    // Флаг создания плана
    
    // Буферы GPU (persistent для переиспользования)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_input_;      // Входной буфер (может быть внешним)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_fft_input_;  // Буфер для FFT (nFFT * beam_count)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_fft_output_; // Буфер для результатов FFT
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_magnitude_;  // Буфер для magnitude (после post-callback)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_maxima_;     // Буфер для максимумов (после reduction)
    
    // Буферы для отладочной версии (без callback'ов)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_selected_complex_;    // Выбранные точки спектра
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> buffer_selected_magnitude_;  // Magnitude выбранных точек
    
    // Userdata буферы для callback'ов
    cl_mem pre_callback_userdata_;         // Userdata для pre-callback
    cl_mem post_callback_userdata_;        // Userdata для post-callback

    // Reduction kernel
    std::shared_ptr<ManagerOpenCL::KernelProgram> reduction_program_;  // Программа для reduction
    cl_kernel reduction_kernel_;           // Kernel для поиска максимумов
    
    // Отладочные kernel'ы (без callback'ов)
    cl_kernel padding_kernel_;             // Kernel для padding данных (основной)
    cl_kernel post_kernel_;                // Kernel для magnitude + select (основной)
    
    // Массивы kernel'ов для ПАРАЛЛЕЛЬНОЙ обработки (по одному на поток)
    static constexpr size_t MAX_PARALLEL_KERNELS = 8;  // Максимум параллельных потоков
    std::vector<cl_kernel> padding_kernels_;           // padding_kernels_[stream_idx]
    std::vector<cl_kernel> post_kernels_;              // post_kernels_[stream_idx]
    bool parallel_kernels_created_ = false;            // Флаг создания параллельных kernel'ов
    
    // Профилирование
    struct ProfilingData {
        double upload_time_ms;
        double pre_callback_time_ms;
        double fft_time_ms;
        double post_callback_time_ms;
        double reduction_time_ms;
        double download_time_ms;
        double total_time_ms;
    };
    ProfilingData last_profiling_;
    
    // ═══════════════════════════════════════════════════════════════
    // Batch Processing конфигурация и данные
    // ═══════════════════════════════════════════════════════════════
    
    // Кэшируемые буферы для batch processing (создаются один раз)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_fft_input_;     // FFT input buffer
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_fft_output_;    // FFT output buffer
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_input_buffer_;  // Input copy buffer
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_sel_complex_;   // Selected complex output (DEPRECATED)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_sel_magnitude_; // Selected magnitude output (DEPRECATED)
    std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> batch_maxima_;        // MaxValue output для unified kernel
    size_t batch_buffers_size_;                                  // Текущий размер буферов (num_beams)
    
    // Кэшируемый FFT план для batch processing
    clfftPlanHandle batch_plan_handle_;                         // Handle плана FFT для батчей
    size_t batch_plan_beams_;                                   // Для скольких лучей создан план
    
    // ═══════════════════════════════════════════════════════════════
    // ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА: ресурсы для каждого потока
    // ═══════════════════════════════════════════════════════════════
    
    // Набор ресурсов для одного параллельного потока
    struct ParallelResources {
        std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> fft_input;      // FFT input buffer
        std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> fft_output;     // FFT output buffer
        std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> sel_complex;    // (DEPRECATED)
        std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> sel_magnitude;  // (DEPRECATED)
        std::unique_ptr<ManagerOpenCL::GPUMemoryBuffer> maxima;         // MaxValue output для unified kernel
        clfftPlanHandle plan_handle = 0;                       // FFT план для этого потока
        cl_command_queue queue = nullptr;                      // Command queue для этого потока
        bool initialized = false;
    };
    
    std::vector<ParallelResources> parallel_resources_;       // Ресурсы для параллельных потоков
    size_t num_parallel_streams_ = 3;                         // Количество параллельных потоков
    size_t parallel_buffers_size_ = 0;                        // Размер буферов (num_beams)
    
    // Конфигурация batch processing
    struct BatchConfig {
        double memory_usage_limit = 0.65;    // 60% от доступной памяти
        double batch_size_ratio = 0.22;      // 10% лучей на батч
        size_t min_beams_for_batch = 10;    // Минимум лучей для batch режима
        size_t num_parallel_streams = 3;    // 3 параллельных потока
    };
    BatchConfig batch_config_;
    
    // Профилирование для batch режима (структура определена в public)
    std::vector<BatchProfilingData> batch_profiling_;
    double batch_total_cpu_time_ms_;        // Общее CPU время для всех батчей
    double batch_total_padding_ms_;         // Суммарное время padding
    double batch_total_fft_ms_;             // Суммарное время FFT
    double batch_total_post_ms_;            // Суммарное время post
    bool last_used_batch_mode_;             // Был ли использован batch режим в последнем вызове
    
    // Кэш для планов FFT (ключ: hash параметров)
    struct PlanCacheKey {
        size_t beam_count;
        size_t count_points;
        size_t nFFT;
        size_t out_count_points_fft;
        size_t max_peaks_count;
        
        bool operator==(const PlanCacheKey& other) const {
            return beam_count == other.beam_count &&
                   count_points == other.count_points &&
                   nFFT == other.nFFT &&
                   out_count_points_fft == other.out_count_points_fft &&
                   max_peaks_count == other.max_peaks_count;
        }
    };
    
    struct PlanCacheKeyHash {
        size_t operator()(const PlanCacheKey& key) const {
            return std::hash<size_t>()(key.beam_count) ^
                   (std::hash<size_t>()(key.count_points) << 1) ^
                   (std::hash<size_t>()(key.nFFT) << 2) ^
                   (std::hash<size_t>()(key.out_count_points_fft) << 3) ^
                   (std::hash<size_t>()(key.max_peaks_count) << 4);
        }
    };
    
    static std::unordered_map<PlanCacheKey, clfftPlanHandle, PlanCacheKeyHash> plan_cache_;
    static std::mutex plan_cache_mutex_;
};

} // namespace antenna_fft

