#pragma once

/**
 * @file spectrum_maxima_finder.h
 * @brief Класс для поиска максимума спектра FFT с параболической интерполяцией
 *
 * @section beginner 📚 ДЛЯ НАЧИНАЮЩИХ (простое объяснение)
 *
 * **Что делает этот класс?**
 *
 * SpectrumMaximaFinder — это "умный искатель частот". Представьте, что у вас
 * есть сигнал (например, звук или радиоволна), и вы хотите узнать какая
 * частота в нём самая сильная.
 *
 * **Аналогия:** Как эквалайзер в музыкальном плеере показывает какие частоты
 * громче — низкие (басы) или высокие. Этот класс находит САМУЮ громкую частоту.
 *
 * **Как использовать (3 шага):**
 * 1. Создать finder: `SpectrumMaximaFinder finder(&backend);`
 * 2. Подготовить данные в структуре InputData
 * 3. Вызвать: `auto results = finder.Process(input);`
 *
 * **Что получите:** Для каждой антенны — частоту (Гц) и амплитуду максимума.
 *
 * @section expert 🔬 ДЛЯ СПЕЦИАЛИСТОВ (техническое описание)
 *
 * **Алгоритм обработки:**
 * 1. **Pre-callback** (на GPU): zero-padding n_point → nFFT, repeat_count копий
 * 2. **clFFT** (batched): Complex-to-Complex FFT размером nFFT
 * 3. **Post-kernel** (reduction): поиск max(|FFT[i]|²) + параболическая интерполяция
 *
 * **Параболическая интерполяция** уточняет частоту до долей бина:
 * ```
 * δ = 0.5 * (L - R) / (L - 2*C + R)
 * freq = (bin + δ) * sample_rate / nFFT
 * ```
 * где L, C, R — амплитуды соседних бинов.
 *
 * **Batch processing:** При нехватке GPU памяти автоматически разбивает
 * обработку на батчи через BatchManager (memory_limit по умолчанию 80%).
 *
 * **Профилирование:** Интеграция с GPUProfiler — записывает Upload, FFT,
 * PostKernel, Download с детализацией по 5 стадиям OpenCL event.
 *
 * **Поддерживаемые типы данных:**
 * - `std::vector<std::complex<float>>` — CPU данные (upload на GPU)
 * - `cl_mem` — OpenCL буфер (zero-copy, GPU→GPU)
 * - `void*` — SVM указатель (zero-copy)
 *
 * АРХИТЕКТУРА (ПЛАНИРУЕТСЯ):
 * Текущая версия использует OpenCL напрямую (cl_context, clFFT).
 * Будущая версия будет использовать Strategy Pattern:
 *
 *   SpectrumMaximaFinder (фасад)
 *          │
 *          └── ISpectrumProcessor (интерфейс)
 *                  ├── SpectrumProcessorOpenCL (clFFT)
 *                  └── SpectrumProcessorROCm   (hipFFT)
 *
 * См. MemoryBank/specs/dual_backend_architecture.md
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-06
 */

#include "interface/i_backend.hpp"
#include "interface/spectrum_maxima_types.h"
#include "interface/spectrum_input_data.hpp"
#include "kernels/fft_kernel_sources.hpp"
#include "kernels/all_maxima_kernel_sources.hpp"
#include "services/batch_manager.hpp"

#include <CL/cl.h>
#include <type_traits>
#include <clFFT.h>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Класс SpectrumMaximaFinder
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class SpectrumMaximaFinder
 * @brief Поиск максимума спектра после FFT с параболической интерполяцией
 *
 * Алгоритм:
 * 1. Pre-callback: padding n_point → nFFT с нулями
 * 2. FFT: выполнение clFFT с встроенным pre-callback
 * 3. Post-kernel: поиск максимума + парабола (ОТДЕЛЬНЫЙ kernel)
 *
 * Почему post-kernel отдельный?
 * - Нужна редукция (поиск максимума среди всех точек)
 * - Использует __local memory и barrier()
 * - Невозможно реализовать как post-callback (он видит только 1 элемент)
 *
 * Использование:
 * @code
 * SpectrumParams params;
 * params.antenna_count = 5;
 * params.n_point = 1000;
 * params.repeat_count = 2;
 * params.sample_rate = 1000.0f;
 *
 * SpectrumMaximaFinder finder(params, backend);
 * finder.Initialize();
 *
 * auto results = finder.Process(input_data);
 * auto profiling = finder.GetProfilingData();
 * @endcode
 */
class SpectrumMaximaFinder {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // Конструктор / Деструктор
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Конструктор (СТАРЫЙ API — deprecated)
     * @param params Параметры обработки
     * @param backend Указатель на DrvGPU backend (не владеет)
     * @deprecated Используйте SpectrumMaximaFinder(IBackend*) + Process(InputData<T>, ProcessingParams)
     */
    [[deprecated("Use SpectrumMaximaFinder(IBackend*) + Process(InputData<T>, ProcessingParams)")]]
    explicit SpectrumMaximaFinder(const SpectrumParams& params,
                                   drv_gpu_lib::IBackend* backend);

    /**
     * @brief Конструктор (НОВЫЙ API)
     * @param backend Указатель на DrvGPU backend (не владеет)
     *
     * Параметры обработки передаются в Process() вместе с данными.
     * Это позволяет обрабатывать данные с разными параметрами.
     */
    explicit SpectrumMaximaFinder(drv_gpu_lib::IBackend* backend);

    /**
     * @brief Деструктор (освобождает GPU ресурсы)
     */
    ~SpectrumMaximaFinder();

    // Запрет копирования
    SpectrumMaximaFinder(const SpectrumMaximaFinder&) = delete;
    SpectrumMaximaFinder& operator=(const SpectrumMaximaFinder&) = delete;

    // Разрешение перемещения
    SpectrumMaximaFinder(SpectrumMaximaFinder&&) noexcept;
    SpectrumMaximaFinder& operator=(SpectrumMaximaFinder&&) noexcept;

    // ═══════════════════════════════════════════════════════════════════════
    // Публичный интерфейс
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Инициализация GPU ресурсов
     *
     * Создаёт:
     * - Буферы GPU (pre_callback_userdata, fft_input/output, maxima)
     * - FFT план с pre-callback
     * - Компилирует post-kernel
     *
     * @throws std::runtime_error при ошибке инициализации
     */
    void Initialize();

    // ═══════════════════════════════════════════════════════════════════════
    // НОВЫЙ API — шаблонный Process (единственный публичный метод обработки)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Универсальный метод обработки (НОВЫЙ API)
     *
     * @tparam T Тип данных:
     *   - std::vector<std::complex<float>> — CPU вектор
     *   - cl_mem — OpenCL буфер (от заказчика или генератора)
     *   - void* — SVM указатель
     *
     * @param input Входные данные + параметры обработки (antenna_count, n_point, data,
     *              repeat_count, sample_rate, search_range, memory_limit)
     * @param mode Режим поиска пиков (ONE_PEAK по умолчанию)
     * @param driver Тип драйвера (ROCM по умолчанию)
     * @return Вектор результатов: antenna_count SpectrumResult
     *
     * Пример использования:
     * @code
     * SpectrumMaximaFinder finder(&backend);
     *
     * // CPU данные
     * InputData<std::vector<std::complex<float>>> input{
     *     .antenna_count = 256,
     *     .n_point = 1300000,
     *     .data = my_data,
     *     .repeat_count = 2,
     *     .sample_rate = 1000.0f
     * };
     * auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
     *
     * // GPU данные (cl_mem)
     * InputData<cl_mem> gpu_input{
     *     .antenna_count = 256,
     *     .n_point = 1300000,
     *     .data = my_cl_mem,
     *     .gpu_memory_bytes = buffer_size,
     *     .repeat_count = 2,
     *     .sample_rate = 1000.0f
     * };
     * auto results2 = finder.Process(gpu_input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
     * @endcode
     */
    template<typename T>
    std::vector<SpectrumResult> Process(
        const InputData<T>& input,
        PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
        DriverType driver = DriverType::ROCm);

    // ═══════════════════════════════════════════════════════════════════════
    // FindAllMaxima — поиск ВСЕХ локальных максимумов
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Полный pipeline: Input → FFT → Detect → Scan → Compact (НОВЫЙ API)
     *
     * Аналог Process(), но вместо одного пика находит ВСЕ локальные максимумы.
     * Все лучи обрабатываются параллельно одним вызовом!
     *
     * Pipeline: Zero-Pad → clFFT(pre+post callback) → Detect → Scan → Compact
     *
     * @tparam T Тип данных (vector<complex<float>>, cl_mem, void*)
     * @param input Входные данные (сырой сигнал, НЕ FFT!)
     * @param dest Куда вывести результат (CPU/GPU/ALL)
     * @param driver Тип backend (OPENCL, ROCM)
     * @param search_start Начало диапазона поиска (0 = default = 1, пропуск DC)
     * @param search_end Конец диапазона поиска (0 = default = nFFT/2)
     * @return AllMaximaResult с позициями и амплитудами всех максимумов
     *
     * @code
     * InputData<cl_mem> input{
     *     .antenna_count = 256,
     *     .n_point = 1024,
     *     .data = gpu_signal,             // Сырой сигнал (НЕ FFT!)
     *     .gpu_memory_bytes = 256 * 1024 * sizeof(complex<float>),
     *     .sample_rate = 1000.0f
     * };
     * auto result = finder.FindAllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);
     * @endcode
     */
    template<typename T>
    AllMaximaResult FindAllMaxima(
        const InputData<T>& input,
        OutputDestination dest = OutputDestination::CPU,
        DriverType driver = DriverType::OPENCL,
        uint32_t search_start = 0,
        uint32_t search_end = 0);

    /**
     * @brief Поиск ВСЕХ максимумов в готовых FFT данных (без FFT)
     *
     * Принимает уже посчитанный FFT спектр (complex float2) и находит
     * все локальные максимумы. FFT НЕ выполняется!
     *
     * Pipeline: ComputeMagnitudes → Detect → Scan → Compact
     *
     * @tparam T Тип данных:
     *   - std::vector<std::complex<float>> — FFT результат на CPU (будет upload)
     *   - cl_mem — FFT результат уже на GPU (zero-copy)
     *
     * @param input Входные данные. ВАЖНО: input.n_point = размер FFT (nFFT),
     *              input.data = комплексный спектр (beam_count * nFFT float2)
     * @param dest Куда вывести результат (CPU/GPU/ALL)
     * @param driver Тип backend (OPENCL, ROCM)
     * @param search_start Начало диапазона поиска (0 = default = 1)
     * @param search_end Конец диапазона поиска (0 = default = nFFT/2)
     * @return AllMaximaResult с позициями и амплитудами всех максимумов
     *
     * @code
     * // Пример: FFT уже посчитан, нужно найти максимумы
     * InputData<cl_mem> fft_input{
     *     .antenna_count = 5,
     *     .n_point = 1024,           // = nFFT (размер FFT!)
     *     .data = gpu_fft_result,    // Уже FFT (complex float2)
     *     .sample_rate = 1000.0f
     * };
     * auto result = finder.AllMaxima(fft_input, OutputDestination::CPU, DriverType::OPENCL);
     * @endcode
     */
    template<typename T>
    AllMaximaResult AllMaxima(
        const InputData<T>& input,
        OutputDestination dest = OutputDestination::CPU,
        DriverType driver = DriverType::OPENCL,
        uint32_t search_start = 0,
        uint32_t search_end = 0);

    /**
     * @brief Найти все максимумы в уже готовом FFT спектре на GPU
     *
     * Pipeline: Detection → Prefix Sum (Scan) → Compaction
     * Все лучи сканируются параллельно (beam-aware parallel scan).
     *
     * @param fft_data cl_mem буфер с результатом FFT (beam_count * nFFT float2)
     * @param beam_count Количество лучей
     * @param nFFT Размер FFT
     * @param sample_rate Частота дискретизации (Гц)
     * @param dest Куда вывести результат (CPU/GPU/ALL)
     * @param search_start Начало диапазона поиска (0 = default = 1, пропуск DC)
     * @param search_end Конец диапазона поиска (0 = default = nFFT/2)
     * @return AllMaximaResult с позициями и амплитудами всех максимумов
     */
    AllMaximaResult FindAllMaxima(
        cl_mem fft_data,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0,
        uint32_t search_end = 0,
        uint32_t beam_offset = 0,
        cl_mem external_out_maxima = nullptr,
        cl_mem external_out_counts = nullptr);

    /**
     * @brief Получить данные профилирования (из GPUProfiler для модуля SpectrumMaxima)
     */
    ProfilingData GetProfilingData() const;

    /**
     * @brief Получить параметры (с вычисленными nFFT и т.д.)
     */
    const SpectrumParams& GetParams() const { return params_; }

    /**
     * @brief Проверить, инициализирован ли объект
     */
    bool IsInitialized() const { return initialized_; }

    /**
     * @brief Вывести информацию о конфигурации
     */
    void PrintInfo() const;

private:
    // ═══════════════════════════════════════════════════════════════════════
    // Внутренние методы для нового API
    // ═══════════════════════════════════════════════════════════════════════

    /// Подготовить параметры из InputData + ProcessingParams
    void PrepareParams(uint32_t antenna_count, uint32_t n_point,
                       const ProcessingParams& proc_params, PeakSearchMode mode);

    /// Обработка CPU данных (vector)
    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data);

    /// Обработка GPU данных (cl_mem) — БЕЗ upload, только GPU→GPU копирование!
    /// @param gpu_memory_bytes Реальный размер буфера на GPU (для расчёта batch size)
    std::vector<SpectrumResult> ProcessFromGPU(
        cl_mem gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0);

    /// Обработка одного batch из GPU буфера
    std::vector<SpectrumResult> ProcessBatchFromGPU(
        cl_mem gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count);
    // ═══════════════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════════════

    /// Вычислить nFFT и другие параметры
    void CalculateFFTSize();

    /// Следующая степень двойки
    static uint32_t NextPowerOf2(uint32_t n);

    /// Рассчитать память на одну антенну (для BatchManager)
    size_t CalculateBytesPerAntenna() const;

    /// Создать GPU буферы
    void AllocateBuffers();

    /// Создать FFT план с pre-callback
    void CreateFFTPlanWithCallback();

    /// Скомпилировать post-kernel
    void CompilePostKernel();

    /// Загрузить данные в GPU
    cl_event UploadData(const std::vector<std::complex<float>>& input_data);

    /// Выполнить FFT
    cl_event ExecuteFFT(cl_event wait_event);

    /// Выполнить post-kernel
    cl_event ExecutePostKernel(cl_event wait_event);

    /// Прочитать результаты (8 MaxValue на луч → vector<SpectrumResult>)
    std::vector<SpectrumResult> ReadResults(cl_event wait_event);

    /// Освободить ресурсы
    void ReleaseResources();

    // ═══════════════════════════════════════════════════════════════════════
    // Batch Processing (для больших данных)
    // ═══════════════════════════════════════════════════════════════════════

    /// Обработать один batch антенн
    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& input_data,
        size_t start_antenna,
        size_t batch_antenna_count);

    /// Перевыделить буферы под новый размер batch
    void ReallocateBuffersForBatch(size_t batch_antenna_count);

    // ═══════════════════════════════════════════════════════════════════════
    // Приватные поля
    // ═══════════════════════════════════════════════════════════════════════

    // Параметры
    SpectrumParams params_;
    bool initialized_ = false;

    // DrvGPU backend
    drv_gpu_lib::IBackend* backend_ = nullptr;

    // OpenCL ресурсы
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    // clFFT
    clfftPlanHandle plan_handle_ = 0;
    bool plan_created_ = false;

    // GPU буферы
    cl_mem pre_callback_userdata_ = nullptr;    ///< [32 байт параметры][входные данные]
    cl_mem fft_input_ = nullptr;                ///< Входной буфер FFT
    cl_mem fft_output_ = nullptr;               ///< Выходной буфер FFT
    cl_mem maxima_output_ = nullptr;            ///< Результаты post-kernel
    cl_mem fft_temp_buffer_ = nullptr;          ///< Временный буфер для clFFT (если требуется)
    cl_mem pinned_staging_buffer_ = nullptr;    ///< Pinned buffer для быстрого upload (DMA)

    // Post-kernel
    cl_program post_program_ = nullptr;
    cl_kernel post_kernel_ = nullptr;

    // Batch processing
    size_t current_batch_size_ = 0;      ///< Размер выделенных буферов (для переиспользования)
    size_t actual_batch_size_ = 0;       ///< Реальный размер текущего batch (для ReadResults)
    size_t plan_batch_size_ = 0;         ///< Размер batch, для которого создан план (для reuse)
    size_t fft_temp_buffer_size_ = 0;    ///< Размер текущего temp buffer (для reuse)
    size_t pinned_buffer_size_ = 0;      ///< Размер текущего pinned buffer (для reuse)
    bool clfft_initialized_ = false;     ///< clFFT инициализирован для этого экземпляра

    // Константы
    static constexpr size_t PRE_CALLBACK_HEADER_SIZE = 32;  ///< Размер заголовка userdata
    static constexpr size_t LOCAL_SIZE = 256;               ///< Размер work-group для post-kernel

    /// Структура заголовка для pre-callback userdata
    /// Должна быть 32 байта для выравнивания GPU
    struct PreCallbackHeader {
        uint32_t beam_count;      ///< Количество антенн/лучей в batch
        uint32_t count_points;    ///< Количество точек на антенну (n_point)
        uint32_t nFFT;            ///< Размер FFT
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
        uint32_t padding4;
        uint32_t padding5;
    };
    static_assert(sizeof(PreCallbackHeader) == 32, "PreCallbackHeader должен быть 32 байта");

    /// Записать заголовок в pre_callback_userdata_
    void WritePreCallbackHeader(size_t batch_count);

    // ═══════════════════════════════════════════════════════════════════════
    // AllMaxima — приватные методы и поля
    // ═══════════════════════════════════════════════════════════════════════

    /// Скомпилировать kernel'ы для AllMaxima pipeline
    void CompileAllMaximaKernels();

    /// Выполнить beam-aware prefix sum (параллельно по всем лучам!)
    /// @param input Входной массив [beam_count * n_per_beam] uint
    /// @param output Выходной массив [beam_count * n_per_beam] uint
    /// @param n_per_beam Элементов на луч (nFFT)
    /// @param beam_count Количество лучей
    /// @param wait_event Событие для ожидания
    /// @return cl_event завершения scan
    cl_event ExecutePrefixSum(cl_mem input, cl_mem output,
                              size_t n_per_beam, size_t beam_count,
                              cl_event wait_event);

    // ═══════════════════════════════════════════════════════════════════════
    // Full Pipeline FindAllMaxima — приватные методы
    // ═══════════════════════════════════════════════════════════════════════

    /// Полный pipeline из CPU данных: Upload → FFT → Detect → Scan → Compact
    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end);

    /// Полный pipeline из GPU данных: Copy → FFT → Detect → Scan → Compact
    AllMaximaResult FindAllMaximaFromGPUPipeline(
        cl_mem gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end);

    /// AllMaxima из CPU FFT данных: Upload → ComputeMag → Detect → Scan → Compact
    AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end);

    /// Освободить AllMaxima ресурсы
    void ReleaseAllMaximaResources();

    // AllMaxima kernel'ы
    cl_program all_maxima_program_ = nullptr;
    cl_kernel detect_kernel_ = nullptr;
    cl_kernel compute_magnitudes_kernel_ = nullptr;  ///< Для старого API (FFT на GPU)

    cl_program prefix_sum_program_ = nullptr;
    cl_kernel block_scan_kernel_ = nullptr;
    cl_kernel block_add_kernel_ = nullptr;

    cl_program compact_program_ = nullptr;
    cl_kernel compact_kernel_ = nullptr;

    bool all_maxima_kernels_compiled_ = false;

    // AllMaxima FFT план (с pre + post callbacks)
    clfftPlanHandle allmax_plan_handle_ = 0;     ///< FFT план с pre+post callback
    bool allmax_plan_created_ = false;
    size_t allmax_plan_batch_size_ = 0;          ///< batch size для текущего allmax плана
    cl_mem magnitudes_buffer_ = nullptr;         ///< Pre-computed |FFT[i]| (beam_count * nFFT float)
    size_t magnitudes_buffer_size_ = 0;          ///< Размер magnitudes_buffer_ в элементах

    /// Создать FFT план с pre-callback + post-callback (для AllMaxima pipeline)
    void CreateAllMaximaFFTPlan(size_t batch_count);

    /// Выполнить FFT с AllMaxima планом (pre+post callbacks)
    cl_event ExecuteAllMaximaFFT(cl_event wait_event);

    /// Выделить/переиспользовать magnitudes_buffer_
    void EnsureMagnitudesBuffer(size_t total_elements);

    /// Размер блока для prefix sum (2 * local_size)
    static constexpr size_t SCAN_LOCAL_SIZE = 256;
    static constexpr size_t SCAN_BLOCK_SIZE = SCAN_LOCAL_SIZE * 2;  // 512
};

// ════════════════════════════════════════════════════════════════════════════
// Реализация шаблонного метода Process<T>
// (должна быть в header из-за шаблона)
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
std::vector<SpectrumResult> SpectrumMaximaFinder::Process(
    const InputData<T>& input,
    PeakSearchMode mode,
    DriverType driver)
{
    // 1. Создать временный ProcessingParams из полей InputData
    ProcessingParams proc_params{
        input.repeat_count,
        input.sample_rate,
        input.search_range,
        input.memory_limit
    };

    // 2. Подготовить параметры
    PrepareParams(input.antenna_count, input.n_point, proc_params, mode);

    // 3. Диспетчеризация по типу данных
    if constexpr (is_cpu_vector_v<T>) {
        // CPU данные — вызываем Initialize() и затем стандартный upload
        if (!initialized_) {
            Initialize();
        }
        return ProcessFromCPU(input.data);
    }
    else if constexpr (std::is_same_v<T, cl_mem>) {
        // GPU данные (cl_mem) — НЕ вызываем Initialize()!
        // ProcessFromGPU() сам управляет буферами с учётом уже занятой памяти.
        // Это позволяет правильно рассчитать batch size когда данные УЖЕ на GPU.
        return ProcessFromGPU(input.data, input.antenna_count, input.n_point,
                              input.ActualGpuMemory());
    }
    else if constexpr (is_svm_pointer_v<T>) {
        // SVM данные — TODO: реализовать позже
        throw std::runtime_error("SVM input not implemented yet");
    }
    else {
        // Неизвестный тип
        static_assert(is_cpu_vector_v<T> || std::is_same_v<T, cl_mem> || is_svm_pointer_v<T>,
                      "Unsupported input data type. Use vector<complex<float>>, cl_mem, or void*");
    }

    // driver пока не используется (будет для ROCm)
    (void)driver;
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация шаблонного метода FindAllMaxima<T>
// (должна быть в header из-за шаблона)
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
AllMaximaResult SpectrumMaximaFinder::FindAllMaxima(
    const InputData<T>& input,
    OutputDestination dest,
    DriverType driver,
    uint32_t search_start,
    uint32_t search_end)
{
    (void)driver;  // TODO: ROCm backend

    // 1. Подготовить параметры (как в Process)
    ProcessingParams proc_params{
        input.repeat_count,
        input.sample_rate,
        input.search_range,
        input.memory_limit
    };

    PrepareParams(input.antenna_count, input.n_point, proc_params, PeakSearchMode::ALL_MAXIMA);

    // 2. Диспетчеризация по типу данных
    if constexpr (is_cpu_vector_v<T>) {
        // CPU данные — upload → FFT → AllMaxima
        if (!initialized_) {
            Initialize();
        }
        return FindAllMaximaFromCPU(input.data, dest, search_start, search_end);
    }
    else if constexpr (std::is_same_v<T, cl_mem>) {
        // GPU данные (cl_mem) — copy → FFT → AllMaxima
        return FindAllMaximaFromGPUPipeline(input.data, input.antenna_count, input.n_point,
                                             input.ActualGpuMemory(), dest, search_start, search_end);
    }
    else if constexpr (is_svm_pointer_v<T>) {
        throw std::runtime_error("SVM input not implemented yet for FindAllMaxima");
    }
    else {
        static_assert(is_cpu_vector_v<T> || std::is_same_v<T, cl_mem> || is_svm_pointer_v<T>,
                      "Unsupported input data type");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Реализация шаблонного метода AllMaxima<T>
// (FFT данные → Detect → Scan → Compact, БЕЗ FFT!)
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
AllMaximaResult SpectrumMaximaFinder::AllMaxima(
    const InputData<T>& input,
    OutputDestination dest,
    DriverType driver,
    uint32_t search_start,
    uint32_t search_end)
{
    (void)driver;  // TODO: ROCm backend

    // input.n_point = nFFT (данные уже FFT, padding не нужен!)
    const uint32_t nFFT = input.n_point;
    const uint32_t beam_count = input.antenna_count;
    const float sample_rate = input.sample_rate;

    if constexpr (is_cpu_vector_v<T>) {
        // CPU FFT данные — upload → ComputeMag → Detect → Scan → Compact
        return AllMaximaFromCPU(input.data, beam_count, nFFT,
                                sample_rate, dest, search_start, search_end);
    }
    else if constexpr (std::is_same_v<T, cl_mem>) {
        // GPU FFT данные — прямой вызов (данные уже на GPU)
        return FindAllMaxima(input.data, beam_count, nFFT,
                             sample_rate, dest, search_start, search_end);
    }
    else if constexpr (is_svm_pointer_v<T>) {
        throw std::runtime_error("SVM input not implemented yet for AllMaxima");
    }
    else {
        static_assert(is_cpu_vector_v<T> || std::is_same_v<T, cl_mem> || is_svm_pointer_v<T>,
                      "Unsupported input data type");
    }
}

} // namespace antenna_fft
