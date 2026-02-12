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
        DriverType driver = DriverType::ROCM);

    /**
     * @brief Получить данные профилирования последнего вызова
     */
    const ProfilingData& GetProfilingData() const { return profiling_; }

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
    /// @param send_to_profiler при true — отправить событие read в GPUProfiler (все 5 полей OpenCL)
    std::vector<SpectrumResult> ReadResults(cl_event wait_event, bool send_to_profiler = false);

    /// Профилирование события
    double ProfileEvent(cl_event event, const char* name);

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

    // Профилирование
    ProfilingData profiling_;

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

} // namespace antenna_fft
