#pragma once

/**
 * @file fft_processor.hpp
 * @brief FFTProcessor — FFT 1/n лучей с вариантами вывода
 *
 * Отдельный класс от SpectrumMaximaFinder.
 * SpectrumMaximaFinder ищет МАКСИМУМЫ спектра (пики).
 * FFTProcessor возвращает ПОЛНЫЙ спектр в нужном формате.
 *
 * Поддерживаемые режимы вывода:
 * - COMPLEX:              raw complex FFT spectrum
 * - MAGNITUDE_PHASE:      |FFT|, phase(FFT)
 * - MAGNITUDE_PHASE_FREQ: |FFT|, phase, freq_hz = bin * fs / nFFT
 *
 * Использование:
 * @code
 * // Создать процессор
 * FFTProcessor fft(&backend);
 *
 * // Подготовить данные
 * FFTProcessorParams params;
 * params.beam_count = 256;
 * params.n_point = 1024;
 * params.sample_rate = 1000.0f;
 * params.output_mode = FFTOutputMode::MAGNITUDE_PHASE_FREQ;
 *
 * // Обработка CPU данных
 * std::vector<std::complex<float>> data = ...;
 * auto results = fft.ProcessComplex(data, params);  // -> vector<FFTComplexResult>
 * auto results2 = fft.ProcessMagPhase(data, params); // -> vector<FFTMagPhaseResult>
 *
 * // Обработка GPU данных (cl_mem)
 * cl_mem gpu_data = ...;
 * auto results3 = fft.ProcessComplex(gpu_data, params);
 *
 * // Обработка с профилированием (benchmark):
 * std::vector<std::pair<const char*, cl_event>> events;
 * auto results4 = fft.ProcessComplex(data, params, &events);
 * // events содержит {"Upload", ev1}, {"FFT", ev2}, {"Download", ev3}
 * @endcode
 *
 * @note Production-код — ЧИСТЫЙ. Профилирование опционально через prof_events.
 *       Без prof_events — ноль overhead, cl_event освобождаются внутри.
 *       С prof_events — события собираются для внешнего профилирования.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "fft_processor_types.hpp"
#include "interface/i_backend.hpp"
#include "kernels/fft_processor_kernels.hpp"
#include "services/batch_manager.hpp"
#include "services/gpu_profiler.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <complex>
#include <vector>
#include <utility>
#include <cstdint>

namespace fft_processor {

class FFTProcessor {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // Конструктор / Деструктор
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Конструктор
     * @param backend Указатель на IBackend (не владеет)
     */
    explicit FFTProcessor(drv_gpu_lib::IBackend* backend);

    ~FFTProcessor();

    // Запрет копирования
    FFTProcessor(const FFTProcessor&) = delete;
    FFTProcessor& operator=(const FFTProcessor&) = delete;

    // Перемещение
    FFTProcessor(FFTProcessor&& other) noexcept;
    FFTProcessor& operator=(FFTProcessor&& other) noexcept;

    // ═══════════════════════════════════════════════════════════════════════
    // Публичный API — Complex output
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief FFT с комплексным выводом (CPU данные)
     * @param data Входные данные: beam_count * n_point complex<float>
     * @param params Параметры FFT
     * @param prof_events Если != nullptr — собирает cl_event'ы для профилирования
     *        (вызывающий отвечает за clReleaseEvent). nullptr = без профилирования.
     * @return Вектор FFTComplexResult (один на луч)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    /**
     * @brief FFT с комплексным выводом (GPU данные)
     * @param gpu_data OpenCL буфер с данными
     * @param params Параметры FFT
     * @param gpu_memory_bytes Размер буфера на GPU (0 = auto)
     * @param prof_events Если != nullptr — собирает cl_event'ы для профилирования
     * @return Вектор FFTComplexResult (один на луч)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        cl_mem gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    // ═══════════════════════════════════════════════════════════════════════
    // Публичный API — Magnitude + Phase output
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief FFT с выводом magnitude + phase (CPU данные)
     * @param data Входные данные: beam_count * n_point complex<float>
     * @param params Параметры FFT (output_mode определяет наличие freq)
     * @param prof_events Если != nullptr — собирает cl_event'ы для профилирования
     * @return Вектор FFTMagPhaseResult (один на луч)
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    /**
     * @brief FFT с выводом magnitude + phase (GPU данные)
     * @param gpu_data OpenCL буфер с данными
     * @param params Параметры FFT
     * @param gpu_memory_bytes Размер буфера на GPU (0 = auto)
     * @param prof_events Если != nullptr — собирает cl_event'ы для профилирования
     * @return Вектор FFTMagPhaseResult (один на луч)
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        cl_mem gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    // ═══════════════════════════════════════════════════════════════════════
    // Информация
    // ═══════════════════════════════════════════════════════════════════════

    FFTProfilingData GetProfilingData() const;
    uint32_t GetNFFT() const { return nFFT_; }

private:
    // ═══════════════════════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════════════════════

    static uint32_t NextPowerOf2(uint32_t n);
    void CalculateNFFT(const FFTProcessorParams& params);
    size_t CalculateBytesPerBeam(FFTOutputMode mode) const;

    // ═══════════════════════════════════════════════════════════════════════
    // GPU Resources management
    // ═══════════════════════════════════════════════════════════════════════

    /// Выделить/переиспользовать буферы под batch_size лучей
    void AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode);

    /// Создать FFT план (с pre-callback для padding)
    void CreateFFTPlan(size_t batch_beam_count);

    /// Скомпилировать post-processing kernel (magnitude+phase)
    void CompileMagPhaseKernel();

    /// Записать заголовок pre-callback
    void WritePreCallbackHeader(size_t batch_beam_count);

    /// Освободить все ресурсы
    void ReleaseResources();

    // ═══════════════════════════════════════════════════════════════════════
    // GPU Operations
    // ═══════════════════════════════════════════════════════════════════════

    /// Загрузить CPU данные в pre_callback_userdata_ (после 32-byte header)
    cl_event UploadData(const std::complex<float>* data, size_t count);

    /// Копировать GPU→GPU данные в pre_callback_userdata_
    cl_event CopyGpuData(cl_mem src, size_t src_offset, size_t count);

    /// Выполнить FFT
    cl_event ExecuteFFT(cl_event wait_event);

    /// Выполнить post-processing (complex → mag+phase)
    cl_event ExecuteMagPhaseKernel(cl_event wait_event, size_t beam_count);

    /// Прочитать комплексные результаты
    std::vector<FFTComplexResult> ReadComplexResults(
        cl_event wait_event, size_t beam_count, size_t start_beam,
        float sample_rate,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    /// Прочитать magnitude+phase результаты
    std::vector<FFTMagPhaseResult> ReadMagPhaseResults(
        cl_event wait_event, size_t beam_count, size_t start_beam,
        float sample_rate, bool include_freq,
        std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

    // ═══════════════════════════════════════════════════════════════════════
    // Вспомогательный метод — сохранить или освободить cl_event
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Если prof_events != nullptr — сохранить cl_event.
     *        Иначе — освободить (clReleaseEvent).
     */
    static void CollectOrRelease(cl_event ev, const char* name,
        std::vector<std::pair<const char*, cl_event>>* prof_events)
    {
        if (!ev) return;
        if (prof_events) {
            prof_events->push_back({name, ev});
        } else {
            clReleaseEvent(ev);
        }
    }

    /// Длительность cl_event в мс (CL_PROFILING_COMMAND_START → END). 0 если профилирование недоступно.
    static double EventDurationMs(cl_event ev);

    /// Заполнить last_timing_ из списка events по именам ("Upload"/"FFT"/"PostProcessing"/"Download")
    /// и освободить все события. events очищается после вызова.
    void AccumulateTiming(std::vector<std::pair<const char*, cl_event>>& events);

    // ═══════════════════════════════════════════════════════════════════════
    // Поля
    // ═══════════════════════════════════════════════════════════════════════

    // DrvGPU
    drv_gpu_lib::IBackend* backend_ = nullptr;  ///< Не владеет; должен жить дольше FFTProcessor

    // OpenCL ресурсы
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    // clFFT
    clfftPlanHandle plan_handle_ = 0;
    bool plan_created_ = false;
    bool clfft_initialized_ = false;  ///< clfftSetup() вызывается один раз на всё время жизни объекта

    // GPU буферы
    // ВАЖНО: pre_callback_userdata_ содержит И заголовок (32 байта) И входные данные!
    // Layout: [beam_count:4][count_points:4][nFFT:4][pad×5:20][data: beam_count × n_point × float2]
    // clFFT pre-callback читает из него напрямую (userdata в OpenCL kernel = этот буфер).
    // fft_input_ — "объявленный" входной буфер для clfftEnqueueTransform, его содержимое
    //              НЕ используется — pre-callback перехватывает чтение и даёт данные из userdata.
    cl_mem pre_callback_userdata_ = nullptr;  ///< [32B header][beam_count × n_point × complex<float>]
    cl_mem fft_input_ = nullptr;              ///< "Dummy" input для clFFT; pre-callback подставляет реальные данные
    cl_mem fft_output_ = nullptr;             ///< Результат FFT [beam_count × nFFT × complex<float>]
    cl_mem fft_temp_buffer_ = nullptr;        ///< Scratch-буфер clFFT; размер зависит от nFFT и batch — смотри clfftGetTmpBufSize
    cl_mem mag_output_ = nullptr;             ///< |X[k]| после kernel; [beam_count × nFFT × float]
    cl_mem phase_output_ = nullptr;           ///< arg(X[k]) после kernel; [beam_count × nFFT × float]

    // Post-processing kernel (complex_to_mag_phase)
    cl_program mag_phase_program_ = nullptr;
    cl_kernel mag_phase_kernel_ = nullptr;  ///< Скомпилирован lazily при первом вызове ProcessMagPhase

    // Параметры текущей обработки
    uint32_t nFFT_ = 0;     ///< Актуальный размер FFT = nextPow2(n_point_) × repeat_count
    uint32_t n_point_ = 0;  ///< Реальный размер входа на луч (до padding)

    // Batch management
    size_t current_buffer_beams_ = 0;  ///< Batch size, под который выделены GPU-буферы; если batch.count > этого — realloc
    size_t plan_batch_size_ = 0;       ///< Batch size, под который создан текущий FFT-план; может отличаться от current_buffer_beams_
    size_t fft_temp_buffer_size_ = 0;  ///< Текущий размер fft_temp_buffer_; растёт при необходимости, не уменьшается
    bool has_mag_phase_buffers_ = false;

    // Последнее измеренное время (заполняется в Process* если prof_events == nullptr)
    FFTProfilingData last_timing_;

    // Константы
    static constexpr size_t PRE_CALLBACK_HEADER_SIZE = 32;  ///< Размер заголовка в pre_callback_userdata_: 8 × uint32_t = 32 байта

    // Заголовок в начале pre_callback_userdata_: передаёт параметры в OpenCL pre-callback kernel.
    // Данные начинаются сразу после: (uint32_t*)userdata + 8 == (float2*) данные.
    // ВНИМАНИЕ: порядок полей изменять нельзя — pre-callback kernel читает по фиксированным offset'ам!
    struct PreCallbackHeader {
        uint32_t beam_count;   // header[0]: количество лучей в батче
        uint32_t count_points; // header[1]: n_point — реальный размер входа на луч
        uint32_t nFFT;         // header[2]: целевой размер FFT (со zero-padding)
        uint32_t padding1;     // header[3..7]: выравнивание до 32 байт
        uint32_t padding2;
        uint32_t padding3;
        uint32_t padding4;
        uint32_t padding5;
    };
    static_assert(sizeof(PreCallbackHeader) == 32, "PreCallbackHeader must be 32 bytes");
};

} // namespace fft_processor
