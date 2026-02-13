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
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "fft_processor_types.hpp"
#include "interface/i_backend.hpp"
#include "kernels/fft_processor_kernels.hpp"
#include "services/batch_manager.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <complex>
#include <vector>
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
     * @return Вектор FFTComplexResult (один на луч)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params);

    /**
     * @brief FFT с комплексным выводом (GPU данные)
     * @param gpu_data OpenCL буфер с данными
     * @param params Параметры FFT
     * @param gpu_memory_bytes Размер буфера на GPU (0 = auto)
     * @return Вектор FFTComplexResult (один на луч)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        cl_mem gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0);

    // ═══════════════════════════════════════════════════════════════════════
    // Публичный API — Magnitude + Phase output
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief FFT с выводом magnitude + phase (CPU данные)
     * @param data Входные данные: beam_count * n_point complex<float>
     * @param params Параметры FFT (output_mode определяет наличие freq)
     * @return Вектор FFTMagPhaseResult (один на луч)
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params);

    /**
     * @brief FFT с выводом magnitude + phase (GPU данные)
     * @param gpu_data OpenCL буфер с данными
     * @param params Параметры FFT
     * @param gpu_memory_bytes Размер буфера на GPU (0 = auto)
     * @return Вектор FFTMagPhaseResult (один на луч)
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        cl_mem gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0);

    // ═══════════════════════════════════════════════════════════════════════
    // Информация
    // ═══════════════════════════════════════════════════════════════════════

    const FFTProfilingData& GetProfilingData() const { return profiling_; }
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
        float sample_rate);

    /// Прочитать magnitude+phase результаты
    std::vector<FFTMagPhaseResult> ReadMagPhaseResults(
        cl_event wait_event, size_t beam_count, size_t start_beam,
        float sample_rate, bool include_freq);

    /// Профилирование OpenCL события
    double ProfileEvent(cl_event event);

    // ═══════════════════════════════════════════════════════════════════════
    // Поля
    // ═══════════════════════════════════════════════════════════════════════

    // DrvGPU
    drv_gpu_lib::IBackend* backend_ = nullptr;

    // OpenCL ресурсы
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    // clFFT
    clfftPlanHandle plan_handle_ = 0;
    bool plan_created_ = false;
    bool clfft_initialized_ = false;

    // GPU буферы
    cl_mem pre_callback_userdata_ = nullptr;  ///< [32B header][input data]
    cl_mem fft_input_ = nullptr;              ///< Входной буфер FFT (nFFT * batch)
    cl_mem fft_output_ = nullptr;             ///< Выходной буфер FFT
    cl_mem fft_temp_buffer_ = nullptr;        ///< Temp буфер clFFT
    cl_mem mag_output_ = nullptr;             ///< Магнитуды (nFFT * batch)
    cl_mem phase_output_ = nullptr;           ///< Фазы (nFFT * batch)

    // Post-processing kernel
    cl_program mag_phase_program_ = nullptr;
    cl_kernel mag_phase_kernel_ = nullptr;

    // Параметры текущей обработки
    uint32_t nFFT_ = 0;
    uint32_t n_point_ = 0;

    // Batch management
    size_t current_buffer_beams_ = 0;  ///< Размер выделенных буферов
    size_t plan_batch_size_ = 0;       ///< Batch size текущего FFT плана
    size_t fft_temp_buffer_size_ = 0;
    bool has_mag_phase_buffers_ = false;

    // Профилирование
    FFTProfilingData profiling_;

    // Константы
    static constexpr size_t PRE_CALLBACK_HEADER_SIZE = 32;

    struct PreCallbackHeader {
        uint32_t beam_count;
        uint32_t count_points;
        uint32_t nFFT;
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
        uint32_t padding4;
        uint32_t padding5;
    };
    static_assert(sizeof(PreCallbackHeader) == 32, "PreCallbackHeader must be 32 bytes");
};

} // namespace fft_processor
