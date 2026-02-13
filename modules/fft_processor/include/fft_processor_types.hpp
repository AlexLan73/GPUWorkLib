#pragma once

/**
 * @file fft_processor_types.hpp
 * @brief Типы данных для FFTProcessor — FFT 1/n лучей с вариантами вывода
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include <cstdint>
#include <complex>
#include <vector>

namespace fft_processor {

// ════════════════════════════════════════════════════════════════════════════
// Режимы вывода FFT
// ════════════════════════════════════════════════════════════════════════════

/**
 * @enum FFTOutputMode
 * @brief Формат выходных данных FFT
 */
enum class FFTOutputMode {
    COMPLEX,              ///< Сырой комплексный спектр (float2)
    MAGNITUDE_PHASE,      ///< |FFT|, phase(FFT) — магнитуда и фаза
    MAGNITUDE_PHASE_FREQ  ///< |FFT|, phase, freq_hz = bin * fs / nFFT
};

// ════════════════════════════════════════════════════════════════════════════
// Параметры FFT
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FFTProcessorParams
 * @brief Параметры для FFTProcessor
 */
struct FFTProcessorParams {
    uint32_t beam_count = 1;          ///< Количество лучей (1 или N)
    uint32_t n_point = 0;             ///< Количество входных точек на луч
    float sample_rate = 1000.0f;      ///< Частота дискретизации (Гц)
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;  ///< Режим вывода
    uint32_t repeat_count = 1;        ///< Множитель nFFT (1 = nextPow2(n_point))
    float memory_limit = 0.80f;       ///< Лимит свободной GPU памяти для batch
};

// ════════════════════════════════════════════════════════════════════════════
// Результаты FFT
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FFTBeamResult
 * @brief Результат FFT для одного луча (базовые метаданные)
 */
struct FFTBeamResult {
    uint32_t beam_id = 0;
    uint32_t nFFT = 0;
    float sample_rate = 0.0f;
};

/**
 * @struct FFTComplexResult
 * @brief Результат FFT в комплексном формате (один луч)
 */
struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum;  ///< nFFT комплексных значений
};

/**
 * @struct FFTMagPhaseResult
 * @brief Результат FFT в формате магнитуда + фаза (+ частота) для одного луча
 */
struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude;    ///< nFFT магнитуд
    std::vector<float> phase;        ///< nFFT фаз (радианы)
    std::vector<float> frequency;    ///< nFFT частот (Гц), только при MAGNITUDE_PHASE_FREQ
};

/**
 * @struct FFTProfilingData
 * @brief Профилирование FFT операций
 */
struct FFTProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_processing_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};

} // namespace fft_processor
