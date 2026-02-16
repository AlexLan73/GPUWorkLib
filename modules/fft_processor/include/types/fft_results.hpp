#pragma once

/**
 * @file fft_results.hpp
 * @brief Результаты FFT и локальное профилирование
 *
 * Для централизованного профилирования (JSON, MD) — GPUProfiler + OpenCLProfilingData.
 * FFTProfilingData — локальный формат для GetProfilingData().
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <cstdint>
#include <complex>
#include <vector>

namespace fft_processor {

struct FFTBeamResult {
    uint32_t beam_id = 0;
    uint32_t nFFT = 0;
    float sample_rate = 0.0f;
};

struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum;
};

struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude, phase, frequency;
};

/// Локальные данные профилирования для GetProfilingData()
struct FFTProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_processing_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};

}  // namespace fft_processor
