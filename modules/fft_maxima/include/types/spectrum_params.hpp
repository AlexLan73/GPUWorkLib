#pragma once

/**
 * @file spectrum_params.hpp
 * @brief Параметры поиска максимума спектра
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "types/spectrum_modes.hpp"

#include <cstddef>
#include <cstdint>

namespace antenna_fft {

/**
 * @struct SpectrumParams
 * @brief Параметры для поиска максимума спектра
 */
struct SpectrumParams {
    uint32_t antenna_count = 5;
    uint32_t n_point = 1000;
    uint32_t repeat_count = 2;
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK;
    float memory_limit = 0.80f;
    size_t max_maxima_per_beam = 1000;
    uint32_t nFFT = 0;
    uint32_t base_fft = 0;
};

}  // namespace antenna_fft
