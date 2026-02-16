#pragma once

/**
 * @file fft_params.hpp
 * @brief Параметры FFTProcessor
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "types/fft_modes.hpp"

#include <cstdint>

namespace fft_processor {

struct FFTProcessorParams {
    uint32_t beam_count = 1;
    uint32_t n_point = 0;
    float sample_rate = 1000.0f;
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;
    uint32_t repeat_count = 1;
    float memory_limit = 0.80f;
};

}  // namespace fft_processor
