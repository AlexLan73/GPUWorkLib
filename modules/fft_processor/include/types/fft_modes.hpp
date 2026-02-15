#pragma once

/**
 * @file fft_modes.hpp
 * @brief Режимы вывода FFT
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

namespace fft_processor {

enum class FFTOutputMode {
    COMPLEX,
    MAGNITUDE_PHASE,
    MAGNITUDE_PHASE_FREQ
};

}  // namespace fft_processor
