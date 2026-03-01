#pragma once

/**
 * @file mag_phase_types.hpp
 * @brief Types for ComplexToMagPhaseROCm -- direct complex-to-magnitude+phase conversion
 *
 * Lighter than FFTMagPhaseResult: no nFFT, sample_rate, frequency (no FFT involved).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#include <cstdint>
#include <vector>

namespace fft_processor {

/// Parameters for ComplexToMagPhaseROCm conversion
struct MagPhaseParams {
    uint32_t beam_count   = 1;     ///< Number of parallel beams/channels
    uint32_t n_point      = 0;     ///< Points per beam
    float    memory_limit = 0.80f; ///< Fraction of GPU memory to use (0.0 .. 1.0)
};

/// Result of complex-to-mag/phase conversion (one per beam)
struct MagPhaseResult {
    uint32_t beam_id = 0;         ///< Beam index
    uint32_t n_point = 0;         ///< Points in this beam
    std::vector<float> magnitude;  ///< |z| = sqrt(re^2 + im^2)
    std::vector<float> phase;      ///< arg(z) = atan2(im, re) [radians]
};

}  // namespace fft_processor
