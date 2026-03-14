#pragma once

/**
 * @file i_post_fft_scenario.hpp
 * @brief Interface for post-FFT processing scenarios
 *
 * Each scenario works on the SAME d_spectrum computed once by Window+FFT block.
 * Implementations live in modules/fft_func/ (not in strategies).
 *
 * @date 2026-03-07
 */

#include "result_types.hpp"

#include <cstdint>

#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace strategies {

class IPostFftScenario {
public:
  virtual ~IPostFftScenario() = default;

#if ENABLE_ROCM
  /**
   * @brief Execute scenario on already-computed spectrum
   * @param d_spectrum Complex float spectrum [n_ant x nFFT] on GPU
   * @param d_magnitudes Float magnitudes |spectrum| [n_ant x nFFT] on GPU (may be nullptr)
   * @param n_ant Number of beams
   * @param nFFT FFT size
   * @param sample_rate Sampling rate in Hz
   * @param maxima_limit Max peaks for AllMaxima
   * @param stream HIP stream for async execution
   */
  virtual void execute(
      const void* d_spectrum,
      const void* d_magnitudes,
      uint32_t n_ant,
      uint32_t nFFT,
      float    sample_rate,
      uint32_t maxima_limit,
      hipStream_t stream) = 0;
#endif

  virtual const char* name() const = 0;
};

}  // namespace strategies
