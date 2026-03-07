#pragma once

/**
 * @file weight_generator.hpp
 * @brief Delay-and-sum weight matrix generation for beamforming
 *
 * Formula: W[beam][ant] = exp(-j * 2*pi * f0 * tau_ant) / sqrt(N_ant)
 *
 * Two modes:
 * - Auto-generate from signal parameters
 * - External load from user-provided matrix
 *
 * @date 2026-03-07
 */

#include <complex>
#include <vector>
#include <cstdint>

namespace strategies {

struct WeightParams {
  uint32_t n_ant    = 5;
  double   f0       = 2.0e6;      ///< Signal frequency (Hz)
  double   tau_base = 0.0;        ///< Base delay (s)
  double   tau_step = 100e-6;     ///< Delay step per antenna (s)
};

class WeightGenerator {
public:
  /**
   * @brief Generate Delay-and-sum weight matrix
   * @param params WeightParams with antenna count, frequency, delays
   * @return Flat row-major matrix [n_ant x n_ant] as complex<float>
   *
   * W[beam][ant] = exp(-j * 2*pi * f0 * tau[ant]) / sqrt(N_ant)
   * where tau[ant] = tau_base + ant * tau_step
   */
  static std::vector<std::complex<float>> generate_delay_and_sum(
      const WeightParams& params);

  /**
   * @brief Upload weight matrix to GPU
   * @param backend IBackend for GPU allocation
   * @param weights Flat [n_ant x n_ant] complex<float> matrix
   * @return GPU device pointer (caller must free via backend->Free())
   */
  static void* upload_to_gpu(
      void* backend,  // drv_gpu_lib::IBackend*
      const std::vector<std::complex<float>>& weights);
};

}  // namespace strategies
