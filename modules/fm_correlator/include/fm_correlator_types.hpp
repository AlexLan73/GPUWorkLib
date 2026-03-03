#pragma once

/**
 * @file fm_correlator_types.hpp
 * @brief Types for FM Correlator: params + result
 *
 * FMCorrelatorParams  — FFT size, shifts, signals, LFSR config
 * FMCorrelatorResult  — peaks [S × K × n_kg] with accessor
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace drv_gpu_lib {

struct FMCorrelatorParams {
  size_t fft_size = 32768;
  int num_shifts = 32;
  int num_signals = 5;
  int num_output_points = 2000;
  uint32_t lfsr_polynomial = 0x00400007;  // x^32+x^22+x^2+x+1 (primitive)
  uint32_t lfsr_seed = 0x12345678;
};

struct FMCorrelatorResult {
  std::vector<float> peaks;  // [S * K * n_kg], row-major
  int num_signals = 0;
  int num_shifts = 0;
  int num_output_points = 0;

  float at(int signal, int shift, int point) const {
    return peaks[static_cast<size_t>(
        (signal * num_shifts + shift) * num_output_points + point)];
  }
};

}  // namespace drv_gpu_lib
