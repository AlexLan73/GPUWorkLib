#pragma once

/**
 * @file i_checkpoint_save.hpp
 * @brief Interface for checkpoint saving (Null Object pattern)
 *
 * Production: NullCheckpointSave (all methods = no-op, zero overhead)
 * Debug: CheckpointSave (binary files to Logs/GPU_XX/antenna_processor/)
 *
 * @date 2026-03-07
 */

#include "result_types.hpp"

#include <cstdint>

namespace strategies {

class ICheckpointSave {
public:
  virtual ~ICheckpointSave() = default;

  // C1: after input received
  virtual void save_c1_signal(
      const void* d_data, uint32_t n_ant, uint32_t n_samples,
      float sample_rate, int gpu_id) = 0;

  virtual void save_c1_weights(
      const void* d_weights, uint32_t n_ant, int gpu_id) = 0;

  // C2: after GEMM
  virtual void save_c2_data(
      const void* d_X, uint32_t n_ant, uint32_t n_samples,
      float sample_rate, int gpu_id) = 0;

  virtual void save_c2_stats(
      const statistics::StatisticsResult* pre_stats,
      const statistics::StatisticsResult* post_stats,
      uint32_t n_ant, int gpu_id) = 0;

  // C3: after FFT
  virtual void save_c3_spectrum(
      const void* d_spectrum, uint32_t n_ant, uint32_t nFFT, int gpu_id) = 0;

  virtual void save_c3_minmax(
      const MinMaxResult* results, uint32_t n_ant, int gpu_id) = 0;

  // C4: post-FFT scenario results
  virtual void save_c4_one_max(
      const OneMaxParabolaLite* results, uint32_t n_ant, int gpu_id) = 0;
};

}  // namespace strategies
