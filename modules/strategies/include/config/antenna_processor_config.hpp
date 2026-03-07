#pragma once

/**
 * @file antenna_processor_config.hpp
 * @brief Configuration for AntennaProcessor pipeline
 *
 * @date 2026-03-07
 */

#include "config/post_fft_scenario_mode.hpp"
#include "config/statistics_set.hpp"

#include <cstdint>

namespace strategies {

struct CheckpointSaveConfig {
  bool c1_signal   = false;  ///< Save d_S (large!)
  bool c1_weights  = false;  ///< Save d_W
  bool c2_data     = false;  ///< Save d_X after GEMM (large!)
  bool c2_stats    = false;  ///< Save PRE+POST stats
  bool c3_result   = true;   ///< Save MinMaxResult (cheap)
  bool c3_spectrum = false;  ///< Save full spectrum (huge!)
  bool c4_result   = true;   ///< Save MaxValue peaks (cheap)
  bool json_header = false;  ///< JSON+binary header for debug
};

struct AntennaProcessorConfig {
  // Dimensions
  uint32_t n_ant       = 5;
  uint32_t n_samples   = 8000;
  float    sample_rate  = 12.0e6f;

  // Algorithm
  PostFftScenarioMode scenario_mode = PostFftScenarioMode::ALL_REQUIRED;
  uint32_t maxima_limit             = 1000;
  float    signal_frequency_hz      = 2.0e6f;

  // Statistics at debug points
  StatisticsSet pre_input_stats = StatPreset::P61_ALL;   ///< 2.1: on d_S
  StatisticsSet post_gemm_stats = StatPreset::P61_ALL;   ///< 2.2: on d_X
  StatisticsSet post_fft_stats  = StatPreset::P61_ALL;   ///< 2.3: on |spectrum|

  // Checkpoint (nullptr = NullCheckpointSave, zero overhead)
  const CheckpointSaveConfig* save_cfg = nullptr;

  // Debug mode: enable hipMemcpy D2H at debug points
  bool debug_mode = false;
};

}  // namespace strategies
