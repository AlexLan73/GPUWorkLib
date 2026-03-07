#pragma once

/**
 * @file antenna_processor.hpp
 * @brief AntennaProcessor - abstract base class for antenna array processing
 *
 * Entry point for the strategies pipeline:
 *   d_S (on GPU) -> GEMM -> Window+FFT -> post-FFT scenarios -> AntennaResult
 *
 * @date 2026-03-07
 */

#include "config/antenna_processor_config.hpp"
#include "result_types.hpp"

namespace strategies {

class AntennaProcessor {
public:
  virtual ~AntennaProcessor() = default;

  /**
   * @brief Run full pipeline
   * @param d_S Input signal [n_ant x n_samples] complex float, already on GPU
   * @param d_W Weight matrix [n_ant x n_ant] complex float, already on GPU
   * @return Aggregate result with stats, peaks, minmax, perf
   */
  virtual AntennaResult process(const void* d_S, const void* d_W) = 0;

  // Runtime configuration
  virtual void set_scenario_mode(PostFftScenarioMode mode) = 0;
  virtual void set_pre_input_stats(StatisticsSet stats) = 0;
  virtual void set_post_gemm_stats(StatisticsSet stats) = 0;
  virtual void set_post_fft_stats(StatisticsSet stats) = 0;
  virtual void set_debug_mode(bool enabled) = 0;

  // Info
  virtual const AntennaProcessorConfig& config() const = 0;
  virtual int gpu_id() const = 0;
};

}  // namespace strategies
