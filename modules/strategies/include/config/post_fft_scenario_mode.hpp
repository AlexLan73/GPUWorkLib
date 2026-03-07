#pragma once

/**
 * @file post_fft_scenario_mode.hpp
 * @brief PostFftScenarioMode enum for AntennaProcessor
 *
 * Controls which post-FFT scenarios are executed:
 * - ALL_REQUIRED: production mode, all 3 scenarios
 * - Individual modes for debug/benchmark
 *
 * @date 2026-03-07
 */

#include <cstdint>

namespace strategies {

enum class PostFftScenarioMode : uint8_t {
  ALL_REQUIRED     = 0,  ///< Production: Step2.1 + Step2.2 + Step2.3
  ONE_MAX_PARABOLA = 1,  ///< Step2.1 only: OneMax + 3-point parabola (no phase)
  ALL_MAXIMA       = 2,  ///< Step2.2 only: all local maxima (limit=1000)
  GLOBAL_MINMAX    = 3   ///< Step2.3 only: global MIN + MAX
};

}  // namespace strategies
