#pragma once

/**
 * @file statistics_set.hpp
 * @brief StatField bitmask and StatPreset constants for debug point statistics
 *
 * @date 2026-03-07
 */

#include <cstdint>

namespace strategies {

enum StatField : uint32_t {
  STAT_NONE   = 0,
  STAT_MEAN   = 1 << 0,
  STAT_MEDIAN = 1 << 1,
  STAT_STD    = 1 << 2,
  STAT_VAR    = 1 << 3,
  STAT_MIN    = 1 << 4,
  STAT_MAX    = 1 << 5,
};

using StatisticsSet = uint32_t;

namespace StatPreset {
  constexpr StatisticsSet NONE         = 0;
  constexpr StatisticsSet P61_ALL      = STAT_MEAN | STAT_MEDIAN | STAT_STD
                                       | STAT_VAR  | STAT_MIN    | STAT_MAX;
  constexpr StatisticsSet P62_MEAN_MED = STAT_MEAN | STAT_MEDIAN;
  constexpr StatisticsSet P63_MED_MM   = STAT_MEAN | STAT_MEDIAN | STAT_MIN | STAT_MAX;
  constexpr StatisticsSet P64_STD_VAR  = STAT_STD  | STAT_VAR;
}  // namespace StatPreset

}  // namespace strategies
