#pragma once

/**
 * @file statistics_types.hpp
 * @brief Types and structures for StatisticsProcessor (ROCm)
 *
 * Defines input parameters and result structures for statistical
 * computations on complex float signal data (per-beam).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#include <vector>
#include <complex>
#include <cstdint>

namespace statistics {

// =========================================================================
// Input parameters
// =========================================================================

/**
 * @brief Parameters for statistics computation
 *
 * Input data layout: beam_count * n_point complex<float> (interleaved beams).
 * Each beam is processed independently.
 */
struct StatisticsParams {
  uint32_t beam_count = 1;       ///< Number of beams (antennas)
  uint32_t n_point    = 0;       ///< Samples per beam (complex float)
  size_t   memory_limit = 0;     ///< GPU memory limit (0 = auto)
};

// =========================================================================
// Result structures
// =========================================================================

/**
 * @brief Mean result for one beam (complex)
 */
struct MeanResult {
  uint32_t beam_id = 0;
  std::complex<float> mean{0.0f, 0.0f};  ///< Complex mean
};

/**
 * @brief Full statistics for one beam (mean + variance + std)
 *
 * Computed in a single pass using Welford's algorithm.
 * For complex data: variance and std are computed over magnitudes.
 */
struct StatisticsResult {
  uint32_t beam_id = 0;

  // Complex mean
  std::complex<float> mean{0.0f, 0.0f};

  // Variance and STD over magnitudes
  float variance = 0.0f;          ///< Variance of |z| (magnitude)
  float std_dev  = 0.0f;          ///< Standard deviation of |z|
  float mean_magnitude = 0.0f;    ///< Mean of |z|
};

/**
 * @brief Median result for one beam
 *
 * Median is computed over magnitudes (|z|) of complex samples.
 * Uses radix sort (rocPRIM) + middle element.
 */
struct MedianResult {
  uint32_t beam_id = 0;
  float median_magnitude = 0.0f;  ///< Median of |z|
};

}  // namespace statistics
