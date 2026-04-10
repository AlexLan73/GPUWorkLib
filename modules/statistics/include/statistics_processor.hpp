#pragma once

/**
 * @file statistics_processor.hpp
 * @brief StatisticsProcessor — thin Facade for statistical GPU computations (ROCm/HIP)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 *
 * ROCm-only module. Computes per-beam statistics on complex float GPU data:
 * - Mean (complex) — MeanReductionOp
 * - Median (magnitude) — MedianHistogramOp / MedianRadixSortOp (auto-select)
 * - Variance / STD (magnitude) — WelfordFusedOp / WelfordFloatOp
 * - ComputeStatistics — one-pass mean + variance + std (WelfordFusedOp)
 *
 * Input: beam_count * n_point complex<float> (all antennas at once).
 *
 * PUBLIC API IS UNCHANGED — Python bindings (py_statistics.hpp) work as before.
 *
 * Internal structure:
 *   GpuContext ctx_           — per-module: stream, compiled kernels, shared buffers
 *   MeanReductionOp           — hierarchical complex mean
 *   WelfordFusedOp            — single-pass statistics (complex input)
 *   WelfordFloatOp            — statistics on float magnitudes
 *   MedianRadixSortOp         — rocPRIM sort median (small data)
 *   MedianHistogramOp         — histogram median (large data, float input)
 *   MedianHistogramComplexOp  — histogram median (large data, complex input)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-03-14 (v2 Ref03 Facade)
 */

#if ENABLE_ROCM

#include "statistics_types.hpp"
#include "interface/gpu_context.hpp"
#include "services/profiling_types.hpp"

// Op classes (Layer 5)
#include "operations/mean_reduction_op.hpp"
#include "operations/welford_fused_op.hpp"
#include "operations/welford_float_op.hpp"
#include "operations/median_radix_sort_op.hpp"
#include "operations/median_histogram_op.hpp"
#include "operations/median_histogram_complex_op.hpp"
#include "operations/snr_estimator_op.hpp"  // SNR_05

#include "interface/i_backend.hpp"

#include <complex>
#include <vector>
#include <cstdint>

namespace statistics {

/// ROCm profiling events for ComputeAll methods.
/// Vector of (event_name, timing_data) pairs — same pattern as HeterodyneROCmProfEvents.
using StatisticsROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/// @ingroup grp_statistics
class StatisticsProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Constructor
   * @param backend Pointer to IBackend (non-owning, must be ROCm backend)
   */
  explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);

  ~StatisticsProcessor();

  // No copying
  StatisticsProcessor(const StatisticsProcessor&) = delete;
  StatisticsProcessor& operator=(const StatisticsProcessor&) = delete;

  // Move semantics
  StatisticsProcessor(StatisticsProcessor&& other) noexcept;
  StatisticsProcessor& operator=(StatisticsProcessor&& other) noexcept;

  // =========================================================================
  // Public API -- CPU data (upload -> compute -> download)
  // =========================================================================

  std::vector<MeanResult> ComputeMean(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  std::vector<MedianResult> ComputeMedian(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  std::vector<StatisticsResult> ComputeStatistics(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU data (already on device)
  // =========================================================================

  std::vector<MeanResult> ComputeMean(
      void* gpu_data,
      const StatisticsParams& params);

  std::vector<MedianResult> ComputeMedian(
      void* gpu_data,
      const StatisticsParams& params);

  std::vector<StatisticsResult> ComputeStatistics(
      void* gpu_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- ComputeAll (Statistics + Median in one call)
  // =========================================================================

  /// CPU complex data: upload once → Welford + Median → FullStatisticsResult per beam.
  /// Eliminates double upload vs calling ComputeStatistics + ComputeMedian separately.
  std::vector<FullStatisticsResult> ComputeAll(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// GPU complex data (production path): D2D once → Welford + Median.
  std::vector<FullStatisticsResult> ComputeAll(
      void* gpu_data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// GPU float magnitudes: kMagnitudes → WelfordFloat + Median.
  /// Note: mean field is always {0, 0} (float path has no complex mean).
  std::vector<FullStatisticsResult> ComputeAllFloat(
      void* gpu_float_data,
      const StatisticsParams& params,
      StatisticsROCmProfEvents* prof_events = nullptr);

  /// CPU float magnitudes: convenience wrapper, uploads then calls GPU overload.
  std::vector<FullStatisticsResult> ComputeAllFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU float data (magnitudes already computed)
  // =========================================================================

  std::vector<StatisticsResult> ComputeStatisticsFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  std::vector<MedianResult> ComputeMedianFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- CPU float data (vector<float> wrappers for tests)
  // =========================================================================

  std::vector<StatisticsResult> ComputeStatisticsFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  std::vector<MedianResult> ComputeMedianFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- SNR estimation (SNR_06)
  // =========================================================================

  /**
   * @brief Compute SNR (dB) from CPU data via CA-CFAR
   *
   * Pipeline: upload → gather → FFT(Hann)|X|² → CFAR → median.
   *
   * @param data        CPU complex<float> [n_antennas × n_samples] (row-major)
   * @param n_antennas  Number of antennas
   * @param n_samples   Samples per antenna
   * @param config      SNR estimation config (см. snr_defaults::)
   * @return Result with snr_db_global, used_antennas, used_bins, n_actual
   *
   * @note Result НЕ содержит BranchType — классификация через BranchSelector.
   */
  SnrEstimationResult ComputeSnrDb(
      const std::vector<std::complex<float>>& data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

  /**
   * @brief Compute SNR (dB) from GPU data (production path)
   *
   * Pipeline: gather → FFT(Hann)|X|² → CFAR → median (данные уже на GPU).
   *
   * @param gpu_data    GPU complex<float>* [n_antennas × n_samples]
   * @param n_antennas  Number of antennas
   * @param n_samples   Samples per antenna
   * @param config      SNR estimation config
   */
  SnrEstimationResult ComputeSnrDb(
      void* gpu_data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

private:
  // =========================================================================
  // Ref03: GpuContext + Op instances
  // =========================================================================

  /// Ensure kernels are compiled (lazy, one-time)
  void EnsureCompiled();

  /// Upload CPU complex data to shared kInput buffer
  void UploadComplexData(const std::complex<float>* data, size_t count);

  /// Copy GPU complex data to shared kInput buffer (D2D)
  void CopyComplexGpuData(void* src, size_t count);

  /// Copy GPU float data to shared kMagnitudes buffer (D2D)
  void CopyFloatGpuData(void* src, size_t count);

  /// Upload CPU float data to shared kMagnitudes buffer (H2D async, reuses buffer)
  void UploadFloatData(const float* data, size_t count);

  /// Read MeanResult from kResult buffer
  std::vector<MeanResult> ReadMeanResults(uint32_t beam_count);

  /// Read StatisticsResult from kResult buffer
  std::vector<StatisticsResult> ReadStatisticsResults(uint32_t beam_count);

  /// Read MedianResult from kMediansCompact buffer
  std::vector<MedianResult> ReadMedianResults(uint32_t beam_count);

  /// Merge StatisticsResult + MedianResult into FullStatisticsResult (1-to-1 by index)
  std::vector<FullStatisticsResult> MergeResults(
      const std::vector<StatisticsResult>& stats,
      const std::vector<MedianResult>& medians);

  // ── Members ───────────────────────────────────────────────────────────

  drv_gpu_lib::IBackend* backend_ = nullptr;  ///< сохранён из конструктора (нужен для SnrEstimatorOp::SetupFft)
  drv_gpu_lib::GpuContext ctx_;     ///< Per-module context (stream, kernels, shared bufs)

  // Op instances (Layer 5) — member variables, not unique_ptr
  MeanReductionOp         mean_op_;
  WelfordFusedOp          welford_fused_op_;
  WelfordFloatOp          welford_float_op_;
  MedianRadixSortOp       median_sort_op_;
  MedianHistogramOp       median_hist_op_;
  MedianHistogramComplexOp median_hist_complex_op_;

  // SNR-estimator (SNR_05/06) — ленивая инициализация
  SnrEstimatorOp snr_estimator_op_;
  bool snr_op_initialized_ = false;

  bool compiled_ = false;

  /// Auto-select threshold: n_point > kHistogramThreshold → histogram, else radix sort
  static constexpr size_t kHistogramThreshold = 100'000;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
