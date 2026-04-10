#pragma once

/**
 * @file statistics_types.hpp
 * @brief Types and structures for StatisticsProcessor (ROCm)
 *
 * Defines input parameters and result structures for statistical
 * computations on complex float signal data (per-beam).
 *
 * SNR-estimator types (SNR_01, 2026-04-09):
 *   - BranchType / BranchThresholds
 *   - SnrEstimationConfig / SnrEstimationResult
 *   - snr_defaults:: namespace (calibrated by Python Эксп.5)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-04-09 (v2 — SNR_01)
 */

#include "types/window_type.hpp"  // fft_processor::WindowType (SNR_02b)

#include <vector>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace statistics {

// =========================================================================
// Shared GPU buffer slot assignments for StatisticsProcessor
// =========================================================================

/// Slot indices for GpuContext::RequireShared / GetShared.
/// Each Op reads/writes specific slots; assignments are module-specific
/// and do NOT belong in the generic GpuContext infrastructure.
namespace shared_buf {
  // Existing slots (legacy — НЕ трогать порядок):
  static constexpr size_t kInput          = 0;  ///< complex<float> input data
  static constexpr size_t kMagnitudes     = 1;  ///< float magnitudes |z|
  static constexpr size_t kResult         = 2;  ///< per-beam results (various types)
  static constexpr size_t kMediansCompact = 3;  ///< float[beam_count] compact medians

  // SNR-estimator slots (SNR_01):
  static constexpr size_t kGatherOutput   = 4;  ///< complex gather subset (n_ant_out × n_actual)
  static constexpr size_t kFftMagSquared  = 5;  ///< float |X|² from ProcessMagnitudesToGPU
  static constexpr size_t kSnrPerAntenna  = 6;  ///< float SNR_db per antenna (before median)

  static constexpr size_t kCount          = 7;  ///< total slots used by this module
}  // namespace shared_buf

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

/**
 * @brief Combined statistics + median result for one beam (ComputeAll output)
 *
 * Merges StatisticsResult (Welford) + MedianResult into a single struct.
 * For float input path (ComputeAllFloat): mean is always {0, 0}.
 */
struct FullStatisticsResult {
  uint32_t beam_id = 0;

  std::complex<float> mean{0.0f, 0.0f};  ///< Complex mean (zero for float path)
  float variance         = 0.0f;         ///< Variance of |z|
  float std_dev          = 0.0f;         ///< Standard deviation of |z|
  float mean_magnitude   = 0.0f;         ///< Mean of |z|
  float median_magnitude = 0.0f;         ///< Median of |z|
};

// =========================================================================
// SNR-estimator (SNR_01) — калибровано Python моделью (Эксп.5, 2026-04-09)
// Source: PyPanelAntennas/SNR/results/exp5_thresholds.json
// =========================================================================

/**
 * @brief Default параметры SNR-estimator. Калибровано Python моделью.
 *
 * Изменения vs pre-calibration:
 *   - kGuardBins:  3 → 5   (для Hann window достаточно 5)
 *   - kRefBins:    8 → 16  (для Hann более плавная оценка шума)
 *   - kDefaultWindow: None → Hann (решает sinc sidelobes −27 dB bias)
 */
namespace snr_defaults {
  static constexpr uint32_t kTargetNFft           = 2048;  ///< default N_fft (гибкий, не догма)
  static constexpr uint32_t kGuardBins            = 5;     ///< калибровано для Hann
  static constexpr uint32_t kRefBins              = 16;    ///< калибровано для Hann
  static constexpr uint32_t kTargetAntennasMedian = 50;    ///< медиана по ~50 антеннам
  static constexpr float    kHysteresisDb         = 2.0f;  ///< защита от дребезга BranchSelector

  /// Default window — Hann (решает проблему sinc sidelobes).
  /// rect даёт −27 dB bias! Python Эксп.0 показал Hann — оптимальный компромисс.
  static constexpr fft_processor::WindowType kDefaultWindow =
      fft_processor::WindowType::Hann;
}  // namespace snr_defaults

/// Branch category для переключения обработки Low/Mid/High SNR
enum class BranchType {
  Low,   ///< low SNR  (< low_to_mid_db) — слабый сигнал/шум
  Mid,   ///< medium SNR (between thresholds) — обычный
  High,  ///< high SNR (> mid_to_high_db) — сильный сигнал
};

/**
 * @brief Пороги переключения branch'ей для SNR-estimator.
 *
 * Калибровано Python Эксп.5 для Hann + CA-CFAR (mean):
 *   P_correct = 97.9% для (Low < −15 dB, Mid −15..0 dB, High > 0 dB).
 * Hysteresis 2 dB защищает пограничные случаи (2.1% ошибок классификации).
 */
struct BranchThresholds {
  float low_to_mid_db  = 15.0f;  ///< калибровано Python Эксп.5 (было 6.0 pre-calibration)
  float mid_to_high_db = 30.0f;  ///< калибровано Python Эксп.5 (было 12.0 pre-calibration)
  float hysteresis_db  = snr_defaults::kHysteresisDb;
};

/**
 * @brief Config для SNR-estimator.
 *
 * Все поля с `= 0` обозначают auto-режим:
 *   - target_n_fft = 0  → snr_defaults::kTargetNFft (2048)
 *   - step_samples = 0  → ceil(n_samples / target_n_fft)
 *   - step_antennas = 0 → ceil(n_antennas / kTargetAntennasMedian)
 *
 * @note Result НЕ содержит BranchType — для классификации использовать BranchSelector.
 */
struct SnrEstimationConfig {
  uint32_t target_n_fft  = 0;   ///< 0 → auto (snr_defaults::kTargetNFft = 2048)
  uint32_t step_samples  = 0;   ///< 0 → auto из target_n_fft
  uint32_t step_antennas = 0;   ///< 0 → auto (ceil(n_antennas / kTargetAntennasMedian))
  uint32_t guard_bins = snr_defaults::kGuardBins;  ///< default 5 (калибровано)
  uint32_t ref_bins   = snr_defaults::kRefBins;    ///< default 16 (калибровано)
  bool     search_full_spectrum = true;            ///< false → search in [0..nFFT/2]

  /// Window function для FFT pre-processing.
  /// Hann (default) решает проблему sinc sidelobes.
  fft_processor::WindowType window = snr_defaults::kDefaultWindow;

  bool     with_dechirp = false;     ///< reserved: встроить дечирп в pipeline
  BranchThresholds thresholds;       ///< калиброванные пороги

  /**
   * @brief Validate config invariants.
   * @throws std::invalid_argument при нарушении (ref window >= nFFT, etc.)
   */
  void Validate() const {
    uint32_t nfft_effective = (target_n_fft > 0) ? target_n_fft : snr_defaults::kTargetNFft;
    // ref window должен помещаться в nFFT с запасом на peak
    if (2u * (guard_bins + ref_bins) + 1u >= nfft_effective) {
      throw std::invalid_argument(
          "SnrEstimationConfig: ref window (2*(guard+ref)+1=" +
          std::to_string(2u * (guard_bins + ref_bins) + 1u) +
          ") >= target_n_fft (" + std::to_string(nfft_effective) + ")");
    }
    if (thresholds.low_to_mid_db >= thresholds.mid_to_high_db) {
      throw std::invalid_argument(
          "SnrEstimationConfig: low_to_mid_db (" +
          std::to_string(thresholds.low_to_mid_db) +
          ") >= mid_to_high_db (" +
          std::to_string(thresholds.mid_to_high_db) + ")");
    }
    if (thresholds.hysteresis_db < 0.0f) {
      throw std::invalid_argument("SnrEstimationConfig: hysteresis_db < 0");
    }
  }
};

/**
 * @brief Result SNR-estimator.
 *
 * @note НЕТ поля BranchType! Классификация делается caller'ом через BranchSelector
 *       (stateful + hysteresis — facade StatisticsProcessor остаётся stateless).
 */
struct SnrEstimationResult {
  float snr_db_global = 0.0f;               ///< медиана SNR_db по used_antennas
  std::vector<float> snr_db_per_antenna;    ///< per-antenna (опционально, может быть пустым)
  uint32_t used_antennas      = 0;          ///< фактическое число антенн после децимации
  uint32_t used_bins          = 0;          ///< фактический nFFT (после NextPowerOf2)
  uint32_t actual_step_samples = 0;         ///< фактический step_samples после auto
  uint32_t n_actual           = 0;          ///< число сэмплов после децимации (до padding)
};

}  // namespace statistics
