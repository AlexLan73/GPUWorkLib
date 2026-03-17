#pragma once

/**
 * @file range_angle_params.hpp
 * @brief RangeAngle module — processing parameters
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "range_angle_types.hpp"
#include <cstdint>

namespace range_angle {

struct RangeAngleParams {
  // ── Antenna array ─────────────────────────────────────────────────────
  uint32_t n_ant_az  = 16;   ///< Number of azimuth antennas
  uint32_t n_ant_el  = 16;   ///< Number of elevation antennas

  // ── Signal parameters ─────────────────────────────────────────────────
  uint32_t n_samples   = 1'300'000;  ///< Samples per antenna per CPI
  float    f_start     = -5e6f;      ///< LFM start frequency, Hz
  float    f_end       = +5e6f;      ///< LFM end frequency, Hz
  float    sample_rate = 12e6f;      ///< ADC sample rate, Hz

  // ── Range FFT ─────────────────────────────────────────────────────────
  /// FFT size for range axis. 0 = auto (next power of 2 >= n_samples)
  uint32_t nfft_range = 0;

  // ── Spatial ───────────────────────────────────────────────────────────
  float antenna_spacing = 0.345f;   ///< Inter-element spacing, metres
  float carrier_freq    = 435e6f;   ///< RF carrier frequency, Hz

  // ── Peak search ───────────────────────────────────────────────────────
  PeakSearchMode peak_mode = PeakSearchMode::kTop1;
  uint32_t       n_peaks   = 1;

  // ── Computed fields (filled by RangeAngleProcessor::SetParams) ────────
  uint32_t n_range_bins = 0;    ///< Effective range bins after FFT
  float    range_res_m  = 0.f;  ///< Range resolution, metres

  // ── Helpers ───────────────────────────────────────────────────────────
  uint32_t get_n_antennas() const { return n_ant_az * n_ant_el; }
  float    get_bandwidth()  const { return f_end - f_start; }
  float    get_duration()   const { return static_cast<float>(n_samples) / sample_rate; }
  float    get_chirp_rate() const { return get_bandwidth() / get_duration(); }
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
