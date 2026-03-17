#pragma once

/**
 * @file range_angle_types.hpp
 * @brief RangeAngle module — types, enums, result structs, shared buffer indices
 *
 * Ref03 Unified Architecture: Layer 1 support (shared buffer index constants).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include <cstdint>
#include <string>
#include <vector>

namespace range_angle {

// ============================================================================
// Enums
// ============================================================================

enum class PeakSearchMode {
  kTop1,   ///< Find single strongest target
  kTopN,   ///< Find N strongest targets
};

// ============================================================================
// Result structs
// ============================================================================

/// Single detected target info
struct TargetInfo {
  float range_m      = 0.f;   ///< Range in metres
  float angle_az_deg = 0.f;   ///< Azimuth angle, degrees
  float angle_el_deg = 0.f;   ///< Elevation angle, degrees
  float range_bin    = 0.f;   ///< Fractional range bin index
  float az_bin       = 0.f;   ///< Fractional azimuth bin index
  float el_bin       = 0.f;   ///< Fractional elevation bin index
  float power_db     = 0.f;   ///< Peak power, dB
  float snr_db       = 0.f;   ///< Estimated SNR, dB
};

/// Full processing result
struct RangeAngleResult {
  bool     success          = false;
  uint32_t n_range_bins     = 0;
  uint32_t n_ant_az         = 0;
  uint32_t n_ant_el         = 0;

  /// Power cube on host (downloaded if download_result=true), size: n_range_bins * n_ant_az * n_ant_el
  std::vector<float> power_cube;

  /// Device pointer to power cube (valid until next Process call)
  void* gpu_power_cube = nullptr;

  std::vector<TargetInfo> targets;
  std::string error_message;
};

// ============================================================================
// Shared buffer slot indices (used with GpuContext::RequireShared)
// ============================================================================

namespace shared_buf {
  static constexpr size_t kInput      = 0;  ///< Raw IQ input (n_ant * n_samples * complex<float>)
  static constexpr size_t kRef        = 1;  ///< Reference LFM chirp signal
  static constexpr size_t kDechirped  = 2;  ///< After dechirp+window (n_ant * n_samples * complex<float>)
  static constexpr size_t kRangeFFT   = 3;  ///< After range FFT (n_ant * nfft_range * complex<float>)
  static constexpr size_t kTransposed = 4;  ///< Transposed layout for beam FFT
  static constexpr size_t kBeamFFT    = 5;  ///< After beam-forming FFT (n_range_bins * n_az * n_el)
  static constexpr size_t kPowerCube  = 6;  ///< |BeamFFT|^2 power cube (float)
  static constexpr size_t kCount      = 7;  ///< Total number of shared buffer slots
}  // namespace shared_buf

}  // namespace range_angle

#endif  // ENABLE_ROCM
