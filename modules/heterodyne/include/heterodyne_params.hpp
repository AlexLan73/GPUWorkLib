#pragma once

/**
 * @file heterodyne_params.hpp
 * @brief Parameters and result types for LFM dechirp processing
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include <cstdint>
#include <vector>
#include <string>

namespace drv_gpu_lib {

/** LFM parameters for dechirp */
struct HeterodyneParams {
  float f_start      = 0.0f;      // Hz, LFM start frequency
  float f_end        = 1e6f;      // Hz, LFM end frequency (B = f_end - f_start)
  float sample_rate  = 12e6f;     // Hz, fs
  int   num_samples  = 4000;      // N, samples per antenna
  int   num_antennas = 5;         // number of antennas (channels)

  // Derived (computed by class):
  // T   = num_samples / sample_rate
  // mu  = (f_end - f_start) / T        [Hz/s, chirp rate]
  // bin_width = sample_rate / num_samples [Hz/bin]

  float GetBandwidth() const { return f_end - f_start; }
  float GetDuration() const { return static_cast<float>(num_samples) / sample_rate; }
  float GetChirpRate() const { return GetBandwidth() / GetDuration(); }
  float GetBinWidth() const { return sample_rate / static_cast<float>(num_samples); }
};

/** Result per antenna */
struct AntennaDechirpResult {
  int   antenna_idx    = 0;
  float f_beat_hz      = 0.0f;    // beat frequency [Hz]
  float f_beat_bin     = 0.0f;    // bin (fractional, parabolic interp.)
  float range_m        = 0.0f;    // range R = c*T*f_beat/(2B) [m]
  float peak_amplitude = 0.0f;    // peak amplitude
  float peak_snr_db    = 0.0f;    // peak SNR [dB]
};

/** Overall dechirp result */
struct HeterodyneResult {
  bool  success = false;
  std::vector<AntennaDechirpResult> antennas;   // per-antenna result
  std::vector<float> max_positions;             // all maxima positions (for control)
  std::string error_message;

  // Helper: R = c * T * f_beat / (2 * B)
  static float CalcRange(float f_beat, float sample_rate,
                          int num_samples, float bandwidth) {
    float T = static_cast<float>(num_samples) / sample_rate;
    return (3e8f * T * f_beat) / (2.0f * bandwidth);
  }
};

}  // namespace drv_gpu_lib