#pragma once

/**
 * @file range_angle_processor.hpp
 * @brief RangeAngleProcessor — Facade for LFM range-angle processing pipeline (ROCm/HIP)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 *
 * Processing pipeline:
 *   1. DechirpWindowOp  — multiply by conjugate LFM reference + Hann window
 *   2. RangeFftOp       — batched 1-D forward FFT along range axis (hipFFT)
 *   3. TransposeOp      — reorder [n_ant × n_range_bins] → [n_range_bins × n_ant]
 *   4. BeamFftOp        — 2-D spatial FFT (azimuth × elevation) + fftshift + |.|^2
 *   5. PeakSearchOp     — find peak(s) in power cube → TargetInfo list
 *
 * Public API:
 *   processor.SetParams(params);
 *   RangeAngleResult result = processor.Process(iq_data);
 *
 * ROCm-only module. All GPU memory managed via GpuContext shared buffers.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "range_angle_params.hpp"
#include "range_angle_types.hpp"
#include "operations/dechirp_window_op.hpp"
#include "operations/range_fft_op.hpp"
#include "operations/transpose_op.hpp"
#include "operations/beam_fft_op.hpp"
#include "operations/peak_search_op.hpp"

#include "interface/gpu_context.hpp"
#include "interface/i_backend.hpp"

#include <complex>
#include <cstdint>
#include <vector>

namespace range_angle {

/// @ingroup grp_range_angle
class RangeAngleProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Construct processor for a given GPU backend
   * @param backend Non-owning pointer to IBackend (must be ROCm, initialized)
   */
  explicit RangeAngleProcessor(drv_gpu_lib::IBackend* backend);

  ~RangeAngleProcessor();

  // No copy
  RangeAngleProcessor(const RangeAngleProcessor&)            = delete;
  RangeAngleProcessor& operator=(const RangeAngleProcessor&) = delete;

  // Move semantics
  RangeAngleProcessor(RangeAngleProcessor&& other) noexcept;
  RangeAngleProcessor& operator=(RangeAngleProcessor&& other) noexcept;

  // =========================================================================
  // Configuration
  // =========================================================================

  /**
   * @brief Set processing parameters
   *
   * Computes derived fields (n_range_bins, range_res_m, nfft_range).
   * Call before Process(). Invalidates compiled state if dimensions changed.
   */
  void SetParams(const RangeAngleParams& params);

  const RangeAngleParams& GetParams() const { return params_; }

  // =========================================================================
  // Processing
  // =========================================================================

  /**
   * @brief Run full range-angle processing pipeline
   * @param data           Input IQ data: n_ant * n_samples complex<float>,
   *                       layout: antenna-major (antenna 0 first, then antenna 1, ...)
   * @param download_result If true, copy power_cube from GPU to result.power_cube
   * @return RangeAngleResult with detected targets (and optionally power cube)
   */
  RangeAngleResult Process(
      const std::vector<std::complex<float>>& data,
      bool download_result = true);

private:
  // =========================================================================
  // Internal helpers
  // =========================================================================

  /// Lazy kernel compilation (hiprtc, once per param change)
  void EnsureCompiled();

  /// Build reference LFM chirp on GPU (lazy, once per params)
  void BuildRefSignal();

  /// Upload CPU IQ data to shared kInput buffer
  void UploadData(const std::complex<float>* data, size_t count);

  /// Compute derived params (n_range_bins, range_res_m, nfft_range)
  void ComputeDerivedParams();

  // =========================================================================
  // Members
  // =========================================================================

  drv_gpu_lib::GpuContext ctx_;   ///< Per-module context (Layer 1)

  // Op instances (Layer 5)
  DechirpWindowOp  dechirp_op_;
  RangeFftOp       range_fft_op_;
  TransposeOp      transpose_op_;
  BeamFftOp        beam_fft_op_;
  PeakSearchOp     peak_op_;

  RangeAngleParams params_;
  bool compiled_   = false;
  bool ref_built_  = false;
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
