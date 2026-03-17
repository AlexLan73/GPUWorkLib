#pragma once

/**
 * @file peak_search_op.hpp
 * @brief PeakSearchOp — |.|^2 power cube + CPU argmax3D + target coordinate conversion
 *
 * Ref03 Unified Architecture: Layer 5 (Concrete Operation).
 *
 * Input:  shared_buf::kTransposed — complex beam FFT output [n_range_bins × n_az × n_el]
 * Output: shared_buf::kPowerCube  — float power cube (allocated here)
 *         result_                 — RangeAngleResult with targets list
 *
 * Pipeline:
 *   1. magnitude_sq_kernel: complex→float power (on GPU)
 *   2. hipMemcpy D2H (if download_result): copy power cube to host
 *   3. CPU argmax3D + parabolic interpolation
 *   4. Convert bins to (range_m, az_deg, el_deg)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "range_angle_types.hpp"
#include "range_angle_params.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace range_angle {

class PeakSearchOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "PeakSearch"; }

  /**
   * @brief Execute power cube computation + peak search.
   * @param p               Processing parameters
   * @param download_result If true, copy power_cube to result_.power_cube
   */
  void Execute(const RangeAngleParams& p, bool download_result) {
    result_ = RangeAngleResult{};

    uint32_t n_rbins = p.n_range_bins;
    uint32_t n_az    = p.n_ant_az;
    uint32_t n_el    = p.n_ant_el;
    uint32_t total   = n_rbins * n_az * n_el;

    // Allocate kPowerCube shared buffer (float)
    void* d_beam = ctx_->GetShared(shared_buf::kTransposed);
    void* d_pow  = ctx_->RequireShared(
        shared_buf::kPowerCube,
        static_cast<size_t>(total) * sizeof(float));

    if (!d_beam || !d_pow) {
      result_.success       = false;
      result_.error_message = "PeakSearchOp: null GPU buffer";
      return;
    }

    // Step 1: |z|^2 on GPU
    uint32_t grid_x = (total + 255u) / 256u;
    void* args[] = { &d_beam, &d_pow, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("magnitude_sq_kernel"),
        grid_x, 1, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      result_.success       = false;
      result_.error_message = std::string("PeakSearchOp magnitude_sq_kernel: ") +
                              hipGetErrorString(err);
      return;
    }

    // Step 2: Synchronize (we need results on CPU)
    err = hipStreamSynchronize(stream());
    if (err != hipSuccess) {
      result_.success       = false;
      result_.error_message = std::string("PeakSearchOp sync: ") + hipGetErrorString(err);
      return;
    }

    // Step 3: Download power cube to host
    std::vector<float> h_power(total);
    err = hipMemcpy(h_power.data(), d_pow,
                    static_cast<size_t>(total) * sizeof(float),
                    hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      result_.success       = false;
      result_.error_message = std::string("PeakSearchOp D2H: ") +
                              hipGetErrorString(err);
      return;
    }

    // Fill result power cube if requested
    if (download_result) {
      result_.power_cube = h_power;
    }
    result_.gpu_power_cube = d_pow;
    result_.n_range_bins   = n_rbins;
    result_.n_ant_az       = n_az;
    result_.n_ant_el       = n_el;

    if (total == 0 || n_rbins == 0) {
      result_.success = true;
      return;
    }

    // Step 4: CPU argmax3D
    uint32_t best_r = 0, best_az = 0, best_el = 0;
    float    best_val = -std::numeric_limits<float>::max();

    for (uint32_t r = 0; r < n_rbins; ++r) {
      for (uint32_t az = 0; az < n_az; ++az) {
        for (uint32_t el = 0; el < n_el; ++el) {
          float v = h_power[r * n_az * n_el + az * n_el + el];
          if (v > best_val) {
            best_val = v;
            best_r   = r;
            best_az  = az;
            best_el  = el;
          }
        }
      }
    }

    // Step 5: Parabolic interpolation (fractional bin)
    auto parabolic = [](float vm1, float v0, float vp1) -> float {
      float denom = 2.f * (2.f * v0 - vm1 - vp1);
      if (fabsf(denom) < 1e-12f) return 0.f;
      return (vp1 - vm1) / denom;
    };

    auto get = [&](int r, int az, int el) -> float {
      r  = std::max(0, std::min((int)n_rbins - 1, r));
      az = std::max(0, std::min((int)n_az    - 1, az));
      el = std::max(0, std::min((int)n_el    - 1, el));
      return h_power[r * n_az * n_el + az * n_el + el];
    };

    float frac_r  = parabolic(
        get(best_r - 1, best_az, best_el),
        get(best_r,     best_az, best_el),
        get(best_r + 1, best_az, best_el));
    float frac_az = parabolic(
        get(best_r, best_az - 1, best_el),
        get(best_r, best_az,     best_el),
        get(best_r, best_az + 1, best_el));
    float frac_el = parabolic(
        get(best_r, best_az, best_el - 1),
        get(best_r, best_az, best_el    ),
        get(best_r, best_az, best_el + 1));

    float r_frac  = static_cast<float>(best_r)  + frac_r;
    float az_frac = static_cast<float>(best_az) + frac_az;
    float el_frac = static_cast<float>(best_el) + frac_el;

    // Step 6: Convert bins to physical units
    // Range: bin index → metres
    //   range_m = r_frac * range_res_m
    float range_m = r_frac * p.range_res_m;

    // Angle: az/el FFT bin → sine space → degrees
    //   sin(theta) = (bin - N/2) / N * lambda / d
    //   lambda = c / carrier_freq
    //   Normalised spatial frequency: f_az = (az_frac - n_az/2) / n_az
    float lambda  = 3e8f / p.carrier_freq;
    float d       = p.antenna_spacing;
    float f_az    = (az_frac - static_cast<float>(n_az) * 0.5f) /
                    static_cast<float>(n_az);
    float f_el    = (el_frac - static_cast<float>(n_el) * 0.5f) /
                    static_cast<float>(n_el);
    float sin_az  = f_az * lambda / d;
    float sin_el  = f_el * lambda / d;
    // Clamp to valid sine range
    sin_az = std::max(-1.f, std::min(1.f, sin_az));
    sin_el = std::max(-1.f, std::min(1.f, sin_el));
    float az_deg  = asinf(sin_az) * (180.f / 3.14159265f);
    float el_deg  = asinf(sin_el) * (180.f / 3.14159265f);

    // Power in dB
    float power_db = (best_val > 0.f) ? 10.f * log10f(best_val) : -999.f;

    TargetInfo tgt;
    tgt.range_m      = range_m;
    tgt.angle_az_deg = az_deg;
    tgt.angle_el_deg = el_deg;
    tgt.range_bin    = r_frac;
    tgt.az_bin       = az_frac;
    tgt.el_bin       = el_frac;
    tgt.power_db     = power_db;
    tgt.snr_db       = 0.f;  // not computed

    result_.targets.push_back(tgt);
    result_.success = true;
  }

  const RangeAngleResult& GetResult() const { return result_; }
  RangeAngleResult&       GetResult()       { return result_; }

protected:
  void OnRelease() override {
    result_ = RangeAngleResult{};
  }

private:
  RangeAngleResult result_;
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
