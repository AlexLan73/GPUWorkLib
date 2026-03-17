#pragma once

/**
 * @file beam_fft_op.hpp
 * @brief BeamFftOp — 2-D batched beam-forming FFT (azimuth × elevation) + fftshift
 *
 * Ref03 Unified Architecture: Layer 5 (Concrete Operation).
 *
 * Input:  shared_buf::kTransposed — [n_range_bins × n_ant_az × n_ant_el]
 * Output: shared_buf::kBeamFFT    — [n_range_bins × n_az × n_el] after 2-D FFT + fftshift
 *
 * Uses hipFFT 2-D batched plan (batch = n_range_bins).
 * FFT is in-place on kTransposed buffer (reused as kBeamFFT via same pointer).
 * After FFT: fftshift2d_kernel reorders quadrants in-place.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "range_angle_types.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace range_angle {

class BeamFftOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "BeamFFT"; }

  /**
   * @brief Create 2-D batched hipFFT plan for beam-forming FFT.
   * @param n_az         Azimuth FFT size (= n_ant_az)
   * @param n_el         Elevation FFT size (= n_ant_el)
   * @param n_range_bins Batch size
   */
  void InitPlan(uint32_t n_az, uint32_t n_el, uint32_t n_range_bins) {
    if (plan_ != 0) {
      hipfftDestroy(plan_);
      plan_ = 0;
    }

    n_az_         = n_az;
    n_el_         = n_el;
    n_range_bins_ = n_range_bins;

    // hipfftPlanMany for 2-D C2C batched plan
    int rank      = 2;
    int dims[2]   = { static_cast<int>(n_az), static_cast<int>(n_el) };
    int inembed[2] = { static_cast<int>(n_az), static_cast<int>(n_el) };
    int onembed[2] = { static_cast<int>(n_az), static_cast<int>(n_el) };
    int istride   = 1;
    int ostride   = 1;
    int idist     = static_cast<int>(n_az * n_el);
    int odist     = static_cast<int>(n_az * n_el);

    hipfftResult res = hipfftPlanMany(
        &plan_, rank, dims,
        inembed, istride, idist,
        onembed, ostride, odist,
        HIPFFT_C2C,
        static_cast<int>(n_range_bins));
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "BeamFftOp::InitPlan hipfftPlanMany failed: " + std::to_string(res));
    }

    res = hipfftSetStream(plan_, stream());
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "BeamFftOp::InitPlan hipfftSetStream failed: " + std::to_string(res));
    }
  }

  /**
   * @brief Execute 2-D beam-forming FFT in-place, then fftshift2d in-place.
   *
   * Reads/writes shared_buf::kTransposed (in-place).
   * The fftshift2d swaps quadrants of each range bin's 2-D spectrum.
   */
  void Execute() {
    // 2-D FFT in-place on kTransposed
    void* d_buf = ctx_->GetShared(shared_buf::kTransposed);
    if (!d_buf) {
      throw std::runtime_error("BeamFftOp::Execute: kTransposed buffer is null");
    }
    if (plan_ == 0) {
      throw std::runtime_error("BeamFftOp::Execute: plan not initialized");
    }

    hipfftResult res = hipfftExecC2C(
        plan_,
        static_cast<hipfftComplex*>(d_buf),
        static_cast<hipfftComplex*>(d_buf),
        HIPFFT_FORWARD);
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "BeamFftOp::Execute hipfftExecC2C failed: " + std::to_string(res));
    }

    // fftshift2d in-place
    // Grid: (n_range_bins, n_az/2, n_el/2)  Block: (1, 1, 1)
    uint32_t half_az = n_az_ >> 1u;
    uint32_t half_el = n_el_ >> 1u;

    if (half_az == 0 || half_el == 0) return;  // degenerate case

    void* args[] = { &d_buf, &n_range_bins_, &n_az_, &n_el_ };

    hipError_t err = hipModuleLaunchKernel(
        kernel("fftshift2d_kernel"),
        n_range_bins_, half_az, half_el,
        1, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("BeamFftOp::Execute fftshift2d_kernel: ") +
          hipGetErrorString(err));
    }
  }

protected:
  void OnRelease() override {
    if (plan_ != 0) {
      hipfftDestroy(plan_);
      plan_ = 0;
    }
    n_az_ = n_el_ = n_range_bins_ = 0;
  }

private:
  hipfftHandle plan_         = 0;
  uint32_t     n_az_         = 0;
  uint32_t     n_el_         = 0;
  uint32_t     n_range_bins_ = 0;
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
