#pragma once

/**
 * @file range_fft_op.hpp
 * @brief RangeFftOp — batched 1-D forward FFT along range axis (hipFFT)
 *
 * Ref03 Unified Architecture: Layer 5 (Concrete Operation).
 *
 * Input:  shared_buf::kDechirped — [n_ant * nfft_r]  complex<float>
 * Output: shared_buf::kRangeFFT  — [n_ant * nfft_r]  complex<float>
 *         (in-place: output overwrites the same buffer via kRangeFFT slot)
 *
 * Actually performs in-place: reads kDechirped, writes to kRangeFFT which
 * the processor pre-allocates at same size. The FFT itself is in-place
 * (src == dst pointer) to avoid extra allocation.
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

class RangeFftOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "RangeFFT"; }

  /**
   * @brief Create batched 1-D hipFFT plan.
   * @param nfft_r  Range FFT size (power of 2)
   * @param batch   Number of antennas (batch count)
   */
  void InitPlan(uint32_t nfft_r, uint32_t batch) {
    if (plan_ != 0) {
      hipfftDestroy(plan_);
      plan_ = 0;
    }

    nfft_r_ = nfft_r;
    batch_  = batch;

    hipfftResult res = hipfftPlan1d(&plan_,
                                    static_cast<int>(nfft_r),
                                    HIPFFT_C2C,
                                    static_cast<int>(batch));
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "RangeFftOp::InitPlan hipfftPlan1d failed: " + std::to_string(res));
    }

    res = hipfftSetStream(plan_, stream());
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "RangeFftOp::InitPlan hipfftSetStream failed: " + std::to_string(res));
    }
  }

  /**
   * @brief Execute batched forward C2C FFT in-place on kDechirped buffer.
   *        Result written to same buffer (kDechirped == kRangeFFT in-place).
   */
  void Execute() {
    void* d_buf = ctx_->GetShared(shared_buf::kDechirped);
    if (!d_buf) {
      throw std::runtime_error("RangeFftOp::Execute: kDechirped buffer is null");
    }
    if (plan_ == 0) {
      throw std::runtime_error("RangeFftOp::Execute: plan not initialized");
    }

    hipfftResult res = hipfftExecC2C(
        plan_,
        static_cast<hipfftComplex*>(d_buf),
        static_cast<hipfftComplex*>(d_buf),
        HIPFFT_FORWARD);
    if (res != HIPFFT_SUCCESS) {
      throw std::runtime_error(
          "RangeFftOp::Execute hipfftExecC2C failed: " + std::to_string(res));
    }
  }

protected:
  void OnRelease() override {
    if (plan_ != 0) {
      hipfftDestroy(plan_);
      plan_ = 0;
    }
    nfft_r_ = 0;
    batch_  = 0;
  }

private:
  hipfftHandle plan_  = 0;
  uint32_t     nfft_r_ = 0;
  uint32_t     batch_  = 0;
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
