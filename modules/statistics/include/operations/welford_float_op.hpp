#pragma once

/**
 * @file welford_float_op.hpp
 * @brief WelfordFloatOp — Welford statistics on float magnitudes (already computed)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from StatisticsProcessor::ExecuteWelfordFloatKernel().
 *
 * For cases where magnitudes are pre-computed (float input, not complex).
 * Computes mean(|z|) + variance(|z|) + std(|z|).
 *
 * Kernels: welford_float
 * Private buffers: BufferSet<0>
 * Shared buffers: reads kMagnitudes, writes kResult
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "services/buffer_set.hpp"
#include "interface/gpu_context.hpp"
#include "statistics_types.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace statistics {

class WelfordFloatOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "WelfordFloat"; }

  /**
   * @brief Execute Welford on float magnitudes
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Reads kMagnitudes (float), writes kResult (5 floats per beam:
   * mean_re=0, mean_im=0, mean_mag, variance, std_dev).
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    void* magnitudes_buf = ctx_->GetShared(shared_buf::kMagnitudes);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));

    // 3 shared arrays × (kBlockSize+1) floats: mean_mag, M2, count
    size_t shared_mem = 3 * (kBlockSize + 1) * sizeof(float);

    void* args[] = { &magnitudes_buf, &result_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("welford_float"),
        bc, 1, 1,
        kBlockSize, 1, 1,
        shared_mem, stream(),
        args, nullptr);

    if (err != hipSuccess) {
      throw std::runtime_error("WelfordFloatOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
  drv_gpu_lib::BufferSet<0> bufs_;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
