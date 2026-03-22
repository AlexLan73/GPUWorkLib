#pragma once

/**
 * @file welford_fused_op.hpp
 * @brief WelfordFusedOp — single-pass Welford statistics on complex input
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from StatisticsProcessor::ExecuteWelfordFusedKernel().
 *
 * Computes mean(complex) + mean(|z|) + variance(|z|) + std(|z|) in ONE pass.
 * Reads input directly — no separate magnitudes kernel needed.
 *
 * Kernels: welford_fused
 * Private buffers: BufferSet<0> — no private buffers
 * Shared buffers: reads kInput, writes kResult
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

class WelfordFusedOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "WelfordFused"; }

  /**
   * @brief Execute fused Welford kernel
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Reads kInput (complex<float>), writes kResult (5 floats per beam:
   * mean_re, mean_im, mean_mag, variance, std_dev).
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    void* input_buf  = ctx_->GetShared(shared_buf::kInput);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));

    // 5 shared arrays × (kBlockSize+1) floats: sum_re, sum_im, mean_mag, M2, count
    size_t shared_mem = 5 * (kBlockSize + 1) * sizeof(float);

    void* args[] = { &input_buf, &result_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("welford_fused"),
        bc, 1, 1,           // one block per beam
        kBlockSize, 1, 1,
        shared_mem, stream(),
        args, nullptr);

    if (err != hipSuccess) {
      throw std::runtime_error("WelfordFusedOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
  drv_gpu_lib::BufferSet<0> bufs_;  // no private buffers
};

}  // namespace statistics

#endif  // ENABLE_ROCM
