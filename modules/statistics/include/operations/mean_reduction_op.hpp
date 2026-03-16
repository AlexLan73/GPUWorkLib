#pragma once

/**
 * @file mean_reduction_op.hpp
 * @brief MeanReductionOp — hierarchical complex mean reduction on GPU
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from StatisticsProcessor::ExecuteMeanReduction().
 *
 * Kernels: mean_reduce_phase1, mean_reduce_final
 * Private buffers: BufferSet<1> — reduce_buf (partial sums)
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

class MeanReductionOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MeanReduction"; }

  /**
   * @brief Execute complex mean reduction
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Reads ctx_->GetShared(kInput), writes ctx_->GetShared(kResult).
   * Result: beam_count × float2 (re, im) in kResult buffer.
   */
  void Execute(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);

    // Double-load: each block covers 2 × kBlockSize elements
    unsigned int blocks_per_beam = (np + kDoubleLoadElements - 1) / kDoubleLoadElements;

    // Allocate private reduce buffer
    size_t reduce_count = beam_count * blocks_per_beam;
    bufs_.Require(kReduce, reduce_count * 2 * sizeof(float));  // float2

    void* input_buf  = ctx_->GetShared(shared_buf::kInput);
    void* reduce_buf = bufs_.Get(kReduce);
    void* result_buf = ctx_->RequireShared(
        shared_buf::kResult,
        beam_count * 5 * sizeof(float));  // max: WelfordResult (5 floats)

    // Phase 1: block-level reduction (2D grid: blocks_per_beam × beam_count)
    void* args1[] = { &input_buf, &reduce_buf, &bc, &np };

    hipError_t err = hipModuleLaunchKernel(
        kernel("mean_reduce_phase1"),
        blocks_per_beam, bc, 1,
        kBlockSize, 1, 1,
        0, stream(),
        args1, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MeanReductionOp phase1: " +
                                std::string(hipGetErrorString(err)));
    }

    // Phase 2: final reduction (one block per beam)
    unsigned int final_block = 1;
    while (final_block < blocks_per_beam && final_block < kBlockSize) {
      final_block *= 2;
    }
    if (final_block > kBlockSize) final_block = kBlockSize;

    void* args2[] = { &reduce_buf, &result_buf, &bc, &blocks_per_beam, &np };

    err = hipModuleLaunchKernel(
        kernel("mean_reduce_final"),
        bc, 1, 1,
        final_block, 1, 1,
        0, stream(),
        args2, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MeanReductionOp final: " +
                                std::string(hipGetErrorString(err)));
    }
  }

protected:
  void OnRelease() override { bufs_.ReleaseAll(); }

private:
  static constexpr unsigned int kBlockSize = 256;
  static constexpr unsigned int kDoubleLoadElements = kBlockSize * 2;

  enum Buf { kReduce, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
