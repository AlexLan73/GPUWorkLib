#pragma once

/**
 * @file median_histogram_op.hpp
 * @brief MedianHistogramOp — exact median via 4-pass byte-wise histogram (float input)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from StatisticsProcessor::ExecuteHistogramMedian() with is_complex=false.
 *
 * For large datasets (n_point > 100K) — O(n) instead of O(n log n) sort.
 * 4 passes: each narrows to the correct byte of the median value.
 *
 * Kernels: histogram_median_pass, find_median_bucket
 * Private buffers: BufferSet<3> — hist_buf, target_prefix, target_value
 * Shared buffers: reads kMagnitudes, writes kMediansCompact
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
#include <vector>
#include <cstring>
#include <algorithm>

namespace statistics {

class MedianHistogramOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MedianHistogram"; }

  /**
   * @brief Execute histogram-based median on float magnitudes
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Reads kMagnitudes (float), writes kMediansCompact (float[beam_count]).
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count);

    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int median_rank = np / 2;

    unsigned int blocks_per_beam = std::min(
        static_cast<unsigned int>((n_point + kBlockSize - 1) / kBlockSize),
        1024u);

    void* data_ptr = ctx_->GetShared(shared_buf::kMagnitudes);
    void* hist_buf = bufs_.Get(kHist);
    void* prefix   = bufs_.Get(kPrefix);
    void* value    = bufs_.Get(kValue);

    // Zero out target buffers
    hipMemsetAsync(prefix, 0, beam_count * sizeof(unsigned int), stream());
    hipMemsetAsync(value,  0, beam_count * sizeof(unsigned int), stream());

    hipFunction_t hist_kernel = kernel("histogram_median_pass");
    hipFunction_t bucket_kernel = kernel("find_median_bucket");

    // 4 passes: byte 0 (MSB) through byte 3 (LSB)
    for (unsigned int pass = 0; pass < 4; ++pass) {
      // Clear histogram bins
      hipMemsetAsync(hist_buf, 0,
                     beam_count * 256 * sizeof(unsigned int), stream());

      // Histogram kernel: 2D grid (blocks_per_beam × beam_count)
      void* hist_args[] = { &data_ptr, &hist_buf, &np, &bc, &pass, &value };
      hipError_t err = hipModuleLaunchKernel(
          hist_kernel,
          blocks_per_beam, bc, 1,
          kBlockSize, 1, 1,
          0, stream(),
          hist_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramOp histogram pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }

      // Find median bucket: 1 block per beam, 1 thread
      void* bucket_args[] = { &hist_buf, &prefix, &value, &median_rank, &pass };
      err = hipModuleLaunchKernel(
          bucket_kernel,
          bc, 1, 1, 1, 1, 1,
          0, stream(),
          bucket_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramOp bucket pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }
    }

    // Convert uint32 → float on host (beam_count is small, typically ≤256)
    ConvertResultToFloat(beam_count);
  }

protected:
  void OnRelease() override { bufs_.ReleaseAll(); }

private:
  static constexpr unsigned int kBlockSize = 256;

  enum Buf { kHist, kPrefix, kValue, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;

  void AllocatePrivateBuffers(size_t beam_count) {
    bufs_.Require(kHist,   beam_count * 256 * sizeof(unsigned int));
    bufs_.Require(kPrefix, beam_count * sizeof(unsigned int));
    bufs_.Require(kValue,  beam_count * sizeof(unsigned int));

    ctx_->RequireShared(shared_buf::kMediansCompact,
                        beam_count * sizeof(float));
  }

  void ConvertResultToFloat(size_t beam_count) {
    std::vector<unsigned int> target_host(beam_count);
    hipError_t err = hipMemcpyDtoH(target_host.data(), bufs_.Get(kValue),
                                    beam_count * sizeof(unsigned int));
    if (err != hipSuccess) {
      throw std::runtime_error("MedianHistogramOp: DtoH target_value failed");
    }

    std::vector<float> medians_host(beam_count);
    for (size_t b = 0; b < beam_count; ++b) {
      unsigned int u = target_host[b] ^ 0x80000000u;
      float val;
      std::memcpy(&val, &u, sizeof(float));
      medians_host[b] = val;
    }

    err = hipMemcpyHtoD(
        ctx_->GetShared(shared_buf::kMediansCompact),
        medians_host.data(),
        beam_count * sizeof(float));
    if (err != hipSuccess) {
      throw std::runtime_error("MedianHistogramOp: HtoD medians failed");
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
