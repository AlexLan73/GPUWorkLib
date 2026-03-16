#pragma once

/**
 * @file median_histogram_complex_op.hpp
 * @brief MedianHistogramComplexOp — exact median via histogram on complex input
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from StatisticsProcessor::ExecuteHistogramMedian() with is_complex=true.
 *
 * Identical to MedianHistogramOp but uses histogram_median_pass_complex kernel
 * which computes |z| on-the-fly from complex input (no separate magnitudes step).
 *
 * Kernels: histogram_median_pass_complex, find_median_bucket
 * Private buffers: BufferSet<3> — hist_buf, target_prefix, target_value
 * Shared buffers: reads kInput, writes kMediansCompact
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

class MedianHistogramComplexOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MedianHistogramComplex"; }

  /**
   * @brief Execute histogram-based median on complex input
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Reads kInput (complex<float>), writes kMediansCompact (float[beam_count]).
   * Computes |z| on-the-fly inside the histogram kernel.
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count);

    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int median_rank = np / 2;

    unsigned int blocks_per_beam = std::min(
        static_cast<unsigned int>((n_point + kBlockSize - 1) / kBlockSize),
        1024u);

    void* data_ptr = ctx_->GetShared(shared_buf::kInput);
    void* hist_buf = bufs_.Get(kHist);
    void* prefix   = bufs_.Get(kPrefix);
    void* value    = bufs_.Get(kValue);

    // Zero out target buffers
    hipMemsetAsync(prefix, 0, beam_count * sizeof(unsigned int), stream());
    hipMemsetAsync(value,  0, beam_count * sizeof(unsigned int), stream());

    hipFunction_t hist_kernel = kernel("histogram_median_pass_complex");
    hipFunction_t bucket_kernel = kernel("find_median_bucket");

    for (unsigned int pass = 0; pass < 4; ++pass) {
      hipMemsetAsync(hist_buf, 0,
                     beam_count * 256 * sizeof(unsigned int), stream());

      void* hist_args[] = { &data_ptr, &hist_buf, &np, &bc, &pass, &value };
      hipError_t err = hipModuleLaunchKernel(
          hist_kernel,
          blocks_per_beam, bc, 1,
          kBlockSize, 1, 1,
          0, stream(),
          hist_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramComplexOp histogram pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }

      void* bucket_args[] = { &hist_buf, &prefix, &value, &median_rank, &pass };
      err = hipModuleLaunchKernel(
          bucket_kernel,
          bc, 1, 1, 1, 1, 1,
          0, stream(),
          bucket_args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("MedianHistogramComplexOp bucket pass " +
                                  std::to_string(pass) + ": " +
                                  hipGetErrorString(err));
      }
    }

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
      throw std::runtime_error("MedianHistogramComplexOp: DtoH target_value failed");
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
      throw std::runtime_error("MedianHistogramComplexOp: HtoD medians failed");
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
