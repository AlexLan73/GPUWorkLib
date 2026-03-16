#pragma once

/**
 * @file median_radix_sort_op.hpp
 * @brief MedianRadixSortOp — median via rocPRIM segmented radix sort
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from: ExecuteMagnitudesKernel + ExecuteMedianSort + ExecuteExtractMediansKernel.
 *
 * Pipeline: compute_magnitudes → segmented_radix_sort → extract_medians
 * Used when n_point <= kHistogramThreshold (small data — sort is faster).
 *
 * Kernels: compute_magnitudes, extract_medians
 * External: gpu_sort::ExecuteSort (statistics_sort_gpu.hip — not modified)
 * Private buffers: BufferSet<3> — sort_buf, sort_temp_buf, offsets_buf
 * Shared buffers: reads kInput, writes kMagnitudes + kMediansCompact
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "services/buffer_set.hpp"
#include "interface/gpu_context.hpp"
#include "statistics_types.hpp"
#include "statistics_sort_gpu.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace statistics {

class MedianRadixSortOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MedianRadixSort"; }

  /**
   * @brief Execute radix sort median pipeline (complex input)
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Full pipeline: magnitudes → sort → extract_medians.
   * Reads kInput, writes kMediansCompact (float[beam_count]).
   */
  void Execute(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count, n_point);

    size_t total = beam_count * n_point;
    ExecuteMagnitudes(total);
    ExecuteSort(beam_count, n_point, total);
    ExecuteExtractMedians(beam_count, n_point);
  }

  /**
   * @brief Execute radix sort median on pre-computed magnitudes (float input)
   * @param beam_count Number of beams
   * @param n_point Samples per beam
   *
   * Skips magnitudes kernel. Reads kMagnitudes, writes kMediansCompact.
   */
  void ExecuteFloat(size_t beam_count, size_t n_point) {
    AllocatePrivateBuffers(beam_count, n_point);

    size_t total = beam_count * n_point;
    ExecuteSort(beam_count, n_point, total);
    ExecuteExtractMedians(beam_count, n_point);
  }

protected:
  void OnRelease() override {
    bufs_.ReleaseAll();
    sort_temp_size_ = 0;
    current_beams_ = 0;
    current_n_point_ = 0;
  }

private:
  static constexpr unsigned int kBlockSize = 256;

  enum Buf { kSortBuf, kSortTemp, kOffsets, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;

  size_t sort_temp_size_ = 0;
  size_t current_beams_ = 0;
  size_t current_n_point_ = 0;

  void AllocatePrivateBuffers(size_t beam_count, size_t n_point) {
    if (beam_count == current_beams_ && n_point == current_n_point_
        && bufs_.Get(kSortBuf)) {
      return;  // already allocated
    }

    size_t total = beam_count * n_point;

    // Sort output buffer
    bufs_.Require(kSortBuf, total * sizeof(float));

    // Offsets buffer + async upload
    bufs_.Require(kOffsets, (beam_count + 1) * sizeof(unsigned int));
    {
      std::vector<unsigned int> host_offsets(beam_count + 1);
      for (size_t i = 0; i <= beam_count; ++i) {
        host_offsets[i] = static_cast<unsigned int>(i * n_point);
      }
      hipError_t err = hipMemcpyAsync(
          bufs_.Get(kOffsets), host_offsets.data(),
          (beam_count + 1) * sizeof(unsigned int),
          hipMemcpyHostToDevice, stream());
      if (err != hipSuccess) {
        throw std::runtime_error("MedianRadixSortOp: offsets upload failed");
      }
    }

    // Query sort temp size + allocate
    {
      auto* d_offsets = static_cast<const unsigned int*>(bufs_.Get(kOffsets));
      hipError_t err = gpu_sort::QuerySortTempSize(
          sort_temp_size_, d_offsets, d_offsets + 1,
          static_cast<unsigned int>(total),
          static_cast<unsigned int>(beam_count),
          stream());
      if (err != hipSuccess) {
        throw std::runtime_error("MedianRadixSortOp: QuerySortTempSize failed");
      }
      if (sort_temp_size_ > 0) {
        bufs_.Require(kSortTemp, sort_temp_size_);
      }
    }

    // Ensure shared magnitudes buffer exists
    ctx_->RequireShared(shared_buf::kMagnitudes,
                        total * sizeof(float));
    // Ensure shared medians_compact buffer exists
    ctx_->RequireShared(shared_buf::kMediansCompact,
                        beam_count * sizeof(float));

    current_beams_ = beam_count;
    current_n_point_ = n_point;
  }

  void ExecuteMagnitudes(size_t total_elements) {
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int grid = (total + kBlockSize * 2 - 1) / (kBlockSize * 2);

    void* input_buf = ctx_->GetShared(shared_buf::kInput);
    void* mag_buf   = ctx_->GetShared(shared_buf::kMagnitudes);

    void* args[] = { &input_buf, &mag_buf, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("compute_magnitudes"),
        grid, 1, 1, kBlockSize, 1, 1,
        0, stream(), args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp magnitudes: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  void ExecuteSort(size_t beam_count, size_t n_point, size_t total) {
    auto* d_offsets = static_cast<const unsigned int*>(bufs_.Get(kOffsets));
    auto* mag_buf   = static_cast<const float*>(
        ctx_->GetShared(shared_buf::kMagnitudes));
    auto* sort_buf  = static_cast<float*>(bufs_.Get(kSortBuf));

    hipError_t err = gpu_sort::ExecuteSort(
        bufs_.Get(kSortTemp), sort_temp_size_,
        mag_buf, sort_buf,
        d_offsets, d_offsets + 1,
        static_cast<unsigned int>(total),
        static_cast<unsigned int>(beam_count),
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp sort: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  void ExecuteExtractMedians(size_t beam_count, size_t n_point) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = static_cast<unsigned int>(n_point);
    unsigned int grid = (bc + kBlockSize - 1) / kBlockSize;

    void* sort_buf    = bufs_.Get(kSortBuf);
    void* medians_buf = ctx_->GetShared(shared_buf::kMediansCompact);

    void* args[] = { &sort_buf, &medians_buf, &np, &bc };

    hipError_t err = hipModuleLaunchKernel(
        kernel("extract_medians"),
        grid, 1, 1, kBlockSize, 1, 1,
        0, stream(), args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MedianRadixSortOp extract: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
