#pragma once

/**
 * @file transpose_op.hpp
 * @brief TransposeOp — GPU tiled matrix transpose for beam-forming axis reorder
 *
 * Ref03 Unified Architecture: Layer 5 (Concrete Operation).
 *
 * Input:  shared_buf::kDechirped  — [n_ant × n_range_bins] complex<float>
 *         (reuses kDechirped after in-place range FFT)
 * Output: shared_buf::kTransposed — [n_range_bins × n_ant] complex<float>
 *
 * Kernel: transpose_complex_kernel — tiled 32×32 with LDS +1 padding
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "range_angle_types.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace range_angle {

class TransposeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Transpose"; }

  /**
   * @brief Execute matrix transpose [n_ant × n_range_bins] → [n_range_bins × n_ant]
   * @param n_ant        Source rows (number of antennas)
   * @param n_range_bins Source columns (range FFT bins)
   */
  void Execute(uint32_t n_ant, uint32_t n_range_bins) {
    void* d_in  = ctx_->GetShared(shared_buf::kDechirped);   // post-FFT data
    void* d_out = ctx_->GetShared(shared_buf::kTransposed);

    if (!d_in || !d_out) {
      throw std::runtime_error("TransposeOp::Execute: null shared buffer");
    }

    // Grid: (ceil(n_cols/32), ceil(n_rows/32), 1)  Block: (32, 32, 1)
    // n_rows = n_ant, n_cols = n_range_bins
    uint32_t grid_x = (n_range_bins + 31u) / 32u;
    uint32_t grid_y = (n_ant        + 31u) / 32u;

    void* args[] = { &d_in, &d_out, &n_ant, &n_range_bins };

    hipError_t err = hipModuleLaunchKernel(
        kernel("transpose_complex_kernel"),
        grid_x, grid_y, 1,
        32, 32, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("TransposeOp::Execute transpose_complex_kernel: ") +
          hipGetErrorString(err));
    }
  }

protected:
  void OnRelease() override {
    // No private buffers
  }
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
