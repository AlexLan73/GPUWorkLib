#pragma once

/**
 * @file dechirp_window_op.hpp
 * @brief DechirpWindowOp — multiply input by conjugate LFM reference + Hamming window
 *
 * Ref03 Unified Architecture: Layer 5 (Concrete Operation).
 *
 * Manages:
 *   d_window_  — Hamming window (float[n_samples]), filled once in OnInitialize
 *
 * Reads shared buffers:
 *   shared_buf::kInput  — raw IQ   [n_ant * n_samples * complex<float>]
 *   shared_buf::kRef    — LFM ref  [n_samples * complex<float>]
 *
 * Writes shared buffer:
 *   shared_buf::kDechirped — dechirped [n_ant * nfft_r * complex<float>] (with zero-pad)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "services/buffer_set.hpp"
#include "interface/gpu_context.hpp"
#include "range_angle_types.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace range_angle {

class DechirpWindowOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "DechirpWindow"; }

  /**
   * @brief Allocate Hamming window on GPU and fill via hamming_window_kernel.
   * @param n_samples Number of samples (window length)
   *
   * Call once after GpuKernelOp::Initialize(ctx) is done.
   * Re-allocates if n_samples changes.
   */
  void InitWindow(uint32_t n_samples) {
    n_samples_ = n_samples;

    // Allocate / re-allocate window buffer
    bufs_.Require(kWindow, static_cast<size_t>(n_samples) * sizeof(float));
    float* d_win = static_cast<float*>(bufs_.Get(kWindow));

    // Launch hamming_window_kernel
    // Grid: (ceil(n_samples/256), 1, 1)  Block: (256, 1, 1)
    uint32_t grid_x = (n_samples + 255u) / 256u;
    void* args[] = { &d_win, &n_samples };

    hipError_t err = hipModuleLaunchKernel(
        kernel("hamming_window_kernel"),
        grid_x, 1, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("DechirpWindowOp::InitWindow hamming_window_kernel: ") +
          hipGetErrorString(err));
    }
  }

  /**
   * @brief Execute dechirp + Hamming window + zero-pad on all antennas.
   * @param n_ant     Total antennas (n_ant_az * n_ant_el)
   * @param n_samples Samples per antenna
   * @param nfft_r    Range FFT size (output width, >= n_samples for zero-pad)
   */
  void Execute(uint32_t n_ant, uint32_t n_samples, uint32_t nfft_r) {
    void* d_rx  = ctx_->GetShared(shared_buf::kInput);
    void* d_ref = ctx_->GetShared(shared_buf::kRef);
    void* d_out = ctx_->GetShared(shared_buf::kDechirped);
    void* d_win = bufs_.Get(kWindow);

    if (!d_rx || !d_ref || !d_out || !d_win) {
      throw std::runtime_error("DechirpWindowOp::Execute: null shared buffer");
    }

    // Grid: (ceil(nfft_r/256), n_ant, 1)  Block: (256, 1, 1)
    uint32_t grid_x = (nfft_r + 255u) / 256u;

    void* args[] = { &d_rx, &d_ref, &d_win, &d_out,
                     &n_ant, &n_samples, &nfft_r };

    hipError_t err = hipModuleLaunchKernel(
        kernel("dechirp_window_kernel"),
        grid_x, n_ant, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error(
          std::string("DechirpWindowOp::Execute dechirp_window_kernel: ") +
          hipGetErrorString(err));
    }
  }

protected:
  void OnRelease() override {
    bufs_.ReleaseAll();
    n_samples_ = 0;
  }

private:
  enum Buf { kWindow, kBufCount };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;
  uint32_t n_samples_ = 0;
};

}  // namespace range_angle

#endif  // ENABLE_ROCM
