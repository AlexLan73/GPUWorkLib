#pragma once

/**
 * @file pad_data_op.hpp
 * @brief PadDataOp — zero-padding kernel for FFT input
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from FFTProcessorROCm::ExecutePadKernel().
 *
 * Pipeline: memset(fft_input, 0) + pad_data kernel (input → fft_input)
 * Kernel: pad_data — copies n_point samples per beam, rest is already zeroed.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

class PadDataOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "PadData"; }

  /**
   * @brief Execute zero-padding: input_buf → fft_input_buf
   * @param input_buf Device pointer to raw input [beam_count × n_point × complex]
   * @param fft_input_buf Device pointer to padded output [beam_count × nFFT × complex]
   * @param beam_count Number of beams in this batch
   * @param n_point Samples per beam (before padding)
   * @param nFFT FFT size (after padding, power of 2)
   */
  void Execute(void* input_buf, void* fft_input_buf,
               size_t beam_count, uint32_t n_point, uint32_t nFFT) {
    // Zero entire padded buffer first (async)
    hipError_t err = hipMemsetAsync(
        fft_input_buf, 0,
        beam_count * nFFT * sizeof(float) * 2,  // complex<float> = 2 floats
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error("PadDataOp memset: " +
                                std::string(hipGetErrorString(err)));
    }

    // 2D grid: X covers n_point samples, Y = beam_id
    unsigned int block_x = 256;
    unsigned int grid_x = static_cast<unsigned int>((n_point + block_x - 1) / block_x);
    unsigned int grid_y = static_cast<unsigned int>(beam_count);

    unsigned int np = n_point;
    unsigned int nf = nFFT;

    void* args[] = { &input_buf, &fft_input_buf, &np, &nf };

    err = hipModuleLaunchKernel(
        kernel("pad_data"),
        grid_x, grid_y, 1,
        block_x, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("PadDataOp kernel: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
