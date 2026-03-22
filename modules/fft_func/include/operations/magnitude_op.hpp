#pragma once

/**
 * @file magnitude_op.hpp
 * @brief MagnitudeOp — complex-to-magnitude conversion kernel (no phase)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from ComplexToMagPhaseROCm::ExecuteMagnitudeKernel().
 *
 * Kernel: complex_to_magnitude — magnitude * inv_n, float output.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

class MagnitudeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Magnitude"; }

  /**
   * @brief Execute complex → magnitude conversion (no phase, normalized)
   * @param input Device pointer to complex data [total × complex<float>]
   * @param output Device pointer to float output [total × float]
   * @param total_elements beam_count × n_point
   * @param inv_n Normalization factor (1.0 = no norm, 1/n_point, or custom)
   */
  void Execute(void* input, void* output, size_t total_elements, float inv_n) {
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int block_size = 256;
    unsigned int grid_size = (total + block_size - 1) / block_size;

    void* args[] = { &input, &output, &inv_n, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("complex_to_magnitude"),
        grid_size, 1, 1,
        block_size, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MagnitudeOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
