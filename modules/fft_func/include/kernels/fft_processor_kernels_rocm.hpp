#pragma once

/**
 * @file fft_processor_kernels_rocm.hpp
 * @brief HIP kernel sources for FFTProcessorROCm
 *
 * Contains:
 * - Pad kernel (n_point -> nFFT zero-padding for batch FFT)
 * - Post-processing kernel (complex -> magnitude + phase)
 *
 * Kernels are compiled at runtime via hiprtc.
 * Uses custom float2_t struct to avoid hiprtc built-in type issues.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

namespace fft_processor {
namespace kernels {

/**
 * @brief Combined HIP kernel source: pad_data + complex_to_mag_phase
 *
 * pad_data:
 *   Pads input data from n_point to nFFT with zeros for batch FFT.
 *   Each thread handles one complex element in the output.
 *
 * complex_to_mag_phase:
 *   Converts complex FFT output to magnitude and phase.
 *   Each thread processes one complex element.
 */
inline const char* GetHIPKernelSource() {
    return R"HIP(

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct float2_t {
    float x;
    float y;
};

// 2D grid: blockIdx.y == beam_id — eliminates int div/mod per thread.
// Zeros in fft_input handled by hipMemsetAsync before launch — no divergent else-branch.
// __launch_bounds__(256) improves occupancy.
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int beam_id = blockIdx.y;
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;  // Only copy valid points; rest already zero

    output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
}

// Interleaved {mag, phase} output — caller does single DtoH instead of two transfers.
__launch_bounds__(256)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float2_t* __restrict__ mag_phase,   // interleaved: .x=mag, .y=phase
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = fft_output[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

)HIP";
}

}  // namespace kernels
}  // namespace fft_processor

#endif  // ENABLE_ROCM
