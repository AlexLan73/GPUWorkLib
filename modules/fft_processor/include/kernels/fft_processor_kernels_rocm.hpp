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

struct float2_t {
    float x;
    float y;
};

extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int beam_count,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = beam_count * nFFT;
    if (gid >= total) return;

    unsigned int beam_id = gid / nFFT;
    unsigned int pos     = gid % nFFT;

    if (pos < n_point && beam_id < beam_count) {
        output[gid] = input[beam_id * n_point + pos];
    } else {
        float2_t zero;
        zero.x = 0.0f;
        zero.y = 0.0f;
        output[gid] = zero;
    }
}

extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float* __restrict__ mag_output,
    float* __restrict__ phase_output,
    unsigned int beam_count,
    unsigned int nFFT)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = beam_count * nFFT;
    if (gid >= total) return;

    float2_t z = fft_output[gid];
    mag_output[gid]   = sqrtf(z.x * z.x + z.y * z.y);
    phase_output[gid]  = atan2f(z.y, z.x);
}

)HIP";
}

}  // namespace kernels
}  // namespace fft_processor

#endif  // ENABLE_ROCM
