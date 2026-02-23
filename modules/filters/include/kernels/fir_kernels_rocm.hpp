#pragma once

/**
 * @file fir_kernels_rocm.hpp
 * @brief HIP kernel sources for FIR filter (ROCm)
 *
 * Port of fir_kernels.hpp (OpenCL) to HIP/ROCm.
 * Uses hiprtc for runtime compilation.
 *
 * Kernel: fir_filter_cf32
 *   - 1D grid: total_points = channels * points
 *   - Each thread: one output sample y[ch][n]
 *   - y[ch][n] = sum_{k=0}^{num_taps-1} h[k] * x[ch][n-k]
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

namespace filters {
namespace kernels {

/**
 * @brief Get FIR kernel source (HIP/ROCm)
 *
 * Single kernel handles all coefficient sizes (no __constant vs __global split).
 * HIP L1/L2 cache provides similar performance to OpenCL __constant memory.
 *
 * @return Pointer to kernel source string
 */
inline const char* GetFirDirectSource_rocm() {
  return R"HIP(

// ─────────────────────────────────────────────────────────────────────────
// Custom float2 for hiprtc compatibility
// ─────────────────────────────────────────────────────────────────────────

struct float2_t {
    float x;
    float y;
};

// ─────────────────────────────────────────────────────────────────────────
// FIR Direct-Form Convolution (HIP)
//
// Layout: [channels * points] float2_t (channel-sequential)
//   input[ch * points + n] = complex sample (ch, n)
//
// 1D grid: gid = ch * points + n
// ─────────────────────────────────────────────────────────────────────────

extern "C" __global__ void fir_filter_cf32(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    const float* __restrict__ coeffs,
    unsigned int num_taps,
    unsigned int channels,
    unsigned int points)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = channels * points;
    if (gid >= total) return;

    unsigned int ch = gid / points;
    unsigned int n  = gid % points;
    unsigned int base = ch * points;

    float acc_x = 0.0f;
    float acc_y = 0.0f;

    for (unsigned int k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2_t x = input[base + (unsigned int)idx];
            float h = coeffs[k];
            acc_x += h * x.x;
            acc_y += h * x.y;
        }
    }

    float2_t result;
    result.x = acc_x;
    result.y = acc_y;
    output[base + n] = result;
}

)HIP";
}

}  // namespace kernels
}  // namespace filters
