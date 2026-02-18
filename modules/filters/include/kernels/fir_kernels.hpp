#pragma once

/**
 * @file fir_kernels.hpp
 * @brief OpenCL kernel sources for FIR filter
 *
 * Direct-form convolution: each work-item computes one output sample.
 * 2D NDRange: (channels, points) for multi-channel parallel processing.
 *
 * Coefficients stored in __constant memory (fast broadcast, <=64KB).
 * Complex signal: float2 (x=Re, y=Im), real coefficients.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

namespace filters {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// FIR Direct-Form Convolution (complex float2 signal, real coefficients)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Get FIR kernel source (OpenCL)
 *
 * Kernel: fir_filter_cf32
 *   - NDRange 2D: global_size = {channels, points_padded}
 *   - Each work-item: one output sample y[ch][n]
 *   - y[ch][n] = sum_{k=0}^{num_taps-1} h[k] * x[ch][n-k]
 *   - Boundary: x[ch][n-k] = 0 if (n-k) < 0
 *
 * @return Pointer to kernel source string
 */
inline const char* GetFirDirectSource_opencl() {
  return R"CL(

// ─────────────────────────────────────────────────────────────────────────
// FIR Direct-Form Convolution
//
// Layout: [channels * points] float2 (channel-sequential)
//   input[ch * points + n] = complex sample (ch, n)
//
// Coefficients: real float[] in __constant memory
//   y.re[n] = sum(h[k] * x.re[n-k])
//   y.im[n] = sum(h[k] * x.im[n-k])
// ─────────────────────────────────────────────────────────────────────────

__kernel void fir_filter_cf32(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  coeffs,
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);   // channel index
    const uint n  = get_global_id(1);   // sample index

    if (n >= points) return;

    const uint base = ch * points;

    float2 acc = (float2)(0.0f, 0.0f);

    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2 x = input[base + (uint)idx];
            float  h = coeffs[k];
            acc.x += h * x.x;
            acc.y += h * x.y;
        }
    }

    output[base + n] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// FIR with __global coefficients (for num_taps > 16000)
// Same algorithm, but coefficients in global memory
// ─────────────────────────────────────────────────────────────────────────

__kernel void fir_filter_cf32_global(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __global const float*  coeffs,
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);
    const uint n  = get_global_id(1);

    if (n >= points) return;

    const uint base = ch * points;

    float2 acc = (float2)(0.0f, 0.0f);

    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2 x = input[base + (uint)idx];
            float  h = coeffs[k];
            acc.x += h * x.x;
            acc.y += h * x.y;
        }
    }

    output[base + n] = acc;
}

)CL";
}

} // namespace kernels
} // namespace filters
