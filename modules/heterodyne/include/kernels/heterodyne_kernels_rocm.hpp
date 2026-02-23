#pragma once

/**
 * @file heterodyne_kernels_rocm.hpp
 * @brief HIP kernel sources for heterodyne dechirp processing
 *
 * Contains:
 * - dechirp_multiply: s_dc = conj(s_rx * s_ref) on GPU
 * - dechirp_correct:  frequency correction exp(j * phase_step * n)
 *
 * Port of dechirp_multiply.cl and dechirp_correct.cl (OpenCL -> HIP).
 * Embedded as raw strings for hiprtc runtime compilation.
 *
 * Optimizations preserved from OpenCL version:
 *   OPT-5: 1D kernel launch (gid = ant*N + n)
 *   OPT-6: phase_step precomputed on CPU (no division in kernel)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

namespace drv_gpu_lib {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// Dechirp multiply kernel source
// ════════════════════════════════════════════════════════════════════════════

inline const char* GetDechirpMultiplySource_rocm() {
  return R"HIP(

// ════════════════════════════════════════════════════════════════════════════
// float2_t — complex number (hiprtc has no built-in float2)
// ════════════════════════════════════════════════════════════════════════════

struct float2_t {
    float x;
    float y;
};

// ════════════════════════════════════════════════════════════════════════════
// dechirp_multiply: output[gid] = conj(rx[gid] * ref[n])
//
// Each thread processes 1 sample of 1 antenna.
// OPT-5: 1D global: total = num_antennas * num_samples
//   ant = gid / num_samples, n = gid % num_samples
//
// ref = conj(s_tx) from LfmConjugateGenerator (broadcast to all antennas).
// conj(rx * ref) = conj(s_rx * conj(s_tx)) = conj(s_rx) * s_tx
// This produces POSITIVE beat frequency: f_beat = +mu*tau
//
// Math: conj((a+jb)(c+jd)) = (ac-bd) - j(ad+bc)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void dechirp_multiply(
    const float2_t* __restrict__ rx,       // [num_antennas * num_samples]
    const float2_t* __restrict__ ref,      // [num_samples] broadcast
          float2_t* __restrict__ dc_out,   // [num_antennas * num_samples]
    const int num_samples,
    const int total_elements)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    const int n = gid % num_samples;       // sample index

    const float2_t rx_v = rx[gid];
    const float2_t re_v = ref[n];          // broadcast: one ref for all antennas

    // conj(rx * ref): gives positive beat frequency
    float2_t out;
    out.x =  rx_v.x * re_v.x - rx_v.y * re_v.y;   // Re:  a*c - b*d
    out.y = -rx_v.x * re_v.y - rx_v.y * re_v.x;    // Im: -(a*d + b*c)
    dc_out[gid] = out;
}

)HIP";
}

// ════════════════════════════════════════════════════════════════════════════
// Dechirp correct kernel source
// ════════════════════════════════════════════════════════════════════════════

inline const char* GetDechirpCorrectSource_rocm() {
  return R"HIP(

// ════════════════════════════════════════════════════════════════════════════
// float2_t — complex number
// ════════════════════════════════════════════════════════════════════════════

struct float2_t {
    float x;
    float y;
};

// ════════════════════════════════════════════════════════════════════════════
// dechirp_correct: output = input * exp(j * phase_step * n)
//
// After dechirp, signal has a tone at f_beat.
// Multiplying by exp(j * phase_step * n) shifts spectrum to DC (0 Hz).
//
// OPT-5: 1D global: total = num_antennas * num_samples
// OPT-6: phase_step[] precomputed on CPU = -2*pi*f_beat/sample_rate
//         kernel uses phase = phase_step[ant] * n (no division)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void dechirp_correct(
    const float2_t* __restrict__ dc_in,       // [num_antennas * num_samples]
          float2_t* __restrict__ corrected,   // [num_antennas * num_samples]
    const float*    __restrict__ phase_step,  // [-2*pi*f_beat/fs] per antenna
    const int num_samples,
    const int total_elements)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    const int ant = gid / num_samples;
    const int n   = gid % num_samples;

    // OPT-6: phase = phase_step[ant] * n (no division in kernel)
    const float phase = phase_step[ant] * (float)n;

    const float cos_p = cosf(phase);
    const float sin_p = sinf(phase);

    // Complex multiply: corrected = dc_in * exp(j*phase)
    const float2_t in = dc_in[gid];
    float2_t out;
    out.x = in.x * cos_p - in.y * sin_p;
    out.y = in.y * cos_p + in.x * sin_p;
    corrected[gid] = out;
}

)HIP";
}

}  // namespace kernels
}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
