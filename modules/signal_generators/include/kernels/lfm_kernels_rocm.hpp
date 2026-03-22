#pragma once

/**
 * @file lfm_kernels_rocm.hpp
 * @brief HIP kernel source for LFM generator (ROCm port of lfm_kernel.cl)
 *
 * s(t) = amplitude * exp(j * (pi * chirp_rate * t^2 + 2*pi*f_start*t))
 *
 * Embedded as raw string for hiprtc compilation.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

namespace signal_gen {
namespace kernels {

inline const char* GetLfmSource_rocm() {
  return R"HIP(

struct float2_t {
    float x;
    float y;
};

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__ __launch_bounds__(256)
void generate_lfm(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = amplitude * cos_val;
    out.y = amplitude * sin_val;
    output[gid] = out;
}

extern "C" __global__ __launch_bounds__(256)
void generate_lfm_real(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    float2_t out;
    out.x = amplitude * __cosf(phase);
    out.y = 0.0f;
    output[gid] = out;
}

// Conjugate LFM: s_ref*(t) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
// Used as reference for dechirp: s_dc = s_rx * s_ref*
// Difference from generate_lfm: negative phase sign (conjugate)
extern "C" __global__ __launch_bounds__(256)
void generate_lfm_conjugate(
    float2_t* __restrict__ output,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_point) return;

    const float t = (float)i / sample_rate;
    // Negative phase → conjugate of LFM
    const float phase = -(M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t);

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = cos_val;  // amplitude = 1.0 for reference signal
    out.y = sin_val;
    output[i] = out;
}

)HIP";
}

}  // namespace kernels
}  // namespace signal_gen
