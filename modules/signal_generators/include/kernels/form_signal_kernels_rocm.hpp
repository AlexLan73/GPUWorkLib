#pragma once

/**
 * @file form_signal_kernels_rocm.hpp
 * @brief HIP kernel source for FormSignalGenerator (ROCm port)
 *
 * Port of form_signal.cl + prng.cl to HIP.
 * Embedded as raw string for hiprtc compilation.
 *
 * X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
 *   + an*norm*(randn_re + j*randn_im)
 * X = 0  if t < 0 or t > ti - dt
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

namespace signal_gen {
namespace kernels {

inline const char* GetFormSignalSource_rocm() {
  return R"HIP(

// ════════════════════════════════════════════════════════════════════════════
// float2_t — complex float (hiprtc has no built-in float2)
// ════════════════════════════════════════════════════════════════════════════

struct float2_t {
    float x;
    float y;
};

// ════════════════════════════════════════════════════════════════════════════
// Philox-2x32-10 PRNG (counter-based)
// ════════════════════════════════════════════════════════════════════════════

struct uint2_t {
    unsigned int x;
    unsigned int y;
};

__device__ unsigned int mulhi32(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
}

__device__ uint2_t philox2x32_round(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_M = 0xD2511F53u;
    unsigned int hi = mulhi32(ctr.x, PHILOX_M);
    unsigned int lo = ctr.x * PHILOX_M;
    uint2_t result;
    result.x = hi ^ key ^ ctr.y;
    result.y = lo;
    return result;
}

__device__ uint2_t philox2x32_10(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_BUMP = 0x9E3779B9u;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key);
    return ctr;
}

__device__ float philox_uniform(unsigned int id, unsigned int seed) {
    uint2_t ctr;
    ctr.x = id;
    ctr.y = seed;
    uint2_t rnd = philox2x32_10(ctr, 0xAB12CD34u);
    return (float)(rnd.x) / 4294967296.0f;
}

// ════════════════════════════════════════════════════════════════════════════
// generate_form_signal kernel
// ════════════════════════════════════════════════════════════════════════════

extern "C" __global__ __launch_bounds__(256)
void generate_form_signal(
    float2_t* __restrict__ output,
    unsigned int antennas,
    unsigned int points,
    float dt,
    float ti,
    float f0,
    float amplitude,
    float noise_amplitude,
    float phase_offset,
    float fdev,
    float norm_val,
    float tau_base,
    float tau_step,
    float tau_min,
    float tau_max,
    unsigned int tau_seed,
    unsigned int noise_seed,
    unsigned int tau_mode)
{
    // 2D grid: blockIdx.x = sample blocks, blockIdx.y = antenna
    const unsigned int sample_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int antenna_id = blockIdx.y;
    if (sample_id >= points || antenna_id >= antennas) return;

    const unsigned int gid = antenna_id * points + sample_id;

    // Tau per-channel
    float tau;
    if (tau_mode == 0u) {
        // FIXED
        tau = tau_base;
    } else if (tau_mode == 1u) {
        // LINEAR
        tau = tau_base + (float)antenna_id * tau_step;
    } else {
        // RANDOM
        float u = philox_uniform(antenna_id, tau_seed);
        tau = tau_min + u * (tau_max - tau_min);
    }

    // Time
    float t = (float)sample_id * dt + tau;

    // Window: X=0 if outside [0, ti-dt]
    if (t < 0.0f || t > ti - dt) {
        float2_t zero;
        zero.x = 0.0f;
        zero.y = 0.0f;
        output[gid] = zero;
        return;
    }

    // Signal phase: 2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi
    const float PI_F = 3.14159265358979f;
    float t_centered = t - ti * 0.5f;
    float phase = 2.0f * PI_F * f0 * t
                + PI_F * fdev / ti * (t_centered * t_centered)
                + phase_offset;

    float sin_phase, cos_phase;
    __sincosf(phase, &sin_phase, &cos_phase);
    float sig_re = amplitude * norm_val * cos_phase;
    float sig_im = amplitude * norm_val * sin_phase;

    // Noise (Philox + Box-Muller)
    float noise_re = 0.0f;
    float noise_im = 0.0f;

    if (noise_amplitude > 0.0f) {
        uint2_t n_ctr;
        n_ctr.x = gid;
        n_ctr.y = noise_seed;
        uint2_t n_rnd = philox2x32_10(n_ctr, 0xCD9E8D57u);

        float u1 = (float)(n_rnd.x) / 4294967296.0f + 1e-10f;
        float u2 = (float)(n_rnd.y) / 4294967296.0f;

        float r = __fsqrt_rn(-2.0f * __logf(u1));
        float theta = 2.0f * PI_F * u2;

        float sin_theta, cos_theta;
        __sincosf(theta, &sin_theta, &cos_theta);
        noise_re = noise_amplitude * norm_val * r * cos_theta;
        noise_im = noise_amplitude * norm_val * r * sin_theta;
    }

    float2_t result;
    result.x = sig_re + noise_re;
    result.y = sig_im + noise_im;
    output[gid] = result;
}

)HIP";
}

}  // namespace kernels
}  // namespace signal_gen
