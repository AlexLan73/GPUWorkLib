/**
 * @file form_signal.cl
 * @brief FormSignal kernel: getX formula with Philox noise
 *
 * Requires: prng.cl (prepended at compile time)
 *
 * X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
 *   + an*norm*(randn_re + j*randn_im)
 * X = 0  if t < 0 or t > ti - dt
 *
 * gid = antenna_id * points + sample_id
 *
 * Оптимизации:
 *   - sincos() вместо отдельных cos/sin (signal + noise)
 *   - native_sqrt / native_log для Box-Muller
 *   - restrict на указателях
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

__kernel void generate_form_signal(
    __global float2* restrict output,
    const uint antennas,
    const uint points,
    const float dt,
    const float ti,
    const float f0,
    const float amplitude,
    const float noise_amplitude,
    const float phase_offset,
    const float fdev,
    const float norm_val,
    const float tau_base,
    const float tau_step,
    const float tau_min,
    const float tau_max,
    const uint tau_seed,
    const uint noise_seed,
    const uint tau_mode)
{
    const uint sample_id  = get_global_id(0);
    const uint antenna_id = get_global_id(1);
    if (sample_id >= points || antenna_id >= antennas) return;

    const uint gid = antenna_id * points + sample_id;

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
    int in_window = (t >= 0.0f && t <= ti - dt) ? 1 : 0;

    if (in_window == 0) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    // Signal phase: 2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi
    float t_centered = t - ti * 0.5f;
    float phase = 2.0f * M_PI_F * f0 * t
                + M_PI_F * fdev / ti * (t_centered * t_centered)
                + phase_offset;

    float cos_phase, sin_phase;
    sin_phase = sincos(phase, &cos_phase);
    float sig_re = amplitude * norm_val * cos_phase;
    float sig_im = amplitude * norm_val * sin_phase;

    // Noise (Philox + Box-Muller)
    float noise_re = 0.0f;
    float noise_im = 0.0f;

    if (noise_amplitude > 0.0f) {
        uint2 n_ctr = (uint2)(gid, noise_seed);
        uint2 n_rnd = philox2x32_10(n_ctr, 0xCD9E8D57u);

        float u1 = (float)(n_rnd.x) / 4294967296.0f + 1e-10f;
        float u2 = (float)(n_rnd.y) / 4294967296.0f;

        float r = native_sqrt(-2.0f * native_log(u1));
        float theta = 2.0f * M_PI_F * u2;

        float cos_theta, sin_theta;
        sin_theta = sincos(theta, &cos_theta);
        noise_re = noise_amplitude * norm_val * r * cos_theta;
        noise_im = noise_amplitude * norm_val * r * sin_theta;
    }

    output[gid] = (float2)(sig_re + noise_re, sig_im + noise_im);
}
