/**
 * @file lfm_analytical_delay.cl
 * @brief LFM signal with per-antenna analytical fractional delay
 *
 * Formula:
 *   t_local = t - tau
 *   phase = pi * chirp_rate * t_local^2 + 2*pi * f_start * t_local
 *   output = amplitude * exp(j * phase)  if t >= tau, else 0
 *
 * No PRNG required.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

// Complex IQ variant
__kernel void generate_lfm_analytical_delay(
    __global float2* output,
    __global const float* delay_us,
    const uint antennas,
    const uint points,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const uint gid = get_global_id(0);
    const uint total = antennas * points;
    if (gid >= total) return;

    const uint antenna_id = gid / points;
    const uint sample_id  = gid % points;

    // Time in seconds
    float t = (float)sample_id / sample_rate;

    // Delay in seconds
    float tau = delay_us[antenna_id] * 1e-6f;

    // Before signal arrives -> zero
    if (t < tau) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    // Local time after delay
    float t_local = t - tau;

    // LFM phase: pi * k * t_local^2 + 2*pi * f_start * t_local
    float phase = M_PI_F * chirp_rate * t_local * t_local
                + 2.0f * M_PI_F * f_start * t_local;

    output[gid] = (float2)(amplitude * cos(phase), amplitude * sin(phase));
}

// Real-only variant (imag = 0)
__kernel void generate_lfm_analytical_delay_real(
    __global float2* output,
    __global const float* delay_us,
    const uint antennas,
    const uint points,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const uint gid = get_global_id(0);
    const uint total = antennas * points;
    if (gid >= total) return;

    const uint antenna_id = gid / points;
    const uint sample_id  = gid % points;

    float t = (float)sample_id / sample_rate;
    float tau = delay_us[antenna_id] * 1e-6f;

    if (t < tau) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    float t_local = t - tau;
    float phase = M_PI_F * chirp_rate * t_local * t_local
                + 2.0f * M_PI_F * f_start * t_local;

    output[gid] = (float2)(amplitude * cos(phase), 0.0f);
}
