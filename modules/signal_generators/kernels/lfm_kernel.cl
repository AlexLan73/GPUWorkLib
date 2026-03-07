/**
 * @file lfm_kernel.cl
 * @brief OpenCL kernel: LFM (chirp) signal generation
 *
 * s(t) = amplitude * exp(j * (pi * chirp_rate * t^2 + 2*pi*f_start*t))
 * For multi-beam: all beams generate the same LFM signal.
 *
 * Оптимизации:
 *   - sincos() вместо отдельных cos/sin
 *   - restrict на указателях
 *   - native_cos для real-only варианта
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

__kernel void generate_lfm(
    __global float2* restrict output,
    const uint beam_count,
    const uint n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const size_t sample_id = get_global_id(0);
    const size_t beam_id   = get_global_id(1);
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const size_t gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;

    // phase = pi*k*t^2 + 2*pi*f_start*t
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    float cos_val;
    float sin_val = sincos(phase, &cos_val);

    output[gid] = (float2)(amplitude * cos_val, amplitude * sin_val);
}

/**
 * @brief LFM real-only (imag = 0)
 */
__kernel void generate_lfm_real(
    __global float2* restrict output,
    const uint beam_count,
    const uint n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const size_t sample_id = get_global_id(0);
    const size_t beam_id   = get_global_id(1);
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const size_t gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    output[gid] = (float2)(amplitude * native_cos(phase), 0.0f);
}
