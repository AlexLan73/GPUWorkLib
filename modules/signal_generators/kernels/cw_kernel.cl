/**
 * @file cw_kernel.cl
 * @brief OpenCL kernel: CW (синусоида) генерация
 *
 * s(t) = amplitude * exp(j * (2*pi*freq*t + initial_phase))
 * Для multi-beam: freq_i = base_freq + beam_id * freq_step
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

__kernel void generate_cw(
    __global float2* output,
    const uint beam_count,
    const uint n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const size_t gid = get_global_id(0);
    const size_t beam_id = gid / n_point;
    const size_t sample_id = gid % n_point;

    if (beam_id >= beam_count) return;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    output[gid] = (float2)(amplitude * cos(phase), amplitude * sin(phase));
}

/**
 * @brief Real-only CW: imag = 0
 */
__kernel void generate_cw_real(
    __global float2* output,
    const uint beam_count,
    const uint n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const size_t gid = get_global_id(0);
    const size_t beam_id = gid / n_point;
    const size_t sample_id = gid % n_point;

    if (beam_id >= beam_count) return;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    output[gid] = (float2)(amplitude * cos(phase), 0.0f);
}
