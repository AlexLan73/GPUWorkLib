/**
 * @file lfm_conjugate.cl
 * @brief Conjugate LFM reference signal: conj(s_tx) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
 *
 * Used as reference for dechirp processing:
 *   s_dc = s_rx(t) * conj(s_tx(t))
 *   Result is a tone at f_beat = mu * tau
 *
 * Оптимизации:
 *   - sincos() вместо отдельных cos/sin
 *   - restrict на указателях
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

__kernel void generate_lfm_conjugate(
    __global float2* restrict output,
    const uint        num_samples,
    const float       sample_rate,
    const float       f_start,
    const float       chirp_rate,
    const float       amplitude)
{
    const uint n = get_global_id(0);
    if (n >= num_samples) return;

    float t = (float)n / sample_rate;

    // conj(LFM) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
    float phase = -(M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t);

    float cos_val;
    float sin_val = sincos(phase, &cos_val);

    output[n] = (float2)(amplitude * cos_val, amplitude * sin_val);
}
