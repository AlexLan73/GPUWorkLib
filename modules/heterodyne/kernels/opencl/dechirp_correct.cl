/**
 * @file dechirp_correct.cl
 * @brief Frequency correction: output = input * exp(j * phase_step * n)
 *
 * After dechirp, signal has a tone at f_beat.
 * Multiplying by exp(j * phase_step * n) shifts spectrum to DC (0 Hz).
 * Result: spectrum peak at 0 Hz (verification).
 *
 * OPT-5: 1D global: (num_antennas * num_samples)
 * OPT-6: phase_step[] precomputed on CPU = -2*pi*f_beat/sample_rate
 *         kernel uses phase = phase_step[ant] * n (no division)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */
__kernel void dechirp_correct(
    __global const float2* dc_in,        // dechirped [num_antennas * num_samples]
    __global       float2* corrected,    // output [num_antennas * num_samples]
    __global const float*  phase_step,   // precomputed: -2*pi*f_beat/fs per antenna
    const int   num_samples
) {
    int gid = get_global_id(0);
    int ant = gid / num_samples;
    int n   = gid % num_samples;

    // OPT-6: phase = phase_step[ant] * n  (no division by sample_rate in kernel)
    float phase = phase_step[ant] * (float)n;

    // native_cos/native_sin — быстрые приближения SFU (~2x быстрее cos/sin)
    float2 corr;
    corr.x = native_cos(phase);
    corr.y = native_sin(phase);

    // Complex multiply
    float2 in = dc_in[gid];
    corrected[gid].x = in.x * corr.x - in.y * corr.y;
    corrected[gid].y = in.y * corr.x + in.x * corr.y;
}
