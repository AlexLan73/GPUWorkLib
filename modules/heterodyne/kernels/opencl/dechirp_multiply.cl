/**
 * @file dechirp_multiply.cl
 * @brief Dechirp: output[ant][n] = conj(rx[ant][n] * ref[n])
 *
 * Each work-item processes 1 sample of 1 antenna.
 * OPT-5: 1D global: (num_antennas * num_samples)
 *   ant = gid / num_samples, n = gid % num_samples
 *
 * ref = conj(s_tx) from LfmConjugateGenerator.
 * conj(rx * ref) = conj(s_rx * conj(s_tx)) = conj(s_rx) * s_tx
 * This produces POSITIVE beat frequency: f_beat = +mu*tau
 * so that FFT peak is in the lower half and correction exp(-j*2*pi*f*t) works.
 *
 * Math: conj((a+jb)(c+jd)) = conj((ac-bd)+j(ad+bc)) = (ac-bd) - j(ad+bc)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */
__kernel void dechirp_multiply(
    __global const float2* rx,          // [num_antennas * num_samples]
    __global const float2* ref,         // [num_samples] - conj(s_tx), broadcast
    __global       float2* dc_out,      // [num_antennas * num_samples]
    const int num_samples
) {
    int gid = get_global_id(0);
    int n   = gid % num_samples;       // sample index

    float2 rx_v = rx[gid];
    float2 re_v = ref[n];              // broadcast: one ref for all antennas

    // conj(rx * ref): gives positive beat frequency
    dc_out[gid].x =  rx_v.x * re_v.x - rx_v.y * re_v.y;   // Re:  a*c - b*d
    dc_out[gid].y = -rx_v.x * re_v.y - rx_v.y * re_v.x;   // Im: -(a*d + b*c)
}
