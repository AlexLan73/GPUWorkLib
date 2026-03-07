// ============================================================================
// lch_farrow_delay.cl — Fractional delay kernel (Lagrange 48x5) + optional noise
//
// 2D NDRange: dim0=sample_id, dim1=antenna_id
// Algorithm:
//   read_pos = sample_id - delay_us[antenna] * 1e-6 * sample_rate
//   center = floor(read_pos), frac = read_pos - center
//   row = (uint)(frac * 48) % 48
//   output[n] = sum(L[row][k] * input[center-1+k], k=0..4)
// ============================================================================

// Philox-2x32-10: counter-based PRNG
uint2 philox2x32_round(uint2 ctr, uint key) {
    const uint PHILOX_M = 0xD2511F53u;
    uint hi = mul_hi(ctr.x, PHILOX_M);
    uint lo = ctr.x * PHILOX_M;
    return (uint2)(hi ^ key ^ ctr.y, lo);
}

uint2 philox2x32_10(uint2 ctr, uint key) {
    const uint PHILOX_BUMP = 0x9E3779B9u;
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

// ============================================================================
// LCH Farrow: fractional delay (Lagrange 48x5)
// ============================================================================

__kernel void lch_farrow_delay(
    __global const float2* input,
    __global float2* output,
    __constant float* lagrange_matrix,
    __global const float* delay_us,
    const uint antennas,
    const uint points,
    const float sample_rate,
    const float noise_amplitude,
    const float norm_val,
    const uint noise_seed)
{
    // 2D NDRange: dim0=sample, dim1=antenna (eliminates div/mod ~40 cycles/thread)
    const uint antenna_id = get_global_id(1);
    const uint sample_id  = get_global_id(0);
    if (sample_id >= points) return;
    const uint gid = antenna_id * points + sample_id;

    // delay in samples
    float delay_samples = delay_us[antenna_id] * 1e-6f * sample_rate;
    float read_pos = (float)sample_id - delay_samples;

    // Before signal start -> zero
    if (read_pos < 0.0f) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    // center = floor(read_pos), frac = read_pos - center
    int center = (int)floor(read_pos);
    float frac = read_pos - (float)center;
    uint row = ((uint)(frac * 48.0f)) % 48u;

    // 5 Lagrange coefficients
    float L0 = lagrange_matrix[row * 5u + 0u];
    float L1 = lagrange_matrix[row * 5u + 1u];
    float L2 = lagrange_matrix[row * 5u + 2u];
    float L3 = lagrange_matrix[row * 5u + 3u];
    float L4 = lagrange_matrix[row * 5u + 4u];

    // Read 5 input samples around center (center-1 .. center+3)
    uint base = antenna_id * points;

    #define READ_SAMPLE(idx) \
        (((idx) >= 0 && (idx) < (int)points) ? \
         input[base + (uint)(idx)] : (float2)(0.0f, 0.0f))

    float2 s0 = READ_SAMPLE(center - 1);
    float2 s1 = READ_SAMPLE(center);
    float2 s2 = READ_SAMPLE(center + 1);
    float2 s3 = READ_SAMPLE(center + 2);
    float2 s4 = READ_SAMPLE(center + 3);

    #undef READ_SAMPLE

    // 5-point Lagrange interpolation
    float2 result = L0 * s0 + L1 * s1 + L2 * s2 + L3 * s3 + L4 * s4;

    // Optional noise (Philox + Box-Muller)
    if (noise_amplitude > 0.0f) {
        uint2 n_ctr = (uint2)(gid, noise_seed);
        uint2 n_rnd = philox2x32_10(n_ctr, 0xCD9E8D57u);

        float u1 = (float)(n_rnd.x) / 4294967296.0f + 1e-10f;
        float u2 = (float)(n_rnd.y) / 4294967296.0f;

        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * M_PI_F * u2;

        float noise_re = noise_amplitude * norm_val * r * cos(theta);
        float noise_im = noise_amplitude * norm_val * r * sin(theta);

        result += (float2)(noise_re, noise_im);
    }

    output[gid] = result;
}
