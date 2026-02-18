/**
 * @file delayed_form_signal.cl
 * @brief Fractional delay kernel: Lagrange 5-point interpolation + noise
 *
 * Requires: prng.cl (prepended at compile time)
 *
 * Algorithm (DelayedFormSignal_Kernel_CORRECT):
 *   delay_us -> delay_samples = delay_us * 1e-6 * sample_rate
 *   read_pos = sample_id - delay_samples   -- position to read from input
 *   center = floor(read_pos)               -- integer center of 5-point window
 *   frac = read_pos - center              -- fractional part [0, 1)
 *   row = uint(frac * 48) % 48             -- matrix row
 *   output[n] = sum(k=0..4) L[row][k] * input[center - 1 + k]
 *   (boundary samples = 0)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

__kernel void apply_fractional_delay(
    __global const float2* input,        // Clean signal (from FormSignalGenerator)
    __global float2* output,             // Result: delayed + noise
    __constant float* lagrange_matrix,   // 48x5 = 240 floats
    __global const float* delay_us,      // Per-antenna delays (us)
    const uint antennas,
    const uint points,
    const float sample_rate,
    const float noise_amplitude,
    const float norm_val,
    const uint noise_seed)
{
    const uint gid = get_global_id(0);
    const uint total = antennas * points;
    if (gid >= total) return;

    const uint antenna_id = gid / points;
    const uint sample_id  = gid % points;

    // Convert delay (us) to samples
    float delay_samples = delay_us[antenna_id] * 1e-6f * sample_rate;

    // read_pos = sample_id - delay_samples (position to read from input)
    float read_pos = (float)sample_id - delay_samples;

    // Before signal arrives -> zero
    if (read_pos < 0.0f) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    // center = floor(read_pos), frac = read_pos - center
    int center = (int)floor(read_pos);
    float frac = read_pos - (float)center;
    uint row = ((uint)(frac * 48.0f)) % 48u;

    // 5 Lagrange coefficients for this row
    float L0 = lagrange_matrix[row * 5u + 0u];
    float L1 = lagrange_matrix[row * 5u + 1u];
    float L2 = lagrange_matrix[row * 5u + 2u];
    float L3 = lagrange_matrix[row * 5u + 3u];
    float L4 = lagrange_matrix[row * 5u + 4u];

    // Read 5 input samples around center (center-1, center, center+1, center+2, center+3)
    uint base = antenna_id * points;

    // Helper: read with zero-padding at boundaries
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

    // Add noise (Philox + Box-Muller) if needed
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
