

// ─────────────────────────────────────────────────────────────────────────
// IIR Biquad Cascade - Direct Form II Transposed
//
// One work-item = one channel (all samples, all sections)
// Complex signal (float2), real biquad coefficients
//
// SOS matrix layout: sos[section * 5 + {0:b0, 1:b1, 2:b2, 3:a1, 4:a2}]
//
// Direct Form II Transposed:
//   y[n] = b0*x[n] + w1
//   w1   = b1*x[n] - a1*y[n] + w2
//   w2   = b2*x[n] - a2*y[n]
// ─────────────────────────────────────────────────────────────────────────

__kernel void iir_biquad_cascade_cf32(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  sos_matrix,
    const uint num_sections,
    const uint points)
{
    const uint ch   = get_global_id(0);
    const uint base = ch * points;

    for (uint sec = 0; sec < num_sections; sec++) {

        // Load biquad coefficients for this section
        float b0 = sos_matrix[sec * 5u + 0u];
        float b1 = sos_matrix[sec * 5u + 1u];
        float b2 = sos_matrix[sec * 5u + 2u];
        float a1 = sos_matrix[sec * 5u + 3u];
        float a2 = sos_matrix[sec * 5u + 4u];

        // State (Direct Form II Transposed)
        float2 w1 = (float2)(0.0f, 0.0f);
        float2 w2 = (float2)(0.0f, 0.0f);

        // Source pointer: first section reads input, subsequent read output (in-place cascade).
        // Extracted outside points-loop to eliminate per-sample branch.
        __global const float2* src = (sec == 0u)
            ? (__global const float2*)input
            : (__global const float2*)output;

        for (uint n = 0; n < points; n++) {
            float2 x = src[base + n];

            // DFII Transposed
            float2 y;
            y.x = b0 * x.x + w1.x;
            y.y = b0 * x.y + w1.y;

            w1.x = b1 * x.x - a1 * y.x + w2.x;
            w1.y = b1 * x.y - a1 * y.y + w2.y;

            w2.x = b2 * x.x - a2 * y.x;
            w2.y = b2 * x.y - a2 * y.y;

            output[base + n] = y;
        }
    }
}

