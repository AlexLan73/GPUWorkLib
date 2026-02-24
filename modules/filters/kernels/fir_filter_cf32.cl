

// ─────────────────────────────────────────────────────────────────────────
// FIR Direct-Form Convolution
//
// Layout: [channels * points] float2 (channel-sequential)
//   input[ch * points + n] = complex sample (ch, n)
//
// Coefficients: real float[] in __constant memory
//   y.re[n] = sum(h[k] * x.re[n-k])
//   y.im[n] = sum(h[k] * x.im[n-k])
// ─────────────────────────────────────────────────────────────────────────

__kernel void fir_filter_cf32(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  coeffs,
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);   // channel index
    const uint n  = get_global_id(1);   // sample index

    if (n >= points) return;

    const uint base = ch * points;

    float2 acc = (float2)(0.0f, 0.0f);

    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2 x = input[base + (uint)idx];
            float  h = coeffs[k];
            acc.x += h * x.x;
            acc.y += h * x.y;
        }
    }

    output[base + n] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// FIR with __global coefficients (for num_taps > 16000)
// Same algorithm, but coefficients in global memory
// ─────────────────────────────────────────────────────────────────────────

__kernel void fir_filter_cf32_global(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __global const float*  coeffs,
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);
    const uint n  = get_global_id(1);

    if (n >= points) return;

    const uint base = ch * points;

    float2 acc = (float2)(0.0f, 0.0f);

    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2 x = input[base + (uint)idx];
            float  h = coeffs[k];
            acc.x += h * x.x;
            acc.y += h * x.y;
        }
    }

    output[base + n] = acc;
}

