

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
    __constant     float*  coeffs,  // __constant: кешируется, до 64KB (~16000 тапов)
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);   // channel index
    const uint n  = get_global_id(1);   // sample index

    if (n >= points) return;

    const uint base = ch * points;

    float2 acc = (float2)(0.0f, 0.0f);

    // Branch-free inner loop: limit k to valid range [0, min(num_taps, n+1))
    const uint k_max = min(num_taps, n + 1u);
    for (uint k = 0; k < k_max; k++) {
        float2 x = input[base + n - k];
        float  h = coeffs[k];
        acc.x += h * x.x;
        acc.y += h * x.y;
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

    const uint k_max = min(num_taps, n + 1u);
    for (uint k = 0; k < k_max; k++) {
        float2 x = input[base + n - k];
        float  h = coeffs[k];
        acc.x += h * x.x;
        acc.y += h * x.y;
    }

    output[base + n] = acc;
}

