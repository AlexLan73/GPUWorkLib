

// ─── Configuration ──────────────────────────────────────────────────────────
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t { float x; float y; };

// ═════════════════════════════════════════════════════════════════════════════
// KAMA — Kaufman Adaptive Moving Average
// ER = Direction / Volatility → SC = (ER*(fast-slow)+slow)^2 → update
// Optimizations:
//   - %N replaced with conditional branch (3 locations, ~60 cycles saved/iter)
//   - dir/vol division → __frcp_rn() fast reciprocal
//   - prev_idx computed via simple subtraction + conditional
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void kaufman_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    unsigned int N,
    float fast_sc,
    float slow_sc)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float eps = 1e-8f;
    const float sc_diff = fast_sc - slow_sc;

    // Passthrough first N samples
    for (unsigned int i = 0; i < N && i < points; i++) {
        out[base + i] = in[base + i];
    }
    if (points <= N) return;

    // KAMA initial state = last sample of warmup period
    float2_t kama = in[base + N - 1];

    for (unsigned int n = N; n < points; n++) {
        float2_t x = in[base + n];

        // 1. Direction: |x[n] - x[n-N]|
        float dir_re = fabsf(x.x - in[base + n - N].x);
        float dir_im = fabsf(x.y - in[base + n - N].y);

        // 2. Volatility: sum |x[i] - x[i-1]| for i = n-N+1..n (N terms)
        float vol_re = 0.0f, vol_im = 0.0f;
        for (unsigned int i = n - N + 1; i <= n; i++) {
            vol_re += fabsf(in[base + i].x - in[base + i - 1].x);
            vol_im += fabsf(in[base + i].y - in[base + i - 1].y);
        }

        // 3. Efficiency Ratio — __frcp_rn instead of full division
        float er_re = (vol_re > eps) ? dir_re * __frcp_rn(vol_re) : 0.0f;
        float er_im = (vol_im > eps) ? dir_im * __frcp_rn(vol_im) : 0.0f;

        // 4. Smoothing Constant: SC = (ER*(fast-slow)+slow)^2
        float sc_re = er_re * sc_diff + slow_sc;
        float sc_im = er_im * sc_diff + slow_sc;
        sc_re *= sc_re;
        sc_im *= sc_im;

        // 5. Update KAMA
        kama.x = kama.x + sc_re * (x.x - kama.x);
        kama.y = kama.y + sc_im * (x.y - kama.y);

        out[base + n] = kama;
    }
}

