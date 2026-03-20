#!/usr/bin/env python3
"""
Test: KaufmanFilterROCm — GPU KAMA (Kaufman Adaptive MA) vs numpy reference

Filter: KAMA — speed adapts automatically based on Efficiency Ratio (ER):
  ER ≈ 1 (trending signal)  → fast tracking (alpha ≈ fast_sc)
  ER ≈ 0 (noisy signal)    → slow tracking  (alpha ≈ slow_sc)
GPU class: gpuworklib.KaufmanFilterROCm

Tests:
  1. test_kaufman_basic          — random complex signal, GPU vs numpy reference
  2. test_kaufman_multi_channel  — 8-channel, each vs reference
  3. test_kaufman_trend_signal   — linear trend → ER ≈ 1 → KAMA follows closely
  4. test_kaufman_noise_signal   — white noise  → ER ≈ 0 → KAMA barely moves
  5. test_kaufman_adaptive_transition — trend→noise→trend: adaptive behaviour
  6. test_kaufman_channel_independence — 256 channels with unique slopes
  7. test_kaufman_step_demo      — step signal visual demo (CPU-only)
  8. test_kaufman_properties     — is_ready(), get_params()

Algorithm:
  fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
  For n >= er_period (first er_period points: passthrough):
    direction  = |x[n] - x[n - N]|
    volatility = sum(|x[i] - x[i-1]|, i=n-N+1..n)   [rolling update O(1)]
    ER  = direction / volatility  (0 if volatility < eps)
    SC  = (ER * (fast_sc - slow_sc) + slow_sc) ^ 2
    KAMA += SC * (x[n] - KAMA)

Note:
  Tolerance GPU vs numpy: < 1e-4 (float32).
  er_period ≤ 128 (GPU ring-buffer register limit).
  GPU API:
    kauf = gpuworklib.KaufmanFilterROCm(ctx)
    kauf.set_params(er_period=10, fast=2, slow=30)
    kauf.process(data)              # 1D or 2D (channels, points) complex64
    kauf.is_ready()
    kauf.get_params()               # dict: {er_period, fast_period, slow_period}

Usage:
  python Python_test/filters/test_kaufman_rocm.py

Author: Kodo (AI Assistant)
Date: 2026-03-04
"""

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _subdir in ["build/python", "build/python/Release", "build/python/Debug", "build/debian-radeon9070/python"]:
    _p = os.path.join(PROJECT_ROOT, _subdir)
    if os.path.exists(_p):
        sys.path.insert(0, _p)
        break

try:
    import gpuworklib
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found. Skipping GPU tests.")

# ============================================================================
# Parameters
# ============================================================================

POINTS      = 4096
CHANNELS    = 8
ER_PERIOD   = 10    # N — Kaufman default
FAST_PERIOD = 2     # fast EMA period (ER=1)
SLOW_PERIOD = 30    # slow EMA period (ER=0)
ATOL        = 1e-4  # float32 tolerance

# Precomputed SCs for reference
FAST_SC = 2.0 / (FAST_PERIOD + 1)   # = 2/3 ≈ 0.6667
SLOW_SC = 2.0 / (SLOW_PERIOD + 1)   # = 2/31 ≈ 0.0645

# ============================================================================
# Python reference implementation — rolling volatility, float32, matches GPU
# ============================================================================

def _kaufman_1ch(data: np.ndarray,
                 N: int,
                 fast_sc: float,
                 slow_sc: float) -> np.ndarray:
    """KAMA on a single 1D complex64 channel.

    Matches GPU kernel (kaufman_kernel_rocm.cl) exactly:
      - Delta ring buffer: delta[i] = |data[i+1] - data[i]|, N terms (not value ring)
      - kama initialised to data[N-1]  (last warmup sample, matches kama = in[base+N-1])
      - ER = dir / (vol + eps)  (branchless, mirrors __frcp_rn(vol+eps))
      - Sliding vol update AFTER KAMA write, uses NEXT sample in[n+1]
    Re and Im are processed independently (matching kernel float2_t split).
    """
    n      = len(data)
    out    = np.empty(n, dtype=np.complex64)
    fast   = np.float32(fast_sc)
    slow   = np.float32(slow_sc)
    sc_diff = fast - slow
    eps    = np.float32(1e-8)

    # First N points: passthrough
    for i in range(min(N, n)):
        out[i] = data[i]

    if n <= N:
        return out

    # Initialize delta ring buffer: N deltas |data[i+1]-data[i]| for i=0..N-1
    delta_re = np.zeros(N, dtype=np.float32)
    delta_im = np.zeros(N, dtype=np.float32)
    vol_re = np.float32(0.0)
    vol_im = np.float32(0.0)
    for i in range(N):
        dr = abs(np.float32(data[i + 1].real) - np.float32(data[i].real))
        di = abs(np.float32(data[i + 1].imag) - np.float32(data[i].imag))
        delta_re[i] = dr
        delta_im[i] = di
        vol_re += dr
        vol_im += di

    # KAMA initial state = last warmup sample (mirrors GPU: kama = in[base + N - 1])
    kama_re = np.float32(data[N - 1].real)
    kama_im = np.float32(data[N - 1].imag)
    head = 0

    for idx in range(N, n):
        x_re = np.float32(data[idx].real)
        x_im = np.float32(data[idx].imag)

        # 1. Direction: |x[n] - x[n-N]|
        dir_re = abs(x_re - np.float32(data[idx - N].real))
        dir_im = abs(x_im - np.float32(data[idx - N].imag))

        # 2. Efficiency Ratio: branchless, eps prevents div-by-zero
        er_re = dir_re / (vol_re + eps)
        er_im = dir_im / (vol_im + eps)

        # 3. Smoothing Constant: SC = (ER*(fast-slow)+slow)^2
        sc_re = er_re * sc_diff + slow
        sc_re = sc_re * sc_re
        sc_im = er_im * sc_diff + slow
        sc_im = sc_im * sc_im

        # 4. KAMA update
        kama_re = kama_re + sc_re * (x_re - kama_re)
        kama_im = kama_im + sc_im * (x_im - kama_im)

        out[idx] = np.complex64(complex(kama_re, kama_im))

        # 5. Sliding window update (mirrors GPU: if (n+1 < points))
        if idx + 1 < n:
            new_dr = abs(np.float32(data[idx + 1].real) - x_re)
            new_di = abs(np.float32(data[idx + 1].imag) - x_im)
            vol_re = vol_re + new_dr - delta_re[head]
            vol_im = vol_im + new_di - delta_im[head]
            delta_re[head] = new_dr
            delta_im[head] = new_di
            head = head + 1
            if head >= N:
                head = 0

    return out


def kaufman_ref(data: np.ndarray,
                N: int        = ER_PERIOD,
                fast: int     = FAST_PERIOD,
                slow: int     = SLOW_PERIOD) -> np.ndarray:
    """KAMA reference for 1D or 2D (ch, pts) complex64 input."""
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)

    if data.ndim == 1:
        return _kaufman_1ch(data, N, fsc, ssc)
    return np.stack([_kaufman_1ch(data[ch], N, fsc, ssc)
                     for ch in range(data.shape[0])])


# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_ctx_kauf():
    ctx  = gpuworklib.ROCmGPUContext(0)
    kauf = gpuworklib.KaufmanFilterROCm(ctx)
    return ctx, kauf


# ============================================================================
# Test 1: basic GPU vs reference
# ============================================================================

def test_kaufman_basic():
    """Random complex64 signal (1D): GPU vs numpy KAMA reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  KAMA(N={ER_PERIOD}, fast={FAST_PERIOD}, slow={SLOW_PERIOD}): "
          f"max_diff={max_diff:.2e}, atol={ATOL:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: multi-channel
# ============================================================================

def test_kaufman_multi_channel():
    """8-channel 2D (channels, points): each channel matches numpy reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = np.stack([make_complex_signal(POINTS, seed=ch) for ch in range(CHANNELS)])
    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    assert gpu_out.shape == data.shape
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  KAMA multi-channel shape={data.shape}: max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 3: trend signal → ER ≈ 1 → fast tracking
# ============================================================================

def test_kaufman_trend_signal():
    """Linear trend x[n]=n*0.1 (ER≈1): GPU KAMA tracks closely (CPU ref check).

    On a perfectly linear trend, direction = volatility → ER = 1.
    SC = fast_sc^2 = (2/3)^2 ≈ 0.444 → fast EMA-like response.
    KAMA lag should be < fast_period samples.
    """
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    slope = 0.1
    data  = (np.arange(POINTS, dtype=np.float32) * slope).astype(np.complex64)

    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    # GPU vs reference
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"

    # After warm-up: KAMA must be close to the trend (lag < 2 * fast_period)
    skip     = ER_PERIOD + FAST_PERIOD * 3
    lag_max  = float(np.max(np.abs(gpu_out[skip:].real - data[skip:].real)))
    print(f"  Trend slope={slope}: max_lag={lag_max:.4f} "
          f"(expected < {2 * FAST_PERIOD * slope:.4f})")
    assert lag_max < 2 * FAST_PERIOD * slope + 0.1, \
        f"KAMA lag on trend too large: {lag_max:.4f}"
    print("  PASSED")


# ============================================================================
# Test 4: noise signal → ER ≈ 0 → KAMA barely moves
# ============================================================================

def test_kaufman_noise_signal():
    """White noise (ER≈0): GPU KAMA output std << input std (slow tracking).

    SC = slow_sc^2 = (2/31)^2 ≈ 0.0042 → almost stationary output.
    """
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    rng   = np.random.default_rng(123)
    noise = rng.standard_normal(POINTS).astype(np.float32)
    data  = noise.astype(np.complex64)

    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    # GPU vs reference
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"

    # KAMA std should be much smaller than input std
    skip     = ER_PERIOD * 5
    in_std   = float(np.std(data[skip:].real))
    kama_std = float(np.std(gpu_out[skip:].real))
    ratio    = kama_std / in_std
    print(f"  Noise: input_std={in_std:.4f}, KAMA_std={kama_std:.4f}, ratio={ratio:.4f}")
    assert ratio < 0.2, \
        f"KAMA should suppress noise: std_ratio={ratio:.4f} (expected < 0.2)"
    print("  PASSED")


# ============================================================================
# Test 5: adaptive transition trend → noise → trend
# ============================================================================

def test_kaufman_adaptive_transition():
    """Trend→noise→trend: KAMA adapts speed automatically (CPU ref + GPU).

    Phase 1 [0..511]:   linear trend 0→1 → ER≈1, KAMA tracks fast
    Phase 2 [512..1023]: white noise σ=0.2 → ER≈0, KAMA barely moves
    Phase 3 [1024..2047]: step to 1.0 → ER→1, KAMA converges
    """
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    rng  = np.random.default_rng(77)
    data = np.zeros(2048, dtype=np.complex64)

    # Phase 1: trend 0 → 1 over 512 samples
    data[:512] = (np.linspace(0.0, 1.0, 512, dtype=np.float32)
                  + 1j * np.zeros(512, dtype=np.float32))
    # Phase 2: noise around 1.0 (same level as end of trend → ER≈0 → KAMA barely moves)
    noise_ph2  = rng.standard_normal(512).astype(np.float32) * 0.2
    data[512:1024] = (1.0 + noise_ph2).astype(np.complex64)
    # Phase 3: step at 1.0 + tiny noise
    noise_ph3 = rng.standard_normal(1024).astype(np.float32) * 0.02
    data[1024:] = (1.0 + noise_ph3).astype(np.complex64)

    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    # GPU vs reference
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), f"GPU vs ref max_diff={max_diff:.4e}"

    # Phase 1 end: KAMA should be close to 1.0
    kama_end_trend = float(gpu_out[510].real)
    assert abs(kama_end_trend - 1.0) < 0.05, \
        f"Phase 1 end: KAMA={kama_end_trend:.4f}, expected ≈ 1.0"

    # Phase 2: KAMA almost unchanged in steady noise (ER≈0 → SC≈slow_sc^2≈0.004)
    # Check from t=700 (after initial convergence to noise mean) to t=1000
    kama_noise_start = float(gpu_out[700].real)
    kama_noise_end   = float(gpu_out[1000].real)
    delta_noise      = abs(kama_noise_end - kama_noise_start)
    print(f"  Phase 2 KAMA drift (steady, t=700..1000): "
          f"{kama_noise_start:.4f} → {kama_noise_end:.4f}, Δ={delta_noise:.4f}")
    assert delta_noise < 0.10, \
        f"KAMA should barely move in steady noise: delta={delta_noise:.4f}"

    # Phase 3: KAMA converges to 1.0 within ~30 samples (trend → fast SC)
    kama_ph3_late = float(gpu_out[1060].real)
    print(f"  Phase 3: KAMA at +36 samples = {kama_ph3_late:.4f} (expected ≈ 1.0)")
    assert abs(kama_ph3_late - 1.0) < 0.1, \
        f"Phase 3 convergence: KAMA={kama_ph3_late:.4f}, expected ≈ 1.0"

    print("  PASSED")


# ============================================================================
# Test 6: channel independence (256 channels with distinct slopes)
# ============================================================================

def test_kaufman_channel_independence():
    """256 channels with unique trend slopes: GPU states must not cross-contaminate."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    N_CH   = 256
    slopes = np.arange(N_CH, dtype=np.float32) * 0.01
    data   = np.zeros((N_CH, POINTS), dtype=np.complex64)
    for ch in range(N_CH):
        data[ch] = (slopes[ch] * np.arange(POINTS, dtype=np.float32)).astype(np.complex64)

    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)
    gpu_out = kauf.process(data)
    ref     = kaufman_ref(data, ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), \
        f"Channel independence failed, max_diff={max_diff:.4e}"

    # Each channel's KAMA at the end should reflect its own slope
    skip = ER_PERIOD + FAST_PERIOD * 5
    for ch in [0, 50, 128, 255]:
        expected = float(slopes[ch] * POINTS)
        got      = float(gpu_out[ch, -1].real)
        err      = abs(got - expected)
        print(f"  ch={ch:3d}: slope={slopes[ch]:.3f}, "
              f"expected_end={expected:.2f}, KAMA={got:.2f}, err={err:.2f}")
        if slopes[ch] > 0:  # ch=0 is zero → KAMA stays 0
            assert err / (expected + 1e-6) < 0.05, \
                f"ch={ch}: KAMA end={got:.2f}, expected {expected:.2f}"

    print("  PASSED")


# ============================================================================
# Test 7: step signal demo (CPU-only — validates Python reference math)
# ============================================================================

def test_kaufman_step_demo():
    """Step [0×20, 1×50, 0×50]: KAMA adapts to step instantly (CPU ref only).

    No GPU required. Verifies:
      - KAMA reaches plateau ≈ 1.0 at the step
      - KAMA stable in noise: demonstrates adaptive advantage vs fixed MA
    """
    N      = 10
    points = 120
    sig    = np.zeros(points, dtype=np.complex64)
    sig[20:70] = np.complex64(1.0 + 0j)

    fsc = FAST_SC
    ssc = SLOW_SC
    out = kaufman_ref(sig, N, FAST_PERIOD, SLOW_PERIOD)

    # At plateau (t=55): KAMA ≈ 1.0
    val_plateau = float(out[55].real)
    assert abs(val_plateau - 1.0) < 0.01, \
        f"KAMA plateau at t=55: {val_plateau:.4f}, expected ≈ 1.0"

    # At step onset (t=20): KAMA jumps with SC = fast_sc^2 ≈ 0.444
    # KAMA[20] = KAMA[19] + fast_sc^2 * (1 - KAMA[19])
    # Before step KAMA is near 0, so KAMA[20] ≈ fast_sc^2
    val_step = float(out[20].real)
    expected_step = fsc ** 2  # ≈ 0.444
    print(f"  KAMA at step onset t=20: {val_step:.4f}, expected≈{expected_step:.4f}")
    assert abs(val_step - expected_step) < 0.05, \
        f"KAMA first step: {val_step:.4f}, expected ≈ {expected_step:.4f}"

    # Print visual table
    print(f"\n  {'t':>4} | input | KAMA(N={N}, fast={FAST_PERIOD}, slow={SLOW_PERIOD})")
    print(f"  {'':-^4}-+-{'':-^5}-+-{'':-^35}")
    for t in [0, 5, 10, 15, 20, 22, 25, 28, 30, 35, 40, 50, 65, 70, 75, 80, 90, 110, 119]:
        mark = ""
        if t == 20: mark = " <-- step up"
        if t == 70: mark = " <-- step down"
        print(f"  {t:>4d} | {sig[t].real:>5.1f} | {out[t].real:>7.4f}{mark}")

    # Adaptive advantage: trend-noise-trend test with CPU reference
    rng = np.random.default_rng(55)
    data_tnt = np.zeros(200, dtype=np.complex64)
    data_tnt[:60]   = (np.linspace(0, 1, 60, dtype=np.float32)).astype(np.complex64)
    data_tnt[60:120] = (rng.standard_normal(60).astype(np.float32) * 0.3).astype(np.complex64)
    data_tnt[120:]  = np.complex64(1.0 + 0j)

    out_tnt = kaufman_ref(data_tnt, N, FAST_PERIOD, SLOW_PERIOD)

    # In noise phase: KAMA should be nearly unchanged
    kama_noise_start = float(out_tnt[70].real)
    kama_noise_end   = float(out_tnt[115].real)
    delta = abs(kama_noise_end - kama_noise_start)
    print(f"\n  Trend→noise→trend: KAMA noise drift = {delta:.5f} (expected < 0.05)")
    assert delta < 0.05, f"KAMA should not move much in noise: delta={delta:.5f}"

    print("  PASSED")


# ============================================================================
# Test 8: properties
# ============================================================================

def test_kaufman_properties():
    """KaufmanFilterROCm: is_ready(), get_params() returns correct er/fast/slow."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    _, kauf = make_ctx_kauf()
    kauf.set_params(ER_PERIOD, FAST_PERIOD, SLOW_PERIOD)

    assert kauf.is_ready(), "is_ready() should be True after set_params"

    params = kauf.get_params()
    assert params['er_period']   == ER_PERIOD,   f"er_period={params['er_period']}"
    assert params['fast_period'] == FAST_PERIOD, f"fast_period={params['fast_period']}"
    assert params['slow_period'] == SLOW_PERIOD, f"slow_period={params['slow_period']}"

    print(f"  is_ready={kauf.is_ready()}, params={params}")
    print(f"  fast_sc={FAST_SC:.4f}, slow_sc={SLOW_SC:.4f}")
    print(f"  SC(ER=1)={FAST_SC**2:.4f}  SC(ER=0)={SLOW_SC**2:.5f}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  KaufmanFilterROCm (KAMA) — Python Test')
    print(SEP)
    print(f'  HAS_GPU={HAS_GPU}')
    print(f'  Params: N={ER_PERIOD}, fast={FAST_PERIOD}, slow={SLOW_PERIOD}')
    print(f'  fast_sc={FAST_SC:.4f}, slow_sc={SLOW_SC:.4f}')
    print(f'  SC(ER=1)={FAST_SC**2:.4f}  (fast EMA)')
    print(f'  SC(ER=0)={SLOW_SC**2:.5f} (almost stationary)')

    passed, failed = 0, 0

    def run(label, fn):
        global passed, failed
        print(f'\n[{label}] {fn.__doc__.splitlines()[0]}')
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
            failed += 1

    run('1', test_kaufman_basic)
    run('2', test_kaufman_multi_channel)
    run('3', test_kaufman_trend_signal)
    run('4', test_kaufman_noise_signal)
    run('5', test_kaufman_adaptive_transition)
    run('6', test_kaufman_channel_independence)
    run('7', test_kaufman_step_demo)
    run('8', test_kaufman_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
