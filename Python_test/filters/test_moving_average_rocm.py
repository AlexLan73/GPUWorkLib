#!/usr/bin/env python3
"""
Test: MovingAverageFilterROCm — GPU скользящие средние (ROCm) vs numpy reference

Filters: SMA, EMA, MMA (Wilder), DEMA, TEMA
GPU class: gpuworklib.MovingAverageFilterROCm

Tests:
  1. test_ema_basic          — EMA(N=10), 1D complex signal vs numpy reference
  2. test_sma_basic          — SMA(N=8),  1D complex signal vs numpy reference
  3. test_mma_basic          — MMA(N=10, alpha=1/N) vs numpy reference
  4. test_dema_basic         — DEMA(N=10) = 2*EMA1 - EMA2 vs numpy reference
  5. test_tema_basic         — TEMA(N=10) = 3*EMA1 - 3*EMA2 + EMA3 vs numpy reference
  6. test_multi_channel      — 8-channel EMA: each channel vs reference
  7. test_impulse_response   — EMA: delta input → exponential decay check
  8. test_channel_independence— 256 channels with distinct signals
  9. test_step_response_demo — step signal [0×20, 1×50, 0×50]: visual demo (CPU-only)
  10. test_properties        — is_ready(), get_window_size(), get_type()

Note:
  Tolerance GPU vs numpy reference: < 1e-4 (float32).
  Python references implement float32 arithmetic to match GPU kernels exactly.
  GPU API (Python binding):
    ma = gpuworklib.MovingAverageFilterROCm(ctx)
    ma.set_params("EMA", window_size)   # type: "SMA"/"EMA"/"MMA"/"DEMA"/"TEMA"
    ma.process(data)                    # 1D or 2D (channels, points) complex64
    ma.is_ready()                       # bool
    ma.get_window_size()                # int
    ma.get_type()                       # MAType enum / string

Usage:
  python Python_test/filters/test_moving_average_rocm.py
  pytest Python_test/filters/test_moving_average_rocm.py -v

Author: Kodo (AI Assistant)
Date: 2026-03-04
"""

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'debian-radeon9070', 'python'))

try:
    import gpuworklib
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found. Skipping GPU tests.")

# ============================================================================
# Parameters
# ============================================================================

POINTS   = 4096
CHANNELS = 8
N_WIN    = 10    # window size for EMA / MMA / DEMA / TEMA
N_SMA    = 8     # window size for SMA (must be ≤ 128, GPU ring buffer limit)
ATOL     = 1e-4  # float32 tolerance GPU vs Python reference

# ============================================================================
# Python reference implementations — float32 arithmetic, match GPU kernels
# ============================================================================

def _ema_1ch(data: np.ndarray, alpha: float) -> np.ndarray:
    """EMA on a single 1D complex64 channel, float32 arithmetic.

    State init: state = data[0]  (matches GPU kernel: state = in[base])
    Recurrence: state = alpha*x + (1-alpha)*state
    """
    a  = np.float32(alpha)
    om = np.float32(1.0) - a
    n  = len(data)
    out = np.empty(n, dtype=np.complex64)
    sr = np.float32(data[0].real)
    si = np.float32(data[0].imag)
    out[0] = np.complex64(complex(sr, si))
    for i in range(1, n):
        sr = a * np.float32(data[i].real) + om * sr
        si = a * np.float32(data[i].imag) + om * si
        out[i] = np.complex64(complex(sr, si))
    return out


def ema_ref(data: np.ndarray, N: int) -> np.ndarray:
    """EMA reference: alpha = 2/(N+1). Works for 1D or 2D (ch, pts) input."""
    alpha = 2.0 / (N + 1)
    if data.ndim == 1:
        return _ema_1ch(data, alpha)
    return np.stack([_ema_1ch(data[ch], alpha) for ch in range(data.shape[0])])


def mma_ref(data: np.ndarray, N: int) -> np.ndarray:
    """MMA (Wilder's Smoothed MA) reference: alpha = 1/N."""
    alpha = 1.0 / N
    if data.ndim == 1:
        return _ema_1ch(data, alpha)
    return np.stack([_ema_1ch(data[ch], alpha) for ch in range(data.shape[0])])


def _sma_1ch(data: np.ndarray, N: int) -> np.ndarray:
    """SMA on a single 1D complex64 channel — ring-buffer approach, float32.

    Matches GPU sma_kernel:
      n < N : partial average  out[n] = sum[0..n] / (n+1)
      n >= N: sliding update   out[n] = (sum + x[n] - x[n-N]) / N
    """
    n    = len(data)
    out  = np.empty(n, dtype=np.complex64)
    buf  = np.zeros(N, dtype=np.complex64)
    sr   = np.float32(0.0)
    si   = np.float32(0.0)
    head = 0
    inv_N = np.float32(1.0 / N)
    for i in range(n):
        xr = np.float32(data[i].real)
        xi = np.float32(data[i].imag)
        if i < N:
            buf[i] = data[i]
            sr += xr
            si += xi
            out[i] = np.complex64(complex(sr / np.float32(i + 1),
                                          si / np.float32(i + 1)))
        else:
            or_ = np.float32(buf[head].real)
            oi_ = np.float32(buf[head].imag)
            buf[head] = data[i]
            head += 1
            if head >= N:
                head = 0
            sr += xr - or_
            si += xi - oi_
            out[i] = np.complex64(complex(sr * inv_N, si * inv_N))
    return out


def sma_ref(data: np.ndarray, N: int) -> np.ndarray:
    """SMA reference (N ≤ 128). Works for 1D or 2D (ch, pts) input."""
    if data.ndim == 1:
        return _sma_1ch(data, N)
    return np.stack([_sma_1ch(data[ch], N) for ch in range(data.shape[0])])


def dema_ref(data: np.ndarray, N: int) -> np.ndarray:
    """DEMA = 2*EMA1 - EMA2 per channel.

    Two-pass approach is mathematically equivalent to GPU single-pass kernel:
      GPU:  ema1 = alpha*x + (1-a)*ema1;  ema2 = alpha*ema1 + (1-a)*ema2
      Both start at data[0], give identical sequence.
    """
    alpha = 2.0 / (N + 1)

    def _dema_1ch(d: np.ndarray) -> np.ndarray:
        ema1 = _ema_1ch(d, alpha)
        ema2 = _ema_1ch(ema1, alpha)
        return (np.float32(2.0) * ema1 - ema2).astype(np.complex64)

    if data.ndim == 1:
        return _dema_1ch(data)
    return np.stack([_dema_1ch(data[ch]) for ch in range(data.shape[0])])


def tema_ref(data: np.ndarray, N: int) -> np.ndarray:
    """TEMA = 3*EMA1 - 3*EMA2 + EMA3 per channel (three-pass, equiv to GPU)."""
    alpha = 2.0 / (N + 1)

    def _tema_1ch(d: np.ndarray) -> np.ndarray:
        ema1 = _ema_1ch(d, alpha)
        ema2 = _ema_1ch(ema1, alpha)
        ema3 = _ema_1ch(ema2, alpha)
        return (np.float32(3.0) * ema1
                - np.float32(3.0) * ema2
                + ema3).astype(np.complex64)

    if data.ndim == 1:
        return _tema_1ch(data)
    return np.stack([_tema_1ch(data[ch]) for ch in range(data.shape[0])])


# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_ctx_ma():
    ctx = gpuworklib.ROCmGPUContext(0)
    ma  = gpuworklib.MovingAverageFilterROCm(ctx)
    return ctx, ma


# ============================================================================
# Test 1: EMA basic
# ============================================================================

def test_ema_basic():
    """EMA(N=10) single channel 1D: GPU vs numpy reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, ma = make_ctx_ma()
    ma.set_params("EMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = ema_ref(data, N_WIN)

    assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  EMA(N={N_WIN}): max_diff={max_diff:.2e}, atol={ATOL:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"EMA max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: SMA basic
# ============================================================================

def test_sma_basic():
    """SMA(N=8) single channel 1D: GPU vs numpy ring-buffer reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, ma = make_ctx_ma()
    ma.set_params("SMA", N_SMA)
    gpu_out = ma.process(data)
    ref     = sma_ref(data, N_SMA)

    assert gpu_out.shape == data.shape
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  SMA(N={N_SMA}): max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"SMA max_diff={max_diff:.4e} > {ATOL}"

    # Partial averages at the warmup region
    for k in range(1, N_SMA):
        expected = np.mean(data[:k+1])
        diff = abs(float(gpu_out[k]) - float(expected))
        assert diff < ATOL * 10, f"SMA warmup at k={k}: diff={diff:.4e}"

    print("  PASSED")


# ============================================================================
# Test 3: MMA basic
# ============================================================================

def test_mma_basic():
    """MMA(N=10, alpha=1/N, Wilder's) single channel 1D: GPU vs numpy reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, ma = make_ctx_ma()
    ma.set_params("MMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = mma_ref(data, N_WIN)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  MMA(N={N_WIN}, alpha=1/N): max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"MMA max_diff={max_diff:.4e} > {ATOL}"

    # MMA must be slower than EMA at the same N (smaller alpha → smoother)
    ref_ema = ema_ref(data, N_WIN)
    # EMA alpha=2/11≈0.182 > MMA alpha=0.1 → EMA reacts faster
    # After warmup, std(MMA) should be lower than std(EMA) — smoother
    skip = N_WIN * 3
    assert float(np.std(np.abs(ref[skip:]))) < float(np.std(np.abs(ref_ema[skip:]))), \
        "MMA should be smoother (smaller std) than EMA at same N"
    print("  PASSED")


# ============================================================================
# Test 4: DEMA basic
# ============================================================================

def test_dema_basic():
    """DEMA(N=10) = 2*EMA1 - EMA2: GPU single-pass vs numpy two-pass reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, ma = make_ctx_ma()
    ma.set_params("DEMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = dema_ref(data, N_WIN)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  DEMA(N={N_WIN}): max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"DEMA max_diff={max_diff:.4e} > {ATOL}"

    # DEMA leads EMA on transitions — verify by comparing lag on a step signal
    step = np.zeros(200, dtype=np.complex64)
    step[50:] = np.complex64(1.0 + 0j)
    ema_step  = ema_ref(step, N_WIN)
    dema_step = dema_ref(step, N_WIN)
    # At n=60 (10 samples after step): DEMA should be ahead of EMA
    assert float(dema_step[60].real) > float(ema_step[60].real), \
        "DEMA should lead EMA on rising edge"
    print("  PASSED")


# ============================================================================
# Test 5: TEMA basic
# ============================================================================

def test_tema_basic():
    """TEMA(N=10) = 3*EMA1 - 3*EMA2 + EMA3: GPU vs numpy reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    _, ma = make_ctx_ma()
    ma.set_params("TEMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = tema_ref(data, N_WIN)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  TEMA(N={N_WIN}): max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"TEMA max_diff={max_diff:.4e} > {ATOL}"

    # TEMA is fastest: verify TEMA leads DEMA leads EMA on a step
    step = np.zeros(200, dtype=np.complex64)
    step[50:] = np.complex64(1.0 + 0j)
    ema_s  = ema_ref(step,  N_WIN)
    dema_s = dema_ref(step, N_WIN)
    tema_s = tema_ref(step, N_WIN)
    assert float(tema_s[60].real) > float(dema_s[60].real) > float(ema_s[60].real), \
        "Response speed order: TEMA > DEMA > EMA"
    print("  PASSED")


# ============================================================================
# Test 6: multi-channel
# ============================================================================

def test_multi_channel():
    """EMA multi-channel 2D (channels, points): each channel matches reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = np.stack([make_complex_signal(POINTS, seed=ch) for ch in range(CHANNELS)])
    _, ma = make_ctx_ma()
    ma.set_params("EMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = ema_ref(data, N_WIN)

    assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  EMA multi-channel shape={data.shape}: max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"
    print("  PASSED")


# ============================================================================
# Test 7: impulse response
# ============================================================================

def test_impulse_response():
    """EMA impulse: delta at n=0 → exponential decay y[n] = (1-alpha)^n."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    # GPU kernel: state = data[0]; out[0] = state
    # so with delta input: out[0]=1, out[n] = (1-alpha)^n for n>0
    data = np.zeros(POINTS, dtype=np.complex64)
    data[0] = np.complex64(1.0 + 0j)

    _, ma = make_ctx_ma()
    ma.set_params("EMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = ema_ref(data, N_WIN)

    alpha     = np.float32(2.0 / (N_WIN + 1))
    one_minus = np.float32(1.0) - alpha

    print(f"  EMA(N={N_WIN}): alpha={float(alpha):.4f}, (1-alpha)={float(one_minus):.4f}")
    for n in [1, 2, 5, 10, 20]:
        expected = float(one_minus ** n)
        got_gpu  = float(gpu_out[n].real)
        got_ref  = float(ref[n].real)
        diff_gpu = abs(got_gpu - expected)
        print(f"    y[{n:2d}]: expected={expected:.6f}, GPU={got_gpu:.6f}, ref={got_ref:.6f}, "
              f"diff={diff_gpu:.2e}")
        assert diff_gpu < ATOL * 10, f"impulse decay at n={n}: diff={diff_gpu:.4e}"

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    assert np.allclose(gpu_out, ref, atol=ATOL), f"GPU vs ref max_diff={max_diff:.4e}"
    print("  PASSED")


# ============================================================================
# Test 8: channel independence (256 channels)
# ============================================================================

def test_channel_independence():
    """256 channels with distinct signals: GPU states must not cross-contaminate."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    N_CH = 256
    data = np.stack([make_complex_signal(POINTS, seed=ch * 7) for ch in range(N_CH)])
    _, ma = make_ctx_ma()
    ma.set_params("EMA", N_WIN)
    gpu_out = ma.process(data)
    ref     = ema_ref(data, N_WIN)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  256-ch EMA: max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), \
        f"Channel independence failed, max_diff={max_diff:.4e}"
    print("  PASSED")


# ============================================================================
# Test 9: step response demo (CPU-only — validates Python reference math)
# ============================================================================

def test_step_response_demo():
    """Step signal [0×20, 1×50, 0×50]: compare SMA/EMA/MMA/DEMA/TEMA (CPU refs).

    No GPU required. Validates:
      - All filters reach plateau ≈ 1.0 by mid-segment (t=55)
      - Response speed order: TEMA > DEMA > EMA > MMA, SMA = N-step ramp
    """
    N      = 10
    points = 120
    sig    = np.zeros(points, dtype=np.complex64)
    sig[20:70] = np.complex64(1.0 + 0j)

    out_sma  = sma_ref(sig, N)
    out_ema  = ema_ref(sig, N)
    out_mma  = mma_ref(sig, N)
    out_dema = dema_ref(sig, N)
    out_tema = tema_ref(sig, N)

    # All filters must reach ~1.0 at plateau (t=55, well inside [20..70])
    for name, out in [("SMA",  out_sma),  ("EMA",  out_ema),
                      ("MMA",  out_mma),  ("DEMA", out_dema),
                      ("TEMA", out_tema)]:
        val = float(out[55].real)
        assert abs(val - 1.0) < 0.05, \
            f"{name}(N={N}) mid-plateau={val:.4f}, expected ≈ 1.0"

    # Speed order at the rising edge (t=23, 3 samples after step)
    assert float(out_tema[23].real) > float(out_dema[23].real), "TEMA leads DEMA"
    assert float(out_dema[23].real) > float(out_ema[23].real),  "DEMA leads EMA"
    assert float(out_ema[23].real)  > float(out_mma[23].real),  "EMA leads MMA"

    # SMA reaches plateau exactly at t=20+N=30 (N-sample ramp, then flat)
    assert abs(float(out_sma[30].real) - 1.0) < 1e-5, "SMA plateau at t=30"

    # Print visual table
    print(f"\n  {'t':>4} | input | SMA   | EMA   | MMA   | DEMA  | TEMA")
    print(f"  {'':-^4}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}")
    for t in [0, 5, 10, 20, 25, 30, 35, 40, 50, 65, 70, 80, 110]:
        print(f"  {t:>4d} | {sig[t].real:>5.1f} "
              f"| {out_sma[t].real:>5.3f} "
              f"| {out_ema[t].real:>5.3f} "
              f"| {out_mma[t].real:>5.3f} "
              f"| {out_dema[t].real:>5.3f} "
              f"| {out_tema[t].real:>5.3f}")
    print("  PASSED")


# ============================================================================
# Test 10: properties
# ============================================================================

def test_properties():
    """MovingAverageFilterROCm: is_ready(), get_window_size(), get_type()."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    _, ma = make_ctx_ma()
    ma.set_params("EMA", N_WIN)

    assert ma.is_ready(), "is_ready() should be True after set_params"
    wsize = ma.get_window_size()
    assert wsize == N_WIN, f"get_window_size()={wsize} != {N_WIN}"
    mtype = str(ma.get_type()).upper()
    assert "EMA" in mtype, f"get_type()='{mtype}' should contain 'EMA'"

    print(f"  is_ready={ma.is_ready()}, window_size={wsize}, type={mtype}")
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  MovingAverageFilterROCm — Python Test')
    print(SEP)
    print(f'  HAS_GPU={HAS_GPU}')

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

    run('1',  test_ema_basic)
    run('2',  test_sma_basic)
    run('3',  test_mma_basic)
    run('4',  test_dema_basic)
    run('5',  test_tema_basic)
    run('6',  test_multi_channel)
    run('7',  test_impulse_response)
    run('8',  test_channel_independence)
    run('9',  test_step_response_demo)
    run('10', test_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
