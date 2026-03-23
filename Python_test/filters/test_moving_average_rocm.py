#!/usr/bin/env python3
"""
Test: MovingAverageFilterROCm — GPU скользящие средние (ROCm) vs numpy reference

Filters: SMA, EMA, MMA (Wilder), DEMA, TEMA
GPU class: gpuworklib.MovingAverageFilterROCm

Classes:
  TestMovingAverageMath  — CPU reference math (no GPU required)
    1. test_step_response_demo — step [0×20,1×50,0×50]: plateau ≈ 1.0, speed order

  TestMovingAverageROCm  — GPU vs numpy reference (ROCm required)
    2. test_channel_independence — 256 channels: no cross-contamination
    3. test_dema_basic           — DEMA(N=10) = 2*EMA1 - EMA2
    4. test_ema_basic            — EMA(N=10), 1D complex signal
    5. test_impulse_response     — EMA: delta → exponential decay check
    6. test_mma_basic            — MMA(N=10, alpha=1/N, Wilder's)
    7. test_multi_channel        — 8-channel EMA
    8. test_properties           — is_ready(), get_window_size(), get_type()
    9. test_sma_basic            — SMA(N=8), 1D complex signal
   10. test_tema_basic           — TEMA(N=10) = 3*EMA1 - 3*EMA2 + EMA3

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

Author: Kodo (AI Assistant)
Date: 2026-03-04 (TestRunner: 2026-03-23)
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.gpu_loader import GPULoader
from common.runner import TestRunner, SkipTest

gpuworklib = GPULoader.get()
HAS_GPU = gpuworklib is not None
if not HAS_GPU:
    print(f"WARNING: gpuworklib not found. (searched: {GPULoader.loaded_from()})")

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


# ============================================================================
# TestMovingAverageMath — CPU reference math (no GPU required)
# ============================================================================

class TestMovingAverageMath:
    """Validates Python reference math — step response and speed ordering."""

    def test_step_response_demo(self):
        """Step [0×20, 1×50, 0×50]: plateau ≈ 1.0, response speed order."""
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

        # Speed order at the rising edge (t=23, 3 samples after step onset)
        assert float(out_tema[23].real) > float(out_dema[23].real), "TEMA leads DEMA"
        assert float(out_dema[23].real) > float(out_ema[23].real),  "DEMA leads EMA"
        assert float(out_ema[23].real)  > float(out_mma[23].real),  "EMA leads MMA"

        # SMA reaches plateau exactly at t=20+N=30 (N-sample ramp, then flat)
        assert abs(float(out_sma[30].real) - 1.0) < 1e-5, "SMA plateau at t=30"

        # Print visual table (informational, not a validation)
        print(f"\n  {'t':>4} | input | SMA   | EMA   | MMA   | DEMA  | TEMA")
        print(f"  {'':-^4}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}-+-{'':-^5}")
        for t in [0, 5, 10, 20, 25, 30, 35, 40, 50, 65, 70, 80, 110]:
            print(f"  {t:>4d} | {sig[t].real:>5.1f} "
                  f"| {out_sma[t].real:>5.3f} "
                  f"| {out_ema[t].real:>5.3f} "
                  f"| {out_mma[t].real:>5.3f} "
                  f"| {out_dema[t].real:>5.3f} "
                  f"| {out_tema[t].real:>5.3f}")


# ============================================================================
# TestMovingAverageROCm — GPU vs numpy reference (ROCm required)
# ============================================================================

class TestMovingAverageROCm:
    """GPU MovingAverageFilterROCm vs numpy reference (ROCm required)."""

    def setUp(self):
        if not HAS_GPU:
            raise SkipTest("gpuworklib not found — ROCm GPU required")
        self.ctx = gpuworklib.ROCmGPUContext(0)
        self.ma  = gpuworklib.MovingAverageFilterROCm(self.ctx)

    def test_channel_independence(self):
        """256 channels with distinct signals: GPU states must not cross-contaminate."""
        N_CH = 256
        data = np.stack([make_complex_signal(POINTS, seed=ch * 7) for ch in range(N_CH)])
        self.ma.set_params("EMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = ema_ref(data, N_WIN)

        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  256-ch EMA: max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), \
            f"Channel independence failed, max_diff={max_diff:.4e}"

    def test_dema_basic(self):
        """DEMA(N=10) = 2*EMA1 - EMA2: GPU single-pass vs numpy two-pass reference."""
        data = make_complex_signal(POINTS)
        self.ma.set_params("DEMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = dema_ref(data, N_WIN)

        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  DEMA(N={N_WIN}): max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"DEMA max_diff={max_diff:.4e} > {ATOL}"

        # DEMA leads EMA on a step edge
        step = np.zeros(200, dtype=np.complex64)
        step[50:] = np.complex64(1.0 + 0j)
        ema_step  = ema_ref(step, N_WIN)
        dema_step = dema_ref(step, N_WIN)
        assert float(dema_step[60].real) > float(ema_step[60].real), \
            "DEMA should lead EMA on rising edge"

    def test_ema_basic(self):
        """EMA(N=10) single channel 1D: GPU vs numpy reference."""
        data = make_complex_signal(POINTS)
        self.ma.set_params("EMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = ema_ref(data, N_WIN)

        assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  EMA(N={N_WIN}): max_diff={max_diff:.2e}, atol={ATOL:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"EMA max_diff={max_diff:.4e} > {ATOL}"

    def test_impulse_response(self):
        """EMA impulse: delta at n=0 → exponential decay y[n] = (1-alpha)^n."""
        # GPU kernel: state = data[0]; out[0] = state
        # With delta input: out[0]=1, out[n] = (1-alpha)^n for n>0
        data = np.zeros(POINTS, dtype=np.complex64)
        data[0] = np.complex64(1.0 + 0j)

        self.ma.set_params("EMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = ema_ref(data, N_WIN)

        alpha     = np.float32(2.0 / (N_WIN + 1))
        one_minus = np.float32(1.0) - alpha

        print(f"\n  EMA(N={N_WIN}): alpha={float(alpha):.4f}, (1-alpha)={float(one_minus):.4f}")
        for n in [1, 2, 5, 10, 20]:
            expected = float(one_minus ** n)
            got_gpu  = float(gpu_out[n].real)
            diff_gpu = abs(got_gpu - expected)
            print(f"    y[{n:2d}]: expected={expected:.6f}, GPU={got_gpu:.6f}, "
                  f"diff={diff_gpu:.2e}")
            assert diff_gpu < ATOL * 10, f"impulse decay at n={n}: diff={diff_gpu:.4e}"

        max_diff = float(np.max(np.abs(gpu_out - ref)))
        assert np.allclose(gpu_out, ref, atol=ATOL), f"GPU vs ref max_diff={max_diff:.4e}"

    def test_mma_basic(self):
        """MMA(N=10, alpha=1/N, Wilder's) single channel 1D: GPU vs numpy reference."""
        data = make_complex_signal(POINTS)
        self.ma.set_params("MMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = mma_ref(data, N_WIN)

        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  MMA(N={N_WIN}, alpha=1/N): max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"MMA max_diff={max_diff:.4e} > {ATOL}"

        # MMA must be smoother than EMA at the same N (smaller alpha → less tracking)
        ref_ema = ema_ref(data, N_WIN)
        skip = N_WIN * 3
        assert float(np.std(np.abs(ref[skip:]))) < float(np.std(np.abs(ref_ema[skip:]))), \
            "MMA should be smoother (smaller std) than EMA at same N"

    def test_multi_channel(self):
        """EMA multi-channel 2D (channels, points): each channel matches reference."""
        data = np.stack([make_complex_signal(POINTS, seed=ch) for ch in range(CHANNELS)])
        self.ma.set_params("EMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = ema_ref(data, N_WIN)

        assert gpu_out.shape == data.shape, f"shape: {gpu_out.shape} vs {data.shape}"
        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  EMA multi-channel shape={data.shape}: max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"max_diff={max_diff:.4e} > {ATOL}"

    def test_properties(self):
        """is_ready(), get_window_size(), get_type() after set_params."""
        self.ma.set_params("EMA", N_WIN)

        assert self.ma.is_ready(), "is_ready() should be True after set_params"
        wsize = self.ma.get_window_size()
        assert wsize == N_WIN, f"get_window_size()={wsize} != {N_WIN}"
        mtype = str(self.ma.get_type()).upper()
        assert "EMA" in mtype, f"get_type()='{mtype}' should contain 'EMA'"

        print(f"\n  is_ready={self.ma.is_ready()}, window_size={wsize}, type={mtype}")

    def test_sma_basic(self):
        """SMA(N=8) single channel 1D: GPU vs numpy ring-buffer reference."""
        data = make_complex_signal(POINTS)
        self.ma.set_params("SMA", N_SMA)
        gpu_out = self.ma.process(data)
        ref     = sma_ref(data, N_SMA)

        assert gpu_out.shape == data.shape
        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  SMA(N={N_SMA}): max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"SMA max_diff={max_diff:.4e} > {ATOL}"

        # Partial averages at the warmup region
        for k in range(1, N_SMA):
            expected = np.mean(data[:k+1])
            diff = abs(float(gpu_out[k]) - float(expected))
            assert diff < ATOL * 10, f"SMA warmup at k={k}: diff={diff:.4e}"

    def test_tema_basic(self):
        """TEMA(N=10) = 3*EMA1 - 3*EMA2 + EMA3: GPU vs numpy reference."""
        data = make_complex_signal(POINTS)
        self.ma.set_params("TEMA", N_WIN)
        gpu_out = self.ma.process(data)
        ref     = tema_ref(data, N_WIN)

        max_diff = float(np.max(np.abs(gpu_out - ref)))
        print(f"\n  TEMA(N={N_WIN}): max_diff={max_diff:.2e}")
        assert np.allclose(gpu_out, ref, atol=ATOL), f"TEMA max_diff={max_diff:.4e} > {ATOL}"

        # TEMA is fastest: verify TEMA > DEMA > EMA on a step edge (t=53)
        step = np.zeros(200, dtype=np.complex64)
        step[50:] = np.complex64(1.0 + 0j)
        ema_s  = ema_ref(step,  N_WIN)
        dema_s = dema_ref(step, N_WIN)
        tema_s = tema_ref(step, N_WIN)
        t_check = 53
        assert float(tema_s[t_check].real) > float(dema_s[t_check].real) > float(ema_s[t_check].real), \
            (f"Response speed at t={t_check}: "
             f"TEMA={tema_s[t_check].real:.4f} DEMA={dema_s[t_check].real:.4f} "
             f"EMA={ema_s[t_check].real:.4f} — expected TEMA > DEMA > EMA")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    runner = TestRunner()

    results = []
    results += runner.run(TestMovingAverageMath())
    results += runner.run(TestMovingAverageROCm())

    runner.print_summary(results)
