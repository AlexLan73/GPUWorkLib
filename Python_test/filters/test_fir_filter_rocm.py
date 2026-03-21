#!/usr/bin/env python3
"""
Test: FirFilterROCm — GPU FIR filter (ROCm) vs scipy reference

Tests:
  1. single_channel_basic    — 1D complex signal vs scipy.signal.lfilter
  2. multi_channel           — 2D (channels, points) vs scipy.signal.lfilter
  3. all_pass (delta)        — identity filter: y = x
  4. lowpass_attenuation     — high-frequency component attenuated
  5. properties              — num_taps, coefficients, repr

Usage:
  python Python_test/filters/test_fir_filter_rocm.py

Author: Kodo (AI Assistant)
Date: 2026-02-24
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.gpu_loader import GPULoader
from common.runner import SkipTest

gpuworklib = GPULoader.get()
HAS_GPU = gpuworklib is not None
if not HAS_GPU:
    print(f"WARNING: gpuworklib not found. Skipping GPU tests. (searched: {GPULoader.loaded_from()})")

try:
    import scipy.signal as ss
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Skipping validation tests.")

# ============================================================================
# Parameters
# ============================================================================

SAMPLE_RATE = 50_000.0    # Hz
POINTS      = 4096
CHANNELS    = 8
FIR_TAPS    = 64
FIR_CUTOFF  = 0.1         # normalized (0-1, Nyquist=1)
F_LOW       = 200.0       # Hz — pass band
F_HIGH      = 8_000.0     # Hz — stop band
ATOL        = 1e-4        # absolute tolerance (float32)

# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_two_tone(n: int, fs: float, f_low: float, f_high: float) -> np.ndarray:
    """Single-channel two-tone complex signal."""
    t = np.arange(n, dtype=np.float32) / np.float32(fs)
    sig = (np.cos(2 * np.pi * f_low * t) + 0.5 * np.cos(2 * np.pi * f_high * t)
           + 1j * (np.sin(2 * np.pi * f_low * t) + 0.5 * np.sin(2 * np.pi * f_high * t)))
    return sig.astype(np.complex64)


def scipy_fir_ref(coeffs: list, data: np.ndarray) -> np.ndarray:
    """scipy.signal.lfilter reference for FIR (single or multi-channel)."""
    if data.ndim == 1:
        return ss.lfilter(coeffs, [1.0], data).astype(np.complex64)
    # multi-channel: apply per row
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        out[ch] = ss.lfilter(coeffs, [1.0], data[ch]).astype(np.complex64)
    return out


def make_ctx_fir():
    """Create ROCm context and FIR filter."""
    ctx = gpuworklib.ROCmGPUContext(0)
    fir = gpuworklib.FirFilterROCm(ctx)
    return ctx, fir


# ============================================================================
# Test 1: single channel basic
# ============================================================================

def test_fir_single_channel_basic():
    """FIR single-channel 1D: GPU result matches scipy.lfilter."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    coeffs = ss.firwin(FIR_TAPS, FIR_CUTOFF).tolist()
    data = make_complex_signal(POINTS)

    ctx, fir = make_ctx_fir()
    fir.set_coefficients(coeffs)
    gpu_out = fir.process(data)

    ref = scipy_fir_ref(coeffs, data)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  num_taps={fir.num_taps}, max_diff={max_diff:.2e}, atol={ATOL:.2e}")
    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape} vs {data.shape}"
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: multi-channel 2D
# ============================================================================

def test_fir_multi_channel():
    """FIR multi-channel 2D (channels, points): each channel matches scipy.lfilter."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    coeffs = ss.firwin(FIR_TAPS, FIR_CUTOFF).tolist()
    data = np.zeros((CHANNELS, POINTS), dtype=np.complex64)
    for ch in range(CHANNELS):
        data[ch] = make_complex_signal(POINTS, seed=ch)

    ctx, fir = make_ctx_fir()
    fir.set_coefficients(coeffs)
    gpu_out = fir.process(data)

    ref = scipy_fir_ref(coeffs, data)

    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  shape={data.shape}, num_taps={fir.num_taps}, max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 3: identity (delta) filter
# ============================================================================

def test_fir_all_pass():
    """FIR delta filter [1.0]: output equals input."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    coeffs = [1.0]
    data = make_complex_signal(POINTS)

    ctx, fir = make_ctx_fir()
    fir.set_coefficients(coeffs)
    gpu_out = fir.process(data)

    max_diff = float(np.max(np.abs(gpu_out - data)))
    print(f"  max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, data, atol=ATOL), f"identity FIR failed, max diff={max_diff:.4e}"
    print("  PASSED")


# ============================================================================
# Test 4: low-pass attenuation
# ============================================================================

def test_fir_lowpass_attenuation():
    """FIR lowpass: high-frequency component power is reduced."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    coeffs = ss.firwin(FIR_TAPS, FIR_CUTOFF).tolist()
    data = make_two_tone(POINTS, SAMPLE_RATE, F_LOW, F_HIGH)

    ctx, fir = make_ctx_fir()
    fir.set_coefficients(coeffs)
    gpu_out = fir.process(data)

    # Compare power: skip transient (first FIR_TAPS samples)
    skip = FIR_TAPS
    in_power  = float(np.mean(np.abs(data[skip:])**2))
    out_power = float(np.mean(np.abs(gpu_out[skip:])**2))
    ratio = out_power / in_power
    print(f"  in_power={in_power:.4f}, out_power={out_power:.4f}, ratio={ratio:.4f}")
    # Low-pass should reduce total power (high-freq 0.5x component is attenuated)
    assert ratio < 0.9, f"Expected attenuation, got power ratio={ratio:.4f}"
    print("  PASSED")


# ============================================================================
# Test 5: properties
# ============================================================================

def test_fir_properties():
    """FirFilterROCm: num_taps, coefficients, __repr__ are correct."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    coeffs = [0.25, 0.5, 0.25]
    ctx, fir = make_ctx_fir()
    fir.set_coefficients(coeffs)

    assert fir.num_taps == len(coeffs), f"num_taps={fir.num_taps} != {len(coeffs)}"
    got_coeffs = list(fir.coefficients)
    for i, (got, exp) in enumerate(zip(got_coeffs, coeffs)):
        assert abs(got - exp) < 1e-6, f"coeff[{i}]: got={got}, expected={exp}"
    print(f"  repr={repr(fir)}")
    assert "FirFilterROCm" in repr(fir)
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  FirFilterROCm — Python Test')
    print(SEP)
    print(f'  HAS_GPU={HAS_GPU}, HAS_SCIPY={HAS_SCIPY}')

    passed, failed = 0, 0

    def run(label, fn):
        global passed, failed
        print(f'\n[{label}] {fn.__doc__}')
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
            failed += 1

    run('1', test_fir_single_channel_basic)
    run('2', test_fir_multi_channel)
    run('3', test_fir_all_pass)
    run('4', test_fir_lowpass_attenuation)
    run('5', test_fir_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
