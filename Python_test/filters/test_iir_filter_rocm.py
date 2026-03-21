#!/usr/bin/env python3
"""
Test: IirFilterROCm — GPU IIR biquad cascade (ROCm) vs scipy reference

Tests:
  1. single_channel_basic    — 1D complex signal vs scipy.signal.sosfilt
  2. multi_channel           — 2D (channels, points) vs scipy.signal.sosfilt
  3. passthrough (zeroes)    — zero input → zero output
  4. section_attenuation     — LP filter attenuates high frequency
  5. properties              — num_sections, sections, __repr__

Note: GPU IIR is most efficient with many channels (>= 8).
      For single channel the overhead of GPU launch is significant.

Usage:
  python Python_test/filters/test_iir_filter_rocm.py

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
IIR_ORDER   = 2
IIR_CUTOFF  = 0.1         # normalized (0-1, Nyquist=1)
F_LOW       = 200.0       # Hz — pass band
F_HIGH      = 8_000.0     # Hz — stop band
ATOL        = 1e-3        # IIR has larger accumulated error than FIR

# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_two_tone(n: int, fs: float, f_low: float, f_high: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / np.float32(fs)
    sig = (np.cos(2 * np.pi * f_low * t) + 0.5 * np.cos(2 * np.pi * f_high * t)
           + 1j * (np.sin(2 * np.pi * f_low * t) + 0.5 * np.sin(2 * np.pi * f_high * t)))
    return sig.astype(np.complex64)


def sos_to_sections(sos: np.ndarray) -> list:
    """
    Convert scipy sos array to list of biquad dicts for IirFilterROCm.
    SOS format: [b0, b1, b2, a0, a1, a2] where a0=1 always.
    """
    sections = []
    for row in sos:
        sections.append({
            'b0': float(row[0]),
            'b1': float(row[1]),
            'b2': float(row[2]),
            'a1': float(row[4]),   # skip a0 (always 1)
            'a2': float(row[5]),
        })
    return sections


def scipy_iir_ref(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    """scipy.signal.sosfilt reference (single or multi-channel)."""
    if data.ndim == 1:
        return ss.sosfilt(sos, data).astype(np.complex64)
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        out[ch] = ss.sosfilt(sos, data[ch]).astype(np.complex64)
    return out


def make_ctx_iir():
    ctx = gpuworklib.ROCmGPUContext(0)
    iir = gpuworklib.IirFilterROCm(ctx)
    return ctx, iir


# ============================================================================
# Test 1: single channel basic
# ============================================================================

def test_iir_single_channel_basic():
    """IIR single-channel 1D: GPU result matches scipy.sosfilt."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    sos = ss.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)
    data = make_complex_signal(POINTS)

    ctx, iir = make_ctx_iir()
    iir.set_sections(sections)
    gpu_out = iir.process(data)

    ref = scipy_iir_ref(sos, data)

    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  num_sections={iir.num_sections}, max_diff={max_diff:.2e}, atol={ATOL:.2e}")
    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape} vs {data.shape}"
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: multi-channel 2D
# ============================================================================

def test_iir_multi_channel():
    """IIR multi-channel 2D (channels, points): each channel matches scipy.sosfilt."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    sos = ss.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)
    data = np.zeros((CHANNELS, POINTS), dtype=np.complex64)
    for ch in range(CHANNELS):
        data[ch] = make_complex_signal(POINTS, seed=ch)

    ctx, iir = make_ctx_iir()
    iir.set_sections(sections)
    gpu_out = iir.process(data)

    ref = scipy_iir_ref(sos, data)

    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape} vs {data.shape}"
    max_diff = float(np.max(np.abs(gpu_out - ref)))
    print(f"  shape={data.shape}, num_sections={iir.num_sections}, max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, ref, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 3: zero input → zero output
# ============================================================================

def test_iir_zero_input():
    """IIR with zero input: output is zero (no initial state)."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    sos = ss.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)
    data = np.zeros(POINTS, dtype=np.complex64)

    ctx, iir = make_ctx_iir()
    iir.set_sections(sections)
    gpu_out = iir.process(data)

    max_val = float(np.max(np.abs(gpu_out)))
    print(f"  max_val={max_val:.2e}")
    assert max_val < 1e-6, f"Expected zero output, got max={max_val:.4e}"
    print("  PASSED")


# ============================================================================
# Test 4: low-pass attenuation
# ============================================================================

def test_iir_lowpass_attenuation():
    """IIR lowpass: high-frequency component is significantly attenuated."""
    if not HAS_GPU or not HAS_SCIPY:
        print("  SKIP: no GPU or scipy")
        return

    sos = ss.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)
    data = make_two_tone(POINTS, SAMPLE_RATE, F_LOW, F_HIGH)

    ctx, iir = make_ctx_iir()
    iir.set_sections(sections)
    gpu_out = iir.process(data)

    # Compare power: skip transient (first ~100 samples for IIR)
    skip = 100
    in_power  = float(np.mean(np.abs(data[skip:])**2))
    out_power = float(np.mean(np.abs(gpu_out[skip:])**2))
    ratio = out_power / in_power
    print(f"  in_power={in_power:.4f}, out_power={out_power:.4f}, ratio={ratio:.4f}")
    assert ratio < 0.9, f"Expected attenuation, got power ratio={ratio:.4f}"
    print("  PASSED")


# ============================================================================
# Test 5: properties
# ============================================================================

def test_iir_properties():
    """IirFilterROCm: num_sections, sections dict, __repr__ are correct."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    raw_sections = [
        {'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.56, 'a2': 0.64}
    ]

    ctx, iir = make_ctx_iir()
    iir.set_sections(raw_sections)

    assert iir.num_sections == 1, f"num_sections={iir.num_sections} != 1"
    got = list(iir.sections)[0]
    for key in ('b0', 'b1', 'b2', 'a1', 'a2'):
        assert abs(got[key] - raw_sections[0][key]) < 1e-5, (
            f"section[{key}]: got={got[key]}, expected={raw_sections[0][key]}")

    print(f"  repr={repr(iir)}")
    assert "IirFilterROCm" in repr(iir)
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  IirFilterROCm — Python Test')
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

    run('1', test_iir_single_channel_basic)
    run('2', test_iir_multi_channel)
    run('3', test_iir_zero_input)
    run('4', test_iir_lowpass_attenuation)
    run('5', test_iir_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
