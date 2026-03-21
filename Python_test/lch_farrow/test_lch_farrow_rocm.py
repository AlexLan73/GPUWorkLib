#!/usr/bin/env python3
"""
Test: LchFarrowROCm — GPU Lagrange 48x5 fractional delay (ROCm) vs CPU reference

Algorithm:
  delay_samples = delay_us * 1e-6 * sample_rate
  read_pos(n)   = n - delay_samples
  center        = floor(read_pos)
  frac          = read_pos - center
  row           = int(frac * 48) % 48
  output[n]     = sum(L[row][k] * input[center - 1 + k],  k=0..4)
  (boundary: skip if index < 0 or >= N)

Tests:
  1. zero_delay        — output equals input (identity, row 0)
  2. integer_delay     — output = shifted input by D samples
  3. fractional_delay  — GPU vs CPU Lagrange reference
  4. multi_antenna     — 4 antennas with different delays
  5. properties        — sample_rate, delays, __repr__

Usage:
  python Python_test/lch_farrow/test_lch_farrow_rocm.py

Author: Kodo (AI Assistant)
Date: 2026-02-24
"""

import sys
import os
import json
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

PROJECT_ROOT = os.path.dirname(_PT_DIR)

from common.gpu_loader import GPULoader
from common.runner import SkipTest

gpuworklib = GPULoader.get()
HAS_GPU = gpuworklib is not None
if not HAS_GPU:
    print(f"WARNING: gpuworklib not found. Skipping GPU tests. (searched: {GPULoader.loaded_from()})")

# ============================================================================
# Load Lagrange matrix 48×5
# ============================================================================

MATRIX_PATH = os.path.join(PROJECT_ROOT, 'modules', 'lch_farrow', 'lagrange_matrix_48x5.json')

def load_lagrange_matrix() -> np.ndarray:
    """Load 48x5 Lagrange matrix from JSON. Returns (48, 5) float32 array."""
    with open(MATRIX_PATH, 'r') as f:
        data = json.load(f)
    arr = np.array(data['data'], dtype=np.float32)
    return arr.reshape(data['rows'], data['columns'])


# ============================================================================
# CPU reference — exact match of C++ ProcessCpu
# ============================================================================

def cpu_lch_farrow(signal: np.ndarray, delay_us: float, sample_rate: float,
                   L_matrix: np.ndarray) -> np.ndarray:
    """
    CPU reference for LchFarrow fractional delay.
    Matches C++ ProcessCpu exactly (including boundary handling).

    Args:
        signal:      input complex64 (N,)
        delay_us:    delay in microseconds
        sample_rate: Hz
        L_matrix:    (48, 5) Lagrange coefficients

    Returns:
        output complex64 (N,), zeros where read_pos < 0
    """
    N = len(signal)
    output = np.zeros(N, dtype=np.complex64)
    delay_samples = np.float32(delay_us * 1e-6 * sample_rate)

    for n in range(N):
        read_pos = np.float32(n) - delay_samples
        if read_pos < 0.0:
            continue

        center = int(np.floor(read_pos))
        frac   = float(read_pos) - center
        row    = int(frac * 48.0) % 48

        L = L_matrix[row]   # shape (5,)
        val = np.complex64(0.0)
        for k in range(5):
            idx = center - 1 + k
            if 0 <= idx < N:
                val += np.complex64(L[k]) * signal[idx]

        output[n] = val

    return output


# ============================================================================
# Parameters
# ============================================================================

SAMPLE_RATE = 1_000_000.0   # 1 MHz
POINTS      = 512
ATOL_INT    = 1e-4          # integer delay (exact shift)
ATOL_FRAC   = 5e-3          # fractional delay (float32 rounding)

# ============================================================================
# Helpers
# ============================================================================

def make_complex_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def make_ctx_lch(sample_rate=SAMPLE_RATE):
    ctx = gpuworklib.ROCmGPUContext(0)
    lch = gpuworklib.LchFarrowROCm(ctx)
    lch.set_sample_rate(sample_rate)
    return ctx, lch


# ============================================================================
# Test 1: zero delay
# ============================================================================

def test_zero_delay():
    """Zero delay: output equals input (identity)."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    data = make_complex_signal(POINTS)
    ctx, lch = make_ctx_lch()
    lch.set_delays([0.0])    # 0 µs
    gpu_out = lch.process(data)

    max_diff = float(np.max(np.abs(gpu_out - data)))
    print(f"  N={POINTS}, delay=0 µs, max_diff={max_diff:.2e}")
    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape}"
    assert np.allclose(gpu_out, data, atol=ATOL_INT), (
        f"zero delay not identity, max diff={max_diff:.4e}")
    print("  PASSED")


# ============================================================================
# Test 2: integer delay
# ============================================================================

def test_integer_delay():
    """Integer delay (5 µs at 1 MHz = 5 samples): output = shift(input, 5)."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    DELAY_US = 5.0
    D = int(DELAY_US * 1e-6 * SAMPLE_RATE)   # = 5 samples

    data = make_complex_signal(POINTS)
    ctx, lch = make_ctx_lch()
    lch.set_delays([DELAY_US])
    gpu_out = lch.process(data)

    # Expected: output[n] = input[n - D] for n >= D, else 0
    ref = np.zeros(POINTS, dtype=np.complex64)
    ref[D:] = data[:POINTS - D]

    max_diff = float(np.max(np.abs(gpu_out[D:] - ref[D:])))
    print(f"  delay={DELAY_US} µs = {D} samples, max_diff={max_diff:.2e}")
    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape}"
    assert np.allclose(gpu_out[D:], ref[D:], atol=ATOL_INT), (
        f"integer delay mismatch, max diff={max_diff:.4e}")
    print("  PASSED")


# ============================================================================
# Test 3: fractional delay vs CPU reference
# ============================================================================

def test_fractional_delay_vs_cpu():
    """Fractional delay (2.7 µs): GPU matches CPU Lagrange reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    DELAY_US = 2.7
    try:
        L_matrix = load_lagrange_matrix()
    except FileNotFoundError:
        print(f"  SKIP: matrix not found at {MATRIX_PATH}")
        return

    data = make_complex_signal(POINTS)
    ctx, lch = make_ctx_lch()
    lch.set_delays([DELAY_US])
    gpu_out = lch.process(data)

    cpu_ref = cpu_lch_farrow(data, DELAY_US, SAMPLE_RATE, L_matrix)

    delay_samples = DELAY_US * 1e-6 * SAMPLE_RATE   # 2.7
    skip = int(np.ceil(delay_samples)) + 2           # skip transient boundary

    max_diff = float(np.max(np.abs(gpu_out[skip:] - cpu_ref[skip:])))
    print(f"  delay={DELAY_US} µs ({delay_samples:.2f} samples), max_diff={max_diff:.2e}")
    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape}"
    assert np.allclose(gpu_out[skip:], cpu_ref[skip:], atol=ATOL_FRAC), (
        f"fractional delay mismatch, max diff={max_diff:.4e} > atol={ATOL_FRAC}")
    print("  PASSED")


# ============================================================================
# Test 4: multi-antenna
# ============================================================================

def test_multi_antenna():
    """Multi-antenna (4 channels, different delays): each matches CPU reference."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    try:
        L_matrix = load_lagrange_matrix()
    except FileNotFoundError:
        print(f"  SKIP: matrix not found at {MATRIX_PATH}")
        return

    delays_us = [0.0, 1.5, 3.3, 5.25]   # µs per antenna (avoid exact integer delays — GPU float32 boundary)
    n_ant = len(delays_us)
    data = np.zeros((n_ant, POINTS), dtype=np.complex64)
    for i in range(n_ant):
        data[i] = make_complex_signal(POINTS, seed=i * 7)

    ctx, lch = make_ctx_lch()
    lch.set_delays(delays_us)
    gpu_out = lch.process(data)   # shape (n_ant, points)

    assert gpu_out.shape == data.shape, f"shape mismatch: {gpu_out.shape}"

    max_diff_all = 0.0
    for i, delay_us in enumerate(delays_us):
        cpu_ref = cpu_lch_farrow(data[i], delay_us, SAMPLE_RATE, L_matrix)
        ds = delay_us * 1e-6 * SAMPLE_RATE
        skip = int(np.ceil(ds)) + 2
        diff = float(np.max(np.abs(gpu_out[i, skip:] - cpu_ref[skip:])))
        max_diff_all = max(max_diff_all, diff)
        print(f"    ant[{i}] delay={delay_us} µs, max_diff={diff:.2e}")

    assert max_diff_all < ATOL_FRAC, (
        f"multi-antenna max diff={max_diff_all:.4e} > atol={ATOL_FRAC}")
    print("  PASSED")


# ============================================================================
# Test 5: properties
# ============================================================================

def test_properties():
    """LchFarrowROCm: sample_rate, delays, __repr__ are correct."""
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

    delays = [1.0, 2.5, 0.0]
    ctx, lch = make_ctx_lch(sample_rate=SAMPLE_RATE)
    lch.set_delays(delays)

    assert abs(lch.sample_rate - SAMPLE_RATE) < 1.0, (
        f"sample_rate={lch.sample_rate} != {SAMPLE_RATE}")

    got_delays = list(lch.delays)
    assert len(got_delays) == len(delays), f"delays len mismatch: {got_delays}"
    for i, (got, exp) in enumerate(zip(got_delays, delays)):
        assert abs(got - exp) < 1e-5, f"delay[{i}]: got={got}, expected={exp}"

    print(f"  repr={repr(lch)}")
    assert "LchFarrowROCm" in repr(lch)
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  LchFarrowROCm — Python Test')
    print(SEP)
    print(f'  HAS_GPU={HAS_GPU}')
    print(f'  SAMPLE_RATE={SAMPLE_RATE/1e6:.1f} MHz  N={POINTS}')
    print(f'  Matrix: {MATRIX_PATH}')

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

    run('1', test_zero_delay)
    run('2', test_integer_delay)
    run('3', test_fractional_delay_vs_cpu)
    run('4', test_multi_antenna)
    run('5', test_properties)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed} passed', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
