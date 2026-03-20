#!/usr/bin/env python3
"""
Test: HeterodyneROCm — GPU LFM heterodyne processor (ROCm) vs NumPy reference

Operations:
  Dechirp: s_dc = conj(rx * ref)  element-wise
  Correct: output = dc * exp(j * (-2π * f_beat / fs) * n)

Tests:
  1. dechirp_vs_numpy      — conj(rx * ref) matches numpy reference
  2. dechirp_multi_antenna — multi-antenna (5 ant) dechirp
  3. correct_zero_beat     — zero beat freq: output equals input
  4. correct_vs_numpy      — frequency correction matches numpy formula
  5. params_dict           — set_params / params property correct
  6. dechirp_correct_chain — dechirp + correct pipeline end-to-end

Usage:
  python Python_test/heterodyne/test_heterodyne_rocm.py
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager
from common.runner import SkipTest

# ============================================================================
# Default LFM parameters
# ============================================================================

F_START      = 0.0
F_END        = 2_000_000.0   # 2 MHz bandwidth
SAMPLE_RATE  = 12_000_000.0  # 12 MHz
NUM_SAMPLES  = 8_000
NUM_ANTENNAS = 5
ATOL         = 1e-4

# ============================================================================
# NumPy reference formulas
# ============================================================================

def ref_dechirp(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Reference dechirp: conj(rx * ref)
    ref is broadcast: shape (num_samples,) → applied to each antenna block.
    """
    num_samples = ref.shape[0]
    total = rx.shape[0]
    num_antennas = total // num_samples

    out = np.zeros(total, dtype=np.complex64)
    for ant in range(num_antennas):
        block = rx[ant * num_samples : (ant + 1) * num_samples]
        out[ant * num_samples : (ant + 1) * num_samples] = np.conj(block * ref)
    return out


def ref_correct(dc: np.ndarray, f_beat_hz: list, sample_rate: float,
                num_samples: int) -> np.ndarray:
    """
    Reference correct: dc[ant, n] * exp(j * (-2π * f_beat[ant] / fs) * n)
    phase_step = -2π * f_beat / fs  (precomputed, same as C++ OPT-6)
    """
    num_antennas = len(f_beat_hz)
    out = np.zeros_like(dc)
    n = np.arange(num_samples, dtype=np.float32)
    for ant in range(num_antennas):
        phase_step = np.float32(-2.0 * np.pi * f_beat_hz[ant] / sample_rate)
        correction = np.exp(1j * phase_step * n).astype(np.complex64)
        out[ant * num_samples : (ant + 1) * num_samples] = (
            dc[ant * num_samples : (ant + 1) * num_samples] * correction
        )
    return out


# ============================================================================
# Helpers
# ============================================================================

def make_random_signal(total: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(total) + 1j * rng.standard_normal(total)).astype(np.complex64)


def _make_het(num_samples=NUM_SAMPLES, num_antennas=NUM_ANTENNAS):
    """Создать HeterodyneROCm через GPUContextManager (синглтон)."""
    ctx = GPUContextManager.get_rocm()
    if ctx is None:
        raise SkipTest("ROCm недоступен")
    gw = GPULoader.get()
    het = gw.HeterodyneROCm(ctx)
    het.set_params(
        f_start=F_START, f_end=F_END,
        sample_rate=SAMPLE_RATE,
        num_samples=num_samples,
        num_antennas=num_antennas
    )
    return ctx, het


def _check_gpu():
    """Проверить доступность GPU — бросает SkipTest если нет."""
    gw = GPULoader.get()
    if gw is None or not hasattr(gw, 'HeterodyneROCm'):
        raise SkipTest("gpuworklib/HeterodyneROCm недоступен")
    ctx = GPUContextManager.get_rocm()
    if ctx is None:
        raise SkipTest("ROCm недоступен")


# ============================================================================
# Test 1: dechirp vs numpy
# ============================================================================

def test_dechirp_vs_numpy():
    """Dechirp: GPU conj(rx*ref) matches numpy reference."""
    _check_gpu()

    N   = 1024
    ant = 3
    total = ant * N
    rx  = make_random_signal(total, seed=1)
    ref = make_random_signal(N, seed=2)

    ctx, het = _make_het(num_samples=N, num_antennas=ant)
    gpu_dc = het.dechirp(rx, ref)

    np_dc = ref_dechirp(rx, ref)
    max_diff = float(np.max(np.abs(gpu_dc - np_dc)))
    print(f"  antennas={ant}, N={N}, max_diff={max_diff:.2e}")
    assert gpu_dc.shape == (total,), f"shape mismatch: {gpu_dc.shape}"
    assert np.allclose(gpu_dc, np_dc, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 2: dechirp multi-antenna (default params)
# ============================================================================

def test_dechirp_multi_antenna():
    """Dechirp multi-antenna: 5 antennas, 8000 samples."""
    _check_gpu()

    total = NUM_ANTENNAS * NUM_SAMPLES
    rx  = make_random_signal(total, seed=10)
    ref = make_random_signal(NUM_SAMPLES, seed=20)

    ctx, het = _make_het()
    gpu_dc = het.dechirp(rx, ref)

    np_dc = ref_dechirp(rx, ref)
    max_diff = float(np.max(np.abs(gpu_dc - np_dc)))
    print(f"  antennas={NUM_ANTENNAS}, N={NUM_SAMPLES}, max_diff={max_diff:.2e}")
    assert gpu_dc.shape == (total,), f"shape mismatch: {gpu_dc.shape}"
    assert np.allclose(gpu_dc, np_dc, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 3: correct with zero beat frequency
# ============================================================================

def test_correct_zero_beat():
    """Correct with zero beat: output equals input."""
    _check_gpu()

    N   = 512
    ant = 2
    total = ant * N
    dc = make_random_signal(total, seed=5)
    f_beat = [0.0] * ant

    ctx, het = _make_het(num_samples=N, num_antennas=ant)
    gpu_out = het.correct(dc, f_beat)

    max_diff = float(np.max(np.abs(gpu_out - dc)))
    print(f"  antennas={ant}, N={N}, max_diff={max_diff:.2e}")
    assert np.allclose(gpu_out, dc, atol=ATOL), f"zero beat not identity, max diff={max_diff:.4e}"
    print("  PASSED")


# ============================================================================
# Test 4: correct vs numpy formula
# ============================================================================

def test_correct_vs_numpy():
    """Correct: GPU exp(j*phase_step*n) matches numpy reference."""
    _check_gpu()

    N   = 2048
    ant = 3
    total = ant * N
    dc = make_random_signal(total, seed=7)
    f_beat = [12_345.0, -8_765.0, 50_000.0]  # Hz, one per antenna

    ctx, het = _make_het(num_samples=N, num_antennas=ant)
    gpu_out = het.correct(dc, f_beat)

    np_out = ref_correct(dc, f_beat, SAMPLE_RATE, N)
    max_diff = float(np.max(np.abs(gpu_out - np_out)))
    print(f"  antennas={ant}, N={N}, f_beat={f_beat}, max_diff={max_diff:.2e}")
    assert gpu_out.shape == (total,), f"shape mismatch: {gpu_out.shape}"
    assert np.allclose(gpu_out, np_out, atol=ATOL), f"max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Test 5: params dict
# ============================================================================

def test_params_dict():
    """HeterodyneROCm.params returns correct values."""
    _check_gpu()

    ctx, het = _make_het()
    p = het.params

    assert abs(p['f_start']     - F_START)      < 1.0, f"f_start={p['f_start']}"
    assert abs(p['f_end']       - F_END)         < 1.0, f"f_end={p['f_end']}"
    assert abs(p['sample_rate'] - SAMPLE_RATE)   < 1.0, f"sample_rate={p['sample_rate']}"
    assert p['num_samples']  == NUM_SAMPLES,  f"num_samples={p['num_samples']}"
    assert p['num_antennas'] == NUM_ANTENNAS, f"num_antennas={p['num_antennas']}"

    bw = abs(F_END - F_START)
    assert abs(p['bandwidth'] - bw) < 1.0, f"bandwidth={p['bandwidth']}"

    print(f"  bandwidth={p['bandwidth']/1e6:.2f} MHz, chirp_rate={p['chirp_rate']:.2e}")
    print("  PASSED")


# ============================================================================
# Test 6: dechirp + correct chain
# ============================================================================

def test_dechirp_correct_chain():
    """Pipeline: dechirp → correct produces expected numpy result."""
    _check_gpu()

    N   = 1024
    ant = 2
    total = ant * N
    rx  = make_random_signal(total, seed=11)
    ref = make_random_signal(N, seed=22)
    f_beat = [1000.0, -2000.0]

    ctx, het = _make_het(num_samples=N, num_antennas=ant)

    # GPU pipeline
    dc_gpu  = het.dechirp(rx, ref)
    out_gpu = het.correct(dc_gpu, f_beat)

    # NumPy pipeline
    dc_np  = ref_dechirp(rx, ref)
    out_np = ref_correct(dc_np, f_beat, SAMPLE_RATE, N)

    max_diff = float(np.max(np.abs(out_gpu - out_np)))
    print(f"  antennas={ant}, N={N}, max_diff={max_diff:.2e}")
    assert np.allclose(out_gpu, out_np, atol=ATOL), f"chain max diff={max_diff:.4e} > atol={ATOL}"
    print("  PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    SEP = '=' * 60
    print(SEP)
    print('  HeterodyneROCm — Python Test')
    print(SEP)
    print(f'  F_START={F_START/1e6:.1f} MHz  F_END={F_END/1e6:.1f} MHz')
    print(f'  SAMPLE_RATE={SAMPLE_RATE/1e6:.0f} MHz  N={NUM_SAMPLES}  ANT={NUM_ANTENNAS}')

    passed, failed, skipped = 0, 0, 0

    def run(label, fn):
        global passed, failed, skipped
        print(f'\n[{label}] {fn.__doc__}')
        try:
            fn()
            passed += 1
        except SkipTest as e:
            print(f'  SKIP: {e}')
            skipped += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
            failed += 1

    run('1', test_dechirp_vs_numpy)
    run('2', test_dechirp_multi_antenna)
    run('3', test_correct_zero_beat)
    run('4', test_correct_vs_numpy)
    run('5', test_params_dict)
    run('6', test_dechirp_correct_chain)

    print(f'\n{SEP}')
    print(f'  Results: {passed}/{passed + failed + skipped} passed', end='')
    if skipped:
        print(f'  ({skipped} skipped)', end='')
    print('  — ALL PASSED ✓' if not failed else f'  ({failed} FAILED)')
    print(SEP)
