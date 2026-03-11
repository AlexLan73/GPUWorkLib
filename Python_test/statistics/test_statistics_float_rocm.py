#!/usr/bin/env python3
"""
Statistics Float ROCm — Python validation test
================================================

Validates StatisticsProcessor.compute_statistics_float and compute_median_float
against NumPy reference.

Tests:
  1. stats_float_single_beam  — compute_statistics_float, single beam, vs np.mean/np.std
  2. stats_float_multi_beam   — compute_statistics_float, 4 beams
  3. median_float_single_beam — compute_median_float, sorted data, vs np.median
  4. median_float_multi_beam  — compute_median_float, 4 beams
  5. stats_float_constant     — constant signal: std≈0, mean=value
  6. pipeline_mag_to_stats    — ComplexToMagROCm → compute_statistics_float
  7. pipeline_mag_to_median   — ComplexToMagROCm → compute_median_float

Usage:
  pytest Python_test/statistics/test_statistics_float_rocm.py -v
  PYTHONPATH=build/debian-radeon9070/python pytest ...

Author: Kodo (AI Assistant)
Date: 2026-03-11
"""

import numpy as np
import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def stats_proc(gw, rocm_ctx):
    if not hasattr(gw, "StatisticsProcessor"):
        pytest.skip("StatisticsProcessor not available")
    return gw.StatisticsProcessor(rocm_ctx)


@pytest.fixture(scope="module")
def mag_proc(gw, rocm_ctx):
    if not hasattr(gw, "ComplexToMagROCm"):
        pytest.skip("ComplexToMagROCm not available")
    return gw.ComplexToMagROCm(rocm_ctx)


# ============================================================================
# Helpers
# ============================================================================

RNG = np.random.default_rng(42)


def make_float_data(n_beams: int, n_points: int, lo: float = 0.5, hi: float = 5.0) -> np.ndarray:
    return RNG.uniform(lo, hi, size=(n_beams, n_points)).astype(np.float32)


def make_iq(n: int, amplitude: float = 1.0, fs: float = 44100.0, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / fs
    return (amplitude * np.exp(1j * 2.0 * np.pi * freq * t)).astype(np.complex64)


# ============================================================================
# Tests
# ============================================================================


def test_stats_float_single_beam(stats_proc):
    """compute_statistics_float: single beam vs np.mean, np.std."""
    n = 4096
    data = RNG.uniform(1.0, 5.0, n).astype(np.float32)
    results = stats_proc.compute_statistics_float(data, beam_count=1)
    assert len(results) == 1

    ref_mean = float(np.mean(data))
    ref_std = float(np.std(data))

    r = results[0]
    assert abs(r["mean_magnitude"] - ref_mean) < 0.01 * ref_mean + 1e-5, \
        f"mean_magnitude={r['mean_magnitude']:.4f} vs ref={ref_mean:.4f}"
    assert abs(r["std_dev"] - ref_std) < 0.01 * ref_std + 1e-5, \
        f"std_dev={r['std_dev']:.4f} vs ref={ref_std:.4f}"


def test_stats_float_multi_beam(stats_proc):
    """compute_statistics_float: 4 beams, verify per-beam mean/std."""
    n_beams, n_pts = 4, 2048
    data = make_float_data(n_beams, n_pts)
    results = stats_proc.compute_statistics_float(data, beam_count=n_beams)
    assert len(results) == n_beams

    for b in range(n_beams):
        ref_mean = float(np.mean(data[b]))
        ref_std = float(np.std(data[b]))
        r = results[b]
        assert abs(r["mean_magnitude"] - ref_mean) < 0.02 * ref_mean + 1e-5, \
            f"beam {b}: mean {r['mean_magnitude']:.4f} vs {ref_mean:.4f}"
        assert abs(r["std_dev"] - ref_std) < 0.02 * ref_std + 1e-5, \
            f"beam {b}: std {r['std_dev']:.4f} vs {ref_std:.4f}"


def test_median_float_single_beam(stats_proc):
    """compute_median_float: sorted data 0..N-1, median ≈ (N-1)/2."""
    n = 1024
    data = np.arange(n, dtype=np.float32)
    np.random.shuffle(data)  # shuffle for realism
    results = stats_proc.compute_median_float(data, beam_count=1)
    assert len(results) == 1

    ref_median = float(np.median(data))
    assert abs(results[0]["median_magnitude"] - ref_median) < 1.0, \
        f"median={results[0]['median_magnitude']:.1f} vs ref={ref_median:.1f}"


def test_median_float_multi_beam(stats_proc):
    """compute_median_float: 4 beams."""
    n_beams, n_pts = 4, 512
    data = make_float_data(n_beams, n_pts, lo=0.0, hi=100.0)
    results = stats_proc.compute_median_float(data, beam_count=n_beams)
    assert len(results) == n_beams

    for b in range(n_beams):
        ref_median = float(np.median(data[b]))
        assert abs(results[b]["median_magnitude"] - ref_median) < 1.0, \
            f"beam {b}: median {results[b]['median_magnitude']:.1f} vs {ref_median:.1f}"


def test_stats_float_constant(stats_proc):
    """Constant signal: std≈0, mean≈value."""
    n = 2048
    value = 3.14
    data = np.full(n, value, dtype=np.float32)
    results = stats_proc.compute_statistics_float(data, beam_count=1)
    assert len(results) == 1
    assert abs(results[0]["mean_magnitude"] - value) < 1e-3, \
        f"mean={results[0]['mean_magnitude']:.4f}, expected {value}"
    assert results[0]["std_dev"] < 5e-3, \
        f"std_dev={results[0]['std_dev']:.6f} should be ~0 for constant signal"


def test_pipeline_mag_to_stats(mag_proc, stats_proc):
    """Pipeline: ComplexToMagROCm → compute_statistics_float."""
    n = 4096
    amplitude = 2.5
    iq = make_iq(n, amplitude)

    # Step 1: compute magnitudes on GPU
    mag = mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)

    # Step 2: statistics on float magnitudes
    results = stats_proc.compute_statistics_float(mag, beam_count=1)
    assert len(results) == 1

    # For constant-amplitude sinusoid: mean_magnitude ≈ amplitude
    mean_mag = results[0]["mean_magnitude"]
    assert abs(mean_mag - amplitude) < 0.05, \
        f"mean_magnitude={mean_mag:.4f}, expected≈{amplitude}"

    # std should be very small
    assert results[0]["std_dev"] < 0.05, \
        f"std_dev={results[0]['std_dev']:.4f} too large"


def test_pipeline_mag_to_median(mag_proc, stats_proc):
    """Pipeline: ComplexToMagROCm → compute_median_float."""
    n = 2048
    amplitude = 1.5
    iq = make_iq(n, amplitude)

    mag = mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)
    results = stats_proc.compute_median_float(mag, beam_count=1)
    assert len(results) == 1

    median = results[0]["median_magnitude"]
    assert abs(median - amplitude) < 0.05, \
        f"median={median:.4f}, expected≈{amplitude}"
