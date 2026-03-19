#!/usr/bin/env python3
"""
ProcessMagnitude ROCm — Python validation test
================================================

Validates ComplexToMagROCm.process_magnitude results against NumPy reference.

Tests:
  1. single_beam_no_norm   — single beam, norm_coeff=1, result ≈ np.abs(iq)
  2. single_beam_norm_n    — norm_coeff=-1 (÷n_point), result ≈ np.abs(iq)/n
  3. multi_beam            — 4 beams, norm_coeff=1, each beam verified
  4. zero_signal           — all-zero input, magnitude = 0
  5. norm_coeff_custom     — norm_coeff=0.5, result ≈ np.abs(iq)*0.5
  6. pipeline_to_stats     — process_magnitude → compute_statistics_float (pipeline)
  7. pipeline_to_median    — process_magnitude → compute_median_float (pipeline)

Usage:
  python Python_test/fft_func/test_process_magnitude_rocm.py
  PYTHONPATH=build/debian-radeon9070/python python ...

Author: Kodo (AI Assistant)
Date: 2026-03-11
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner
from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager


# ============================================================================
# Helpers
# ============================================================================

FS = 44100.0
FREQ = 1000.0


def make_iq(n: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate complex sinusoid IQ signal."""
    t = np.arange(n, dtype=np.float32) / FS
    ph = 2.0 * np.pi * FREQ * t
    return (amplitude * np.exp(1j * ph)).astype(np.complex64)


# ============================================================================
# Tests
# ============================================================================

class TestProcessMagnitude:
    """Tests for ComplexToMagROCm.process_magnitude."""

    def setUp(self):
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCm недоступен")
        if not hasattr(gw, "ComplexToMagROCm"):
            raise SkipTest("ComplexToMagROCm not available in this build")
        self._mag_proc = gw.ComplexToMagROCm(ctx)
        if not hasattr(gw, "StatisticsProcessor"):
            raise SkipTest("StatisticsProcessor not available in this build")
        self._stats_proc = gw.StatisticsProcessor(ctx)

    def test_single_beam_no_norm(self):
        """Single beam, norm_coeff=1: result ≈ np.abs(iq)."""
        n = 4096
        iq = make_iq(n)
        result = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)
        ref = np.abs(iq).astype(np.float32)
        assert result.shape == (n,), f"Expected ({n},), got {result.shape}"
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_single_beam_norm_by_n(self):
        """norm_coeff=-1 (÷n_point): result ≈ np.abs(iq) / n."""
        n = 2048
        amplitude = 3.0
        iq = make_iq(n, amplitude)
        result = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=-1.0)
        ref = (np.abs(iq) / n).astype(np.float32)
        assert result.shape == (n,)
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-7)

    def test_multi_beam(self):
        """4 beams × 2048 points, norm_coeff=1."""
        n = 2048
        beam_count = 4
        iq = np.zeros((beam_count, n), dtype=np.complex64)
        for b in range(beam_count):
            iq[b] = make_iq(n, amplitude=float(b + 1))

        result = self._mag_proc.process_magnitude(iq, beam_count=beam_count, norm_coeff=1.0)
        assert result.shape == (beam_count, n), f"Expected ({beam_count},{n}), got {result.shape}"

        for b in range(beam_count):
            ref = np.abs(iq[b]).astype(np.float32)
            np.testing.assert_allclose(result[b], ref, rtol=1e-4, atol=1e-5,
                                        err_msg=f"beam {b} mismatch")

    def test_zero_signal(self):
        """Zero input → magnitude = 0."""
        n = 512
        iq = np.zeros(n, dtype=np.complex64)
        result = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)
        assert result.shape == (n,)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_norm_coeff_custom(self):
        """norm_coeff=0.5: result ≈ np.abs(iq) * 0.5."""
        n = 1024
        iq = make_iq(n, amplitude=2.0)
        result = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=0.5)
        ref = (np.abs(iq) * 0.5).astype(np.float32)
        assert result.shape == (n,)
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_pipeline_process_magnitude_to_stats(self):
        """Pipeline: process_magnitude → compute_statistics_float."""
        n = 4096
        amplitude = 2.0
        iq = make_iq(n, amplitude)
        mag = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)

        results = self._stats_proc.compute_statistics_float(mag, beam_count=1)
        assert len(results) == 1
        mean_mag = results[0]["mean_magnitude"]
        assert abs(mean_mag - amplitude) < 0.05 * amplitude, \
            f"mean_magnitude={mean_mag:.4f}, expected≈{amplitude}"
        assert results[0]["std_dev"] < 0.01, \
            f"std_dev={results[0]['std_dev']:.6f} too large for constant-amplitude sinusoid"

    def test_pipeline_process_magnitude_to_median(self):
        """Pipeline: process_magnitude → compute_median_float."""
        n = 2048
        amplitude = 3.0
        iq = make_iq(n, amplitude)
        mag = self._mag_proc.process_magnitude(iq, beam_count=1, norm_coeff=1.0)

        results = self._stats_proc.compute_median_float(mag, beam_count=1)
        assert len(results) == 1
        median = results[0]["median_magnitude"]
        assert abs(median - amplitude) < 0.05, \
            f"median_magnitude={median:.4f}, expected≈{amplitude}"


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestProcessMagnitude())
    runner.print_summary(results)
