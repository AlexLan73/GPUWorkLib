"""
FM Correlator ROCm — Python tests

Tests:
  test_autocorrelation     — ref vs ref, SNR > 10
  test_shift_pattern       — run_test_pattern, verify peak positions
  test_cpu_vs_gpu_pattern  — numpy circshift vs GPU, atol=1e-4
  test_params_only_mode    — set_params + prepare_reference + run_test_pattern (no data)

Run:
  python Python_test/fm_correlator/test_fm_correlator_rocm.py
  PYTHONPATH=build/python python Python_test/fm_correlator/test_fm_correlator_rocm.py
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner

try:
    import gpuworklib
    HAS_ROCM = hasattr(gpuworklib, 'FMCorrelatorROCm')
except ImportError:
    HAS_ROCM = False


class TestFMCorrelatorROCm:
    """FM Correlator ROCm tests — skip if ROCm not available."""

    def setUp(self):
        if not HAS_ROCM:
            raise SkipTest("ROCm not available or FMCorrelatorROCm not found")
        self._ctx = gpuworklib.ROCmGPUContext(0)

    def test_autocorrelation(self):
        """Autocorrelation: ref vs ref -> peak at j=0, SNR > 10."""
        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=4096, num_shifts=1, num_signals=1,
                        num_output_points=200)
        ref = corr.generate_msequence()
        corr.prepare_reference_from_data(ref)
        peaks = corr.process(ref)

        assert peaks.shape == (1, 1, 200)

        peak_val = peaks[0, 0, 0]
        noise_max = np.max(peaks[0, 0, 1:])
        snr = peak_val / noise_max if noise_max > 0 else 1e6

        print(f"peak={peak_val:.4f} noise={noise_max:.6f} SNR={snr:.1f}")
        assert snr > 10, f"SNR={snr:.1f} < 10"

    def test_shift_pattern(self):
        """GPU test pattern: verify peak positions."""
        N, K, S = 4096, 10, 5
        shift_step = 2

        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=N, num_shifts=K, num_signals=S,
                        num_output_points=200)
        corr.prepare_reference()

        peaks_gpu = corr.run_test_pattern(shift_step)
        assert peaks_gpu.shape == (S, K, 200)

        verified = 0
        for s in range(S):
            for k in range(K):
                expected = (k - s * shift_step) % N
                if expected < 200:
                    actual_peak = np.argmax(peaks_gpu[s, k, :])
                    assert actual_peak == expected, \
                        f"s={s} k={k}: expected peak at {expected}, got {actual_peak}"
                    verified += 1

        print(f"Verified {verified} peak positions")
        assert verified > 0

    def test_cpu_vs_gpu_pattern(self):
        """CPU circshift vs GPU generate_test_inputs — results must match."""
        N, K, S = 4096, 8, 4
        shift_step = 2

        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=N, num_shifts=K, num_signals=S,
                        num_output_points=200)

        ref = corr.generate_msequence()
        corr.prepare_reference_from_data(ref)

        # CPU: numpy circshift
        signals = np.stack([np.roll(ref, -s * shift_step) for s in range(S)])
        peaks_cpu = corr.process(signals.astype(np.float32))

        # GPU: run_test_pattern
        corr.prepare_reference_from_data(ref)
        peaks_gpu = corr.run_test_pattern(shift_step)

        np.testing.assert_allclose(peaks_cpu, peaks_gpu, atol=1e-4,
                                   err_msg="CPU vs GPU peaks mismatch")

    def test_params_only_mode(self):
        """Test mode where only parameters are passed — no data transfer."""
        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=8192, num_shifts=16, num_signals=8,
                        num_output_points=500)
        corr.prepare_reference()
        peaks = corr.run_test_pattern(shift_step=3)

        assert peaks.shape == (8, 16, 500)
        assert np.all(np.isfinite(peaks))
        assert np.max(peaks) > 0


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestFMCorrelatorROCm())
    runner.print_summary(results)
