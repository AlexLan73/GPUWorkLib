"""
Smoke-тест processing адаптеров -- требует GPU (gpuworklib).

Запуск: python Python_test/Core/processing/test_processing_smoke.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from common import GPUContextManager, SkipTest
from common.configs import HeterodyneConfig
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from Core.processing import StatisticsAdapter, HeterodyneAdapter, FftAdapter


class TestProcessingSmoke:
    """Smoke-тест processing адаптеров."""

    def test_statistics_adapter(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="statistics_adapter_smoke")
        adapter = StatisticsAdapter(ctx)
        data = np.random.randn(4, 1024).astype(np.float32)
        stats = adapter.process(data)

        has_keys = all(k in stats for k in ("mean", "std", "median"))
        ok_shape = stats["mean"].shape == (4,)
        result.add(ValidationResult(
            passed=has_keys and ok_shape,
            metric_name="stats_structure",
            actual_value=1.0 if has_keys and ok_shape else 0.0,
            threshold=1.0,
            message=f"keys={list(stats.keys())}, mean_shape={stats['mean'].shape}"
        ))

        # compute_mean_std shortcut
        means, stds = adapter.compute_mean_std(data)
        ok_shortcut = means.shape == (4,) and stds.shape == (4,)
        result.add(ValidationResult(
            passed=ok_shortcut,
            metric_name="compute_mean_std",
            actual_value=1.0 if ok_shortcut else 0.0,
            threshold=1.0,
            message=f"means_shape={means.shape}, stds_shape={stds.shape}"
        ))

        # repr
        ok_repr = "StatisticsAdapter" in repr(adapter)
        result.add(ValidationResult(
            passed=ok_repr,
            metric_name="repr",
            actual_value=1.0 if ok_repr else 0.0,
            threshold=1.0,
            message=repr(adapter)
        ))

        return result

    def test_fft_adapter_complex(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="fft_adapter_complex")
        fft = FftAdapter(ctx, n_fft=4096, mode="complex")
        data = np.random.randn(4096).astype(np.complex64)
        out = fft.process(data)

        ok_dtype = out.dtype == np.complex64
        ok_nfft = fft.n_fft == 4096
        result.add(ValidationResult(
            passed=ok_dtype and ok_nfft,
            metric_name="fft_complex",
            actual_value=1.0 if ok_dtype else 0.0,
            threshold=1.0,
            message=f"dtype={out.dtype}, shape={out.shape}, n_fft={fft.n_fft}"
        ))
        return result

    def test_fft_adapter_magnitude(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="fft_adapter_magnitude")
        fft = FftAdapter(ctx, n_fft=4096, mode="magnitude")
        data = np.random.randn(4096).astype(np.complex64)
        out = fft.process(data)

        ok_dtype = out.dtype == np.float32
        result.add(ValidationResult(
            passed=ok_dtype,
            metric_name="fft_magnitude",
            actual_value=1.0 if ok_dtype else 0.0,
            threshold=1.0,
            message=f"dtype={out.dtype}, shape={out.shape}"
        ))
        return result

    def test_fft_adapter_invalid_mode(self):
        result = TestResult(test_name="fft_adapter_invalid_mode")
        try:
            FftAdapter(None, n_fft=4096, mode="invalid")
            ok = False
            msg = "ValueError not raised"
        except ValueError as e:
            ok = True
            msg = str(e)
        except Exception as e:
            # gpuworklib not loaded -> RuntimeError from _load_gw, which is OK
            # but mode check should happen first
            ok = "mode" in str(e).lower() or isinstance(e, ValueError)
            msg = f"{type(e).__name__}: {e}"
        result.add(ValidationResult(
            passed=ok,
            metric_name="invalid_mode",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=msg
        ))
        return result

    def test_heterodyne_adapter(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="heterodyne_adapter_smoke")
        params = HeterodyneConfig(
            fs=12e6, f0_hz=0.0, fdev_hz=2e6,
            n_samples=8000, n_antennas=5
        )
        het = HeterodyneAdapter(ctx, params)

        ok_ant = het.n_antennas == 5
        ok_bw = abs(het.params.bandwidth - 2e6) < 1.0
        ok_cr = het.params.chirp_rate > 0
        result.add(ValidationResult(
            passed=ok_ant and ok_bw and ok_cr,
            metric_name="het_params",
            actual_value=1.0 if ok_ant and ok_bw and ok_cr else 0.0,
            threshold=1.0,
            message=(f"n_antennas={het.n_antennas}, "
                     f"bandwidth={het.params.bandwidth:.0f}, "
                     f"chirp_rate={het.params.chirp_rate:.2e}")
        ))

        data = np.random.randn(5, 8000).astype(np.complex64)
        out = het.process(data)
        ok_shape = out.shape == (5, 8000)
        ok_dtype = out.dtype == np.complex64
        result.add(ValidationResult(
            passed=ok_shape and ok_dtype,
            metric_name="het_process",
            actual_value=1.0 if ok_shape and ok_dtype else 0.0,
            threshold=1.0,
            message=f"output shape={out.shape}, dtype={out.dtype}"
        ))
        return result


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestProcessingSmoke())
    runner.print_summary(results)
