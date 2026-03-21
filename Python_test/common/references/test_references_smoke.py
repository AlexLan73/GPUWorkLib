"""
Smoke-тест references -- работает без GPU (чистый NumPy).
Проверяет dtype, shape, базовые математические свойства.

Запуск: python Python_test/common/references/test_references_smoke.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from common.runner import TestRunner
from common.result import TestResult, ValidationResult


class TestReferencesSmoke:
    """Smoke-тест references -- все проверки без GPU."""

    def test_signal_refs(self):
        from common.references import SignalReferences
        result = TestResult(test_name="signal_refs")

        # CW
        cw = SignalReferences.cw(12e6, 4096, 2e6)
        result.add(ValidationResult(
            passed=cw.dtype == np.complex64 and cw.shape == (4096,),
            metric_name="cw_shape_dtype",
            actual_value=1.0 if cw.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={cw.dtype}, shape={cw.shape}"
        ))

        # LFM
        lfm = SignalReferences.lfm(12e6, 4096, 0.0, 2e6)
        result.add(ValidationResult(
            passed=lfm.dtype == np.complex64 and lfm.shape == (4096,),
            metric_name="lfm_shape_dtype",
            actual_value=1.0 if lfm.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={lfm.dtype}, shape={lfm.shape}"
        ))

        # LFM with delay
        lfm_d = SignalReferences.lfm_with_delay(12e6, 4096, 0.0, 2e6, 1e-4)
        zeros_before = np.sum(np.abs(lfm_d[:int(12e6 * 1e-4)]) < 1e-10)
        result.add(ValidationResult(
            passed=zeros_before > 0,
            metric_name="lfm_delay_zeros",
            actual_value=float(zeros_before),
            threshold=1.0,
            message=f"zeros before delay: {zeros_before}"
        ))

        # LFM multi-antenna
        delays = np.array([1e-4, 2e-4, 3e-4])
        multi = SignalReferences.lfm_multi_antenna(12e6, 4096, 0.0, 2e6, delays)
        result.add(ValidationResult(
            passed=multi.shape == (3, 4096) and multi.dtype == np.complex64,
            metric_name="lfm_multi_shape",
            actual_value=1.0 if multi.shape == (3, 4096) else 0.0,
            threshold=1.0,
            message=f"shape={multi.shape}"
        ))

        # FormSignal
        form = SignalReferences.form_signal(12e6, 4096, 2e6, 1.0, 0.0, 1e6, 1.0)
        result.add(ValidationResult(
            passed=form.dtype == np.complex64 and form.shape == (4096,),
            metric_name="form_signal_shape",
            actual_value=1.0 if form.shape == (4096,) else 0.0,
            threshold=1.0,
            message=f"dtype={form.dtype}, shape={form.shape}"
        ))

        # Noise
        noise = SignalReferences.noise(4096, seed=42)
        result.add(ValidationResult(
            passed=noise.dtype == np.complex64 and noise.shape == (4096,),
            metric_name="noise_shape_dtype",
            actual_value=1.0 if noise.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={noise.dtype}, mean_abs={np.abs(noise).mean():.4f}"
        ))

        # Dechirp
        s_ref = SignalReferences.lfm(12e6, 4096, 0.0, 2e6)
        dc = SignalReferences.dechirp(lfm, s_ref)
        result.add(ValidationResult(
            passed=dc.dtype == np.complex64 and dc.shape == (4096,),
            metric_name="dechirp_shape",
            actual_value=1.0 if dc.dtype == np.complex64 else 0.0,
            threshold=1.0
        ))

        return result

    def test_statistics_refs(self):
        from common.references import StatisticsReferences as StatsRef
        result = TestResult(test_name="statistics_refs")

        data = np.random.randn(4, 1024).astype(np.float32)
        stats = StatsRef.mean_std_median(data)

        has_keys = all(k in stats for k in ("mean", "std", "median"))
        result.add(ValidationResult(
            passed=has_keys,
            metric_name="stats_keys",
            actual_value=1.0 if has_keys else 0.0,
            threshold=1.0,
            message=f"keys={list(stats.keys())}"
        ))

        ok_shape = stats["mean"].shape == (4,)
        result.add(ValidationResult(
            passed=ok_shape,
            metric_name="stats_shape",
            actual_value=1.0 if ok_shape else 0.0,
            threshold=1.0,
            message=f"mean_shape={stats['mean'].shape}"
        ))

        # 1D input
        data_1d = np.random.randn(512).astype(np.float32)
        m = StatsRef.mean(data_1d)
        result.add(ValidationResult(
            passed=np.isscalar(m) or m.ndim == 0,
            metric_name="stats_1d_scalar",
            actual_value=1.0 if (np.isscalar(m) or m.ndim == 0) else 0.0,
            threshold=1.0,
            message=f"mean={m}"
        ))

        return result

    def test_fft_refs(self):
        from common.references import SignalReferences, FftReferences
        result = TestResult(test_name="fft_refs")

        cw = SignalReferences.cw(12e6, 4096, 2e6)
        peak = FftReferences.peak_freq(cw, 12e6)
        freq_res = 12e6 / 4096
        err = abs(peak - 2e6)
        result.add(ValidationResult(
            passed=err < freq_res,
            metric_name="peak_freq_hz",
            actual_value=err,
            threshold=freq_res,
            message=f"peak={peak:.1f} Hz, expected=2e6 Hz, err={err:.1f}"
        ))

        mag = FftReferences.magnitude(cw)
        result.add(ValidationResult(
            passed=mag.dtype == np.float32 and mag.shape == (4096,),
            metric_name="magnitude_shape",
            actual_value=1.0 if mag.dtype == np.float32 else 0.0,
            threshold=1.0
        ))

        mag_db = FftReferences.magnitude_db(cw)
        result.add(ValidationResult(
            passed=mag_db.dtype == np.float32,
            metric_name="magnitude_db_dtype",
            actual_value=1.0 if mag_db.dtype == np.float32 else 0.0,
            threshold=1.0
        ))

        freqs = FftReferences.freq_axis(4096, 12e6)
        result.add(ValidationResult(
            passed=freqs.shape == (4096,),
            metric_name="freq_axis_shape",
            actual_value=1.0 if freqs.shape == (4096,) else 0.0,
            threshold=1.0
        ))

        return result

    def test_filter_refs(self):
        result = TestResult(test_name="filter_refs")
        try:
            from common.references import FilterReferences
        except ImportError:
            result.add(ValidationResult(
                passed=True,
                metric_name="scipy_skip",
                actual_value=0.0,
                threshold=1.0,
                message="scipy не установлен -- пропуск"
            ))
            return result

        data = np.random.randn(1024).astype(np.complex64)
        filtered = FilterReferences.fir_lowpass(data, fs=50e3, cutoff_hz=1e3, n_taps=64)
        result.add(ValidationResult(
            passed=filtered.dtype == np.complex64 and filtered.shape == data.shape,
            metric_name="fir_lowpass_shape",
            actual_value=1.0 if filtered.shape == data.shape else 0.0,
            threshold=1.0,
            message=f"shape={filtered.shape}"
        ))

        iir_out = FilterReferences.iir_lowpass(data, fs=50e3, cutoff_hz=1e3)
        result.add(ValidationResult(
            passed=iir_out.dtype == np.complex64,
            metric_name="iir_lowpass_dtype",
            actual_value=1.0 if iir_out.dtype == np.complex64 else 0.0,
            threshold=1.0
        ))

        return result


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestReferencesSmoke())
    runner.print_summary(results)
