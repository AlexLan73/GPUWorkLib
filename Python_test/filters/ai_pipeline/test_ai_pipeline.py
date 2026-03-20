"""
test_ai_pipeline.py — тесты AI filter pipeline
=======================================================

Тестирует весь pipeline: MockParser → FilterDesigner → GPU → валидация.
Не требует AI-бэкенда (Groq/Ollama) — используется MockParser.
Не требует matplotlib — только assert.

Запуск:
    python Python_test/filters/ai_pipeline/test_ai_pipeline.py
    PYTHONPATH=build/python python Python_test/filters/ai_pipeline/test_ai_pipeline.py

Тесты:
    test_mock_parser_lowpass      — парсинг "FIR lowpass"
    test_mock_parser_bandpass     — парсинг "IIR bandpass"
    test_mock_parser_highpass     — парсинг "highpass 5kHz"
    test_mock_parser_freq_kHz     — распознавание кГц
    test_filter_designer_fir      — scipy FIR дизайн
    test_filter_designer_iir      — scipy IIR дизайн
    test_fir_apply_scipy          — FilterDesign.apply_scipy()
    test_iir_apply_scipy          — FilterDesign.apply_scipy() IIR
    test_pipeline_fir_gpu         — FIR: MockParser → Design → GPU (skip если нет GPU)
    test_pipeline_iir_gpu         — IIR: MockParser → Design → GPU (skip если нет GPU)
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner
from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager

from .llm_parser import MockParser, FilterSpec, create_parser
from .filter_designer import FilterDesigner, FilterDesign


def make_noise_signal(n: int = 4096, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# MockParser тесты
# ─────────────────────────────────────────────────────────────────────────────

class TestMockParser:
    """Тесты детерминированного regex-парсера."""

    def setUp(self):
        self._parser = MockParser()

    def test_lowpass_fir(self):
        spec = self._parser.parse("FIR lowpass 1kHz", fs=50_000)
        assert spec.filter_class == "fir"
        assert spec.filter_type == "lowpass"
        assert abs(spec.f_cutoff - 1000.0) < 1.0

    def test_bandpass_iir(self):
        spec = self._parser.parse("IIR bandpass 1kHz 5kHz", fs=50_000)
        assert spec.filter_class == "iir"
        assert spec.filter_type == "bandpass"
        assert isinstance(spec.f_cutoff, list)
        assert len(spec.f_cutoff) == 2
        assert abs(spec.f_cutoff[0] - 1000.0) < 1.0
        assert abs(spec.f_cutoff[1] - 5000.0) < 1.0

    def test_highpass_default_iir(self):
        spec = self._parser.parse("highpass 5kHz", fs=50_000)
        assert spec.filter_type == "highpass"
        assert abs(spec.f_cutoff - 5000.0) < 1.0

    def test_kHz_conversion(self):
        spec = self._parser.parse("FIR lowpass 2.5kHz", fs=50_000)
        assert abs(spec.f_cutoff - 2500.0) < 1.0

    def test_butterworth_keyword(self):
        spec = self._parser.parse("butterworth lowpass 3kHz", fs=50_000)
        assert spec.filter_class == "iir"

    def test_order_parsing(self):
        spec = self._parser.parse("IIR lowpass 1kHz order=6", fs=50_000)
        assert spec.order == 6

    def test_normalized_cutoff(self):
        spec = self._parser.parse("FIR lowpass 5kHz", fs=50_000)
        # Nyquist = 25000, нормированный = 5000/25000 = 0.2
        assert abs(spec.normalized_cutoff() - 0.2) < 1e-6

    def test_create_parser_mock(self):
        p = create_parser("mock")
        assert isinstance(p, MockParser)

    def test_backend_name(self):
        assert self._parser.backend_name == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# FilterDesigner тесты
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterDesigner:
    """Тесты scipy-дизайна фильтров."""

    def setUp(self):
        try:
            import scipy.signal  # noqa: F401
        except ImportError:
            raise SkipTest("scipy required")
        parser = MockParser()
        self._designer = FilterDesigner()
        self._fir_spec = parser.parse("FIR lowpass 1kHz", fs=50_000)
        self._iir_spec = parser.parse("butterworth lowpass 2kHz order=4", fs=50_000)
        self._fir_design = self._designer.design(self._fir_spec)
        self._iir_design = self._designer.design(self._iir_spec)
        self._noise = make_noise_signal()

    def test_fir_design_basic(self):
        assert self._fir_design.is_fir
        assert self._fir_design.method == "firwin"
        assert len(self._fir_design.coeffs_b) > 0
        assert abs(self._fir_design.coeffs_a[0] - 1.0) < 1e-9

    def test_fir_n_taps(self):
        assert self._fir_design.n_taps >= 32

    def test_fir_coeffs_symmetric(self):
        """FIR LP должен быть линейно-фазовым (симметричные коэффициенты)."""
        b = self._fir_design.coeffs_b
        np.testing.assert_allclose(b, b[::-1], atol=1e-10,
                                   err_msg="FIR coefficients should be symmetric")

    def test_iir_design_basic(self):
        assert not self._iir_design.is_fir
        assert self._iir_design.method == "butter"
        assert self._iir_design.sos is not None
        assert self._iir_design.sos.shape[1] == 6  # SOS: [b0,b1,b2,a0,a1,a2]

    def test_fir_apply_scipy_shape(self):
        out = self._fir_design.apply_scipy(self._noise)
        assert out.shape == self._noise.shape
        assert out.dtype == self._noise.dtype

    def test_iir_apply_scipy_shape(self):
        out = self._iir_design.apply_scipy(self._noise)
        assert out.shape == self._noise.shape

    def test_fir_lowpass_attenuates_high_freq(self):
        """FIR lowpass должен ослаблять высокие частоты."""
        parser = MockParser()
        spec = parser.parse("FIR lowpass 1kHz", fs=50_000)
        design = self._designer.design(spec)

        fs = 50_000.0
        n = 8192
        t = np.arange(n) / fs

        sig = (np.cos(2 * np.pi * 500 * t) +
               np.cos(2 * np.pi * 10_000 * t)).astype(np.complex64)

        out = design.apply_scipy(sig)
        spec_in = np.abs(np.fft.rfft(sig))
        spec_out = np.abs(np.fft.rfft(out))

        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        idx_pass = np.argmin(np.abs(freqs - 500))
        idx_stop = np.argmin(np.abs(freqs - 10_000))

        pass_ratio = spec_out[idx_pass] / (spec_in[idx_pass] + 1e-10)
        assert pass_ratio > 0.9, f"Pass band too attenuated: {pass_ratio:.3f}"

        stop_ratio = spec_out[idx_stop] / (spec_in[idx_stop] + 1e-10)
        assert stop_ratio < 0.1, f"Stop band not attenuated enough: {stop_ratio:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# GPU integration тесты (skip если нет GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestGPUPipeline:
    """Тесты полного pipeline: ParseSpec → Design → GPU → Validate."""

    def setUp(self):
        try:
            import scipy.signal  # noqa: F401
        except ImportError:
            raise SkipTest("scipy required")
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCm недоступен")
        if not hasattr(gw, "FirFilterROCm"):
            raise SkipTest("FirFilterROCm не доступен")
        self._gw = gw
        self._ctx = ctx
        parser = MockParser()
        designer = FilterDesigner()
        self._fir_design = designer.design(parser.parse("FIR lowpass 1kHz", fs=50_000))
        self._iir_design = designer.design(
            parser.parse("butterworth lowpass 2kHz order=4", fs=50_000)
        )
        self._noise = make_noise_signal()

    def test_fir_gpu_vs_scipy(self):
        """FIR: GPU-результат совпадает с scipy с точностью float32."""
        gw = self._gw
        coeffs = self._fir_design.coeffs_b.astype(np.float32).tolist()
        fir_gpu = gw.FirFilterROCm(self._ctx, coeffs)
        gpu_out = fir_gpu.process(self._noise)

        ref = self._fir_design.apply_scipy(self._noise)

        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4, f"FIR GPU/scipy mismatch: rel_err={rel_err:.2e}"

    def test_iir_gpu_vs_scipy(self):
        """IIR: GPU-результат совпадает с scipy."""
        gw = self._gw
        if not hasattr(gw, "IirFilterROCm"):
            raise SkipTest("IirFilterROCm не доступен")
        b = self._iir_design.coeffs_b.astype(np.float64).tolist()
        a = self._iir_design.coeffs_a.astype(np.float64).tolist()
        iir_gpu = gw.IirFilterROCm(self._ctx, b, a)
        gpu_out = iir_gpu.process(self._noise)

        ref = self._iir_design.apply_scipy(self._noise)

        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4, f"IIR GPU/scipy mismatch: rel_err={rel_err:.2e}"

    def test_full_pipeline_mock(self):
        """Полный pipeline: текст → MockParser → Design → GPU."""
        gw = self._gw
        parser = MockParser()
        designer = FilterDesigner()

        spec = parser.parse("FIR lowpass 2kHz", fs=50_000)
        design = designer.design(spec)

        n = 4096
        rng = np.random.default_rng(0)
        signal = (rng.standard_normal(n) +
                  1j * rng.standard_normal(n)).astype(np.complex64)

        coeffs = design.coeffs_b.astype(np.float32).tolist()
        fir_gpu = gw.FirFilterROCm(self._ctx, coeffs)
        gpu_out = fir_gpu.process(signal)
        ref = design.apply_scipy(signal)

        assert gpu_out.shape == signal.shape
        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestMockParser())
    results += runner.run(TestFilterDesigner())
    results += runner.run(TestGPUPipeline())
    runner.print_summary(results)
