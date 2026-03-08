"""
test_ai_pipeline.py — pytest тесты AI filter pipeline
=======================================================

Тестирует весь pipeline: MockParser → FilterDesigner → GPU → валидация.
Не требует AI-бэкенда (Groq/Ollama) — используется MockParser.
Не требует matplotlib — только assert.

Запуск:
    pytest Python_test/filters/ai_pipeline/test_ai_pipeline.py -v
    pytest Python_test/filters/ai_pipeline/test_ai_pipeline.py -v -k "fir"

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

import numpy as np
import pytest

from .llm_parser import MockParser, FilterSpec, create_parser
from .filter_designer import FilterDesigner, FilterDesign


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def parser():
    return MockParser()


@pytest.fixture(scope="module")
def designer():
    return FilterDesigner()


@pytest.fixture(scope="module")
def fir_spec(parser):
    return parser.parse("FIR lowpass 1kHz", fs=50_000)


@pytest.fixture(scope="module")
def iir_spec(parser):
    return parser.parse("butterworth lowpass 2kHz order=4", fs=50_000)


@pytest.fixture(scope="module")
def fir_design(designer, fir_spec):
    return designer.design(fir_spec)


@pytest.fixture(scope="module")
def iir_design(designer, iir_spec):
    return designer.design(iir_spec)


@pytest.fixture
def noise_signal() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# MockParser тесты
# ─────────────────────────────────────────────────────────────────────────────

class TestMockParser:
    """Тесты детерминированного regex-парсера."""

    def test_lowpass_fir(self, parser):
        spec = parser.parse("FIR lowpass 1kHz", fs=50_000)
        assert spec.filter_class == "fir"
        assert spec.filter_type == "lowpass"
        assert spec.f_cutoff == pytest.approx(1000.0)

    def test_bandpass_iir(self, parser):
        spec = parser.parse("IIR bandpass 1kHz 5kHz", fs=50_000)
        assert spec.filter_class == "iir"
        assert spec.filter_type == "bandpass"
        assert isinstance(spec.f_cutoff, list)
        assert len(spec.f_cutoff) == 2
        assert spec.f_cutoff[0] == pytest.approx(1000.0)
        assert spec.f_cutoff[1] == pytest.approx(5000.0)

    def test_highpass_default_iir(self, parser):
        spec = parser.parse("highpass 5kHz", fs=50_000)
        assert spec.filter_type == "highpass"
        assert spec.f_cutoff == pytest.approx(5000.0)

    def test_kHz_conversion(self, parser):
        spec = parser.parse("FIR lowpass 2.5kHz", fs=50_000)
        assert spec.f_cutoff == pytest.approx(2500.0)

    def test_butterworth_keyword(self, parser):
        spec = parser.parse("butterworth lowpass 3kHz", fs=50_000)
        assert spec.filter_class == "iir"

    def test_order_parsing(self, parser):
        spec = parser.parse("IIR lowpass 1kHz order=6", fs=50_000)
        assert spec.order == 6

    def test_normalized_cutoff(self, parser):
        spec = parser.parse("FIR lowpass 5kHz", fs=50_000)
        # Nyquist = 25000, нормированный = 5000/25000 = 0.2
        assert spec.normalized_cutoff() == pytest.approx(0.2)

    def test_create_parser_mock(self):
        p = create_parser("mock")
        assert isinstance(p, MockParser)

    def test_backend_name(self, parser):
        assert parser.backend_name == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# FilterDesigner тесты
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterDesigner:
    """Тесты scipy-дизайна фильтров."""

    scipy_signal = pytest.importorskip("scipy.signal",
                                        reason="scipy required")

    def test_fir_design_basic(self, fir_design):
        assert fir_design.is_fir
        assert fir_design.method == "firwin"
        assert len(fir_design.coeffs_b) > 0
        assert fir_design.coeffs_a[0] == pytest.approx(1.0)

    def test_fir_n_taps(self, fir_design):
        assert fir_design.n_taps >= 32

    def test_fir_coeffs_symmetric(self, fir_design):
        """FIR LP должен быть линейно-фазовым (симметричные коэффициенты)."""
        b = fir_design.coeffs_b
        np.testing.assert_allclose(b, b[::-1], atol=1e-10,
                                   err_msg="FIR coefficients should be symmetric")

    def test_iir_design_basic(self, iir_design):
        assert not iir_design.is_fir
        assert iir_design.method == "butter"
        assert iir_design.sos is not None
        assert iir_design.sos.shape[1] == 6  # SOS: [b0,b1,b2,a0,a1,a2]

    def test_fir_apply_scipy_shape(self, fir_design, noise_signal):
        out = fir_design.apply_scipy(noise_signal)
        assert out.shape == noise_signal.shape
        assert out.dtype == noise_signal.dtype

    def test_iir_apply_scipy_shape(self, iir_design, noise_signal):
        out = iir_design.apply_scipy(noise_signal)
        assert out.shape == noise_signal.shape

    def test_fir_lowpass_attenuates_high_freq(self, designer, parser):
        """FIR lowpass должен ослаблять высокие частоты."""
        spec = parser.parse("FIR lowpass 1kHz", fs=50_000)
        design = designer.design(spec)

        fs = 50_000.0
        n = 8192
        t = np.arange(n) / fs

        # Сигнал: 500 Hz (в полосе) + 10 kHz (вне полосы)
        sig = (np.cos(2 * np.pi * 500 * t) +
               np.cos(2 * np.pi * 10_000 * t)).astype(np.complex64)

        out = design.apply_scipy(sig)
        spec_in = np.abs(np.fft.rfft(sig))
        spec_out = np.abs(np.fft.rfft(out))

        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        # Индексы для 500 Hz и 10 kHz
        idx_pass = np.argmin(np.abs(freqs - 500))
        idx_stop = np.argmin(np.abs(freqs - 10_000))

        # Полоса пропускания — мало ослабления
        pass_ratio = spec_out[idx_pass] / (spec_in[idx_pass] + 1e-10)
        assert pass_ratio > 0.9, f"Pass band too attenuated: {pass_ratio:.3f}"

        # Полоса задерживания — сильное ослабление
        stop_ratio = spec_out[idx_stop] / (spec_in[idx_stop] + 1e-10)
        assert stop_ratio < 0.1, f"Stop band not attenuated enough: {stop_ratio:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# GPU integration тесты (skip если нет GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestGPUPipeline:
    """Тесты полного pipeline: ParseSpec → Design → GPU → Validate.

    Skip если gpuworklib не найден.
    """

    def test_fir_gpu_vs_scipy(self, gpu_ctx, fir_design, noise_signal):
        """FIR: GPU-результат совпадает с scipy с точностью float32."""
        gw = pytest.importorskip("gpuworklib")  # noqa (handled by fixture)

        # GPU обработка
        coeffs = fir_design.coeffs_b.astype(np.float32).tolist()
        fir_gpu = gw.FirFilterROCm(gpu_ctx, coeffs)
        gpu_out = fir_gpu.process(noise_signal)

        # scipy эталон
        ref = fir_design.apply_scipy(noise_signal)

        # Проверка точности float32
        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4, f"FIR GPU/scipy mismatch: rel_err={rel_err:.2e}"

    def test_iir_gpu_vs_scipy(self, gpu_ctx, iir_design, noise_signal):
        """IIR: GPU-результат совпадает с scipy."""
        gw = pytest.importorskip("gpuworklib")  # noqa

        b = iir_design.coeffs_b.astype(np.float64).tolist()
        a = iir_design.coeffs_a.astype(np.float64).tolist()
        iir_gpu = gw.IirFilterROCm(gpu_ctx, b, a)
        gpu_out = iir_gpu.process(noise_signal)

        ref = iir_design.apply_scipy(noise_signal)

        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4, f"IIR GPU/scipy mismatch: rel_err={rel_err:.2e}"

    def test_full_pipeline_mock(self, gpu_ctx):
        """Полный pipeline: текст → MockParser → Design → GPU."""
        gw = pytest.importorskip("gpuworklib")  # noqa

        parser = MockParser()
        designer = FilterDesigner()

        spec = parser.parse("FIR lowpass 2kHz", fs=50_000)
        design = designer.design(spec)

        n = 4096
        rng = np.random.default_rng(0)
        signal = (rng.standard_normal(n) +
                  1j * rng.standard_normal(n)).astype(np.complex64)

        coeffs = design.coeffs_b.astype(np.float32).tolist()
        fir_gpu = gw.FirFilterROCm(gpu_ctx, coeffs)
        gpu_out = fir_gpu.process(signal)
        ref = design.apply_scipy(signal)

        assert gpu_out.shape == signal.shape
        rel_err = np.max(np.abs(gpu_out - ref)) / (np.max(np.abs(ref)) + 1e-10)
        assert rel_err < 1e-4
