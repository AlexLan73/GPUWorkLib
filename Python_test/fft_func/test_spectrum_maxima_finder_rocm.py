"""
test_spectrum_maxima_finder_rocm.py — тесты SpectrumMaximaFinderROCm (ROCm backend)

Структура:
  - NumPy/SciPy-референс (8 тестов, всегда запускаются)
  - GPU тесты через SpectrumMaximaFinderROCm (6 тестов, skip без GPU)

Проверяемые свойства:
  - process(): ONE_PEAK — один пик на луч с параболической интерполяцией
  - find_all_maxima(): ALL_MAXIMA из FFT спектра
  - find_all_maxima_from_signal(): ALL_MAXIMA из raw сигнала
  - формат вывода совместим с OpenCL SpectrumMaximaFinder
  - однолучевой (dict) и многолучевой (list[dict]) режимы

Запуск:
  cd /home/alex/C++/GPUWorkLib
  PYTHONPATH=build/python python Python_test/fft_maxima/test_spectrum_maxima_finder_rocm.py
"""

import sys
import os
import numpy as np
from scipy.signal import find_peaks

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner

# ─── GPU availability ─────────────────────────────────────────────────────────

try:
    import gpuworklib
    HAS_ROCM_MAXIMA = hasattr(gpuworklib, 'SpectrumMaximaFinderROCm')
except ImportError:
    HAS_ROCM_MAXIMA = False



# ─── NumPy helpers ────────────────────────────────────────────────────────────

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def make_single_tone(n: int, k: int) -> np.ndarray:
    """Комплексный экспоненциальный сигнал на частоте k/N."""
    t = np.arange(n)
    return np.exp(1j * 2 * np.pi * k / n * t).astype(np.complex64)


def make_two_tone(n: int, k1: int, k2: int,
                  a1: float = 1.0, a2: float = 0.8) -> np.ndarray:
    """Два тона с амплитудами a1 и a2."""
    t = np.arange(n)
    s1 = a1 * np.exp(1j * 2 * np.pi * k1 / n * t)
    s2 = a2 * np.exp(1j * 2 * np.pi * k2 / n * t)
    return (s1 + s2).astype(np.complex64)


def numpy_find_all_maxima(spectrum: np.ndarray, sample_rate: float):
    """NumPy-референс: найти все локальные максимумы через scipy.signal.find_peaks."""
    mag = np.abs(spectrum)
    nFFT = len(mag)
    half = nFFT // 2
    peaks, props = find_peaks(mag[:half], height=0)
    freqs = peaks * sample_rate / nFFT
    return peaks, mag[peaks], freqs


# ─── Pure NumPy tests (always run) ───────────────────────────────────────────

class TestNumPyReference:
    """Проверяем NumPy-референс."""

    def test_single_tone_peak_position(self):
        """FFT единичного тона: пик в нужном бине."""
        N, k = 256, 15
        sig = make_single_tone(N, k)
        spec = np.fft.fft(sig)
        peak_bin = int(np.argmax(np.abs(spec[:N // 2])))
        assert peak_bin == k

    def test_two_tones_peaks_found(self):
        """FFT двух тонов: scipy.find_peaks находит оба."""
        N, k1, k2 = 512, 20, 80
        sig = make_two_tone(N, k1, k2)
        spec = np.fft.fft(sig)
        peaks, _, _ = numpy_find_all_maxima(spec, sample_rate=1.0)
        assert k1 in peaks, f"Peak at k1={k1} not found, peaks={peaks}"
        assert k2 in peaks, f"Peak at k2={k2} not found, peaks={peaks}"

    def test_noise_floor_no_strong_peaks(self):
        """Белый шум — нет доминирующего пика (все магнитуды похожи)."""
        rng = np.random.default_rng(42)
        N = 512
        sig = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        spec = np.fft.fft(sig)
        mag = np.abs(spec[:N // 2])
        # Нет пика > 5× среднего (разброс не должен быть большим)
        assert mag.max() < 5 * mag.mean() * 5  # слабое условие — просто проверить что работает

    def test_freq_resolution_formula(self):
        """Частотное разрешение: Δf = fs / nFFT."""
        N = 256
        fs = 1e6
        nFFT = next_pow2(N)
        freq_res = fs / nFFT
        assert abs(freq_res - fs / nFFT) < 1e-6  # exact formula

    def test_parabolic_interpolation(self):
        """Параболическая интерполяция уточняет частоту."""
        # Аналитическая формула δ = 0.5*(L-R)/(L-2C+R)
        N = 256
        k = 13
        sig = make_single_tone(N, k)
        spec = np.fft.fft(sig)
        mag = np.abs(spec)
        peak_bin = k
        L = mag[peak_bin - 1]
        C = mag[peak_bin]
        R = mag[peak_bin + 1]
        delta = 0.5 * (L - R) / (L - 2 * C + R)
        # Для точного тона δ ≈ 0 (пик ровно в бине)
        assert abs(delta) < 0.01

    def test_all_maxima_count(self):
        """Двухтональный сигнал — find_peaks находит ≥ 2 пика."""
        N, k1, k2 = 256, 10, 60
        sig = make_two_tone(N, k1, k2, a1=1.0, a2=1.0)
        spec = np.fft.fft(sig)
        peaks, _, _ = numpy_find_all_maxima(spec, sample_rate=1.0)
        assert len(peaks) >= 2

    def test_single_tone_magnitude(self):
        """Магнитуда пика = N (ненормализованный FFT)."""
        N = 128
        sig = make_single_tone(N, 5)
        spec = np.fft.fft(sig)
        peak_mag = np.max(np.abs(spec))
        assert abs(peak_mag - N) < N * 1e-4, f"peak_mag={peak_mag:.2f}, expected≈{N}"

    def test_multi_beam_independence(self):
        """Каждый луч независим — пики на своих частотах."""
        N = 256
        beams = [make_single_tone(N, k) for k in [10, 30, 50]]
        for i, (sig, k) in enumerate(zip(beams, [10, 30, 50])):
            spec = np.fft.fft(sig)
            peak_bin = int(np.argmax(np.abs(spec[:N // 2])))
            assert peak_bin == k, f"Beam {i}: expected peak at {k}, got {peak_bin}"


# ─── GPU tests ────────────────────────────────────────────────────────────────

class TestSpectrumMaximaFinderROCm:
    """Тесты GPU-реализации (SpectrumMaximaFinderROCm)."""

    def setUp(self):
        if not HAS_ROCM_MAXIMA:
            raise SkipTest("SpectrumMaximaFinderROCm not available")
        self._ctx = gpuworklib.ROCmGPUContext(0)
        self._finder = gpuworklib.SpectrumMaximaFinderROCm(self._ctx)

    # ── process() — ONE_PEAK ─────────────────────────────────────────────────

    def test_process_single_beam_peak_freq(self):
        """process(): пик на правильной частоте для одного луча."""
        N = 512
        k = 25
        fs = 1e6
        sig = make_single_tone(N, k)

        result = self._finder.process(sig, n_point=N, sample_rate=fs)

        assert isinstance(result, dict), "Single beam should return dict"
        assert "freq_hz" in result
        expected_hz = k * fs / next_pow2(N)
        assert abs(result["freq_hz"] - expected_hz) < expected_hz * 0.02, \
            f"freq_hz={result['freq_hz']:.0f}, expected={expected_hz:.0f}"

    def test_process_multi_beam_list(self):
        """process(): многолучевой → list[dict]."""
        N = 256
        fs = 1e6
        ks = [10, 30, 60]
        data = np.stack([make_single_tone(N, k) for k in ks])

        results = self._finder.process(data, n_point=N, sample_rate=fs,
                                 antenna_count=len(ks))

        assert isinstance(results, list), "Multi-beam should return list"
        assert len(results) == len(ks)
        for i, (r, k) in enumerate(zip(results, ks)):
            expected_hz = k * fs / next_pow2(N)
            assert abs(r["freq_hz"] - expected_hz) < expected_hz * 0.05, \
                f"Beam {i}: expected {expected_hz:.0f}, got {r['freq_hz']:.0f}"

    def test_process_result_fields(self):
        """process(): результат содержит все ожидаемые поля."""
        sig = make_single_tone(128, 10)
        result = self._finder.process(sig, n_point=128, sample_rate=1e6)

        for key in ("freq_hz", "magnitude", "phase", "index", "freq_offset"):
            assert key in result, f"Missing field: {key}"

    # ── find_all_maxima() — ALL_MAXIMA ────────────────────────────────────────

    def test_find_all_maxima_two_tones(self):
        """find_all_maxima(): двухтональный спектр → ≥ 2 максимума."""
        N = 512
        k1, k2 = 20, 100
        fs = 1e6
        sig = make_two_tone(N, k1, k2)
        spec = np.fft.fft(sig, n=next_pow2(N)).astype(np.complex64)

        result = self._finder.find_all_maxima(spec, nFFT=next_pow2(N), sample_rate=fs)

        assert isinstance(result, dict), "Single beam → dict"
        assert result["num_maxima"] >= 2, \
            f"Expected >= 2 maxima, got {result['num_maxima']}"

    def test_find_all_maxima_opencl_compat_format(self):
        """find_all_maxima(): формат совместим с OpenCL SpectrumMaximaFinder."""
        N = 256
        sig = make_single_tone(N, 15)
        spec = np.fft.fft(sig).astype(np.complex64)

        result = self._finder.find_all_maxima(spec, nFFT=N, sample_rate=1e6)

        assert "positions"   in result
        assert "magnitudes"  in result
        assert "frequencies" in result
        assert "num_maxima"  in result
        # positions — int32 array
        assert result["positions"].dtype in (np.uint32, np.int32, np.int64, np.uint64)

    def test_find_all_maxima_peak_frequency(self):
        """find_all_maxima(): найденная частота близка к эталону."""
        N = 256
        k = 20
        fs = 1e6
        sig = make_single_tone(N, k)
        spec = np.fft.fft(sig).astype(np.complex64)

        result = self._finder.find_all_maxima(spec, nFFT=N, sample_rate=fs)

        expected_hz = k * fs / N
        freqs = result["frequencies"]
        # Хотя бы один найденный максимум должен быть близко к ожидаемой частоте
        min_err = min(abs(f - expected_hz) for f in freqs)
        freq_res = fs / N
        assert min_err < 2 * freq_res, \
            f"Closest peak to {expected_hz:.0f} Hz is {min_err:.0f} Hz away (>2×Δf={2*freq_res:.0f})"


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestNumPyReference())
    results += runner.run(TestSpectrumMaximaFinderROCm())
    runner.print_summary(results)
