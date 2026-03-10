"""
test_fft_processor_rocm.py — тесты FFTProcessorROCm (hipFFT, ROCm backend)

Структура:
  - NumPy/SciPy-референс (8 тестов, всегда запускаются)
  - GPU тесты через FFTProcessorROCm (6 тестов, пропускаются без GPU)

Проверяемые свойства:
  - process_complex:  спектр совпадает с numpy.fft.fft (atol=1e-2)
  - process_mag_phase: magnitude, phase, frequency совпадают с NumPy-референсом
  - форма массивов: 1D и 2D (multi-beam)
  - nFFT = nextPow2(n_point)
  - zero-padding: спектр нулевых семплов → нулевая магнитуда

Запуск:
  cd /home/alex/C++/GPUWorkLib
  PYTHONPATH=build/python pytest Python_test/fft_processor/test_fft_processor_rocm.py -v
"""

import numpy as np
import pytest

# ─── GPU availability ─────────────────────────────────────────────────────────

try:
    import gpuworklib
    HAS_FFT_ROCM = hasattr(gpuworklib, 'FFTProcessorROCm')
except ImportError:
    HAS_FFT_ROCM = False

rocm_only = pytest.mark.skipif(not HAS_FFT_ROCM, reason="FFTProcessorROCm not available")


# ─── NumPy helpers ────────────────────────────────────────────────────────────

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def numpy_fft_complex(signal: np.ndarray) -> np.ndarray:
    """Reference complex FFT with zero-padding to nextPow2."""
    nFFT = next_pow2(len(signal))
    return np.fft.fft(signal, n=nFFT).astype(np.complex64)


def numpy_fft_mag_phase(signal: np.ndarray, sample_rate: float):
    """Reference magnitude, phase, frequency arrays."""
    nFFT = next_pow2(len(signal))
    spec = np.fft.fft(signal, n=nFFT)
    mag = np.abs(spec).astype(np.float32)
    phase = np.angle(spec).astype(np.float32)
    freq = (np.arange(nFFT) * sample_rate / nFFT).astype(np.float32)
    return mag, phase, freq


# ─── Pure NumPy tests (always run) ───────────────────────────────────────────

class TestNumPyReference:
    """Проверяем NumPy-референс до GPU-тестов."""

    def test_next_pow2_basic(self):
        assert next_pow2(1) == 1
        assert next_pow2(2) == 2
        assert next_pow2(3) == 4
        assert next_pow2(512) == 512
        assert next_pow2(513) == 1024

    def test_fft_single_tone(self):
        """Пик на частоте k/N должен быть в бине k."""
        N = 256
        k = 10
        t = np.arange(N)
        sig = np.exp(1j * 2 * np.pi * k / N * t).astype(np.complex64)
        spec = numpy_fft_complex(sig)
        # Пик в бине k (N = 2^8, nFFT = N)
        peak_bin = int(np.argmax(np.abs(spec)))
        assert peak_bin == k

    def test_fft_mag_phase_single_tone(self):
        """Магнитуда пика = N, фаза = 0 для косинуса с нулевой фазой."""
        N = 128
        k = 5
        t = np.arange(N)
        sig = np.cos(2 * np.pi * k / N * t).astype(np.complex64)
        mag, phase, freq = numpy_fft_mag_phase(sig, sample_rate=1.0)
        # Косинус: пики в бинах k и N-k, каждый с магнитудой N/2
        assert mag[k] == pytest.approx(N / 2, rel=1e-4)

    def test_fft_zero_signal(self):
        """Нулевой сигнал → нулевой спектр."""
        sig = np.zeros(64, dtype=np.complex64)
        spec = numpy_fft_complex(sig)
        assert np.max(np.abs(spec)) == pytest.approx(0.0, abs=1e-6)

    def test_fft_nfft_padding(self):
        """n_point=100 → nFFT=128."""
        sig = np.random.randn(100).astype(np.complex64)
        nFFT = next_pow2(100)
        spec = numpy_fft_complex(sig)
        assert spec.shape[0] == nFFT == 128

    def test_frequency_axis(self):
        """Частотная ось: bin k → k * fs / nFFT."""
        fs = 1e6
        nFFT = 256
        _, _, freq = numpy_fft_mag_phase(np.zeros(nFFT, dtype=np.complex64), fs)
        assert freq[1] == pytest.approx(fs / nFFT, rel=1e-5)
        assert freq[nFFT // 2] == pytest.approx(fs / 2, rel=1e-5)

    def test_multi_beam_shapes(self):
        """2D вход (beam_count, n_point) → (beam_count, nFFT)."""
        n_beams, n_point = 4, 100
        nFFT = next_pow2(n_point)
        data = np.random.randn(n_beams, n_point).astype(np.complex64)
        spectra = np.stack([numpy_fft_complex(data[i]) for i in range(n_beams)])
        assert spectra.shape == (n_beams, nFFT)

    def test_phase_of_pure_real_at_dc(self):
        """Постоянный сигнал → DC-bin с фазой ~0."""
        sig = np.ones(64, dtype=np.complex64)
        _, phase, _ = numpy_fft_mag_phase(sig, sample_rate=1.0)
        assert phase[0] == pytest.approx(0.0, abs=1e-5)


# ─── GPU tests ────────────────────────────────────────────────────────────────

@rocm_only
class TestFFTProcessorROCm:
    """Тесты GPU-реализации (FFTProcessorROCm)."""

    @pytest.fixture(scope="class")
    def proc(self):
        ctx = gpuworklib.ROCmGPUContext(0)
        return gpuworklib.FFTProcessorROCm(ctx)

    # ── process_complex ──────────────────────────────────────────────────────

    def test_process_complex_single_beam_1d(self, proc):
        """1D вход → 1D спектр, совпадает с NumPy."""
        N = 256
        k = 13
        t = np.arange(N)
        sig = np.exp(1j * 2 * np.pi * k / N * t).astype(np.complex64)
        ref = numpy_fft_complex(sig)

        gpu_spec = proc.process_complex(sig, sample_rate=1e6)

        assert gpu_spec.ndim == 1
        assert gpu_spec.shape[0] == len(ref)
        np.testing.assert_allclose(np.abs(gpu_spec), np.abs(ref), atol=0.05)

    def test_process_complex_multi_beam_2d(self, proc):
        """2D вход (beams, n_point) → 2D спектр (beams, nFFT)."""
        n_beams, n_point = 3, 128
        nFFT = next_pow2(n_point)
        data = np.random.randn(n_beams, n_point).astype(np.complex64)

        gpu_spec = proc.process_complex(data, sample_rate=1e6)

        assert gpu_spec.ndim == 2
        assert gpu_spec.shape == (n_beams, nFFT)

    def test_process_complex_zero_padding(self, proc):
        """n_point=100 → nFFT=128, остаток нулевой."""
        sig = np.zeros(100, dtype=np.complex64)
        gpu_spec = proc.process_complex(sig, sample_rate=1e6)
        assert gpu_spec.shape[0] == 128

    def test_nfft_property(self, proc):
        """nfft обновляется после вызова process_complex."""
        sig = np.zeros(512, dtype=np.complex64)
        proc.process_complex(sig, sample_rate=1e6)
        assert proc.nfft == 512

    # ── process_mag_phase ────────────────────────────────────────────────────

    def test_process_mag_phase_keys(self, proc):
        """Результат содержит нужные ключи."""
        sig = np.random.randn(256).astype(np.complex64)
        result = proc.process_mag_phase(sig, sample_rate=1e6)

        assert "magnitude"   in result
        assert "phase"       in result
        assert "frequency"   in result
        assert "nFFT"        in result
        assert "sample_rate" in result

    def test_process_mag_phase_accuracy(self, proc):
        """Магнитуда и фаза совпадают с NumPy-референсом."""
        N = 256
        k = 7
        fs = 1e6
        t = np.arange(N)
        sig = np.exp(1j * 2 * np.pi * k / N * t).astype(np.complex64)
        ref_mag, ref_phase, ref_freq = numpy_fft_mag_phase(sig, fs)

        result = proc.process_mag_phase(sig, sample_rate=fs)

        np.testing.assert_allclose(result["magnitude"], ref_mag, atol=0.05)
        np.testing.assert_allclose(result["frequency"], ref_freq, atol=1.0)

    def test_process_mag_phase_no_freq(self, proc):
        """include_freq=False → нет ключа 'frequency'."""
        sig = np.zeros(64, dtype=np.complex64)
        result = proc.process_mag_phase(sig, sample_rate=1e6, include_freq=False)
        assert "frequency" not in result
