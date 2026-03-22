"""
test_fft_integration.py — интеграционные тесты FFT + SignalGenerator
======================================================================

Тесты 1-3 из оригинального test_gpuworklib.py.

Tests:
  test_multichannel_cw_fft     — CW разных частот → FFT → пик в нужном месте
  test_multichannel_lfm_fft    — LFM → FFT → рассеяный спектр (нет острого пика)
  test_noise_fft_flat_spectrum — Noise → FFT → равномерный спектр

Запуск:
  python Python_test/integration/test_fft_integration.py
  PYTHONPATH=build/python python Python_test/integration/test_fft_integration.py
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
from integration.conftest import make_sig_gen, make_fft_proc


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: CW → FFT → peak at expected frequency
# ─────────────────────────────────────────────────────────────────────────────

class TestCwFftIntegration:
    """CW + FFT: пик спектра на заданной частоте."""

    def setUp(self):
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("GPU context недоступен")
        self._sig_gen = make_sig_gen(gw, ctx)
        self._fft_proc = make_fft_proc(gw, ctx)

    def test_cw_peak_frequency(self):
        """FFT пик CW-сигнала должен быть на частоте f0 ± 1 бин для нескольких частот."""
        fs = 4000.0
        length = 4096

        for f0_hz in [100, 250, 500, 800, 1200]:
            signal = self._sig_gen.generate_cw(freq=f0_hz, fs=fs, length=length)
            spectrum = self._fft_proc.process_complex(signal, sample_rate=fs)

            nfft = len(spectrum)
            freq_axis = np.arange(nfft) * fs / nfft
            mag = np.abs(spectrum)
            peak_idx = int(np.argmax(mag[:nfft // 2]))
            peak_freq = freq_axis[peak_idx]

            bin_size = fs / nfft
            assert abs(peak_freq - f0_hz) <= bin_size * 1.5, (
                f"CW f0={f0_hz} Hz: peak at {peak_freq:.1f} Hz "
                f"(bin_size={bin_size:.2f} Hz)"
            )

    def test_cw_output_shape(self):
        """FFT выходной массив имеет правильную форму."""
        length = 4096
        signal = self._sig_gen.generate_cw(freq=100, fs=4000, length=length)
        spectrum = self._fft_proc.process_complex(signal, sample_rate=4000)
        assert len(spectrum) == length

    def test_cw_signal_complex(self):
        """CW-сигнал комплексный."""
        signal = self._sig_gen.generate_cw(freq=100, fs=4000, length=512)
        arr = np.asarray(signal)
        assert np.iscomplexobj(arr), "CW signal should be complex"

    def test_cw_energy_nonzero(self):
        """CW → FFT: энергия ненулевая."""
        signal = self._sig_gen.generate_cw(freq=500, fs=4000, length=4096)
        spectrum = self._fft_proc.process_complex(signal, sample_rate=4000)
        total_energy = np.sum(np.abs(np.asarray(spectrum)) ** 2)
        assert total_energy > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: LFM → FFT
# ─────────────────────────────────────────────────────────────────────────────

class TestLfmFftIntegration:
    """LFM + FFT: нет острого единственного пика (рассеянный спектр)."""

    def setUp(self):
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("GPU context недоступен")
        self._sig_gen = make_sig_gen(gw, ctx)
        self._fft_proc = make_fft_proc(gw, ctx)

    def test_lfm_spread_spectrum(self):
        """LFM спектр рассеян по полосе — нет доминирующего одного пика."""
        fs = 4000.0
        length = 4096

        signal = self._sig_gen.generate_lfm(
            f_start=100, f_end=1000, fs=fs, length=length
        )
        spectrum = self._fft_proc.process_complex(signal, sample_rate=fs)
        mag = np.abs(np.asarray(spectrum)[:length // 2])

        peak_energy = float(np.max(mag) ** 2)
        total_energy = float(np.sum(mag ** 2))
        peak_ratio = peak_energy / (total_energy + 1e-30)
        assert peak_ratio < 0.3, (
            f"LFM spectrum too concentrated: peak_ratio={peak_ratio:.3f}"
        )

    def test_lfm_output_not_constant(self):
        """LFM сигнал нестационарный — фаза меняется."""
        fs = 4000.0
        length = 4096
        signal = np.asarray(
            self._sig_gen.generate_lfm(f_start=100, f_end=1000, fs=fs, length=length)
        )
        phases = np.angle(signal)
        phase_diffs = np.diff(np.unwrap(phases))
        diff_std = float(np.std(phase_diffs))
        assert diff_std > 0.001, "LFM phase diff should vary"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Noise → FFT → flat spectrum
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseFftIntegration:
    """Noise + FFT: спектр статистически равномерный."""

    def setUp(self):
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("GPU context недоступен")
        self._sig_gen = make_sig_gen(gw, ctx)
        self._fft_proc = make_fft_proc(gw, ctx)

    def test_noise_flat_spectrum(self):
        """Noise FFT: среднее >> std не выполняется для шума."""
        fs = 4000.0
        length = 4096

        signal = self._sig_gen.generate_noise(fs=fs, length=length)
        spectrum = self._fft_proc.process_complex(signal, sample_rate=fs)
        mag = np.abs(np.asarray(spectrum)[:length // 2])

        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))

        cv = std_mag / (mean_mag + 1e-30)
        assert cv < 1.5, (
            f"Noise spectrum not flat: cv={cv:.3f} (mean={mean_mag:.4f}, std={std_mag:.4f})"
        )

    def test_noise_nonzero_energy(self):
        """Noise: ненулевая энергия в спектре."""
        signal = self._sig_gen.generate_noise(fs=4000, length=4096)
        spectrum = self._fft_proc.process_complex(signal, sample_rate=4000)
        assert np.sum(np.abs(np.asarray(spectrum)) ** 2) > 0


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestCwFftIntegration())
    results += runner.run(TestLfmFftIntegration())
    results += runner.run(TestNoiseFftIntegration())
    runner.print_summary(results)
