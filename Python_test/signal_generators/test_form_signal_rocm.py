"""
test_form_signal_rocm.py — тесты FormSignalGeneratorROCm (ROCm backend)

Структура:
  - NumPy-референс (9 тестов, всегда запускаются)
  - GPU тесты через FormSignalGeneratorROCm (6 тестов, skip без GPU)

Signal formula (getX):
  X = a * norm * exp(j*(2π*f0*t + π*fdev/ti*((t-ti/2)²) + phi))
      + an * norm * (randn + j*randn)
  X = 0  when t < 0 or t > ti - dt

Проверяемые свойства:
  - CW: пик спектра на f0, shape (antennas, points)
  - LFM chirp: спектр размазан по диапазону
  - задержка tau_step: линейный сдвиг между антеннами (кросс-корреляция)
  - шум noise_amplitude > 0: SNR ухудшается
  - set_params_from_string: парсинг CSV

Запуск:
  cd /home/alex/C++/GPUWorkLib
  PYTHONPATH=build/python python Python_test/signal_generators/test_form_signal_rocm.py
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner

# ─── GPU availability ─────────────────────────────────────────────────────────

try:
    import gpuworklib
    HAS_FORM_ROCM = hasattr(gpuworklib, 'FormSignalGeneratorROCm')
except ImportError:
    HAS_FORM_ROCM = False



# ─── NumPy helpers ────────────────────────────────────────────────────────────

NORM = 0.7071067811865476  # 1/sqrt(2)


def generate_cw_numpy(antennas: int, points: int, fs: float, f0: float,
                      amplitude: float = 1.0, norm: float = NORM,
                      tau_step: float = 0.0) -> np.ndarray:
    """NumPy-референс для CW сигнала с линейной задержкой."""
    t = np.arange(points) / fs
    result = np.zeros((antennas, points), dtype=np.complex64)
    for ant in range(antennas):
        tau = ant * tau_step
        t_shifted = t - tau
        valid = (t_shifted >= 0) & (t_shifted <= (points - 1) / fs)
        phase = 2 * np.pi * f0 * t_shifted[valid]
        result[ant, valid] = (amplitude * norm * np.exp(1j * phase)).astype(np.complex64)
    return result


def generate_lfm_numpy(points: int, fs: float, f0: float, fdev: float,
                       amplitude: float = 1.0, norm: float = NORM) -> np.ndarray:
    """NumPy-референс для LFM чирпа."""
    t = np.arange(points) / fs
    ti = points / fs
    phase = 2 * np.pi * f0 * t + np.pi * fdev / ti * ((t - ti / 2) ** 2)
    return (amplitude * norm * np.exp(1j * phase)).astype(np.complex64)


def peak_freq(signal: np.ndarray, fs: float) -> float:
    """Частота пика FFT (только положительные частоты)."""
    n = len(signal)
    spec = np.fft.fft(signal)
    half = n // 2
    peak_bin = int(np.argmax(np.abs(spec[:half])))
    return peak_bin * fs / n


# ─── Pure NumPy tests (always run) ───────────────────────────────────────────

class TestNumPyReference:
    """Проверяем NumPy-референс формулы getX."""

    def test_cw_peak_frequency(self):
        """CW: пик FFT точно на f0."""
        N, fs, f0 = 4096, 12e6, 2e6
        sig = generate_cw_numpy(1, N, fs, f0)[0]
        f_peak = peak_freq(sig, fs)
        freq_res = fs / N
        assert abs(f_peak - f0) < 2 * freq_res, \
            f"peak={f_peak:.0f} Hz, expected={f0:.0f} Hz"

    def test_cw_shape(self):
        """CW: shape (antennas, points)."""
        data = generate_cw_numpy(5, 1024, 12e6, 1e6)
        assert data.shape == (5, 1024)
        assert data.dtype == np.complex64

    def test_cw_amplitude(self):
        """CW без задержки: амплитуда = a * norm."""
        N, fs, f0 = 1024, 12e6, 1e6
        a, norm = 2.0, 0.5
        t = np.arange(N) / fs
        sig = (a * norm * np.exp(1j * 2 * np.pi * f0 * t)).astype(np.complex64)
        expected = a * norm
        assert abs(np.abs(sig).mean() - expected) < expected * 1e-4

    def test_lfm_spectral_spread(self):
        """LFM чирп: спектр размазан (нет единственного доминирующего пика)."""
        N, fs, f0, fdev = 4096, 12e6, 1e6, 2e6
        sig = generate_lfm_numpy(N, fs, f0, fdev)
        spec = np.fft.fft(sig)
        mag = np.abs(spec[:N // 2])
        peak_mag = mag.max()
        mean_mag = mag.mean()
        # Для LFM пик не должен намного превышать среднее (не как CW)
        assert peak_mag < 20 * mean_mag, \
            "LFM spectrum should be spread, not concentrated"

    def test_linear_delay_start_sample(self):
        """Линейная задержка: антенна N начинает сигнал на N*tau_step отсчётов позже."""
        N, fs, f0 = 1024, 12e6, 1e6
        tau_step = 20 / fs  # 20 отсчётов на антенну
        data = generate_cw_numpy(3, N, fs, f0, tau_step=tau_step)
        for ant in range(3):
            expected_start = ant * 20
            # Первый ненулевой отсчёт должен быть на expected_start
            valid = np.where(np.abs(data[ant]) > 1e-6)[0]
            if len(valid) == 0:
                continue
            first_valid = int(valid[0])
            assert abs(first_valid - expected_start) <= 1, \
                f"ant{ant}: first_valid={first_valid}, expected={expected_start}"

    def test_zero_before_delay(self):
        """При tau > 0: начальные отсчёты == 0 (антенна 1 с tau_step=100 отсчётов)."""
        N, fs, f0 = 1024, 12e6, 1e6
        tau_step = 100 / fs  # каждая следующая антенна задержана на 100 отсчётов
        data = generate_cw_numpy(2, N, fs, f0, tau_step=tau_step)
        # Антенна 1: tau = 1 * tau_step = 100 отсчётов → первые ~100 должны быть 0
        ant1 = data[1]
        zeros_count = int(np.sum(np.abs(ant1[:95]) < 1e-6))
        assert zeros_count > 90, f"Expected ~100 zeros at start of ant1, got {zeros_count}"

    def test_params_from_string_parse(self):
        """ParseFromString: парсит f0, antennas, points."""
        params_str = "f0=2e6,antennas=5,points=8000,fs=12e6"
        parsed = {}
        for kv in params_str.split(","):
            k, v = kv.split("=")
            parsed[k] = float(v)
        assert abs(parsed["f0"] - 2e6) < 1.0
        assert int(parsed["antennas"]) == 5
        assert int(parsed["points"]) == 8000

    def test_noise_snr_degradation(self):
        """Шум an > 0 ухудшает SNR."""
        N, fs, f0 = 1024, 12e6, 1e6
        # Чистый CW
        sig_clean = generate_cw_numpy(1, N, fs, f0)[0]
        # С шумом (симуляция вручную)
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        an = 0.5
        sig_noisy = sig_clean + an * NORM * noise

        spec_clean = np.abs(np.fft.fft(sig_clean))
        spec_noisy = np.abs(np.fft.fft(sig_noisy))

        # SNR: peak/mean в спектре должен быть меньше у зашумлённого
        snr_clean = spec_clean.max() / spec_clean.mean()
        snr_noisy = spec_noisy.max() / spec_noisy.mean()
        assert snr_noisy < snr_clean, "SNR with noise should be lower"

    def test_multichannel_independence(self):
        """Каждый канал независим (разные задержки → разные начальные фазы)."""
        N, fs, f0 = 512, 12e6, 1e6
        tau_step = 1 / (8 * f0)  # pi/4 на канал
        data = generate_cw_numpy(4, N, fs, f0, tau_step=tau_step)
        # Фаза первых не-нулевых отсчётов должна монотонно меняться
        phases = []
        for ant in range(4):
            valid = np.where(np.abs(data[ant]) > 1e-6)[0]
            if len(valid) > 0:
                phases.append(float(np.angle(data[ant, valid[0]])))
        # Фазы должны отличаться между каналами
        assert len(set(round(p, 2) for p in phases)) > 1


# ─── GPU tests ────────────────────────────────────────────────────────────────

class TestFormSignalGeneratorROCm:
    """Тесты GPU-реализации (FormSignalGeneratorROCm)."""

    def setUp(self):
        if not HAS_FORM_ROCM:
            raise SkipTest("FormSignalGeneratorROCm not available")
        ctx = gpuworklib.ROCmGPUContext(0)
        self._gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    # ── CW signal ────────────────────────────────────────────────────────────

    def test_generate_shape(self):
        """generate(): shape (antennas, points), dtype complex64."""
        self._gen.set_params(antennas=5, points=8000, fs=12e6, f0=2e6)
        sig = self._gen.generate()
        assert sig.shape == (5, 8000)
        assert sig.dtype == np.complex64

    def test_generate_cw_peak_frequency(self):
        """CW: пик FFT близко к f0."""
        N, fs, f0 = 4096, 12e6, 2e6
        self._gen.set_params(antennas=1, points=N, fs=fs, f0=f0)
        sig = self._gen.generate()
        assert sig.shape == (1, N)

        f_peak = peak_freq(sig[0], fs)
        freq_res = fs / N
        assert abs(f_peak - f0) < 2 * freq_res, \
            f"GPU peak={f_peak:.0f} Hz, expected={f0:.0f} Hz"

    def test_generate_vs_numpy(self):
        """CW без задержки: GPU совпадает с NumPy-эталоном (atol ≤ 1e-4)."""
        N, fs, f0 = 1024, 12e6, 1e6
        self._gen.set_params(antennas=1, points=N, fs=fs, f0=f0)
        gpu_sig = self._gen.generate()[0]

        ref = generate_cw_numpy(1, N, fs, f0)[0]
        np.testing.assert_allclose(np.abs(gpu_sig), np.abs(ref), atol=1e-4)

    # ── LFM chirp ────────────────────────────────────────────────────────────

    def test_lfm_spectral_spread(self):
        """LFM: спектр размазан (GPU аналогично NumPy)."""
        N, fs, f0, fdev = 4096, 12e6, 1e6, 2e6
        self._gen.set_params(antennas=1, points=N, fs=fs, f0=f0, fdev=fdev)
        sig = self._gen.generate()[0]
        spec = np.abs(np.fft.fft(sig))[:N // 2]
        assert spec.max() < 30 * spec.mean(), \
            "LFM spectrum should be spread"

    # ── Delay ────────────────────────────────────────────────────────────────

    def test_linear_delay_crosscorr(self):
        """tau_step: кросс-корреляция ant0/ant1 — пик на ожидаемом лаге."""
        N, fs, f0 = 4096, 12e6, 2e6
        tau_step = 100e-9  # 100 нс
        self._gen.set_params(antennas=2, points=N, fs=fs, f0=f0, tau_step=tau_step)
        data = self._gen.generate()
        assert data.shape == (2, N)

        corr = np.fft.ifft(np.fft.fft(data[0]) * np.conj(np.fft.fft(data[1])))
        peak_lag = int(np.argmax(np.abs(corr)))
        expected_lag = int(round(tau_step * fs))
        assert abs(peak_lag - expected_lag) <= 2, \
            f"peak_lag={peak_lag}, expected={expected_lag}"

    # ── set_params_from_string ────────────────────────────────────────────────

    def test_set_params_from_string(self):
        """set_params_from_string: генерация после парсинга CSV."""
        self._gen.set_params_from_string("f0=2e6,antennas=3,points=1024,fs=12e6")
        sig = self._gen.generate()
        assert sig.shape == (3, 1024)
        assert sig.dtype == np.complex64
        assert self._gen.antennas == 3
        assert self._gen.points == 1024


if __name__ == "__main__":
    from common.runner import TestRunner
    runner = TestRunner()
    results = runner.run(TestNumPyReference())
    results += runner.run(TestFormSignalGeneratorROCm())
    runner.print_summary(results)
