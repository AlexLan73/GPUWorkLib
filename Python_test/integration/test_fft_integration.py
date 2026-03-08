"""
test_fft_integration.py — интеграционные тесты FFT + SignalGenerator
======================================================================

Тесты 1-3 из оригинального test_gpuworklib.py.

Tests:
  test_multichannel_cw_fft     — CW разных частот → FFT → пик в нужном месте
  test_multichannel_lfm_fft    — LFM → FFT → рассеяный спектр (нет острого пика)
  test_noise_fft_flat_spectrum — Noise → FFT → равномерный спектр

Запуск:
  pytest Python_test/integration/test_fft_integration.py -v
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: CW → FFT → peak at expected frequency
# ─────────────────────────────────────────────────────────────────────────────

class TestCwFftIntegration:
    """CW + FFT: пик спектра на заданной частоте."""

    @pytest.mark.parametrize("f0_hz", [100, 250, 500, 800, 1200])
    def test_cw_peak_frequency(self, sig_gen, fft_proc, f0_hz):
        """FFT пик CW-сигнала должен быть на частоте f0 ± 1 бин."""
        fs = 4000.0
        length = 4096

        signal = sig_gen.generate_cw(freq=f0_hz, fs=fs, length=length)
        spectrum = fft_proc.process_complex(signal, sample_rate=fs)

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

    def test_cw_output_shape(self, sig_gen, fft_proc):
        """FFT выходной массив имеет правильную форму."""
        length = 4096
        signal = sig_gen.generate_cw(freq=100, fs=4000, length=length)
        spectrum = fft_proc.process_complex(signal, sample_rate=4000)
        assert len(spectrum) == length

    def test_cw_signal_complex(self, sig_gen):
        """CW-сигнал комплексный."""
        signal = sig_gen.generate_cw(freq=100, fs=4000, length=512)
        arr = np.asarray(signal)
        assert np.iscomplexobj(arr), "CW signal should be complex"

    def test_cw_energy_nonzero(self, sig_gen, fft_proc):
        """CW → FFT: энергия ненулевая."""
        signal = sig_gen.generate_cw(freq=500, fs=4000, length=4096)
        spectrum = fft_proc.process_complex(signal, sample_rate=4000)
        total_energy = np.sum(np.abs(np.asarray(spectrum)) ** 2)
        assert total_energy > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: LFM → FFT
# ─────────────────────────────────────────────────────────────────────────────

class TestLfmFftIntegration:
    """LFM + FFT: нет острого единственного пика (рассеянный спектр)."""

    def test_lfm_spread_spectrum(self, sig_gen, fft_proc):
        """LFM спектр рассеян по полосе — нет доминирующего одного пика."""
        fs = 4000.0
        length = 4096

        signal = sig_gen.generate_lfm(
            f_start=100, f_end=1000, fs=fs, length=length
        )
        spectrum = fft_proc.process_complex(signal, sample_rate=fs)
        mag = np.abs(np.asarray(spectrum)[:length // 2])

        # Пик не должен содержать более 30% полной энергии спектра
        peak_energy = float(np.max(mag) ** 2)
        total_energy = float(np.sum(mag ** 2))
        peak_ratio = peak_energy / (total_energy + 1e-30)
        assert peak_ratio < 0.3, (
            f"LFM spectrum too concentrated: peak_ratio={peak_ratio:.3f}"
        )

    def test_lfm_output_not_constant(self, sig_gen):
        """LFM сигнал нестационарный — фаза меняется."""
        fs = 4000.0
        length = 4096
        signal = np.asarray(
            sig_gen.generate_lfm(f_start=100, f_end=1000, fs=fs, length=length)
        )
        # Разность соседних фаз должна быть переменной
        phases = np.angle(signal)
        phase_diffs = np.diff(np.unwrap(phases))
        # Если фаза линейно меняется — дифференциал должен расти
        diff_std = float(np.std(phase_diffs))
        assert diff_std > 0.001, "LFM phase diff should vary"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Noise → FFT → flat spectrum
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseFftIntegration:
    """Noise + FFT: спектр статистически равномерный."""

    def test_noise_flat_spectrum(self, sig_gen, fft_proc):
        """Noise FFT: среднее >> std не выполняется для шума."""
        fs = 4000.0
        length = 4096

        signal = sig_gen.generate_noise(fs=fs, length=length)
        spectrum = fft_proc.process_complex(signal, sample_rate=fs)
        mag = np.abs(np.asarray(spectrum)[:length // 2])

        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))

        # Для шума cv = std/mean должен быть < 1.5 (равномерно рассеян)
        cv = std_mag / (mean_mag + 1e-30)
        assert cv < 1.5, (
            f"Noise spectrum not flat: cv={cv:.3f} (mean={mean_mag:.4f}, std={std_mag:.4f})"
        )

    def test_noise_nonzero_energy(self, sig_gen, fft_proc):
        """Noise: ненулевая энергия в спектре."""
        signal = sig_gen.generate_noise(fs=4000, length=4096)
        spectrum = fft_proc.process_complex(signal, sample_rate=4000)
        assert np.sum(np.abs(np.asarray(spectrum)) ** 2) > 0
