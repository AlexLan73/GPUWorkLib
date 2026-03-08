"""
test_signal_gen_integration.py — интеграционные тесты генераторов + pipeline
=============================================================================

Тесты 4-7 из оригинального test_gpuworklib.py.

Tests:
  test_multichannel_different_frequencies — параллельная генерация
  test_cw_lfm_noise_combined             — три типа сигналов одновременно
  test_from_string_params                — генерация из строки параметров
  test_full_pipeline_fft_analysis        — полный pipeline: Gen → FFT → анализ

Запуск:
  pytest Python_test/integration/test_signal_gen_integration.py -v
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Многоканальная генерация с разными частотами
# ─────────────────────────────────────────────────────────────────────────────

class TestMultichannelGeneration:
    """Несколько каналов с разными параметрами."""

    def test_different_frequencies_cw(self, sig_gen, fft_proc):
        """Разные частоты → разные пики FFT.

        Для каждого канала: FFT пик должен отличаться от соседних.
        """
        fs = 4000.0
        length = 4096
        freqs = [100, 400, 800, 1200]

        peak_freqs = []
        for f0 in freqs:
            signal = sig_gen.generate_cw(freq=f0, fs=fs, length=length)
            spectrum = fft_proc.process_complex(signal, sample_rate=fs)
            mag = np.abs(np.asarray(spectrum)[:length // 2])
            freq_axis = np.arange(length // 2) * fs / length
            peak_freqs.append(float(freq_axis[np.argmax(mag)]))

        # Все пики должны быть различны (разница > 50 Гц между соседними)
        for i in range(len(peak_freqs) - 1):
            diff = peak_freqs[i + 1] - peak_freqs[i]
            assert diff > 50, (
                f"Peaks too close: {peak_freqs[i]:.1f} and {peak_freqs[i+1]:.1f} Hz"
            )

    def test_channels_independent(self, sig_gen, fft_proc):
        """Два канала не влияют друг на друга."""
        fs = 4000.0
        length = 4096

        s1 = np.asarray(sig_gen.generate_cw(freq=200, fs=fs, length=length))
        s2 = np.asarray(sig_gen.generate_cw(freq=800, fs=fs, length=length))

        # Генерировать снова — должны получить то же
        s1b = np.asarray(sig_gen.generate_cw(freq=200, fs=fs, length=length))
        np.testing.assert_allclose(np.abs(s1), np.abs(s1b), rtol=1e-5,
                                    err_msg="CW generation not reproducible")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: CW + LFM + Noise одновременно
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedSignalTypes:
    """Одновременная генерация разных типов сигналов."""

    def test_cw_lfm_noise_different_spectra(self, sig_gen, fft_proc):
        """CW, LFM, Noise имеют статистически разные спектры."""
        fs = 4000.0
        length = 4096

        cw = np.asarray(sig_gen.generate_cw(freq=500, fs=fs, length=length))
        lfm = np.asarray(sig_gen.generate_lfm(f_start=100, f_end=1500, fs=fs, length=length))
        noise = np.asarray(sig_gen.generate_noise(fs=fs, length=length))

        # CW spectrum: доминирующий пик
        cw_spec = np.abs(np.fft.fft(cw)[:length // 2])
        cw_peak_ratio = float(np.max(cw_spec) ** 2 / np.sum(cw_spec ** 2))
        assert cw_peak_ratio > 0.5, f"CW peak ratio too low: {cw_peak_ratio:.3f}"

        # LFM spectrum: рассеян
        lfm_spec = np.abs(np.fft.fft(lfm)[:length // 2])
        lfm_peak_ratio = float(np.max(lfm_spec) ** 2 / np.sum(lfm_spec ** 2))
        assert lfm_peak_ratio < cw_peak_ratio, "LFM should be more spread than CW"

        # Noise: ненулевая энергия
        noise_energy = float(np.sum(np.abs(noise) ** 2))
        assert noise_energy > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Генерация из строки параметров
# ─────────────────────────────────────────────────────────────────────────────

class TestStringParamsGeneration:
    """Генерация сигналов из строковых параметров."""

    def test_form_signal_from_string(self, sig_gen, fft_proc):
        """FormSignal из строки параметров: результат не нулевой."""
        if not hasattr(sig_gen, "generate_from_string"):
            pytest.skip("generate_from_string не доступен")

        params_str = "fs=12e6,f0=2e6,amplitude=1.0,length=4096"
        signal = np.asarray(sig_gen.generate_from_string(params_str))

        assert len(signal) == 4096
        assert np.any(np.abs(signal) > 1e-10), "Signal should be non-zero"

    def test_cw_from_string_matches_direct(self, sig_gen):
        """CW из строки совпадает с прямой генерацией."""
        if not hasattr(sig_gen, "generate_cw_from_string"):
            pytest.skip("generate_cw_from_string не доступен")

        fs, f0, length = 4000.0, 500.0, 1024
        direct = np.asarray(sig_gen.generate_cw(freq=f0, fs=fs, length=length))
        from_str = np.asarray(
            sig_gen.generate_cw_from_string(f"freq={f0},fs={fs},length={length}")
        )
        np.testing.assert_allclose(np.abs(direct), np.abs(from_str), rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Полный pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineIntegration:
    """Полный pipeline: SignalGenerator → FFT → частотный анализ."""

    def test_pipeline_cw_correct_peak(self, sig_gen, fft_proc):
        """Полный pipeline: CW → FFT → пик на правильной частоте."""
        fs = 12e6
        f0 = 2e6
        length = 8192

        signal = sig_gen.generate_cw(freq=f0, fs=fs, length=length)
        spectrum = np.asarray(fft_proc.process_complex(signal, sample_rate=fs))

        # Найти пик
        nfft = len(spectrum)
        freq_axis = np.fft.fftfreq(nfft, d=1.0 / fs)
        half = nfft // 2
        mag = np.abs(spectrum[:half])
        peak_idx = int(np.argmax(mag))
        peak_freq = abs(float(freq_axis[peak_idx]))

        bin_size = fs / nfft
        assert abs(peak_freq - f0) <= bin_size * 2, (
            f"Pipeline: CW f0={f0/1e6:.1f} MHz peak at {peak_freq/1e6:.3f} MHz"
        )

    def test_pipeline_lfm_freq_sweep(self, sig_gen, fft_proc):
        """LFM: частота пика увеличивается во времени (sweep)."""
        fs = 4000.0
        length = 4096
        f_start, f_end = 100, 1500

        signal = np.asarray(
            sig_gen.generate_lfm(f_start=f_start, f_end=f_end, fs=fs, length=length)
        )

        # Разбить на 4 сегмента, проверить что пик растёт
        seg_len = length // 4
        peak_freqs = []
        for i in range(4):
            seg = signal[i * seg_len: (i + 1) * seg_len]
            spec = np.abs(np.fft.fft(seg))
            freqs = np.fft.fftfreq(seg_len, d=1.0 / fs)
            half = seg_len // 2
            peak_freqs.append(abs(float(freqs[np.argmax(spec[:half])])))

        # Пик должен расти от сегмента к сегменту (с допуском)
        assert peak_freqs[-1] > peak_freqs[0], (
            f"LFM peak should increase: {peak_freqs}"
        )
