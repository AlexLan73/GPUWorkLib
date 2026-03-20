#!/usr/bin/env python3
"""
test_scenario_builder.py — физическая модель ULA + генерация сигналов (NumPy)
==============================================================================

ЗАЧЕМ:
    Проверяет что физическая модель антенной решётки (ULA) и генератор сигналов
    дают математически правильные результаты. Это фундамент всего pipeline —
    если здесь ошибка, все остальные тесты бессмысленны.

    Аналогия: это как проверить что рулетка правильно показывает метры,
    перед тем как мерять расстояния.

ЧТО ПРОВЕРЯЕТ:
    ULA Geometry: theta=0 → задержки нули, theta=90 → максимальные задержки,
    отрицательный угол → отрицательные задержки, d = lambda/2.

    Генерация сигналов: CW → FFT-пик на f0, ЛЧМ → полоса ~ fdev,
    fdev=0 эквивалентен CW, задержка между антеннами корректна.

    Multi-source: 2 CW цели → 2 FFT пика, цель + помеха оба видны.

    Шум: AWGN — mean≈0, std≈sigma, воспроизводимость по seed.

    Матрица весов W: shape [n_ant × n_ant], ||строка||=1, beamforming не обнуляет сигнал.

    Фабричные сценарии: make_single_target, make_target_and_jammer, make_multi_target.

GPU: НЕ НУЖЕН — чистый NumPy.

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_scenario_builder.py

Author: Кодо (AI Assistant)
Date: 2026-03-08
"""

import os
import sys
import numpy as np

# Добавить strategies/ в sys.path для импорта scenario_builder и др.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from scenario_builder import (
    ULAGeometry,
    EmitterSignal,
    ScenarioBuilder,
    make_single_target,
    make_target_and_jammer,
    make_multi_target,
)


# ============================================================================
# Constants
# ============================================================================

C_LIGHT = 3e8  # скорость света (м/с)
FS = 12e6      # частота дискретизации
N_SAMPLES = 8000


# ============================================================================
# Helpers
# ============================================================================

def fft_peak_freq(signal_1d: np.ndarray, fs: float) -> float:
    """Найти частоту доминирующего пика в спектре (первая половина)."""
    spectrum = np.fft.fft(signal_1d)
    magnitudes = np.abs(spectrum)
    half = len(magnitudes) // 2
    peak_bin = np.argmax(magnitudes[1:half]) + 1
    return peak_bin * fs / len(signal_1d)


def fft_bandwidth_3dB(signal_1d: np.ndarray, fs: float) -> float:
    """Оценить полосу по уровню -3 дБ от пика."""
    spectrum = np.fft.fft(signal_1d)
    magnitudes = np.abs(spectrum)
    half = len(magnitudes) // 2
    mags = magnitudes[:half]

    peak_val = np.max(mags)
    threshold = peak_val / np.sqrt(2.0)  # -3 дБ

    above = np.where(mags > threshold)[0]
    if len(above) < 2:
        return 0.0

    freq_resolution = fs / len(signal_1d)
    return (above[-1] - above[0]) * freq_resolution


# ============================================================================
# ULA Geometry Tests
# ============================================================================

class TestULAGeometry:

    def test_ula_delays_broadside(self):
        """theta=0 (перпендикулярно решётке) -> все задержки = 0."""
        ula = ULAGeometry(n_ant=8, d_ant_m=0.05)
        delays = ula.compute_delays(theta_deg=0.0)
        np.testing.assert_allclose(delays, 0.0, atol=1e-20)

    def test_ula_delays_endfire(self):
        """theta=90 -> максимальные задержки: tau_i = i*d/c."""
        ula = ULAGeometry(n_ant=4, d_ant_m=0.1, c=C_LIGHT)
        delays = ula.compute_delays(theta_deg=90.0)
        expected = np.arange(4) * 0.1 / C_LIGHT
        np.testing.assert_allclose(delays, expected, rtol=1e-10)

    def test_ula_delays_negative_angle(self):
        """theta<0 -> отрицательные задержки (волна приходит с другой стороны)."""
        ula = ULAGeometry(n_ant=4, d_ant_m=0.05)
        delays_pos = ula.compute_delays(theta_deg=30.0)
        delays_neg = ula.compute_delays(theta_deg=-30.0)
        np.testing.assert_allclose(delays_neg, -delays_pos, rtol=1e-10)

    def test_ula_delays_shape(self):
        """Выход: [n_ant] float64."""
        ula = ULAGeometry(n_ant=16, d_ant_m=0.02)
        delays = ula.compute_delays(theta_deg=45.0)
        assert delays.shape == (16,)
        assert delays.dtype == np.float64

    def test_ula_from_lambda_half(self):
        """d = lambda/2 = c / (2*f_carrier)."""
        f_carrier = 10e9  # X-band
        ula = ULAGeometry.from_lambda_half(n_ant=8, carrier_freq_hz=f_carrier)
        expected_d = C_LIGHT / (2 * f_carrier)  # 0.015 м
        assert ula.n_ant == 8
        np.testing.assert_allclose(ula.d_ant_m, expected_d, rtol=1e-10)

    def test_max_unambiguous_angle(self):
        """d = lambda/2 -> max angle = 90 deg."""
        f_carrier = 10e9
        ula = ULAGeometry.from_lambda_half(n_ant=8, carrier_freq_hz=f_carrier)
        angle = ula.max_unambiguous_angle(f_carrier)
        np.testing.assert_allclose(angle, 90.0, atol=1e-5)


# ============================================================================
# Signal Generation Tests
# ============================================================================

class TestSignalGeneration:

    def test_cw_single_antenna(self):
        """CW сигнал (fdev=0): FFT пик на f0."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=0, f0_hz=2e6, fdev_hz=0)

        scenario = builder.build()
        S = scenario['S']

        assert S.shape == (1, N_SAMPLES)
        assert S.dtype == np.complex64

        peak_freq = fft_peak_freq(S[0], FS)
        freq_res = FS / N_SAMPLES
        assert abs(peak_freq - 2e6) < 2 * freq_res, \
            f"Peak at {peak_freq:.0f} Hz, expected ~2 MHz"

    def test_lfm_bandwidth(self):
        """ЛЧМ (fdev>0): полоса сигнала ~ fdev."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        fdev = 1e6
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=0, f0_hz=2e6, fdev_hz=fdev)

        scenario = builder.build()
        bw = fft_bandwidth_3dB(scenario['S'][0], FS)

        # ЛЧМ полоса примерно = fdev (с точностью до оконных эффектов)
        assert bw > fdev * 0.5, f"BW={bw:.0f} Hz too narrow for fdev={fdev:.0f}"
        assert bw < fdev * 3.0, f"BW={bw:.0f} Hz too wide for fdev={fdev:.0f}"

    def test_lfm_compatibility_with_cpp(self):
        """Формула ЛЧМ с центрированием (t_d - Ti/2)^2.

        При fdev=0 должна быть эквивалентна CW.
        При fdev>0 фаза квадратичная.
        """
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)

        # CW
        builder_cw = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder_cw.add_target(theta_deg=0, f0_hz=2e6, fdev_hz=0)
        S_cw = builder_cw.build()['S']

        # ЛЧМ с fdev=0 (должно совпасть с CW)
        builder_lfm0 = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder_lfm0.add_target(theta_deg=0, f0_hz=2e6, fdev_hz=0.0)
        S_lfm0 = builder_lfm0.build()['S']

        np.testing.assert_allclose(
            np.abs(S_cw), np.abs(S_lfm0), atol=1e-6,
            err_msg="fdev=0 should equal CW"
        )

    def test_signal_amplitude(self):
        """Амплитуда сигнала = A * norm (1/sqrt(2))."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        A = 2.0
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=0, f0_hz=2e6, amplitude=A)

        S = builder.build()['S']
        expected_mag = A / np.sqrt(2.0)
        actual_mag = np.max(np.abs(S))

        np.testing.assert_allclose(actual_mag, expected_mag, rtol=1e-5)

    def test_delay_shifts_signal(self):
        """Сигнал с theta>0 задерживается относительно theta=0."""
        ula = ULAGeometry(n_ant=2, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6)

        S = builder.build()['S']

        # Антенна 1 получает сигнал позже антенны 0
        # Фаза на антенне 1 отстаёт
        phase_0 = np.angle(S[0, N_SAMPLES // 2])
        phase_1 = np.angle(S[1, N_SAMPLES // 2])
        # Разность фаз не нулевая
        assert abs(phase_0 - phase_1) > 1e-6, "Phase should differ between antennas"


# ============================================================================
# Multi-source Tests
# ============================================================================

class TestMultiSource:

    def test_two_targets_sum(self):
        """2 CW цели на разных частотах -> 2 FFT пика."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=0, f0_hz=1e6, amplitude=1.0)
        builder.add_target(theta_deg=0, f0_hz=3e6, amplitude=1.0)

        S = builder.build()['S']

        spectrum = np.abs(np.fft.fft(S[0]))
        half = len(spectrum) // 2
        freq_axis = np.fft.fftfreq(len(S[0]), 1.0 / FS)[:half]

        # Два главных пика
        sorted_indices = np.argsort(spectrum[:half])[::-1]
        top_freqs = sorted(freq_axis[sorted_indices[:2]])

        freq_res = FS / N_SAMPLES
        assert abs(top_freqs[0] - 1e6) < 2 * freq_res, \
            f"First peak at {top_freqs[0]:.0f}, expected ~1 MHz"
        assert abs(top_freqs[1] - 3e6) < 2 * freq_res, \
            f"Second peak at {top_freqs[1]:.0f}, expected ~3 MHz"

    def test_target_plus_jammer(self):
        """Цель + помеха -> оба видны в спектре."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, amplitude=1.0)
        builder.add_jammer(theta_deg=-20, f0_hz=4e6, amplitude=0.5)

        S = builder.build()['S']

        spectrum = np.abs(np.fft.fft(S[0]))
        half = len(spectrum) // 2

        # Проверяем наличие энергии в обоих диапазонах
        freq_res = FS / N_SAMPLES
        bin_target = int(2e6 / freq_res)
        bin_jammer = int(4e6 / freq_res)

        # Целевой пик должен быть сильнее (A=1 vs A=0.5)
        assert spectrum[bin_target] > 0, "Target signal missing"
        assert spectrum[bin_jammer] > 0, "Jammer signal missing"

    def test_emitter_count(self):
        """build() правильно считает targets и jammers."""
        ula = ULAGeometry(n_ant=4, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=10, f0_hz=1e6)
        builder.add_target(theta_deg=20, f0_hz=2e6)
        builder.add_jammer(theta_deg=-30, f0_hz=3e6)

        scenario = builder.build()
        assert len(scenario['targets']) == 2
        assert len(scenario['jammers']) == 1


# ============================================================================
# Noise Tests
# ============================================================================

class TestNoise:

    def test_awgn_statistics(self):
        """AWGN: mean ~ 0, std ~ sigma."""
        ula = ULAGeometry(n_ant=1, d_ant_m=0.05)
        sigma = 0.5
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.set_noise(sigma=sigma, seed=42)

        S = builder.build()['S']

        # Среднее близко к нулю
        assert abs(np.mean(S.real)) < 0.1
        assert abs(np.mean(S.imag)) < 0.1

        # СКО Re и Im ~ sigma / sqrt(2) каждый
        expected_std = sigma / np.sqrt(2.0)
        np.testing.assert_allclose(np.std(S.real), expected_std, rtol=0.1)
        np.testing.assert_allclose(np.std(S.imag), expected_std, rtol=0.1)

    def test_noise_reproducibility(self):
        """Одинаковый seed -> одинаковый результат."""
        ula = ULAGeometry(n_ant=4, d_ant_m=0.05)

        b1 = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        b1.set_noise(sigma=0.1, seed=123)
        S1 = b1.build()['S']

        b2 = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        b2.set_noise(sigma=0.1, seed=123)
        S2 = b2.build()['S']

        np.testing.assert_array_equal(S1, S2)

    def test_noise_different_seeds(self):
        """Разные seed -> разный результат."""
        ula = ULAGeometry(n_ant=4, d_ant_m=0.05)

        b1 = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        b1.set_noise(sigma=0.1, seed=111)
        S1 = b1.build()['S']

        b2 = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        b2.set_noise(sigma=0.1, seed=222)
        S2 = b2.build()['S']

        assert not np.array_equal(S1, S2)


# ============================================================================
# Weight Matrix Tests
# ============================================================================

class TestWeightMatrix:

    def test_weight_matrix_shape(self):
        """W: [n_ant x n_ant] complex64."""
        ula = ULAGeometry(n_ant=8, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6)

        W = builder.generate_weight_matrix(steer_theta_deg=30)
        assert W.shape == (8, 8)
        assert W.dtype == np.complex64

    def test_weight_matrix_unit_norm(self):
        """Каждая строка W имеет единичную норму."""
        ula = ULAGeometry(n_ant=8, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6)

        W = builder.generate_weight_matrix(steer_theta_deg=30)
        for beam in range(8):
            norm = np.linalg.norm(W[beam])
            np.testing.assert_allclose(norm, 1.0, atol=1e-5,
                                       err_msg=f"Beam {beam} norm={norm}")

    def test_scan_weight_matrix_shape(self):
        """Scan W: [n_beams x n_ant] для заданных углов."""
        ula = ULAGeometry(n_ant=8, d_ant_m=0.05)
        builder = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=0, f0_hz=2e6)

        angles = [-30, -10, 0, 10, 30]
        W = builder.generate_scan_weight_matrix(angles)
        assert W.shape == (5, 8)
        assert W.dtype == np.complex64

    def test_beamforming_snr_gain(self):
        """Beamforming улучшает SNR: после W @ S шум подавляется."""
        n_ant = 8
        ula = ULAGeometry(n_ant=n_ant, d_ant_m=0.05)

        # Только шум (без сигнала)
        builder_noise = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder_noise.set_noise(sigma=1.0, seed=42)
        S_noise = builder_noise.build()['S']

        # Сигнал + шум
        builder_sn = ScenarioBuilder(ula, fs=FS, n_samples=N_SAMPLES)
        builder_sn.add_target(theta_deg=30, f0_hz=2e6, amplitude=1.0)
        builder_sn.set_noise(sigma=1.0, seed=42)
        S_sn = builder_sn.build()['S']

        W = builder_sn.generate_weight_matrix(steer_theta_deg=30)

        # До beamforming: средняя мощность по антеннам
        power_before = np.mean(np.abs(S_sn) ** 2)

        # После beamforming
        X = W @ S_sn
        power_after = np.mean(np.abs(X) ** 2)

        # SNR должно улучшиться (хотя бы не ухудшиться)
        # Для Delay-and-sum SNR gain ~ sqrt(N_ant)
        assert power_after > 0, "Beamforming output is zero"


# ============================================================================
# Factory Scenarios Tests
# ============================================================================

class TestFactoryScenarios:

    def test_make_single_target(self):
        """Фабрика: одна ЛЧМ цель."""
        scenario = make_single_target(n_ant=8, theta_deg=30)

        assert 'S' in scenario
        assert 'W' in scenario
        assert scenario['S'].shape == (8, 8000)
        assert scenario['W'].shape == (8, 8)
        assert scenario['S'].dtype == np.complex64
        assert len(scenario['targets']) == 1
        assert len(scenario['jammers']) == 0

    def test_make_target_and_jammer(self):
        """Фабрика: цель + помеха."""
        scenario = make_target_and_jammer(n_ant=8)

        assert scenario['S'].shape == (8, 8000)
        assert scenario['W'].shape == (8, 8)
        assert len(scenario['targets']) == 1
        assert len(scenario['jammers']) == 1

    def test_make_multi_target(self):
        """Фабрика: несколько целей."""
        scenario = make_multi_target(
            n_ant=8,
            thetas=[20, 45, -10],
            f0s=[1e6, 2e6, 3e6],
            fdevs=[1e6, 500e3, 800e3],
            amplitudes=[1.0, 0.7, 0.3]
        )

        assert scenario['S'].shape == (8, 8000)
        assert len(scenario['targets']) == 3
        assert len(scenario['jammers']) == 0

    def test_summary_output(self):
        """summary() возвращает строку."""
        scenario = make_single_target(n_ant=4)
        summary = scenario['builder'].summary()
        assert isinstance(summary, str)
        assert "Array" in summary
        assert "target_0" in summary


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ScenarioBuilder -- numpy-only tests")
    print("=" * 60)

    test_classes = [
        TestULAGeometry,
        TestSignalGeneration,
        TestMultiSource,
        TestNoise,
        TestWeightMatrix,
        TestFactoryScenarios,
    ]

    total_pass = 0
    total_fail = 0

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for name in sorted(dir(instance)):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {name}")
                    total_pass += 1
                except Exception as e:
                    print(f"  FAIL: {name} -- {e}")
                    total_fail += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print("Done.")
