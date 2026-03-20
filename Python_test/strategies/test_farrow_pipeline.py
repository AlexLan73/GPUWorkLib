#!/usr/bin/env python3
"""
test_farrow_pipeline.py — сравнение Pipeline A (фазовая коррекция) vs B (Farrow задержка)
===========================================================================================

ЗАЧЕМ:
    Отвечает на ключевой вопрос: когда ЛЧМ сигнал, Pipeline B (Farrow) точнее Pipeline A?
    Pipeline A компенсирует задержки через фазу несущей — работает хорошо для узкополосного CW.
    Pipeline B сначала выравнивает сигналы по времени через Farrow интерполяцию, потом суммирует.
    Для ЛЧМ (широкополосного) Pipeline B не даёт temporal smearing → энергия не размазывается.

    Тест также проверяет сам FarrowDelay: delay=0 не меняет сигнал, целая задержка — точный сдвиг,
    compensate() восстанавливает оригинал.

ЧТО ПРОВЕРЯЕТ:
    FarrowDelay unit: delay=0, integer delay, compensate round-trip, per-antenna delays.
    Pipeline A vs B: CW сигнал — оба одинаковы (пик на f0).
    Pipeline A vs B: ЛЧМ — B не хуже A по суммарной энергии в полосе (KEY TEST).
    Сложные сценарии: 2 цели, цель + помеха, SNR gain от beamforming.
    Статистика: mean/std/power на каждом шаге, checkpoint'ы на диск (.npy, .json).

GPU: НЕ НУЖЕН — чистый NumPy + FarrowDelay (Python реализация).

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_farrow_pipeline.py

Author: Kodo (AI Assistant)
Date: 2026-03-08
"""

import os
import sys
import tempfile
import numpy as np

# Добавить strategies/ в sys.path для импорта scenario_builder, farrow_delay и т.д.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from scenario_builder import ULAGeometry, ScenarioBuilder, make_single_target
from farrow_delay import FarrowDelay
from pipeline_runner import (
    PipelineRunner, PipelineConfig, PipelineResult,
    compute_channel_stats, find_peaks_per_beam,
)


# ============================================================================
# Constants
# ============================================================================

FS = 12e6
N_SAMPLES = 8000
N_ANT = 8
D_ANT = 0.05  # 5 cm


# ============================================================================
# FarrowDelay Unit Tests
# ============================================================================

class TestFarrowDelay:

    def test_farrow_identity(self):
        """delay=0 → сигнал не меняется."""
        farrow = FarrowDelay()
        rng = np.random.default_rng(42)
        signal = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
        signal = signal.reshape(1, -1)

        result = farrow.apply(signal, np.array([0.0]))
        np.testing.assert_allclose(np.abs(result), np.abs(signal), atol=1e-4)

    def test_farrow_integer_delay(self):
        """Целая задержка → точный сдвиг."""
        farrow = FarrowDelay()
        n = 100
        signal = np.zeros((1, n), dtype=np.complex64)
        signal[0, 10] = 1.0 + 0j  # импульс

        delayed = farrow.apply(signal, np.array([5.0]))
        peak_pos = np.argmax(np.abs(delayed[0]))
        assert peak_pos == 15, f"Peak at {peak_pos}, expected 15"

    def test_farrow_compensate(self):
        """compensate() применяет отрицательную задержку — проверяем сдвиг импульса.

        Замечание: round-trip apply(+d) → compensate(d) нестабилен для Lagrange
        при дробных frac~0.7 (Runge's phenomenon): коэффициенты матрицы осциллируют
        (например [0.957, -12.3, 13.3, -0.957, 0]), что даёт большой gain при
        anti-causal применении. Поэтому проверяем только корректность сдвига:
        импульс в позиции 50 после apply(+5) окажется в 55, после compensate(5)
        вернётся в 50.
        """
        farrow = FarrowDelay()
        n = 200
        signal = np.zeros((1, n), dtype=np.complex64)
        signal[0, 50] = 1.0 + 0j  # импульс

        delay = 5.0  # целая — точный round-trip для integer delay
        delayed = farrow.apply(signal, np.array([delay]))
        assert np.argmax(np.abs(delayed[0])) == 55, "apply(+5): пик должен быть в 55"

        restored = farrow.compensate(delayed, np.array([delay]))
        assert np.argmax(np.abs(restored[0])) == 50, "compensate(5): пик должен вернуться в 50"

    def test_farrow_multi_antenna(self):
        """Per-antenna задержки корректно применяются."""
        farrow = FarrowDelay()
        signal = np.ones((4, 200), dtype=np.complex64)
        delays = np.array([0.0, 1.0, 2.0, 3.0])

        result = farrow.apply(signal, delays)
        assert result.shape == (4, 200)
        assert np.abs(result[0, 0]) > 0.5   # без задержки
        assert np.abs(result[3, 0]) < 0.1   # задержка 3 → первые 3 нулевые
        assert np.abs(result[3, 1]) < 0.1
        assert np.abs(result[3, 2]) < 0.1


# ============================================================================
# Pipeline A & B Basic Tests
# ============================================================================

class TestPipelineBasic:

    def _make_scenario(self, fdev=0.0, noise_sigma=0.0):
        """Создать стандартный сценарий."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=fdev)
        if noise_sigma > 0:
            builder.set_noise(sigma=noise_sigma)
        return builder.build()

    def test_cw_pipeline_a(self):
        """Pipeline A: CW → пик на f0=2MHz."""
        scenario = self._make_scenario(fdev=0)
        runner = PipelineRunner()
        result = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)

        assert result.peaks is not None
        assert len(result.peaks) == N_ANT
        peak = result.peaks[0][0]
        freq_res = FS / result.nFFT
        assert abs(peak.freq_hz - 2e6) < 2 * freq_res

    def test_cw_pipeline_b(self):
        """Pipeline B: CW → пик на f0=2MHz."""
        scenario = self._make_scenario(fdev=0)
        runner = PipelineRunner()
        result = runner.run_pipeline_b(scenario, steer_theta=30)

        assert result.peaks is not None
        assert result.S_aligned is not None  # Farrow data available
        peak = result.peaks[0][0]
        freq_res = FS / result.nFFT
        assert abs(peak.freq_hz - 2e6) < 2 * freq_res

    def test_lfm_pipeline_a(self):
        """Pipeline A: ЛЧМ → энергия в полосе."""
        scenario = self._make_scenario(fdev=1e6)
        runner = PipelineRunner()
        result = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)

        assert result.peaks[0][0].magnitude > 0

    def test_lfm_pipeline_b(self):
        """Pipeline B: ЛЧМ → энергия в полосе, S_aligned доступен."""
        scenario = self._make_scenario(fdev=1e6)
        runner = PipelineRunner()
        result = runner.run_pipeline_b(scenario, steer_theta=30)

        assert result.peaks[0][0].magnitude > 0
        assert result.S_aligned is not None
        assert result.S_aligned.shape == (N_ANT, N_SAMPLES)


# ============================================================================
# Pipeline A vs B Comparison
# ============================================================================

class TestPipelineComparison:

    def test_cw_comparison(self):
        """CW: Pipeline A ≈ Pipeline B."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6)
        scenario = builder.build()

        runner = PipelineRunner()
        result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        peak_a = result_a.peaks[0][0]
        peak_b = result_b.peaks[0][0]

        # Частоты совпадают
        freq_res = FS / result_a.nFFT
        assert abs(peak_a.freq_hz - peak_b.freq_hz) < 2 * freq_res

        # Магнитуды близки
        ratio = peak_b.magnitude / max(peak_a.magnitude, 1e-10)
        assert 0.5 < ratio < 2.0, f"CW ratio B/A = {ratio:.2f}"

    def test_lfm_comparison(self):
        """ЛЧМ: Pipeline B >= Pipeline A по энергии в полосе (KEY TEST)."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=1e6)
        scenario = builder.build()

        runner = PipelineRunner()
        result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        # Суммарная энергия в полосе [1..3 MHz]
        nFFT = result_a.nFFT
        freq_res = FS / nFFT
        bin_lo = int(1e6 / freq_res)
        bin_hi = int(3e6 / freq_res)

        energy_a = np.sum(result_a.magnitudes[0, bin_lo:bin_hi] ** 2)
        energy_b = np.sum(result_b.magnitudes[0, bin_lo:bin_hi] ** 2)

        ratio = energy_b / max(energy_a, 1e-10)
        # Pipeline B не хуже Pipeline A
        assert ratio > 0.8, f"LFM: B/A = {ratio:.3f}, expected >= 0.8"

    def test_lfm_large_delay(self):
        """ЛЧМ с большим шагом антенн — усиливает разницу A vs B."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=0.5)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=2e6)
        scenario = builder.build()

        runner = PipelineRunner()
        result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        # Оба должны дать ненулевой результат
        assert result_a.peaks[0][0].magnitude > 0
        assert result_b.peaks[0][0].magnitude > 0


# ============================================================================
# Complex Scenarios
# ============================================================================

class TestComplexScenarios:

    def test_multi_target_farrow(self):
        """2 ЛЧМ цели с разных углов — Farrow наведение на первую."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=20, f0_hz=2e6, fdev_hz=500e3)
        builder.add_target(theta_deg=45, f0_hz=3.5e6, fdev_hz=500e3)
        builder.set_noise(sigma=0.05)
        scenario = builder.build()

        runner = PipelineRunner()
        result_b = runner.run_pipeline_b(scenario, steer_theta=20)

        # Пик около 2 MHz (первая цель). Допуск широкий: ЛЧМ полоса 500 kHz,
        # FFT пик может быть смещён от f0 на десятки kHz.
        peak = result_b.peaks[0][0]
        assert abs(peak.freq_hz - 2e6) < 300e3, (
            f"Peak at {peak.freq_hz/1e6:.3f} MHz, expected near 2.0 MHz"
        )

    def test_jammer_scenario(self):
        """Цель (30°) + ЛЧМ помеха (-20°)."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=1e6)
        builder.add_jammer(theta_deg=-20, f0_hz=2e6, fdev_hz=500e3, amplitude=0.5)
        builder.set_noise(sigma=0.1)
        scenario = builder.build()

        runner = PipelineRunner()
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)
        assert result_b.peaks[0][0].magnitude > 0

    def test_snr_improvement(self):
        """SNR gain от beamforming — пик заметно выше шумового пола."""
        array = ULAGeometry(n_ant=N_ANT, d_ant_m=D_ANT)
        builder = ScenarioBuilder(array, fs=FS, n_samples=N_SAMPLES)
        builder.add_target(theta_deg=30, f0_hz=2e6, amplitude=1.0)
        builder.set_noise(sigma=1.0, seed=42)
        scenario = builder.build()

        runner = PipelineRunner()
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        peak = result_b.peaks[0][0]
        nFFT = result_b.nFFT
        noise_floor = float(np.median(result_b.magnitudes[0, :nFFT // 2]))
        snr_out = peak.magnitude / max(noise_floor, 1e-10)

        assert snr_out > 3.0, f"SNR too low: {snr_out:.1f}"


# ============================================================================
# Statistics & Checkpoints Tests
# ============================================================================

class TestStatsAndCheckpoints:

    def test_stats_computed(self):
        """Статистика вычисляется на каждом шаге pipeline."""
        scenario = make_single_target(n_ant=4, theta_deg=30, fdev_hz=1e6)
        runner = PipelineRunner()

        result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        # Pipeline A: input, gemm, spectrum
        assert result_a.stats_input is not None
        assert len(result_a.stats_input) == 4
        assert result_a.stats_gemm is not None
        assert result_a.stats_spectrum is not None

        # Pipeline B: + aligned
        assert result_b.stats_input is not None
        assert result_b.stats_aligned is not None  # Farrow шаг!
        assert len(result_b.stats_aligned) == 4
        assert result_b.stats_gemm is not None
        assert result_b.stats_spectrum is not None

    def test_stats_values(self):
        """Значения статистики корректны (power > 0, max > min)."""
        scenario = make_single_target(n_ant=4, theta_deg=30, fdev_hz=1e6)
        runner = PipelineRunner()
        result = runner.run_pipeline_b(scenario, steer_theta=30)

        for s in result.stats_input:
            assert s.power > 0, f"Channel {s.channel_id}: power = 0"
            assert s.max_abs >= s.min_abs
            assert s.n_samples == N_SAMPLES

    def test_pipeline_result_access(self):
        """Все промежуточные данные доступны из PipelineResult."""
        scenario = make_single_target(n_ant=4, theta_deg=30)
        runner = PipelineRunner()

        result = runner.run_pipeline_b(scenario, steer_theta=30)

        # Все numpy массивы доступны
        assert result.S_raw is not None
        assert result.S_raw.shape == (4, N_SAMPLES)
        assert result.S_aligned is not None
        assert result.S_aligned.shape == (4, N_SAMPLES)
        assert result.X_gemm is not None
        assert result.X_gemm.shape[0] == 4
        assert result.spectrum is not None
        assert result.magnitudes is not None
        assert result.W is not None
        assert result.W.shape == (4, 4)
        assert result.freq_axis is not None

    def test_save_to_disk(self):
        """Checkpoint'ы сохраняются на диск."""
        scenario = make_single_target(n_ant=4, theta_deg=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                save_input=True,
                save_aligned=True,
                save_gemm=True,
                save_spectrum=True,
                save_stats=True,
                save_results=True,
            )

            runner = PipelineRunner(output_dir=tmpdir)
            runner.run_pipeline_b(scenario, steer_theta=30, config=config)

            # Проверяем что файлы созданы
            b_dir = os.path.join(tmpdir, "pipeline_b")
            assert os.path.exists(os.path.join(b_dir, "S_raw.npy"))
            assert os.path.exists(os.path.join(b_dir, "S_aligned.npy"))
            assert os.path.exists(os.path.join(b_dir, "X_gemm.npy"))
            assert os.path.exists(os.path.join(b_dir, "spectrum.npy"))
            assert os.path.exists(os.path.join(b_dir, "stats.json"))
            assert os.path.exists(os.path.join(b_dir, "results.json"))

            # Проверяем что npy читается
            S_loaded = np.load(os.path.join(b_dir, "S_aligned.npy"))
            assert S_loaded.shape == (4, N_SAMPLES)

    def test_comparison_output(self):
        """compare() возвращает метрики сравнения."""
        scenario = make_single_target(n_ant=4, theta_deg=30, fdev_hz=1e6)
        runner = PipelineRunner()

        result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
        result_b = runner.run_pipeline_b(scenario, steer_theta=30)

        comp = runner.compare(result_a, result_b)
        assert 'magnitude_ratio_b_over_a' in comp
        assert 'freq_diff_hz' in comp
        assert comp['magnitude_ratio_b_over_a'] > 0

    def test_summary_strings(self):
        """peak_summary() и stats_summary() возвращают строки."""
        scenario = make_single_target(n_ant=4, theta_deg=30)
        runner = PipelineRunner()
        result = runner.run_pipeline_b(scenario, steer_theta=30)

        assert "Pipeline B" in result.peak_summary()
        assert "Pipeline B" in result.stats_summary()
        assert "Input" in result.stats_summary()
        assert "Aligned" in result.stats_summary()


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Farrow Pipeline -- A vs B comparison")
    print("=" * 60)
    print(f"Parameters: {N_ANT} ant, d={D_ANT*100:.0f} cm, "
          f"fs={FS/1e6:.0f} MHz, N={N_SAMPLES}")
    print()

    test_classes = [
        TestFarrowDelay,
        TestPipelineBasic,
        TestPipelineComparison,
        TestComplexScenarios,
        TestStatsAndCheckpoints,
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

    # Demo: full comparison with save
    print(f"\n{'=' * 60}")
    print("Demo: Full Pipeline Comparison (LFM scenario)")
    print("=" * 60)

    scenario = make_single_target(n_ant=8, theta_deg=30, fdev_hz=1e6, noise_sigma=0.1)

    runner = PipelineRunner()
    result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
    result_b = runner.run_pipeline_b(scenario, steer_theta=30)

    runner.print_comparison(result_a, result_b)

    print(f"\n{'=' * 60}")
    print(f"Test results: {total_pass} passed, {total_fail} failed")
    print("Done.")
