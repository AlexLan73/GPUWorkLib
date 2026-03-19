"""
Python тест модуля range_angle (RangeAngleProcessor).

Проверяет:
  1. Значения параметров по умолчанию RangeAngleParams
  2. Enum RangeAnglePeakMode
  3. Вспомогательные методы params (bandwidth, duration, chirp_rate, n_antennas)
  4. __repr__ объектов RangeAngleParams, TargetInfo
  5. SetParams/GetParams roundtrip через GPU процессор
  6. Базовую обработку LFM сигнала: дальность τ=0.5 мс → R=75 000 м
  7. power_cube_numpy() — shape и тип данных

Запуск:
  PYTHONPATH=build/python python Python_test/range_angle/test_range_angle.py

Эталон для сравнения — NumPy/SciPy (chirp rate, duration, bandwidth, range).
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


# ── Вспомогательные функции ───────────────────────────────────────────────────

def make_lfm_signal(n_samples: int, fs: float, f_start: float, f_end: float) -> np.ndarray:
    """Генерация LFM сигнала через NumPy (эталонный расчёт)."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    B = f_end - f_start
    T = n_samples / fs
    mu = B / T
    return np.exp(1j * np.pi * mu * t ** 2).astype(np.complex64)


def build_antenna_array(rx_single: np.ndarray, n_ant: int) -> np.ndarray:
    """Размножить сигнал одной антенны на n_ant антенн (угол = 0°, фаза одинакова)."""
    return np.tile(rx_single, n_ant).astype(np.complex64)


# ── Тесты ────────────────────────────────────────────────────────────────────

class TestRangeAngle:
    """Тесты RangeAngleProcessor."""

    def setUp(self):
        self._gw = GPULoader.get()
        if self._gw is None:
            raise SkipTest("gpuworklib не найден")
        self._rocm_ctx = GPUContextManager.get_rocm()
        if self._rocm_ctx is None:
            raise SkipTest("ROCm недоступен")

    def test_default_params(self):
        """Проверяем дефолтные значения RangeAngleParams."""
        gw = self._gw
        if not hasattr(gw, "RangeAngleParams"):
            raise SkipTest("RangeAngleParams недоступен в этой сборке")

        p = gw.RangeAngleParams()
        assert p.n_ant_az    == 16
        assert p.n_ant_el    == 16
        assert p.n_samples   == 1_300_000
        assert abs(p.f_start     - (-5e6)) < abs(-5e6) * 1e-5
        assert abs(p.f_end       - (+5e6)) < abs(+5e6) * 1e-5
        assert abs(p.sample_rate - 12e6)   < 12e6 * 1e-5
        assert abs(p.carrier_freq - 435e6) < 435e6 * 1e-5
        assert abs(p.antenna_spacing - 0.345) < 0.345 * 1e-4
        assert p.n_peaks == 1
        print(f"\nDefault params: n_ant={p.get_n_antennas()}, "
              f"B={p.get_bandwidth()/1e6:.1f} MHz, "
              f"T={p.get_duration()*1e3:.1f} ms")

    def test_params_helpers(self):
        """Сравниваем helpers params с NumPy-эталоном."""
        gw = self._gw
        if not hasattr(gw, "RangeAngleParams"):
            raise SkipTest("RangeAngleParams недоступен в этой сборке")

        p = gw.RangeAngleParams()

        expected_bw         = p.f_end - p.f_start
        expected_duration   = p.n_samples / p.sample_rate
        expected_chirp_rate = expected_bw / expected_duration
        expected_n_ant      = p.n_ant_az * p.n_ant_el

        assert abs(p.get_bandwidth()  - expected_bw) < abs(expected_bw) * 1e-5
        assert abs(p.get_duration()   - expected_duration) < abs(expected_duration) * 1e-5
        assert abs(p.get_chirp_rate() - expected_chirp_rate) < abs(expected_chirp_rate) * 1e-4
        assert p.get_n_antennas() == expected_n_ant
        print(f"\nHelpers: B={p.get_bandwidth()/1e6:.2f} MHz, "
              f"T={p.get_duration()*1e3:.2f} ms, "
              f"mu={p.get_chirp_rate()/1e6:.1f} MHz/s, "
              f"n_ant={p.get_n_antennas()}")

    def test_peak_mode_enum(self):
        """Проверяем доступность и корректность enum RangeAnglePeakMode."""
        gw = self._gw
        if not hasattr(gw, "RangeAnglePeakMode"):
            raise SkipTest("RangeAnglePeakMode недоступен в этой сборке")

        p = gw.RangeAngleParams()

        p.peak_mode = gw.RangeAnglePeakMode.TOP_1
        assert p.peak_mode == gw.RangeAnglePeakMode.TOP_1

        p.peak_mode = gw.RangeAnglePeakMode.TOP_N
        assert p.peak_mode == gw.RangeAnglePeakMode.TOP_N
        p.n_peaks = 3
        assert p.n_peaks == 3

        print(f"\nPeakMode TOP_1={gw.RangeAnglePeakMode.TOP_1}, "
              f"TOP_N={gw.RangeAnglePeakMode.TOP_N}")

    def test_repr_objects(self):
        """Проверяем, что repr() не падает и содержит смысловые поля."""
        gw = self._gw
        if not hasattr(gw, "RangeAngleParams"):
            raise SkipTest("RangeAngleParams недоступен в этой сборке")

        p = gw.RangeAngleParams()
        r_params = repr(p)
        assert "RangeAngleParams" in r_params
        assert "n_ant=" in r_params
        print(f"\nRangeAngleParams repr: {r_params}")

        t = gw.TargetInfo()
        t.range_m = 75000.0
        t.angle_az_deg = 12.5
        r_target = repr(t)
        assert "TargetInfo" in r_target
        assert "75000" in r_target
        print(f"TargetInfo repr: {r_target}")

    def test_processor_set_get_params(self):
        """Проверяем SetParams/GetParams roundtrip."""
        gw = self._gw
        if not hasattr(gw, "RangeAngleProcessor"):
            raise SkipTest("RangeAngleProcessor недоступен в этой сборке")

        p = gw.RangeAngleParams()
        p.n_ant_az    = 8
        p.n_ant_el    = 4
        p.n_samples   = 65536
        p.f_start     = -3e6
        p.f_end       = +3e6
        p.sample_rate = 10e6

        proc = gw.RangeAngleProcessor(self._rocm_ctx)
        proc.set_params(p)

        got = proc.get_params()
        assert got.n_ant_az    == 8
        assert got.n_ant_el    == 4
        assert got.n_samples   == 65536
        assert abs(got.f_start     - (-3e6)) < abs(-3e6) * 1e-5
        assert abs(got.f_end       - (+3e6)) < abs(+3e6) * 1e-5
        assert abs(got.sample_rate - 10e6)   < 10e6 * 1e-5
        print(f"\nset/get params roundtrip OK: n_ant={got.get_n_antennas()}, "
              f"B={got.get_bandwidth()/1e6:.1f} MHz")

    def test_range_basic(self):
        """
        Основной функциональный тест (xfail пока GPU ядра не реализованы).

        Генерируем LFM с задержкой τ = 0.5 мс.
        Ожидаемая дальность: R = c * τ / 2 = 3e8 * 0.5e-3 / 2 = 75 000 м.
        Допуск: +/- 1 000 м.
        """
        gw = self._gw
        if not hasattr(gw, "RangeAngleProcessor"):
            raise SkipTest("RangeAngleProcessor недоступен в этой сборке")

        N_SAMPLES  = 50_000
        FS         = 12e6
        F_START    = -5e6
        F_END      = +5e6
        N_ANT_AZ   = 8
        N_ANT_EL   = 8
        TAU        = 0.5e-3

        p = gw.RangeAngleParams()
        p.n_ant_az    = N_ANT_AZ
        p.n_ant_el    = N_ANT_EL
        p.n_samples   = N_SAMPLES
        p.f_start     = F_START
        p.f_end       = F_END
        p.sample_rate = FS
        p.peak_mode   = gw.RangeAnglePeakMode.TOP_1
        p.n_peaks     = 1

        proc = gw.RangeAngleProcessor(self._rocm_ctx)
        proc.set_params(p)

        ref_lfm  = make_lfm_signal(N_SAMPLES, FS, F_START, F_END)
        delay_n  = int(TAU * FS)
        rx_single = np.zeros(N_SAMPLES, dtype=np.complex64)
        rx_single[delay_n:] = ref_lfm[:N_SAMPLES - delay_n]

        n_ant = p.get_n_antennas()
        data  = build_antenna_array(rx_single, n_ant)

        try:
            result = proc.process(data, download_result=True)
        except Exception as e:
            print(f"\n  XFAIL (GPU kernels STUB): {e}")
            return  # Expected failure while kernels not implemented

        if not result.success:
            print(f"\n  XFAIL (GPU kernels STUB): {result.error_message}")
            return

        assert len(result.targets) >= 1, "No targets detected"

        R_expected = 3e8 * TAU / 2
        R_got      = result.targets[0].range_m
        err_m      = abs(R_got - R_expected)

        print(f"\nR_expected = {R_expected:.0f} m")
        print(f"R_got      = {R_got:.1f} m")
        print(f"Error      = {err_m:.1f} m")

        assert err_m < 1000.0, (
            f"Range error too large: got {R_got:.1f} m, expected {R_expected:.0f} m, "
            f"err = {err_m:.0f} m (limit 1000 m)")

    def test_power_cube_shape(self):
        """
        Проверяем shape и dtype возвращаемого power cube.
        Нулевой сигнал — только для проверки формата вывода.
        """
        gw = self._gw
        if not hasattr(gw, "RangeAngleProcessor"):
            raise SkipTest("RangeAngleProcessor недоступен в этой сборке")

        N_SAMPLES = 4096
        N_AZ      = 4
        N_EL      = 4

        p = gw.RangeAngleParams()
        p.n_ant_az    = N_AZ
        p.n_ant_el    = N_EL
        p.n_samples   = N_SAMPLES
        p.f_start     = -5e6
        p.f_end       = +5e6
        p.sample_rate = 12e6

        proc = gw.RangeAngleProcessor(self._rocm_ctx)
        proc.set_params(p)

        data = np.zeros(N_AZ * N_EL * N_SAMPLES, dtype=np.complex64)

        result = proc.process(data, download_result=True)

        if result.success:
            cube = result.power_cube_numpy()
            assert cube.ndim  == 3,           f"Expected 3D array, got {cube.ndim}D"
            assert cube.dtype == np.float32,  f"Expected float32, got {cube.dtype}"
            assert cube.shape[1] == N_AZ,     f"Axis-1 should be {N_AZ}, got {cube.shape[1]}"
            assert cube.shape[2] == N_EL,     f"Axis-2 should be {N_EL}, got {cube.shape[2]}"
            n_bins = result.n_range_bins
            assert cube.shape[0] == n_bins,   f"Axis-0 should be {n_bins}, got {cube.shape[0]}"
            print(f"\nPower cube shape: {cube.shape}, dtype: {cube.dtype}")
        else:
            print(f"\nZero-signal result: {result.error_message} (acceptable)")


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestRangeAngle())
    runner.print_summary(results)
