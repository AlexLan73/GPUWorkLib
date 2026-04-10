#!/usr/bin/env python3
"""
test_snr_estimator.py — e2e тест SNR-estimator (SNR_10)
========================================================

Проверяет полный pipeline:
    StatisticsProcessor.compute_snr_db (GPU ROCm)
        → сравнение с numpy reference (cfar_estimator.py)
    BranchSelector.select → Low/Mid/High

Numpy reference берётся из PyPanelAntennas/SNR/ (SNR_00 Python модель).

⚠️ ЗАПРЕЩЕНО pytest! Используется TestRunner + SkipTest из common/runner.py.

Запуск (в понедельник на Debian/AMD):
    "F:/Program Files (x86)/Python314/python.exe" Python_test/statistics/test_snr_estimator.py
    # или на Debian:
    python3 Python_test/statistics/test_snr_estimator.py

Tests:
  test_01_single_antenna_high_snr  — SNR_in=20 dB, numpy cross-check
  test_02_multi_antenna_medium_snr — 50 антенн, SNR_in=5 dB
  test_03_noise_only_branch_low    — чистый шум, BranchSelector→Low

Author: Kodo (AI Assistant)
Date: 2026-04-09
"""

import sys
from pathlib import Path

import numpy as np

# ── Project root + paths ─────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # <repo>/
sys.path.insert(0, str(_ROOT / "Python_test"))
sys.path.insert(0, str(_ROOT / "PyPanelAntennas" / "SNR"))  # numpy reference

# ── Test infrastructure (НЕ pytest!) ─────────────────────────────────────────
from common.runner import TestRunner, SkipTest  # noqa: E402
from common.gpu_loader import GPULoader  # noqa: E402


def _get_gpu_module_or_skip():
    """Загрузить gpuworklib или SkipTest."""
    gw = GPULoader.get()
    if gw is None:
        raise SkipTest("gpuworklib not built — run CMake build first")
    if not hasattr(gw, "StatisticsProcessor"):
        raise SkipTest("StatisticsProcessor not in bindings")
    if not hasattr(gw, "SnrEstimationConfig"):
        raise SkipTest("SNR bindings not built (SNR_07) — rebuild needed")
    return gw


def _import_cfar_or_skip():
    """Импорт numpy reference — пропуск если недоступен."""
    try:
        import cfar_estimator  # type: ignore
        return cfar_estimator
    except ImportError as e:
        raise SkipTest(f"cfar_estimator.py not found: {e}")


# =============================================================================
# Test class (обычный Python class, НЕ pytest)
# =============================================================================

class TestSnrEstimator:
    """e2e тесты SNR-estimator (compute_snr_db)."""

    def __init__(self):
        self.gw = _get_gpu_module_or_skip()
        self.ctx = self.gw.GPUContext()
        self.stat_proc = self.gw.StatisticsProcessor(self.ctx)
        self.cfar_mod = _import_cfar_or_skip()

    # ─────────────────────────────────────────────────────────────────────────
    # test_01 — 1 антенна, SNR_in=20 dB, numpy cross-check
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_single_antenna_high_snr(self):
        n_samp = 5000
        snr_in_db = 20.0
        amplitude = 10.0 ** (snr_in_db / 20.0)  # A = sqrt(SNR_linear)

        rng = np.random.default_rng(42)
        t = np.arange(n_samp, dtype=np.float32)
        signal = (amplitude * np.exp(1j * 2 * np.pi * 0.15 * t)).astype(np.complex64)
        noise = (rng.standard_normal(n_samp) +
                 1j * rng.standard_normal(n_samp)).astype(np.complex64) / np.sqrt(2)
        data = (signal + noise).reshape(1, n_samp)  # (1, 5000)

        # GPU path
        cfg = self.gw.SnrEstimationConfig()
        # defaults: target_n_fft=0→2048, Hann, guard=5, ref=16
        result = self.stat_proc.compute_snr_db(
            data=data, n_antennas=1, n_samples=n_samp, config=cfg)
        gpu_snr_db = result.snr_db_global

        # Numpy reference (Hann + mean CA-CFAR — match GPU defaults)
        ref = self.cfar_mod.CfarEstimator(
            target_n_fft=2048,
            guard_bins=5,        # калибровано (то же что и C++ defaults)
            ref_bins=16,         # калибровано
            search_full_spectrum=True,
            window="hann",
        )
        ref_snr_db = ref.estimate(data[0])

        diff = abs(gpu_snr_db - ref_snr_db)
        print(f"test_01: GPU={gpu_snr_db:.2f} dB, numpy={ref_snr_db:.2f} dB, diff={diff:.2f} dB")

        # Tolerance 1.5 dB: numpy FFT vs rocFFT имеют разные округления на float32
        assert diff < 1.5, f"GPU vs numpy diff > 1.5 dB: {diff}"
        assert gpu_snr_db > 38.0, f"snr_db_global too low: {gpu_snr_db}"
        assert result.used_bins >= 1024  # target_n_fft=2048 → nFFT >= 1024 после децимации

    # ─────────────────────────────────────────────────────────────────────────
    # test_02 — 50 антенн, SNR_in=5 dB, проверка used_antennas
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_multi_antenna_medium_snr(self):
        n_ant = 50
        n_samp = 5000
        snr_in_db = 5.0
        amp = 10.0 ** (snr_in_db / 20.0)

        rng = np.random.default_rng(123)
        data = np.zeros((n_ant, n_samp), dtype=np.complex64)
        t = np.arange(n_samp, dtype=np.float32)
        for a in range(n_ant):
            freq = 0.1 + 0.15 * rng.random()
            sig = amp * np.exp(1j * 2 * np.pi * freq * t)
            noise = (rng.standard_normal(n_samp) +
                     1j * rng.standard_normal(n_samp)) / np.sqrt(2)
            data[a] = (sig + noise).astype(np.complex64)

        cfg = self.gw.SnrEstimationConfig()
        # 50 антенн ≤ kTargetAntennasMedian=50 → step_antennas auto = 1 → used_antennas = 50
        result = self.stat_proc.compute_snr_db(
            data=data, n_antennas=n_ant, n_samples=n_samp, config=cfg)

        print(f"test_02: GPU snr_db={result.snr_db_global:.2f}, "
              f"used_ant={result.used_antennas}, used_bins={result.used_bins}")

        # SNR_fft ≈ 5 + 10*log10(1666) ≈ 37 dB (с bias → 27..42)
        assert 25.0 < result.snr_db_global < 45.0, \
            f"snr_db_global out of range: {result.snr_db_global}"
        assert result.used_antennas == n_ant

    # ─────────────────────────────────────────────────────────────────────────
    # test_03 — только шум → BranchSelector → Low
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_noise_only_branch_low(self):
        n_ant = 50
        n_samp = 5000

        rng = np.random.default_rng(7)
        data = (rng.standard_normal((n_ant, n_samp)) +
                1j * rng.standard_normal((n_ant, n_samp))) / np.sqrt(2)
        data = data.astype(np.complex64)

        cfg = self.gw.SnrEstimationConfig()
        # Defaults: low_to_mid=15, mid_to_high=30 → шум (≈8-10 dB) должен быть Low
        result = self.stat_proc.compute_snr_db(
            data=data, n_antennas=n_ant, n_samples=n_samp, config=cfg)

        print(f"test_03: noise-only snr_db={result.snr_db_global:.2f}")
        # CFAR артефакт на чистом шуме
        assert 3.0 < result.snr_db_global < 18.0

        # BranchSelector → Low
        selector = self.gw.BranchSelector()
        branch = selector.select(result.snr_db_global, cfg.thresholds)
        assert branch == self.gw.BranchType.Low, \
            f"Expected Low, got {branch}"

    # ─────────────────────────────────────────────────────────────────────────
    # test_04 — config.validate() бросает при некорректных параметрах
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_config_validation(self):
        cfg = self.gw.SnrEstimationConfig()
        # ref window слишком большой для малого nFFT
        cfg.target_n_fft = 32
        cfg.guard_bins = 10
        cfg.ref_bins = 20   # 2*(10+20)+1 = 61 > 32 → должно бросать

        raised = False
        try:
            cfg.validate()
        except Exception as e:  # noqa: BLE001
            raised = True
            print(f"test_04: validate raised (expected): {e}")
        assert raised, "cfg.validate() должен был бросить исключение"


# =============================================================================
# main — запуск через TestRunner
# =============================================================================

def main():
    runner = TestRunner()
    try:
        test_obj = TestSnrEstimator()
    except SkipTest as e:
        print(f"[SKIP] Suite init failed: {e}")
        return
    results = runner.run(test_obj)
    runner.print_summary(results)


if __name__ == "__main__":
    main()
