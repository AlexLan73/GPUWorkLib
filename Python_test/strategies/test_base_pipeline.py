"""
test_base_pipeline.py — проверка математики pipeline без GPU (T1)
==================================================================

ЗАЧЕМ:
    Проверяет что алгоритм (GEMM + Hamming + FFT + поиск пика) математически
    правильный на NumPy, прежде чем запускать на GPU.
    Если этот тест падает — сломана математика, не GPU.
    Если этот тест проходит — ошибки на GPU не в алгоритме, а в реализации ядер.

ЧТО ПРОВЕРЯЕТ:
    Полный NumPy pipeline: S → GEMM(W) → Hamming window → FFT → argmax
    Для 4 вариантов сигнала: SIN, LFM_NO_DELAY, LFM_WITH_DELAY, LFM_FARROW.
    Критерии: peak_freq ≈ f0_hz (±2 бина), dynamic_range > 20 дБ.

GPU: НЕ НУЖЕН — чистый NumPy.

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_base_pipeline.py
"""

import sys
import os
import numpy as np

# Добавить strategies/ в sys.path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from test_params import AntennaTestParams, SignalVariant
from signal_generators_strategy import SignalStrategyFactory
from strategy_test_base import StrategyTestBase

_PYTHON_TEST_DIR = os.path.dirname(_DIR)
if _PYTHON_TEST_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_DIR)
from common.result import TestResult, ValidationResult


# ─────────────────────────────────────────────────────────────────────────────
# NumpyPipelineTest — NumPy reference реализация (не требует GPU)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyPipelineTest(StrategyTestBase):
    """Полный NumPy pipeline тест (T1).

    process() запускает _run_numpy_pipeline().
    validate() проверяет peak_freq и dynamic_range.
    """

    MIN_DYNAMIC_RANGE_DB: float = 20.0

    def process(self, data: np.ndarray, ctx) -> dict:
        return self._run_numpy_pipeline(data)

    def validate(self, result: dict, params: AntennaTestParams) -> TestResult:
        tr = TestResult(test_name=self.name)

        peak_freq = result["peak_freq_hz"]
        bin_hz    = params.bin_hz
        freq_err  = abs(peak_freq - params.f0_hz)

        # Проверка 1: частота пика ≈ f0 (только для SIN/CW; LFM без дечирпа не даёт чёткий пик)
        if params.check_peak_freq:
            tr.add(ValidationResult(
                passed    = freq_err < 2.0 * bin_hz,
                metric_name = "peak_freq_error_hz",
                actual_value = freq_err,
                threshold    = 2.0 * bin_hz,
                message      = f"found={peak_freq:.1f} Hz, f0={params.f0_hz:.1f} Hz",
            ))

        # Проверка 2: dynamic_range_dB (beam 0)
        mags = result["magnitudes"]
        max_mag = float(np.max(mags[0, :mags.shape[1] // 2]))
        min_mag = float(np.min(mags[0, :mags.shape[1] // 2]))
        min_safe = max(min_mag, 1e-30)
        dr_db = 20.0 * np.log10(max_mag / min_safe)

        tr.add(ValidationResult(
            passed     = dr_db > self.MIN_DYNAMIC_RANGE_DB,
            metric_name = "dynamic_range_dB",
            actual_value = dr_db,
            threshold    = self.MIN_DYNAMIC_RANGE_DB,
            message      = f"beam0 DR={dr_db:.1f} dB",
        ))

        # Проверка 3: GEMM coherent gain ≈ 1/sqrt(n_ant) * n_ant = sqrt(n_ant)
        X = result["X_gemm"]
        S = self._last_S  # сохранено в process()
        if S is not None and X is not None:
            gain = float(np.mean(np.abs(X)) / max(np.mean(np.abs(S)), 1e-30))
            expected_gain = 1.0 / np.sqrt(params.n_ant)  # identity/sqrt(n) matrix
            gain_err = abs(gain - expected_gain) / max(expected_gain, 1e-30)
            tr.add(ValidationResult(
                passed       = gain_err < 0.1,
                metric_name  = "gemm_gain_error_rel",
                actual_value = gain_err,
                threshold    = 0.1,
                message      = f"gain={gain:.4f}, expected={expected_gain:.4f}",
            ))

        return tr

    def process(self, data: np.ndarray, ctx) -> dict:
        self._last_S = data.copy()
        return self._run_numpy_pipeline(data)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _run_variant(variant: SignalVariant,
                 params: AntennaTestParams | None = None) -> TestResult:
    """Вспомогательная функция: запустить NumpyPipelineTest для variant."""
    if params is None:
        params = AntennaTestParams.small(variant)
    strategy = SignalStrategyFactory.create(variant)
    test     = NumpyPipelineTest(strategy, params)
    return test.run()


def test_sin_full_pipeline():
    """T1: SIN → NumPy GEMM + FFT + peak validation."""
    result = _run_variant(SignalVariant.SIN)
    print(result.summary())
    assert result.passed, result.summary()


def test_lfm_no_delay_pipeline():
    """T1: LFM_NO_DELAY → NumPy pipeline."""
    result = _run_variant(SignalVariant.LFM_NO_DELAY)
    print(result.summary())
    assert result.passed, result.summary()


def test_lfm_delay_pipeline():
    """T1: LFM_WITH_DELAY → NumPy pipeline (целочисленные задержки)."""
    result = _run_variant(SignalVariant.LFM_WITH_DELAY)
    print(result.summary())
    assert result.passed, result.summary()


def test_lfm_farrow_pipeline():
    """T1: LFM_FARROW → NumPy pipeline (дробные задержки Farrow)."""
    result = _run_variant(SignalVariant.LFM_FARROW)
    print(result.summary())
    assert result.passed, result.summary()


def test_all_variants():
    """T1: все 4 варианта сигнала (явный цикл вместо parametrize)."""
    for variant in [
        SignalVariant.SIN,
        SignalVariant.LFM_NO_DELAY,
        SignalVariant.LFM_WITH_DELAY,
        SignalVariant.LFM_FARROW,
    ]:
        result = _run_variant(variant)
        assert result.passed, f"{variant.name}: {result.summary()}"
