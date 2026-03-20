"""
test_debug_steps.py — проверка каждого шага NumPy pipeline по отдельности (T2)
===============================================================================

ЗАЧЕМ:
    test_base_pipeline.py проверяет результат целиком. Этот файл идёт глубже —
    проверяет каждый шаг в отдельности с конкретными числовыми критериями.
    Если base pipeline упал, запусти этот тест чтобы понять на каком шаге ошибка.

    Шаги и что именно проверяется:
    - GEMM: shape (n_ant, n_samples) и coherent gain ≈ 1/sqrt(n_ant)
    - FFT:  peak bin ≈ expected_peak_bin (±2 бина) — только для SIN, не для ЛЧМ без дечирпа
    - OneMax: уточнённая частота через 3-точечную параболическую интерполяцию ≈ f0_hz
    - MinMax: max >= min, dynamic_range > 20 дБ

GPU: НЕ НУЖЕН — чистый NumPy.

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_debug_steps.py
"""

import sys
import os
import numpy as np

_PYTHON_TEST_DIR_EARLY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_TEST_DIR_EARLY not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_DIR_EARLY)
from common.runner import SkipTest

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from test_params import AntennaTestParams, SignalVariant
from signal_generators_strategy import SignalStrategyFactory

_PYTHON_TEST_DIR = os.path.dirname(_DIR)
if _PYTHON_TEST_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_DIR)
from common.result import TestResult, ValidationResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_and_run(variant: SignalVariant,
                      params: AntennaTestParams | None = None):
    """Генерировать сигнал, запустить NumPy pipeline, вернуть dict."""
    import math
    if params is None:
        params = AntennaTestParams.small(variant)

    strategy = SignalStrategyFactory.create(variant)
    S        = strategy.generate(params)

    # W = identity / sqrt(n_ant)
    W = (np.eye(params.n_ant, dtype=np.complex64) / np.sqrt(params.n_ant))
    X = (W @ S).astype(np.complex64)

    # Hamming + FFT
    n = params.n_samples
    nfft = 2 ** math.ceil(math.log2(n))
    win  = np.hamming(n).astype(np.float32)
    Xw   = (X * win[np.newaxis, :]).astype(np.complex64)
    Xp   = np.zeros((params.n_ant, nfft), dtype=np.complex64)
    Xp[:, :n] = Xw
    spec = np.fft.fft(Xp, axis=1).astype(np.complex64)
    mags = np.abs(spec).astype(np.float32)

    return dict(S=S, W=W, X=X, spectrum=spec, magnitudes=mags,
                nfft=nfft, params=params)


def _parabolic_peak(mags: np.ndarray, beam: int = 0,
                    half: bool = True) -> tuple[int, float]:
    """Найти пик и применить 3-точечную параболическую интерполяцию.

    Returns:
        (peak_bin, freq_offset) где offset ∈ [-0.5, +0.5]
    """
    n = mags.shape[1]
    limit = n // 2 if half else n
    peak_bin = int(np.argmax(mags[beam, 1:limit])) + 1
    if peak_bin == 0 or peak_bin >= limit - 1:
        return peak_bin, 0.0
    a, b, c = mags[beam, peak_bin - 1], mags[beam, peak_bin], mags[beam, peak_bin + 1]
    denom = 2.0 * b - a - c
    offset = 0.5 * (c - a) / max(denom, 1e-30) if denom > 0 else 0.0
    return peak_bin, float(np.clip(offset, -0.5, 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# STEP GEMM
# ─────────────────────────────────────────────────────────────────────────────

def test_gemm_shape_and_gain():
    """Step GEMM: shape = (n_ant, n_samples), gain ≈ 1/sqrt(n_ant) — для SIN и LFM_NO_DELAY."""
    for variant in [SignalVariant.SIN, SignalVariant.LFM_NO_DELAY]:
        d = _generate_and_run(variant)
        X, S, params = d["X"], d["S"], d["params"]

        # Shape
        assert X.shape == (params.n_ant, params.n_samples), \
            f"[{variant.name}] GEMM shape mismatch: {X.shape}"

        # Coherent gain (identity/sqrt(n_ant) матрица)
        gain     = float(np.mean(np.abs(X))) / max(float(np.mean(np.abs(S))), 1e-30)
        expected = 1.0 / np.sqrt(params.n_ant)
        err      = abs(gain - expected) / max(expected, 1e-30)
        assert err < 0.1, \
            f"[{variant.name}] GEMM gain error {err:.3f} > 0.1 (gain={gain:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# STEP FFT
# ─────────────────────────────────────────────────────────────────────────────

def test_fft_peak_location():
    """Step FFT: peak bin ≈ expected_peak_bin (±2 бина, только для SIN/CW)."""
    for variant in [SignalVariant.SIN, SignalVariant.LFM_NO_DELAY]:
        d = _generate_and_run(variant)
        mags, nfft, params = d["magnitudes"], d["nfft"], d["params"]

        if not params.check_peak_freq:
            continue  # LFM без дечирпа не даёт чёткий FFT-пик — пропустить

        bin_hz   = params.fs / nfft
        peak_bin = int(np.argmax(mags[0, 1:nfft // 2])) + 1
        found_f  = peak_bin * bin_hz
        err_hz   = abs(found_f - params.f0_hz)

        assert err_hz < 2.0 * bin_hz, \
            f"[{variant.name}] FFT peak freq error {err_hz:.1f} Hz > 2·bin={2*bin_hz:.1f} Hz"


# ─────────────────────────────────────────────────────────────────────────────
# STEP OneMax (паrabolic fit)
# ─────────────────────────────────────────────────────────────────────────────

def test_one_max_accuracy():
    """Step OneMax: refined_freq_hz ≈ f0_hz через 3-точечную параболу (только SIN/CW)."""
    for variant in [SignalVariant.SIN, SignalVariant.LFM_NO_DELAY]:
        d = _generate_and_run(variant)
        mags, nfft, params = d["magnitudes"], d["nfft"], d["params"]

        if not params.check_peak_freq:
            continue  # LFM без дечирпа — пропустить

        bin_hz         = params.fs / nfft
        peak_bin, offs = _parabolic_peak(mags, beam=0)
        refined_freq   = (peak_bin + offs) * bin_hz
        err_hz         = abs(refined_freq - params.f0_hz)

        assert err_hz < 2.0 * bin_hz, \
            f"[{variant.name}] OneMax refined_freq error {err_hz:.1f} Hz > 2·bin={2*bin_hz:.1f} Hz"


# ─────────────────────────────────────────────────────────────────────────────
# STEP MinMax
# ─────────────────────────────────────────────────────────────────────────────

def test_minmax_dynamic_range_loop():
    """Step MinMax: max >= min, dynamic_range > 20 dB — для SIN и LFM_NO_DELAY."""
    for variant in [SignalVariant.SIN, SignalVariant.LFM_NO_DELAY]:
        _test_minmax_dynamic_range_single(variant)


def _test_minmax_dynamic_range_single(variant: SignalVariant):
    """Step MinMax: max >= min, dynamic_range > 20 dB."""
    d    = _generate_and_run(variant)
    mags = d["magnitudes"]
    nfft = d["nfft"]

    max_mag = float(np.max(mags[0, :nfft // 2]))
    min_mag = float(np.min(mags[0, :nfft // 2]))

    assert max_mag >= min_mag, f"max_mag {max_mag:.4f} < min_mag {min_mag:.4f}"

    dr_db = 20.0 * np.log10(max_mag / max(min_mag, 1e-30))
    assert dr_db > 20.0, f"dynamic_range {dr_db:.1f} dB < 20 dB"
