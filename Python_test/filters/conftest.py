"""
conftest.py — фабричные функции для Python_test/filters/
=========================================================

Предоставляет factory functions для тестов фильтров.
"""

import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Параметры тестов по умолчанию
# ─────────────────────────────────────────────────────────────────────────────

_FS = 50_000.0       # Hz
_N_SAMPLES = 4096
_N_CHANNELS = 8
_FIR_TAPS = 64
_FIR_CUTOFF = 0.1    # нормированная (Nyquist=1)
_F_LOW = 200.0       # Hz
_F_HIGH = 8_000.0    # Hz

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filter_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "filters")
os.makedirs(filter_plot_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────────────────────────────────────

def make_fir_coeffs() -> np.ndarray:
    """FIR-коэффициенты (Kaiser window, 64 taps, cutoff=0.1). SkipTest если нет scipy."""
    from common.runner import SkipTest
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise SkipTest("scipy required for FIR coeffs")
    return scipy_signal.firwin(_FIR_TAPS, _FIR_CUTOFF,
                               window="kaiser").astype(np.float32)


def make_iir_coeffs() -> tuple:
    """IIR Butterworth 4-го порядка LP @ 0.1*Nyquist. SkipTest если нет scipy.

    Returns:
        (b, a) — числитель и знаменатель передаточной функции
    """
    from common.runner import SkipTest
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise SkipTest("scipy required for IIR coeffs")
    b, a = scipy_signal.butter(4, _FIR_CUTOFF, btype="low")
    return b.astype(np.float64), a.astype(np.float64)


def make_complex_signal() -> np.ndarray:
    """Одноканальный комплексный сигнал [n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(_N_SAMPLES) +
            1j * rng.standard_normal(_N_SAMPLES)).astype(np.complex64)


def make_multichannel_signal() -> np.ndarray:
    """Многоканальный сигнал [n_channels, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((_N_CHANNELS, _N_SAMPLES)) +
            1j * rng.standard_normal((_N_CHANNELS, _N_SAMPLES))).astype(np.complex64)


def make_two_tone_signal() -> np.ndarray:
    """Двухтональный сигнал: f_low (в полосе) + f_high (вне полосы)."""
    t = np.arange(_N_SAMPLES, dtype=np.float32) / np.float32(_FS)
    sig = (np.cos(2 * np.pi * _F_LOW * t) + 0.5 * np.cos(2 * np.pi * _F_HIGH * t) +
           1j * (np.sin(2 * np.pi * _F_LOW * t) + 0.5 * np.sin(2 * np.pi * _F_HIGH * t)))
    return sig.astype(np.complex64)
