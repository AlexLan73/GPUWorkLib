"""
conftest.py — pytest fixtures для Python_test/filters/
========================================================

Fixtures наследуют gpu_ctx, gw из родительского conftest.py.
Здесь добавляются специфичные для фильтров fixtures.

Fixtures:
    filter_plot_dir — директория для графиков Results/Plots/filters/
    fir_coeffs      — стандартные FIR-коэффициенты для тестов (scipy Kaiser)
    iir_coeffs_b/a  — стандартные IIR-коэффициенты (scipy Butterworth)
    complex_signal  — тестовый комплексный сигнал [n_samples]
    multichannel_signal — тестовый сигнал [n_channels, n_samples]
"""

import os
import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Параметры тестов по умолчанию
# ─────────────────────────────────────────────────────────────────────────────

_FS = 50_000.0       # Hz
_N_SAMPLES = 4096
_N_CHANNELS = 8
_FIR_TAPS = 64
_FIR_CUTOFF = 0.1    # нормированная (Nyquist=1)
_F_LOW = 200.0       # Hz — полоса пропускания
_F_HIGH = 8_000.0    # Hz — полоса задерживания
_ATOL = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def filter_plot_dir(plot_dir) -> str:
    """Директория Results/Plots/filters/."""
    path = os.path.join(plot_dir, "filters")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def fs() -> float:
    """Частота дискретизации для тестов фильтров."""
    return _FS


@pytest.fixture(scope="session")
def fir_coeffs() -> np.ndarray:
    """FIR-коэффициенты (Kaiser window, 64 taps, cutoff=0.1).

    Вычисляется через scipy один раз для всей сессии.
    """
    scipy_signal = pytest.importorskip("scipy.signal",
                                        reason="scipy required for FIR coeffs")
    return scipy_signal.firwin(_FIR_TAPS, _FIR_CUTOFF,
                               window="kaiser").astype(np.float32)


@pytest.fixture(scope="session")
def iir_coeffs():
    """IIR Butterworth 4-го порядка LP @ 0.1*Nyquist.

    Returns:
        (b, a) — числитель и знаменатель передаточной функции
    """
    scipy_signal = pytest.importorskip("scipy.signal",
                                        reason="scipy required for IIR coeffs")
    b, a = scipy_signal.butter(4, _FIR_CUTOFF, btype="low")
    return b.astype(np.float64), a.astype(np.float64)


@pytest.fixture
def complex_signal() -> np.ndarray:
    """Одноканальный комплексный сигнал [n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(_N_SAMPLES) +
            1j * rng.standard_normal(_N_SAMPLES)).astype(np.complex64)


@pytest.fixture
def multichannel_signal() -> np.ndarray:
    """Многоканальный сигнал [n_channels, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((_N_CHANNELS, _N_SAMPLES)) +
            1j * rng.standard_normal((_N_CHANNELS, _N_SAMPLES))).astype(np.complex64)


@pytest.fixture
def two_tone_signal() -> np.ndarray:
    """Двухтональный сигнал: f_low (в полосе) + f_high (вне полосы)."""
    t = np.arange(_N_SAMPLES, dtype=np.float32) / np.float32(_FS)
    sig = (np.cos(2 * np.pi * _F_LOW * t) + 0.5 * np.cos(2 * np.pi * _F_HIGH * t) +
           1j * (np.sin(2 * np.pi * _F_LOW * t) + 0.5 * np.sin(2 * np.pi * _F_HIGH * t)))
    return sig.astype(np.complex64)
