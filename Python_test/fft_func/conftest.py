"""
conftest.py — pytest fixtures для Python_test/fft_maxima/
==========================================================

Fixtures:
    maxima_plot_dir — Results/Plots/fft_maxima/
    maxima_finder   — SpectrumMaximaFinder (GPU, scope=session)
    cw_spectrum     — спектр CW-сигнала с известным пиком
"""

import os
import pytest
import numpy as np

_FS = 12e6
_N_SAMPLES = 4096
_F_PEAK = 2e6      # Гц — ожидаемый пик


@pytest.fixture(scope="session")
def maxima_plot_dir(plot_dir) -> str:
    path = os.path.join(plot_dir, "fft_maxima")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def maxima_finder(gw, gpu_ctx):
    """SpectrumMaximaFinder (skip если нет GPU)."""
    if not hasattr(gw, "SpectrumMaximaFinder"):
        pytest.skip("SpectrumMaximaFinder не доступен в этой сборке")
    return gw.SpectrumMaximaFinder(gpu_ctx)


@pytest.fixture(scope="module")
def cw_spectrum() -> tuple:
    """Спектр CW-сигнала на F_PEAK.

    Returns:
        (spectrum [N_SAMPLES] complex64, expected_freq_hz)
    """
    t = np.arange(_N_SAMPLES) / _FS
    signal = np.exp(1j * 2 * np.pi * _F_PEAK * t).astype(np.complex64)
    spectrum = np.fft.fft(signal).astype(np.complex64)
    return spectrum, _F_PEAK


@pytest.fixture
def noise_spectrum() -> np.ndarray:
    """Спектр шумового сигнала (равномерный)."""
    rng = np.random.default_rng(42)
    signal = (rng.standard_normal(_N_SAMPLES) +
              1j * rng.standard_normal(_N_SAMPLES)).astype(np.complex64)
    return np.fft.fft(signal).astype(np.complex64)
