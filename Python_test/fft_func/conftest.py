"""
conftest.py — фабричные функции для Python_test/fft_func/
==========================================================

Без pytest. Предоставляет factory functions для создания тестовых объектов.

Использование:
    from conftest import make_maxima_finder, make_cw_spectrum, maxima_plot_dir
"""

import os
import numpy as np

_FS = 12e6
_N_SAMPLES = 4096
_F_PEAK = 2e6      # Гц — ожидаемый пик

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
maxima_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "fft_maxima")
os.makedirs(maxima_plot_dir, exist_ok=True)


def make_maxima_finder(gw, gpu_ctx):
    """SpectrumMaximaFinder. SkipTest если нет GPU или класса."""
    from common.runner import SkipTest
    if not hasattr(gw, "SpectrumMaximaFinder"):
        raise SkipTest("SpectrumMaximaFinder не доступен в этой сборке")
    return gw.SpectrumMaximaFinder(gpu_ctx)


def make_cw_spectrum() -> tuple:
    """Спектр CW-сигнала на F_PEAK.

    Returns:
        (spectrum [N_SAMPLES] complex64, expected_freq_hz)
    """
    t = np.arange(_N_SAMPLES) / _FS
    signal = np.exp(1j * 2 * np.pi * _F_PEAK * t).astype(np.complex64)
    spectrum = np.fft.fft(signal).astype(np.complex64)
    return spectrum, _F_PEAK


def make_noise_spectrum() -> np.ndarray:
    """Спектр шумового сигнала (равномерный)."""
    rng = np.random.default_rng(42)
    signal = (rng.standard_normal(_N_SAMPLES) +
              1j * rng.standard_normal(_N_SAMPLES)).astype(np.complex64)
    return np.fft.fft(signal).astype(np.complex64)
