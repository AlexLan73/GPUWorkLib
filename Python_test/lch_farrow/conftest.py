"""
conftest.py — фабричные функции для Python_test/lch_farrow/
============================================================

Предоставляет factory functions для тестов LCH Farrow.
"""

import os
import sys
import numpy as np

# Добавить strategies/ в path для FarrowDelay
_STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), '..', 'strategies')
if os.path.isdir(_STRATEGIES_DIR) and _STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_STRATEGIES_DIR))

_N_ANT = 4
_N_SAMPLES = 4096
_FS = 12e6

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lch_farrow_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "lch_farrow")
os.makedirs(lch_farrow_plot_dir, exist_ok=True)


def make_farrow_proc(gw, gpu_ctx):
    """LchFarrowROCm GPU-процессор. SkipTest если нет GPU или класса."""
    from common.runner import SkipTest
    if not hasattr(gw, "LchFarrowROCm"):
        raise SkipTest("LchFarrowROCm не доступен в этой сборке")
    return gw.LchFarrowROCm(gpu_ctx)


def make_farrow_numpy():
    """FarrowDelay — numpy реализация."""
    from farrow_delay import FarrowDelay
    return FarrowDelay()


def make_test_signal_2d(n_ant: int = _N_ANT,
                        n_samples: int = _N_SAMPLES) -> np.ndarray:
    """Тестовый сигнал [n_ant, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((n_ant, n_samples)) +
            1j * rng.standard_normal((n_ant, n_samples))).astype(np.complex64)


def make_delays_samples(n_ant: int = _N_ANT) -> np.ndarray:
    """Тестовые задержки [n_ant] в отсчётах (с дробной частью)."""
    return np.array([0.0, 1.5, 3.24, 7.8], dtype=np.float64)[:n_ant]


def make_delays_seconds(n_ant: int = _N_ANT, fs: float = _FS) -> np.ndarray:
    """Задержки в секундах."""
    return make_delays_samples(n_ant) / fs
