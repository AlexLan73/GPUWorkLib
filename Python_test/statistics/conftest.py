"""
conftest.py — фабричные функции для Python_test/statistics/
============================================================

Предоставляет factory functions для тестов статистики.
"""

import os
import numpy as np

_N_CH = 8
_N_SAMPLES = 4096

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stats_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "statistics")
os.makedirs(stats_plot_dir, exist_ok=True)


def make_stats_proc(gw, gpu_ctx):
    """StatisticsProcessor. SkipTest если нет GPU или класса."""
    from common.runner import SkipTest
    if not hasattr(gw, "StatisticsProcessor"):
        raise SkipTest("StatisticsProcessor не доступен в этой сборке")
    return gw.StatisticsProcessor(gpu_ctx)


def make_random_matrix() -> np.ndarray:
    """Случайная матрица [n_ch, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((_N_CH, _N_SAMPLES)) +
            1j * rng.standard_normal((_N_CH, _N_SAMPLES))).astype(np.complex64)


def make_real_matrix() -> np.ndarray:
    """Вещественная матрица [n_ch, n_samples] float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_N_CH, _N_SAMPLES)).astype(np.float32)
