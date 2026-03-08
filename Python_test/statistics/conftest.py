"""
conftest.py — pytest fixtures для Python_test/statistics/
==========================================================

Fixtures:
    stats_plot_dir  — Results/Plots/statistics/
    stats_proc      — StatisticsProcessor (GPU, scope=session)
    random_matrix   — случайная матрица [n_ch, n_samples] complex64
"""

import os
import pytest
import numpy as np

_N_CH = 8
_N_SAMPLES = 4096


@pytest.fixture(scope="session")
def stats_plot_dir(plot_dir) -> str:
    path = os.path.join(plot_dir, "statistics")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def stats_proc(gw, gpu_ctx):
    """StatisticsProcessor (skip если нет GPU)."""
    if not hasattr(gw, "StatisticsProcessor"):
        pytest.skip("StatisticsProcessor не доступен в этой сборке")
    return gw.StatisticsProcessor(gpu_ctx)


@pytest.fixture
def random_matrix() -> np.ndarray:
    """Случайная матрица [n_ch, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((_N_CH, _N_SAMPLES)) +
            1j * rng.standard_normal((_N_CH, _N_SAMPLES))).astype(np.complex64)


@pytest.fixture
def real_matrix() -> np.ndarray:
    """Вещественная матрица [n_ch, n_samples] float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_N_CH, _N_SAMPLES)).astype(np.float32)
