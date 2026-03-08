"""
conftest.py — pytest fixtures для Python_test/lch_farrow/
==========================================================

Fixtures:
    lch_farrow_plot_dir — Results/Plots/lch_farrow/
    farrow_proc         — LchFarrowROCm (GPU, scope=session)
    farrow_delay        — FarrowDelay (numpy)
    test_signal_2d      — тестовый сигнал [n_ant, n_samples]
"""

import os
import sys
import pytest
import numpy as np

# Добавить strategies/ в path для FarrowDelay
_STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), '..', 'strategies')
if os.path.isdir(_STRATEGIES_DIR) and _STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_STRATEGIES_DIR))

_N_ANT = 4
_N_SAMPLES = 4096
_FS = 12e6


@pytest.fixture(scope="session")
def lch_farrow_plot_dir(plot_dir) -> str:
    path = os.path.join(plot_dir, "lch_farrow")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def farrow_proc(gw, gpu_ctx):
    """LchFarrowROCm GPU-процессор (skip если нет GPU)."""
    if not hasattr(gw, "LchFarrowROCm"):
        pytest.skip("LchFarrowROCm не доступен в этой сборке")
    return gw.LchFarrowROCm(gpu_ctx)


@pytest.fixture(scope="session")
def farrow_numpy():
    """FarrowDelay — numpy реализация."""
    from farrow_delay import FarrowDelay
    return FarrowDelay()


@pytest.fixture
def test_signal_2d() -> np.ndarray:
    """Тестовый сигнал [n_ant, n_samples] complex64."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((_N_ANT, _N_SAMPLES)) +
            1j * rng.standard_normal((_N_ANT, _N_SAMPLES))).astype(np.complex64)


@pytest.fixture
def delays_samples() -> np.ndarray:
    """Тестовые задержки [n_ant] в отсчётах (с дробной частью)."""
    return np.array([0.0, 1.5, 3.24, 7.8], dtype=np.float64)[:_N_ANT]


@pytest.fixture
def delays_seconds(delays_samples) -> np.ndarray:
    """Задержки в секундах."""
    return delays_samples / _FS
