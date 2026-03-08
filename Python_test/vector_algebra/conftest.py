"""
conftest.py — pytest fixtures для Python_test/vector_algebra/
==============================================================

Fixtures:
    va_plot_dir     — Results/Plots/vector_algebra/
    cholesky_solver — CholeskyInverter (GPU, scope=session)
    spd_matrix      — симметричная положительно-определённая матрица [N×N]
"""

import os
import pytest
import numpy as np

_N = 8   # размер матрицы


@pytest.fixture(scope="session")
def va_plot_dir(plot_dir) -> str:
    path = os.path.join(plot_dir, "vector_algebra")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def cholesky_solver(gw, gpu_ctx):
    """CholeskyInverter (skip если нет GPU)."""
    if not hasattr(gw, "CholeskyInverter"):
        pytest.skip("CholeskyInverter не доступен в этой сборке")
    return gw.CholeskyInverter(gpu_ctx)


@pytest.fixture(scope="module")
def spd_matrix() -> np.ndarray:
    """Симметричная положительно-определённая матрица [N×N] complex128.

    Генерируется как A = R†R + N*I (гарантированно SPD).
    """
    rng = np.random.default_rng(42)
    R = rng.standard_normal((_N, _N)) + 1j * rng.standard_normal((_N, _N))
    A = R.conj().T @ R + _N * np.eye(_N)
    return A.astype(np.complex128)


@pytest.fixture(scope="module")
def spd_matrix_batch(spd_matrix) -> np.ndarray:
    """Батч SPD матриц [batch=4, N, N]."""
    return np.stack([spd_matrix * (i + 1) for i in range(4)])
