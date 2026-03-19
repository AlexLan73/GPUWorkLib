"""
conftest.py — фабричные функции для Python_test/vector_algebra/
================================================================

Без pytest. Предоставляет factory functions для тестов линейной алгебры.
"""

import os
import numpy as np

_N = 8   # размер матрицы по умолчанию

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
va_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "vector_algebra")
os.makedirs(va_plot_dir, exist_ok=True)


def make_cholesky_solver(gw, gpu_ctx):
    """CholeskyInverter. SkipTest если нет GPU или класса."""
    from common.runner import SkipTest
    if not hasattr(gw, "CholeskyInverter"):
        raise SkipTest("CholeskyInverter не доступен в этой сборке")
    return gw.CholeskyInverter(gpu_ctx)


def make_spd_matrix(n: int = _N) -> np.ndarray:
    """Симметричная положительно-определённая матрица [N×N] complex128.

    Генерируется как A = R†R + N*I (гарантированно SPD).
    """
    rng = np.random.default_rng(42)
    R = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = R.conj().T @ R + n * np.eye(n)
    return A.astype(np.complex128)


def make_spd_matrix_batch(n: int = _N, batch: int = 4) -> np.ndarray:
    """Батч SPD матриц [batch, N, N]."""
    base = make_spd_matrix(n)
    return np.stack([base * (i + 1) for i in range(batch)])
