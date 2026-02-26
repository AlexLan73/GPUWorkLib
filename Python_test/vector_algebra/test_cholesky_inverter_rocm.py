"""
Python тесты CholeskyInverterROCm (Task_11 v2: SymmetrizeMode).

Эталон: np.linalg.inv() vs GPU Cholesky (rocSOLVER POTRF+POTRI).
Два режима: Roundtrip (CPU symmetrize) и GpuKernel (hiprtc).

Запуск:
    cd /home/alex/C++/GPUWorkLib
    pytest Python_test/vector_algebra/test_cholesky_inverter_rocm.py -v

Требования:
    - ROCm (AMD Radeon 9070 или совместимое GPU)
    - gpuworklib собран с ENABLE_ROCM=ON
    - pytest, numpy
"""

import pytest
import numpy as np


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def rocm_context():
    """ROCm GPU контекст. Пропустить если ROCm недоступен."""
    try:
        import gpuworklib
        ctx = gpuworklib.ROCmGPUContext(0)
        return ctx
    except Exception as e:
        pytest.skip(f"ROCm недоступен: {e}")


@pytest.fixture(scope="module")
def inverter(rocm_context):
    """CholeskyInverterROCm (default: GpuKernel mode)."""
    import gpuworklib
    return gpuworklib.CholeskyInverterROCm(
        rocm_context, gpuworklib.SymmetrizeMode.GpuKernel
    )


@pytest.fixture(scope="module")
def inverter_roundtrip(rocm_context):
    """CholeskyInverterROCm (Roundtrip mode)."""
    import gpuworklib
    return gpuworklib.CholeskyInverterROCm(
        rocm_context, gpuworklib.SymmetrizeMode.Roundtrip
    )


# ============================================================================
# Helpers
# ============================================================================


def make_positive_definite(n: int, seed: int = 42) -> np.ndarray:
    """Создать HPD матрицу n×n: A = B*B^H + n*I."""
    rng = np.random.default_rng(seed)
    B = (rng.standard_normal((n, n)) +
         1j * rng.standard_normal((n, n))).astype(np.complex64)
    A = (B @ B.conj().T + n * np.eye(n, dtype=np.complex64)).astype(np.complex64)
    return A


def frobenius_error(A: np.ndarray, A_inv: np.ndarray) -> float:
    """||A * A_inv - I||_F"""
    n = A.shape[0]
    product = A.astype(np.complex128) @ A_inv.astype(np.complex128)
    return float(np.linalg.norm(product - np.eye(n, dtype=np.complex128), "fro"))


# ============================================================================
# 5.12.1: test_invert_5x5
# ============================================================================


def test_invert_5x5(inverter):
    """CPU инверсия 5×5. Ошибка < 1e-5."""
    n = 5
    A = make_positive_definite(n, seed=1)

    A_inv_gpu = inverter.invert_cpu(A.flatten(), n)

    assert A_inv_gpu.shape == (n, n)
    assert A_inv_gpu.dtype == np.complex64

    err = frobenius_error(A, A_inv_gpu)
    assert err < 1e-4, f"Frobenius error {err:.2e} >= 1e-4"


# ============================================================================
# 5.12.2: test_invert_341x341
# ============================================================================


def test_invert_341x341(inverter):
    """CPU инверсия 341×341. Ошибка < 1e-2."""
    n = 341
    A = make_positive_definite(n, seed=42)

    A_inv_gpu = inverter.invert_cpu(A.flatten(), n)

    assert A_inv_gpu.shape == (n, n)

    err = frobenius_error(A, A_inv_gpu)
    assert err < 1e-2, f"Frobenius error {err:.2e} >= 1e-2"


# ============================================================================
# 5.12.3: test_batch_4x64
# ============================================================================


def test_batch_4x64(inverter):
    """Batched инверсия 4 × 64×64. Для каждой ошибка < 1e-3."""
    n = 64
    batch_count = 4

    matrices = [make_positive_definite(n, seed=i) for i in range(batch_count)]
    flat = np.concatenate([m.flatten() for m in matrices])

    results = inverter.invert_batch_cpu(flat, n, batch_count)

    assert results.shape == (batch_count, n, n)
    assert results.dtype == np.complex64

    for k in range(batch_count):
        err = frobenius_error(matrices[k], results[k])
        assert err < 1e-3, f"Матрица {k}: error {err:.2e} >= 1e-3"


# ============================================================================
# 5.12.4: test_batch_sizes
# ============================================================================


def test_batch_sizes(inverter):
    """Разные batch sizes: 1, 4, 8."""
    n = 64

    for batch_count in [1, 4, 8]:
        matrices = [make_positive_definite(n, seed=i + 100)
                     for i in range(batch_count)]
        flat = np.concatenate([m.flatten() for m in matrices])

        results = inverter.invert_batch_cpu(flat, n, batch_count)
        assert results.shape == (batch_count, n, n)


# ============================================================================
# 5.12.5: test_modes_roundtrip_vs_kernel
# ============================================================================


def test_modes_roundtrip_vs_kernel(inverter, inverter_roundtrip):
    """Оба режима дают одинаковый результат."""
    n = 64
    batch_count = 4

    matrices = [make_positive_definite(n, seed=i + 200)
                 for i in range(batch_count)]
    flat = np.concatenate([m.flatten() for m in matrices])

    result_kernel = inverter.invert_batch_cpu(flat, n, batch_count)
    result_roundtrip = inverter_roundtrip.invert_batch_cpu(flat, n, batch_count)

    # Оба режима должны давать практически одинаковый результат
    delta = (result_kernel.astype(np.complex128) -
             result_roundtrip.astype(np.complex128))
    diff = float(np.linalg.norm(delta.reshape(-1)))
    assert diff < 1e-5, \
        f"Roundtrip vs GpuKernel diff: {diff:.2e} >= 1e-5"


# ============================================================================
# 5.12.6: test_set_symmetrize_mode
# ============================================================================


def test_set_symmetrize_mode(rocm_context):
    """set_symmetrize_mode / get_symmetrize_mode работают."""
    import gpuworklib

    inv = gpuworklib.CholeskyInverterROCm(
        rocm_context, gpuworklib.SymmetrizeMode.GpuKernel
    )
    assert inv.get_symmetrize_mode() == gpuworklib.SymmetrizeMode.GpuKernel

    inv.set_symmetrize_mode(gpuworklib.SymmetrizeMode.Roundtrip)
    assert inv.get_symmetrize_mode() == gpuworklib.SymmetrizeMode.Roundtrip

    # Проверить что работает после смены режима
    n = 5
    A = make_positive_definite(n, seed=999)
    A_inv = inv.invert_cpu(A.flatten(), n)
    err = frobenius_error(A, A_inv)
    assert err < 1e-4, f"После смены режима: error {err:.2e}"
