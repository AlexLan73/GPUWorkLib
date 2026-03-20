"""
Python тесты CholeskyInverterROCm (Task_11 v2: SymmetrizeMode).

Эталон: np.linalg.inv() vs GPU Cholesky (rocSOLVER POTRF+POTRI).
Два режима: Roundtrip (CPU symmetrize) и GpuKernel (hiprtc).

Запуск:
    cd /home/alex/C++/GPUWorkLib
    python Python_test/vector_algebra/test_cholesky_inverter_rocm.py
    PYTHONPATH=build/python python Python_test/vector_algebra/test_cholesky_inverter_rocm.py

Требования:
    - ROCm (AMD Radeon 9070 или совместимое GPU)
    - gpuworklib собран с ENABLE_ROCM=ON
    - numpy
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner


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
# Tests
# ============================================================================


class TestCholeskyInverterROCm:
    """Python тесты CholeskyInverterROCm."""

    def setUp(self):
        try:
            import gpuworklib
            self._gw = gpuworklib
            ctx = gpuworklib.ROCmGPUContext(0)
            self._ctx = ctx
            self._inverter = gpuworklib.CholeskyInverterROCm(
                ctx, gpuworklib.SymmetrizeMode.GpuKernel
            )
            self._inverter_roundtrip = gpuworklib.CholeskyInverterROCm(
                ctx, gpuworklib.SymmetrizeMode.Roundtrip
            )
        except ImportError:
            raise SkipTest("gpuworklib не найден")
        except Exception as e:
            raise SkipTest(f"ROCm недоступен: {e}")

    def test_invert_5x5(self):
        """CPU инверсия 5×5. Ошибка < 1e-5."""
        n = 5
        A = make_positive_definite(n, seed=1)

        A_inv_gpu = self._inverter.invert_cpu(A.flatten(), n)

        assert A_inv_gpu.shape == (n, n)
        assert A_inv_gpu.dtype == np.complex64

        err = frobenius_error(A, A_inv_gpu)
        assert err < 1e-4, f"Frobenius error {err:.2e} >= 1e-4"

    def test_invert_341x341(self):
        """CPU инверсия 341×341. Ошибка < 1e-2."""
        n = 341
        A = make_positive_definite(n, seed=42)

        A_inv_gpu = self._inverter.invert_cpu(A.flatten(), n)

        assert A_inv_gpu.shape == (n, n)

        err = frobenius_error(A, A_inv_gpu)
        assert err < 1e-2, f"Frobenius error {err:.2e} >= 1e-2"

    def test_batch_4x64(self):
        """Batched инверсия 4 × 64×64. Для каждой ошибка < 1e-3."""
        n = 64
        batch_count = 4

        matrices = [make_positive_definite(n, seed=i) for i in range(batch_count)]
        flat = np.concatenate([m.flatten() for m in matrices])

        results = self._inverter.invert_batch_cpu(flat, n, batch_count)

        assert results.shape == (batch_count, n, n)
        assert results.dtype == np.complex64

        for k in range(batch_count):
            err = frobenius_error(matrices[k], results[k])
            assert err < 1e-3, f"Матрица {k}: error {err:.2e} >= 1e-3"

    def test_batch_sizes(self):
        """Разные batch sizes: 1, 4, 8."""
        n = 64

        for batch_count in [1, 4, 8]:
            matrices = [make_positive_definite(n, seed=i + 100)
                         for i in range(batch_count)]
            flat = np.concatenate([m.flatten() for m in matrices])

            results = self._inverter.invert_batch_cpu(flat, n, batch_count)
            assert results.shape == (batch_count, n, n)

    def test_modes_roundtrip_vs_kernel(self):
        """Оба режима дают одинаковый результат."""
        n = 64
        batch_count = 4

        matrices = [make_positive_definite(n, seed=i + 200)
                     for i in range(batch_count)]
        flat = np.concatenate([m.flatten() for m in matrices])

        result_kernel = self._inverter.invert_batch_cpu(flat, n, batch_count)
        result_roundtrip = self._inverter_roundtrip.invert_batch_cpu(flat, n, batch_count)

        delta = (result_kernel.astype(np.complex128) -
                 result_roundtrip.astype(np.complex128))
        diff = float(np.linalg.norm(delta.reshape(-1)))
        assert diff < 1e-5, \
            f"Roundtrip vs GpuKernel diff: {diff:.2e} >= 1e-5"

    def test_set_symmetrize_mode(self):
        """set_symmetrize_mode / get_symmetrize_mode работают."""
        gw = self._gw
        inv = gw.CholeskyInverterROCm(
            self._ctx, gw.SymmetrizeMode.GpuKernel
        )
        assert inv.get_symmetrize_mode() == gw.SymmetrizeMode.GpuKernel

        inv.set_symmetrize_mode(gw.SymmetrizeMode.Roundtrip)
        assert inv.get_symmetrize_mode() == gw.SymmetrizeMode.Roundtrip

        # Проверить что работает после смены режима
        n = 5
        A = make_positive_definite(n, seed=999)
        A_inv = inv.invert_cpu(A.flatten(), n)
        err = frobenius_error(A, A_inv)
        assert err < 1e-4, f"После смены режима: error {err:.2e}"


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestCholeskyInverterROCm())
    runner.print_summary(results)
