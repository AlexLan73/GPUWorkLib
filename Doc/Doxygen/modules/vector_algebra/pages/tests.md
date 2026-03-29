@page vector_algebra_tests Vector Algebra -- Тесты и бенчмарки

@tableofcontents

@section va_tests_intro Введение

Тестирование модуля **vector_algebra** включает проверку корректности Cholesky-обращения
для матриц различных размеров (от 5x5 до 341x341), batch-режима, различных форматов
входных данных (CPU vector, hipPtr, cl_mem ZeroCopy), сравнение режимов
симметризации (Roundtrip vs GpuKernel) и stage-профилирование.

---

@section va_tests_cpp C++ тесты

@subsection va_tests_cpp_main Основные тесты (test_cholesky_inverter_rocm.hpp)

Файл: `modules/vector_algebra/tests/test_cholesky_inverter_rocm.hpp`

9 тестов, каждый выполняется в двух режимах симметризации (Roundtrip + GpuKernel):

| # | Тест | Описание |
|---|------|----------|
| 1 | IdentityInverse | \f$ I^{-1} = I \f$, проверка тривиального случая |
| 2 | Small5x5 | Малая HPD матрица 5x5, сравнение с NumPy reference |
| 3 | Medium64x64 | Матрица 64x64, \f$ \varepsilon_F < 10^{-4} \f$ |
| 4 | Large341x341 | Большая матрица 341x341 (типичный размер для Capon) |
| 5 | GPUPointerInput | Вход -- `hipFloatComplex*` (данные уже на GPU) |
| 6 | ZeroCopyClMem | Вход -- `cl_mem` (OpenCL -> ROCm ZeroCopy) |
| 7 | BatchSmall4x64 | Batch: 4 матрицы 64x64 |
| 8 | BatchLarge4x256 | Batch: 4 матрицы 256x256 |
| 9 | DiagonalLoading | Плохо обусловленная матрица + \f$ \mu I \f$, POTRF success |

@subsection va_tests_cpp_crossbackend Cross-backend тесты (test_cross_backend_conversion.hpp)

Файл: `modules/vector_algebra/tests/test_cross_backend_conversion.hpp`

| # | Тест | Описание |
|---|------|----------|
| 1 | VectorToResult | `std::vector` -> Invert -> AsVector(): roundtrip |
| 2 | HipPtrToResult | `hipPtr` -> Invert -> AsHipPtr(): zero-copy chain |
| 3 | ClMemToResult | `cl_mem` -> Invert -> AsVector(): cross-backend |
| 4 | OutputFormats | Один Invert, проверка AsVector() + AsHipPtr() |

@subsection va_tests_cpp_benchmark Бенчмарк симметризации (test_benchmark_symmetrize.hpp)

Файл: `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp`

| # | Бенчмарк | Описание |
|---|----------|----------|
| 1 | Roundtrip_341 | Symmetrize Roundtrip, n=341 |
| 2 | GpuKernel_341 | Symmetrize GpuKernel, n=341 |
| 3 | Batch_16x64 | 16 матриц 64x64: Roundtrip vs GpuKernel |

@subsection va_tests_cpp_profiling Stage-профилирование (test_stage_profiling.hpp)

Файл: `modules/vector_algebra/tests/test_stage_profiling.hpp`

Профилирование каждой стадии через `GPUProfiler`:

| Стадия | Описание |
|--------|----------|
| Upload | H2D: копирование HPD матрицы на GPU |
| POTRF | Cholesky factorization (rocsolver_cpotrf) |
| POTRI | Cholesky inversion (rocsolver_cpotri) |
| Symmetrize | Заполнение нижнего треугольника (Roundtrip или GpuKernel) |
| Download | D2H: копирование результата (для AsVector) |

@subsection va_tests_cpp_example Пример C++ теста

@code{.cpp}
// test_cholesky_inverter_rocm.hpp -- Large341x341
void TestLarge341x341(DrvGPU& drv, SymmetrizeMode mode) {
  auto& con = ConsoleOutput::GetInstance();
  const int n = 341;

  vector_algebra::CholeskyInverterROCm inv(drv.GetBackend());
  inv.Initialize(n);
  inv.SetSymmetrizeMode(mode);  // Roundtrip или GpuKernel

  // Генерация HPD: A = X^H * X + n*I
  auto hpd = GenerateHPDMatrix(n);
  auto result = inv.Invert(hpd);
  auto inv_matrix = result.AsVector();

  // Проверка: ||A * A^-1 - I||_F
  auto product = MatMul(hpd, inv_matrix, n);
  double error = FrobeniusError(product, IdentityMatrix(n));

  con.Print("341x341 [" + ModeToString(mode) + "]: "
            "Frobenius error = " + std::to_string(error));
  ASSERT_LT(error, 1e-2);  // float32 limit для n=341
}
@endcode

---

@section va_tests_python Python тесты

@subsection va_tests_python_main Основные тесты (test_cholesky_inverter_rocm.py)

Файл: `Python_test/vector_algebra/test_cholesky_inverter_rocm.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_identity_inverse | \f$ I^{-1} = I \f$ |
| 2 | test_small_5x5 | 5x5 HPD, сравнение с `np.linalg.inv()` |
| 3 | test_medium_64x64 | 64x64, relative error \f$ < 10^{-4} \f$ |
| 4 | test_large_341x341 | 341x341, float32 precision check |
| 5 | test_batch_4x64 | Batch 4 матрицы 64x64 |
| 6 | test_roundtrip_vs_gpukernel | Оба режима дают одинаковый результат |

@subsection va_tests_python_csv CSV-сравнение (test_matrix_csv_comparison.py)

Файл: `Python_test/vector_algebra/test_matrix_csv_comparison.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_csv_reference | Сравнение GPU с CSV-эталоном (MATLAB/Octave) |
| 2 | test_csv_batch | Batch результаты vs CSV |

@subsection va_tests_python_example Пример Python теста

@code{.py}
# test_cholesky_inverter_rocm.py -- test_large_341x341
def test_large_341x341(self):
    """Обращение матрицы 341x341: GPU vs NumPy."""
    n = 341
    # Генерация HPD матрицы
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    X = X.astype(np.complex64)
    A = X @ X.conj().T + n * np.eye(n, dtype=np.complex64)

    # GPU инверсия
    inv_gpu = self.inverter.invert(A)

    # NumPy reference
    inv_np = np.linalg.inv(A)

    # Проверка: ||A * A_inv - I||_F / sqrt(n)
    product = A @ inv_gpu
    error = np.linalg.norm(product - np.eye(n, dtype=np.complex64))
    rel_error = error / np.sqrt(n)

    self.assertLess(rel_error, 1e-2,
                    f"341x341 relative error {rel_error:.2e} too large")

    # Element-wise: GPU vs NumPy
    np.testing.assert_allclose(
        inv_gpu, inv_np, atol=1e-2, rtol=1e-3,
        err_msg="GPU vs NumPy mismatch for 341x341"
    )
@endcode

---

@section va_tests_benchmarks Бенчмарки

@subsection va_tests_bench_symmetrize Бенчмарк симметризации: Roundtrip vs GpuKernel

Сравнение двух режимов для матрицы 341x341:

| Режим | Операции | Ожидаемое время |
|-------|----------|-----------------|
| **Roundtrip** | D2H (341^2 complex = 930 KB) + CPU loop + H2D | ~0.5-2 мс |
| **GpuKernel** | HIP kernel 2D launch (22x22 blocks of 16x16) | ~0.01-0.05 мс |

GpuKernel быстрее в **10-50x** за счёт отсутствия PCIe transfers.

@subsection va_tests_bench_batch Batch бенчмарки

| Конфигурация | POTRF | POTRI | Symmetrize | Total |
|-------------|-------|-------|------------|-------|
| 16 x 64x64 | ~ | ~ | ~ | Batch vs sequential |
| 4 x 256x256 | ~ | ~ | ~ | Batch amortization |

@subsection va_tests_bench_stages Stage-профилирование

Пример вывода `GPUProfiler` для матрицы 341x341:

```
| Stage        | Time (ms) | % Total |
|-------------|-----------|---------|
| Upload      |     0.12  |    3.1% |
| POTRF       |     2.45  |   63.0% |
| POTRI       |     1.15  |   29.5% |
| Symmetrize  |     0.05  |    1.3% |
| Download    |     0.12  |    3.1% |
| Total       |     3.89  |  100.0% |
```

@note POTRF доминирует (\f$ O(n^3/3) \f$). Для больших \f$ n \f$ Upload/Download
становятся пренебрежимо малыми относительно вычислений.

@subsection va_tests_bench_example Запуск бенчмарка

@code{.cpp}
// Из all_test.hpp:
#include "modules/vector_algebra/tests/test_benchmark_symmetrize.hpp"
#include "modules/vector_algebra/tests/test_stage_profiling.hpp"

// В RunAllTests():
test_benchmark_symmetrize::RunBenchmarks(drv);
test_stage_profiling::RunProfiling(drv);
@endcode

---

@section va_tests_tolerances Допустимые погрешности

| Размер \f$ n \f$ | \f$ \|A \cdot A^{-1} - I\|_F / \sqrt{n} \f$ | Примечание |
|---:|---:|---|
| 5 | \f$ < 10^{-5} \f$ | Отличная точность float32 |
| 64 | \f$ < 10^{-4} \f$ | Стандартная точность |
| 256 | \f$ < 10^{-3} \f$ | Приемлемая для Capon |
| 341 | \f$ < 10^{-2} \f$ | Предел float32 (diagonal loading рекомендуется) |

@note Режимы Roundtrip и GpuKernel дают **одинаковые** результаты с точностью до ULP,
так как операция conjugate transpose выполняется точно (нет округления).

---

@section va_tests_see_also Смотрите также

- @ref vector_algebra_overview -- Обзор модуля, классы и быстрый старт
- @ref vector_algebra_formulas -- Математические формулы (Cholesky, инверсия, симметризация)
