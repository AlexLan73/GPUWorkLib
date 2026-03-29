@page vector_algebra_overview Vector Algebra -- Обзор модуля

@tableofcontents

@section va_overview_purpose Назначение

Модуль **vector_algebra** выполняет обращение эрмитовых положительно-определённых (HPD)
матриц через Cholesky-разложение на GPU с помощью rocsolver.
Поддерживает входные данные из CPU (`std::vector`), ROCm device pointer (`hipMalloc`),
и OpenCL `cl_mem` (ZeroCopy через общую виртуальную память).

> **Namespace**: `vector_algebra` | **Backend**: ROCm (rocsolver) | **Статус**: Active

@section va_overview_classes Ключевые классы

| Класс | Описание | Заголовок |
|-------|----------|-----------|
| @ref vector_algebra::CholeskyInverterROCm | Facade: Invert(data) -> CholeskyResult, batch mode | `cholesky_inverter_rocm.hpp` |
| @ref vector_algebra::CholeskyResult | RAII контейнер результата: AsVector() / AsHipPtr() | `cholesky_result.hpp` |
| @ref vector_algebra::MatrixOpsROCm | Low-level rocsolver операции: POTRF, POTRI, symmetrize | `matrix_ops_rocm.hpp` |
| @ref vector_algebra::DiagonalLoadRegularizer | Регуляризация: \f$ A + \mu I \f$ для плохо обусловленных матриц | `diagonal_load_regularizer.hpp` |

@section va_overview_arch Архитектура

```
CholeskyInverterROCm (Facade)
    |
    +-- MatrixOpsROCm (rocsolver operations)
    |       |
    |       +-- rocsolver_cpotrf()  -- Cholesky factorization
    |       +-- rocsolver_cpotri()  -- Cholesky inversion
    |       +-- symmetrize()        -- fill lower triangle
    |
    +-- DiagonalLoadRegularizer
    |       |
    |       +-- A + mu*I (diagonal loading)
    |
    +-- CholeskyResult (RAII output)
            |
            +-- AsVector()  -- copy D2H, return std::vector
            +-- AsHipPtr()  -- return device pointer (zero-copy)
```

Модуль следует 6-слойной архитектуре Ref03:

- **Слой 1** -- `GpuContext`: per-module контекст (hipStream, rocblas/rocsolver handles)
- **Слой 5** -- Concrete Ops: `MatrixOpsROCm`, `DiagonalLoadRegularizer`
- **Слой 6** -- Facade: `CholeskyInverterROCm` + RAII result `CholeskyResult`

@section va_overview_symmetrize Режимы симметризации

После POTRI результат содержит только **верхний треугольник**. Нижний треугольник
заполняется через conjugate transpose. Два режима:

| Режим | Описание | Когда использовать |
|-------|----------|--------------------|
| **Roundtrip** | D2H -> CPU symmetrize -> H2D | Малые матрицы, отладка |
| **GpuKernel** | HIP kernel in-place на GPU | Большие матрицы, production |

@note `GpuKernel` режим не требует копирования данных между host и device,
что даёт значительный выигрыш для больших матриц.

@section va_overview_pipeline Конвейер обработки

```
input HPD matrix (complex64)
         |
    [DiagonalLoadRegularizer]  (опционально: A + mu*I)
         |
    [rocsolver_cpotrf]         (A = U^H * U)
         |
    [rocsolver_cpotri]         (A^-1 из U)
         |
    [symmetrize]               (Roundtrip или GpuKernel)
         |
    CholeskyResult (RAII)
         |
    +-- AsVector()  -- std::vector<complex<float>>
    +-- AsHipPtr()  -- hipFloatComplex*
```

@section va_overview_input_formats Форматы входных данных

| Формат | Метод | Описание |
|--------|-------|----------|
| `std::vector<std::complex<float>>` | `Invert(vector)` | Автоматический upload H2D |
| `hipFloatComplex*` | `Invert(hip_ptr, n)` | Данные уже на ROCm GPU |
| `cl_mem` | `Invert(cl_mem, n)` | OpenCL buffer -> ZeroCopy -> ROCm |

@warning ZeroCopy (cl_mem -> hip) работает только на системах с unified memory
(AMD APU или dGPU с XNACK enabled). На дискретных GPU без XNACK
используется промежуточное копирование.

@section va_overview_quickstart Быстрый старт

@subsection va_overview_cpp C++ пример

@code{.cpp}
#include "modules/vector_algebra/include/cholesky_inverter_rocm.hpp"

// Создание инвертера
vector_algebra::CholeskyInverterROCm inv(backend);
inv.Initialize(n);  // n -- размер матрицы NxN

// Входная HPD матрица (row-major, complex float)
std::vector<std::complex<float>> hpd_matrix = GenerateHPDMatrix(n);

// Обращение через Cholesky
auto result = inv.Invert(hpd_matrix);

// Получение результата
auto inv_matrix = result.AsVector();  // D2H copy, std::vector<complex<float>>

// Проверка: A * A^-1 ~= I
auto product = MatMul(hpd_matrix, inv_matrix, n);
double error = FrobeniusError(product, IdentityMatrix(n));
// error < 1e-4 для well-conditioned matrices
@endcode

@code{.cpp}
// Batch mode: 4 матрицы 64x64
inv.Initialize(64);
std::vector<std::vector<std::complex<float>>> batch(4);
for (auto& m : batch) m = GenerateHPDMatrix(64);

auto results = inv.InvertBatch(batch);
for (auto& r : results) {
  auto inv_m = r.AsVector();
  // ...
}
@endcode

@code{.cpp}
// С регуляризацией (diagonal loading)
vector_algebra::DiagonalLoadRegularizer reg(/*mu=*/1e-6f);
auto regularized = reg.Apply(ill_conditioned_matrix, n);
auto result = inv.Invert(regularized);
@endcode

@subsection va_overview_python Python пример

@code{.py}
import gpuworklib as gw
import numpy as np

# Инициализация GPU контекста
ctx = gw.ROCmGPUContext(0)

# Создание инвертера для матриц 64x64
inv = gw.CholeskyInverterROCm(ctx, n=64)

# Генерация HPD матрицы
A = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
A = A.astype(np.complex64)
hpd = A @ A.conj().T + 64 * np.eye(64, dtype=np.complex64)

# Обращение на GPU
result = inv.invert(hpd)  # np.ndarray complex64, shape=(64, 64)

# Проверка через NumPy
np_inv = np.linalg.inv(hpd)
error = np.linalg.norm(result - np_inv) / np.linalg.norm(np_inv)
print(f"Relative error: {error:.2e}")  # < 1e-4
@endcode

@code{.py}
# Batch: 4 матрицы 256x256
matrices = [generate_hpd(256) for _ in range(4)]
results = inv.invert_batch(matrices)
@endcode

@section va_overview_deps Зависимости

- **DrvGPU** -- управление GPU-контекстом, потоками, памятью
- **ROCm / HIP** -- запуск HIP kernels (symmetrize), hipStream
- **rocsolver** -- POTRF (Cholesky factorization), POTRI (Cholesky inversion)
- **rocblas** -- handle для rocsolver

@section va_overview_see_also Смотрите также

- @ref vector_algebra_formulas -- Математические формулы (Cholesky, инверсия, симметризация)
- @ref vector_algebra_tests -- Тесты, бенчмарки и сравнение режимов
- @ref capon_overview -- Capon beamformer (использует CholeskyInverterROCm для \f$ R^{-1} \f$)
- @ref statistics_overview -- Статистика сигналов
