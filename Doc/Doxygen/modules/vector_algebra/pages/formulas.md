@page vector_algebra_formulas Vector Algebra -- Математические формулы

@tableofcontents

@section va_formulas_intro Введение

Модуль **vector_algebra** реализует обращение эрмитовых положительно-определённых (HPD)
матриц на GPU через Cholesky-разложение (rocsolver). Ниже приведены формулы
для каждого этапа: разложение, обращение, симметризация, регуляризация и метрики ошибок.

---

@section va_formulas_hpd Эрмитова положительно-определённая матрица

Матрица \f$ A \in \mathbb{C}^{n \times n} \f$ называется HPD, если:

1. **Эрмитовость**: \f$ A = A^H \f$, т.е. \f$ a_{ij} = \overline{a_{ji}} \f$
2. **Положительная определённость**: \f$ \mathbf{x}^H A \mathbf{x} > 0 \f$ для любого \f$ \mathbf{x} \ne 0 \f$

@note В радиолокации HPD матрицы возникают как ковариационные матрицы сигналов:
\f$ R = \frac{1}{K} \sum_{k=1}^{K} \mathbf{x}_k \mathbf{x}_k^H \f$.

---

@section va_formulas_cholesky Cholesky-разложение (POTRF)

Любая HPD матрица единственным образом раскладывается:

\f[
A = U^H U
\f]

где \f$ U \f$ -- верхнетреугольная матрица с вещественными положительными диагональными элементами.

@subsection va_formulas_cholesky_elements Элементы U

Диагональные элементы:

\f[
u_{jj} = \sqrt{a_{jj} - \sum_{k=0}^{j-1} |u_{kj}|^2}
\f]

Внедиагональные элементы (для \f$ i < j \f$):

\f[
u_{ij} = \frac{1}{u_{ii}} \left(a_{ij} - \sum_{k=0}^{i-1} \overline{u_{ki}} \cdot u_{kj}\right)
\f]

@subsection va_formulas_cholesky_rocsolver rocsolver реализация

Функция `rocsolver_cpotrf()`:
- **Вход**: HPD матрица \f$ A \f$ (column-major или row-major)
- **Выход**: верхняя треугольная \f$ U \f$ (in-place, перезаписывает верхний треугольник \f$ A \f$)
- **Fill mode**: `rocblas_fill_upper`
- **Сложность**: \f$ O(n^3 / 3) \f$

@warning Если матрица не HPD (например, из-за численных ошибок), POTRF вернёт ненулевой `info`.
В этом случае необходима регуляризация через `DiagonalLoadRegularizer`.

---

@section va_formulas_inversion Обращение матрицы (POTRI)

Из верхнего треугольника \f$ U \f$ (результат POTRF) вычисляется обратная матрица:

\f[
A^{-1} = U^{-1} (U^{-1})^H = (U^H U)^{-1}
\f]

Функция `rocsolver_cpotri()`:
- **Вход**: \f$ U \f$ (результат POTRF, in-place)
- **Выход**: \f$ A^{-1} \f$ -- **только верхний треугольник** (in-place)
- **Сложность**: \f$ O(n^3 / 3) \f$

@note После POTRI нижний треугольник содержит мусор. Необходим этап симметризации.

---

@section va_formulas_symmetrize Симметризация (Conjugate Transpose Fill)

Результат POTRI -- только верхний треугольник \f$ A^{-1} \f$.
Нижний треугольник заполняется через conjugate transpose:

\f[
(A^{-1})_{ji} = \overline{(A^{-1})_{ij}}, \quad \forall\; j > i
\f]

Для каждого элемента ниже диагонали:

\f[
\text{Re}(a_{ji}) = \text{Re}(a_{ij}), \qquad \text{Im}(a_{ji}) = -\text{Im}(a_{ij})
\f]

@subsection va_formulas_symmetrize_roundtrip Режим Roundtrip

1. **D2H**: копирование матрицы с GPU на CPU (`hipMemcpy`)
2. **CPU symmetrize**: цикл по \f$ j > i \f$: \f$ a_{ji} = \overline{a_{ij}} \f$
3. **H2D**: копирование обратно на GPU

Сложность памяти: \f$ O(n^2) \f$ дополнительно на host.

@subsection va_formulas_symmetrize_gpukernel Режим GpuKernel

HIP kernel in-place на GPU:

@code{.cpp}
// Псевдокод HIP kernel
__global__ void symmetrize_kernel(hipFloatComplex* A, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (j > i && i < n && j < n) {
    A[j * n + i] = hipConjf(A[i * n + j]);  // conjugate transpose
  }
}
@endcode

Grid: 2D \f$ \lceil n/16 \rceil \times \lceil n/16 \rceil \f$ блоков по 16x16 потоков.

@note GpuKernel режим не требует копирования D2H/H2D, что критически важно
для больших матриц (n > 100) и batch-операций.

---

@section va_formulas_diagonal_loading Регуляризация (Diagonal Loading)

Для плохо обусловленных матриц (малые собственные значения):

\f[
A_{\text{reg}} = A + \mu I
\f]

где:
- \f$ \mu > 0 \f$ -- параметр регуляризации (loading factor)
- \f$ I \f$ -- единичная матрица \f$ n \times n \f$

@subsection va_formulas_diag_effect Влияние на собственные значения

Если \f$ \lambda_i \f$ -- собственные значения \f$ A \f$, то:

\f[
\lambda_i^{\text{reg}} = \lambda_i + \mu
\f]

Число обусловленности:

\f[
\kappa(A_{\text{reg}}) = \frac{\lambda_{\max} + \mu}{\lambda_{\min} + \mu} \le \kappa(A)
\f]

@subsection va_formulas_diag_choice Выбор \f$ \mu \f$

Типичные значения:
- \f$ \mu = \text{tr}(A) / n \cdot 10^{-6} \f$ -- относительно среднего собственного значения
- \f$ \mu = 10^{-6} \ldots 10^{-3} \f$ -- фиксированное малое значение

@warning Слишком большое \f$ \mu \f$ искажает обратную матрицу. В Capon beamformer
это приводит к расширению луча и потере разрешения.

---

@section va_formulas_error Метрики ошибок

@subsection va_formulas_error_frobenius Норма Фробениуса

Ошибка обращения:

\f[
\varepsilon = \|A \cdot A^{-1} - I\|_F = \sqrt{\sum_{i,j} |[A \cdot A^{-1}]_{ij} - \delta_{ij}|^2}
\f]

где \f$ \delta_{ij} \f$ -- символ Кронекера.

@subsection va_formulas_error_relative Относительная ошибка

\f[
\varepsilon_{\text{rel}} = \frac{\|A \cdot A^{-1} - I\|_F}{\|I\|_F} = \frac{\varepsilon}{\sqrt{n}}
\f]

@subsection va_formulas_error_expected Ожидаемая точность

| Размер \f$ n \f$ | \f$ \kappa(A) \f$ | \f$ \varepsilon_{\text{rel}} \f$ | Примечание |
|---:|---:|---:|---|
| 5 | \f$ 10^2 \f$ | \f$ < 10^{-5} \f$ | Отличная точность |
| 64 | \f$ 10^3 \f$ | \f$ < 10^{-4} \f$ | Хорошая точность |
| 256 | \f$ 10^4 \f$ | \f$ < 10^{-3} \f$ | Приемлемая (float32) |
| 341 | \f$ 10^5 \f$ | \f$ < 10^{-2} \f$ | Рекомендуется diagonal loading |

@note Точность ограничена float32 (complex64). Для матриц с \f$ \kappa > 10^5 \f$
рекомендуется diagonal loading или переход на double precision.

---

@section va_formulas_batch Batch-обработка

Для \f$ M \f$ матриц размера \f$ n \times n \f$:

\f[
A_m^{-1} = (U_m^H U_m)^{-1}, \quad m = 0, 1, \ldots, M-1
\f]

rocsolver поддерживает batched POTRF/POTRI, что позволяет
обрабатывать все матрицы за один вызов API с минимальным overhead.

Эффективная пропускная способность:

\f[
\text{Throughput} = \frac{M \cdot 2n^3 / 3}{\text{time}} \quad [\text{FLOP/s}]
\f]

---

@section va_formulas_see_also Смотрите также

- @ref vector_algebra_overview -- Обзор модуля, классы и быстрый старт
- @ref vector_algebra_tests -- Тесты, бенчмарки и сравнение режимов симметризации
