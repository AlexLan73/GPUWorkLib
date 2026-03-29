@page statistics_formulas Statistics -- Математические формулы

@tableofcontents

@section stat_formulas_intro Введение

Модуль **statistics** реализует вычисление статистик на комплексных IQ-данных,
распределённых по лучам (beams). Все вычисления выполняются на GPU через
специализированные HIP-ядра. Ниже приведены формулы для каждой операции.

---

@section stat_formulas_complex_mean Комплексное среднее (MeanReductionOp)

Для каждого луча \f$ b \f$ с \f$ N \f$ комплексными отсчётами:

\f[
\bar{z}_b = \frac{1}{N} \sum_{n=0}^{N-1} z_{b,n}, \quad z_{b,n} = x_{b,n} + j\,y_{b,n}
\f]

Среднее вычисляется **покомпонентно** (Re и Im отдельно):

\f[
\text{Re}(\bar{z}_b) = \frac{1}{N} \sum_{n=0}^{N-1} x_{b,n}, \qquad
\text{Im}(\bar{z}_b) = \frac{1}{N} \sum_{n=0}^{N-1} y_{b,n}
\f]

@subsection stat_formulas_mean_reduction Иерархическая редукция

GPU-реализация использует **multi-pass parallel reduction**:

1. **Первый проход**: каждый warp суммирует 256 элементов, результат -- partial sums
2. **Последующие проходы**: редукция partial sums до единственного значения
3. **Финальное деление**: нормировка на \f$ N \f$

Количество проходов: \f$ \lceil \log_{256}(N) \rceil \f$.

@note Для `float2` (complex) редукция выполняется одновременно по `.x` и `.y` компонентам.

---

@section stat_formulas_magnitude Модуль комплексного числа

Магнитуда используется как промежуточный шаг для Welford и Median:

\f[
|z| = \sqrt{\text{Re}(z)^2 + \text{Im}(z)^2}
\f]

На GPU вычисляется через HIP intrinsic `hipSqrtf(x*x + y*y)` или эквивалент.

---

@section stat_formulas_welford Welford one-pass (WelfordFusedOp)

Вычисление mean_magnitude, variance и std_dev за **один проход** по данным.

@subsection stat_formulas_welford_sums Суммы

Для каждого луча \f$ b \f$ вычисляются две суммы по магнитудам:

\f[
S_1 = \sum_{n=0}^{N-1}|z_{b,n}|, \qquad S_2 = \sum_{n=0}^{N-1}|z_{b,n}|^2
\f]

@subsection stat_formulas_welford_stats Статистики из сумм

\f[
M_b = \frac{S_1}{N} \quad \text{(средняя магнитуда)}
\f]

\f[
\text{Var}_b = \frac{S_2}{N} - M_b^2 \quad \text{(дисперсия, population)}
\f]

\f[
\text{STD}_b = \sqrt{\max(\text{Var}_b,\; 0)} \quad \text{(СКО)}
\f]

@note Используется **population variance** (ddof=0), что соответствует `np.var(x, ddof=0)` в NumPy.

@warning \f$ \max(\text{Var}_b, 0) \f$ -- защита от отрицательных значений, которые могут возникнуть
из-за ошибок округления float32 при вычитании близких величин.

@subsection stat_formulas_welford_fused Fused kernel

Ядро `welford_fused` объединяет:
1. Вычисление \f$ |z| \f$ из `float2`
2. Параллельную редукцию \f$ S_1 \f$ и \f$ S_2 \f$ (shared memory)
3. Финальное вычисление \f$ M_b, \text{Var}_b, \text{STD}_b \f$

Это позволяет избежать промежуточного буфера магнитуд и дополнительного прохода по памяти.

@subsection stat_formulas_welford_float WelfordFloatOp

Для данных, уже представленных как float magnitudes (не complex):

\f[
M_b = \frac{1}{N} \sum_{n=0}^{N-1} a_{b,n}, \quad
\text{Var}_b = \frac{1}{N} \sum_{n=0}^{N-1} a_{b,n}^2 - M_b^2
\f]

где \f$ a_{b,n} \f$ -- float-значение (например, результат ProcessMagnitude).

---

@section stat_formulas_median Медиана (MedianRadixSortOp / MedianHistogramOp)

Медиана по магнитудам для каждого луча:

\f[
\text{median}_b = \text{sorted}(\{|z_{b,0}|, |z_{b,1}|, \ldots, |z_{b,N-1}|\})[\lfloor N/2 \rfloor]
\f]

Используется **NumPy convention**: для чётного \f$ N \f$ берётся элемент \f$ \lfloor N/2 \rfloor \f$
(lower median), без усреднения двух центральных элементов.

@subsection stat_formulas_median_radix Radix Sort подход

Для малых и средних массивов (\f$ N < 256\text{K} \f$):

1. Вычисление \f$ |z| \f$ для каждого элемента
2. Segmented radix sort через rocPRIM (per-beam сегменты)
3. Выбор \f$ \lfloor N/2 \rfloor \f$-го элемента из отсортированного массива

Сложность: \f$ O(N \cdot W) \f$, где \f$ W = 32 \f$ -- ширина ключа float (кол-во бит).

@subsection stat_formulas_median_bitonic Bitonic Sort (малые массивы)

Для \f$ N < 256 \f$: bitonic sort в регистрах одного warp.

\f[
\text{Comparisons} = \frac{N \log_2^2(N)}{4} \quad \text{(bitonic network)}
\f]

@subsection stat_formulas_median_histogram Histogram подход

Для больших массивов (\f$ N \ge 256\text{K} \f$):

**Итеративное сужение диапазона:**

1. Найти \f$ [\min, \max] \f$ по всем \f$ |z| \f$ луча
2. Разбить на \f$ B = 256 \f$ бинов, ширина бина: \f$ w = \frac{\max - \min}{B} \f$
3. Построить гистограмму, найти бин, содержащий \f$ \lfloor N/2 \rfloor \f$-й элемент (prefix sum)
4. Сузить диапазон до этого бина: \f$ [\min', \max'] = [\min + k \cdot w, \min + (k+1) \cdot w] \f$
5. Повторять, пока \f$ w < \varepsilon \f$ (точность float32)

Сложность: \f$ O(N \cdot P) \f$, где \f$ P \approx 4{-}6 \f$ проходов -- значительно лучше сортировки.

---

@section stat_formulas_full_pipeline Полный конвейер ComputeAll

\f[
\text{ComputeAll}(z) = \begin{cases}
\bar{z}_b & \text{MeanReductionOp} \\
M_b, \text{Var}_b, \text{STD}_b & \text{WelfordFusedOp} \\
\text{median}_b & \text{MedianRadixSortOp или MedianHistogramOp}
\end{cases}
\f]

Все три операции запускаются последовательно на одном HIP-потоке для данного луча,
результаты агрегируются в `FullStatisticsResult`.

---

@section stat_formulas_see_also Смотрите также

- @ref statistics_overview -- Обзор модуля, классы и быстрый старт
- @ref statistics_tests -- Тесты и бенчмарки
