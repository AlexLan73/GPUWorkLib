@page lch_farrow_formulas lch_farrow — Математические формулы

@tableofcontents

@section lch_farrow_formulas_intro Введение

Модуль `lch_farrow` реализует дробную задержку комплексных сигналов через
5-точечную интерполяцию Лагранжа. Ключевая оптимизация — предвычисленная
матрица коэффициентов 48x5, заменяющая вычисление полиномов в runtime.

@section lch_farrow_formulas_total_delay Полная задержка

Задержка в отсчётах разделяется на целую и дробную части:

\f[
\tau_{\text{total}} = D + \mu, \quad D \in \mathbb{Z},\; \mu \in [0, 1)
\f]

где:
- \f$ \tau_{\text{total}} \f$ — полная задержка в отсчётах
- \f$ D \f$ — целая часть задержки (integer delay) — реализуется сдвигом индекса
- \f$ \mu \f$ — дробная часть задержки (fractional delay) — реализуется интерполяцией

Пересчёт из микросекунд: \f$ \tau = \text{delay\_us} \times f_s \times 10^{-6} \f$

@section lch_farrow_formulas_lagrange 5-точечная интерполяция Лагранжа

Интерполированное значение для отсчёта \f$ n \f$:

\f[
y[n] = \sum_{k=0}^{4} L_k(\mu) \cdot x[n - D - k + 2]
\f]

где:
- \f$ x[\cdot] \f$ — входной сигнал (комплексный)
- \f$ L_k(\mu) \f$ — базисные полиномы Лагранжа 4-го порядка
- \f$ D \f$ — целая часть задержки
- Окно интерполяции: 5 точек с центром в \f$ n - D \f$

@section lch_farrow_formulas_basis Базисные полиномы Лагранжа

Пять базисных полиномов \f$ L_k(\mu) \f$ для \f$ \mu \in [0, 1) \f$:

\f[
L_0(\mu) = \frac{-\mu(\mu-1)(\mu-2)(\mu-3)}{24}
\f]

\f[
L_1(\mu) = \frac{\mu(\mu+1)(\mu-1)(\mu-2)}{-6}
\f]

\f[
L_2(\mu) = \frac{(\mu+2)(\mu+1)(\mu-1)(\mu-2)}{4}
\f]

\f[
L_3(\mu) = \frac{\mu(\mu+2)(\mu+1)(\mu-1)}{-6}
\f]

\f[
L_4(\mu) = \frac{\mu(\mu+2)(\mu+1)(\mu-2)}{24}
\f]

@note Общая формула базисных полиномов Лагранжа:

\f[
L_k(\mu) = \prod_{\substack{j=0 \\ j \neq k}}^{4} \frac{\mu - \mu_j}{\mu_k - \mu_j}
\f]

с узлами \f$ \mu_j = j - 2 \f$ (т.е. \f$ \{-2, -1, 0, 1, 2\} \f$).

@section lch_farrow_formulas_properties Свойства полиномов

Базисные полиномы Лагранжа обладают свойствами:

1. **Единичное разбиение**: \f$ \sum_{k=0}^{4} L_k(\mu) = 1 \f$ для всех \f$ \mu \f$
2. **Интерполяция**: \f$ L_k(\mu_j) = \delta_{kj} \f$ (Кронекер)
3. **Точность**: полином 4-го порядка воспроизводит без ошибки полиномы до 4-й степени включительно

@section lch_farrow_formulas_discrete_matrix Дискретная матрица коэффициентов

Вместо вычисления полиномов в runtime используется предвычисленная **lookup-таблица 48x5**:

\f[
\text{coeff}[r][k] = L_k\!\left(\frac{r}{48}\right), \quad r = 0, \ldots, 47, \quad k = 0, \ldots, 4
\f]

48 уровней квантования дробной части \f$ \mu \f$:

\f[
\text{row} = \lfloor \mu \times 48 \rfloor
\f]

Погрешность квантования:

\f[
\Delta\mu_{\max} = \frac{1}{2 \times 48} \approx 0.0104
\f]

@note 48 уровней обеспечивают достаточную точность для типичных задач ЦОС.
При необходимости количество уровней можно увеличить (например, 96 или 192).

@section lch_farrow_formulas_interpolation Схема интерполяции

Для каждого выходного отсчёта \f$ n \f$ антенны \f$ a \f$:

\f[
y_a[n] = \sum_{k=0}^{4} \text{coeff}[\text{row}_a][k] \cdot x_a[n - D_a - k + 2]
\f]

где:
- \f$ D_a = \lfloor \tau_a \rfloor \f$ — целая часть задержки антенны \f$ a \f$
- \f$ \mu_a = \tau_a - D_a \f$ — дробная часть
- \f$ \text{row}_a = \lfloor \mu_a \times 48 \rfloor \f$ — индекс строки в таблице

@section lch_farrow_formulas_gpu GPU-оптимизация

Ключевые оптимизации в ядре `lch_farrow_delay`:

1. **Без div/mod в kernel**: все индексы (\f$ D_a \f$, row_a) предвычислены на хосте
2. **Coalesced memory access**: 2D grid (x=sample, y=antenna) — потоки одного warp читают соседние отсчёты
3. **Constant memory**: матрица 48x5 = 240 float в constant memory (быстрый broadcast)
4. **5 MAC операций**: на каждый выходной отсчёт — минимум арифметики

@section lch_farrow_formulas_summary Сводка

| Параметр | Значение |
|----------|----------|
| Порядок интерполяции | 4 (5 точек) |
| Квантование \f$ \mu \f$ | 48 уровней |
| Размер таблицы | 48 × 5 = 240 float |
| Погрешность | \f$ \Delta\mu < 0.0104 \f$ |
| Операций на отсчёт | 5 MAC (multiply-accumulate) |

@see lch_farrow_overview
@see lch_farrow_tests
