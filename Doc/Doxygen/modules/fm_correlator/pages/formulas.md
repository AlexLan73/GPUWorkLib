@page fm_correlator_formulas fm_correlator — Математические формулы

@tableofcontents

@section fm_correlator_formulas_intro Введение

Модуль `fm_correlator` реализует частотную (FFT-based) корреляцию принятых сигналов
с циклически сдвинутыми копиями M-последовательности. Все вычисления используют
Hermitian symmetry для экономии памяти и вычислений.

@section fm_correlator_formulas_mseq M-последовательность (LFSR)

Генерация псевдослучайной максимальной длины последовательности через 32-bit LFSR
с полиномом обратной связи `0x00400007`:

\f[
\text{ref}[i] = \begin{cases} +1.0, & \text{bit}_{31} = 1 \\ -1.0, & \text{bit}_{31} = 0 \end{cases}
\f]

Свойства M-последовательности:
- Длина: \f$ 2^{32} - 1 \f$ (полный период)
- Баланс: ~50% значений +1.0, ~50% значений -1.0
- Автокорреляция: импульсная (высокий пик при нулевом сдвиге, низкий уровень боковых лепестков)
- Воспроизводимость: одинаковый seed → одинаковая последовательность

@note Полином `0x00400007` = \f$ x^{32} + x^{22} + x^2 + x + 1 \f$ — примитивный полином,
гарантирующий максимальный период.

@section fm_correlator_formulas_cyclic_shift Циклический сдвиг

Формирование K циклически сдвинутых копий опорной последовательности:

\f[
\text{ref\_complex}[k][i] = \big(\text{ref}[(i+k) \bmod N],\; 0\big), \quad k = 0, \ldots, K-1
\f]

где:
- \f$ k \f$ — номер циклического сдвига (гипотеза задержки)
- \f$ N \f$ — размер FFT
- Мнимая часть = 0 (действительный сигнал → комплексное представление)

Реализация: HIP kernel `apply_cyclic_shifts` — параллельно формирует все K копий.

@section fm_correlator_formulas_correlation Корреляция в частотной области

Корреляция через умножение спектров (теорема о свёртке):

@subsection fm_correlator_formulas_fft_forward Прямое FFT (R2C)

\f[
\text{ref\_fft}[k] = \text{FFT}\{\text{ref\_complex}[k]\}, \quad k = 0, \ldots, K-1
\f]

\f[
\text{inp\_fft}[s] = \text{FFT}\{\text{input}[s]\}, \quad s = 0, \ldots, S-1
\f]

@note R2C FFT: для действительного входа длины \f$ N \f$ выход содержит только \f$ N/2+1 \f$ бинов
(Hermitian symmetry: \f$ X[N-k] = \overline{X[k]} \f$). Экономия памяти 2x.

@subsection fm_correlator_formulas_conj_multiply Сопряжённое умножение

\f[
\text{corr\_fft}[s][k] = \overline{\text{ref\_fft}[k]} \cdot \text{inp\_fft}[s]
\f]

Реализация: HIP kernel `multiply_conj_fused` — 3D grid (N/2+1, K, S),
одновременная обработка всех комбинаций сигналов и сдвигов.

@subsection fm_correlator_formulas_ifft Обратное FFT (C2R)

\f[
\text{corr\_time}[s][k] = \text{IFFT}\{\text{corr\_fft}[s][k]\}
\f]

@subsection fm_correlator_formulas_peaks Извлечение пиков

\f[
\text{peaks}[s][k][j] = \frac{|\text{corr\_time}[s][k][j]|}{N}, \quad j = 0, \ldots, n_{kg}-1
\f]

Реализация: HIP kernel `extract_magnitudes_real` — bitwise abs (1 инструкция, без ветвлений).

@section fm_correlator_formulas_hermitian Hermitian Symmetry

Для действительного входного сигнала длины \f$ N \f$:

\f[
X[k] = \overline{X[N-k]}, \quad k = 1, \ldots, N/2-1
\f]

Следствия:
- R2C FFT выдаёт только \f$ N/2+1 \f$ комплексных бинов (вместо \f$ N \f$)
- Сопряжённое умножение выполняется только для \f$ N/2+1 \f$ бинов
- C2R IFFT восстанавливает полный действительный сигнал длины \f$ N \f$
- **Экономия**: 2x по памяти, ~2x по вычислениям

@section fm_correlator_formulas_summary Сводка формул

| Этап | Формула | Размерность |
|------|---------|-------------|
| M-seq | LFSR, poly=0x00400007 | [N] float |
| Cyclic shift | \f$ \text{ref}[(i+k) \bmod N] \f$ | [K × N] float2 |
| R2C FFT | \f$ \text{FFT}\{x\} \f$ | [K × (N/2+1)] complex |
| Conj multiply | \f$ \overline{R} \cdot I \f$ | [S × K × (N/2+1)] complex |
| C2R IFFT | \f$ \text{IFFT}\{X\} \f$ | [S × K × N] float |
| Peaks | \f$ |x| / N \f$ | [S × K × n_kg] float |

@see fm_correlator_overview
@see fm_correlator_tests
