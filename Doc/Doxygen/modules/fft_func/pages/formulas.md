@page fft_func_formulas FFT Functions — Математика

@tableofcontents

@section fft_math_dft 1. Дискретное преобразование Фурье (DFT)

Для сигнала \f$ x[n] \f$ длиной \f$ N \f$:

\f[
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi k n}{N}}, \quad k = 0, 1, \ldots, N-1
\f]

**Частотная ось**:

\f[
f_k = k \cdot \frac{f_s}{N_{FFT}}, \quad k = 0, 1, \ldots, \frac{N_{FFT}}{2}
\f]

где \f$ f_s \f$ — частота дискретизации, \f$ N_{FFT} \f$ — размер FFT (после zero-padding).

@section fft_math_zeropad 2. Zero-Padding

Если входной сигнал имеет \f$ N_{point} \f$ отсчётов, а FFT выполняется для \f$ N_{FFT} \ge N_{point} \f$:

\f[
x_{padded}[n] = \begin{cases}
x[n], & 0 \le n < N_{point} \\
0,    & N_{point} \le n < N_{FFT}
\end{cases}
\f]

**Эффект**: увеличение разрешения частотной оси без добавления новой информации.

\f$ N_{FFT} = 2^{\lceil \log_2 N_{point} \rceil} \f$ — всегда степень двойки.

Kernel `pad_data` на HIP выполняет это параллельно:
@code{.cpp}
// Каждый поток обрабатывает один отсчёт одного луча
global_id = beam * nFFT + sample;
if (sample < n_point): fft_buf[global_id] = input[beam * n_point + sample];
else:                  fft_buf[global_id] = {0.0f, 0.0f};
@endcode

@section fft_math_magphase 3. Амплитуда и фаза

После FFT получаем комплексный спектр \f$ X[k] = Re[k] + j \cdot Im[k] \f$.

**Амплитуда** (magnitude):

\f[
|X[k]| = \sqrt{Re[k]^2 + Im[k]^2}
\f]

На GPU используется `__fsqrt_rn` — округление к ближайшему (IEEE 754 compliant).

**Фаза** (phase):

\f[
\angle X[k] = \text{atan2}(Im[k],\ Re[k]) \in (-\pi, \pi]
\f]

**Нормировка** для сравнения с CPU FFT:

\f[
|X[k]|_{norm} = \frac{|X[k]|}{N_{FFT}}
\f]

@section fft_math_batch 4. Batch FFT (несколько лучей)

Для \f$ B \f$ лучей одновременно (`beam_count`):

\f[
X_b[k] = \sum_{n=0}^{N_{FFT}-1} x_b[n] \cdot e^{-j \frac{2\pi k n}{N_{FFT}}}, \quad b = 0, \ldots, B-1
\f]

hipFFT выполняет batch FFT одним вызовом:
@code{.cpp}
hipfftExecC2C(plan, fft_buf, fft_buf, HIPFFT_FORWARD);
// plan создан с: batch = beam_count, n = {nFFT}
@endcode

@note In-place: входной буфер = выходной буфер (`BufferSet<3>::kFftBuf`).

@section fft_math_peak 5. Поиск максимума спектра

@subsection fft_math_one_peak ONE_PEAK — один максимум

Kernel `post_kernel` выполняет:

1. **Reduction**: находит глобальный максимум \f$ |X[k]| \f$ по всем \f$ k \f$
2. **Параболическая интерполяция** для уточнения частоты:

\f[
\delta = \frac{1}{2} \cdot \frac{|X[k-1]| - |X[k+1]|}{|X[k-1]| - 2|X[k]| + |X[k+1]|}
\f]

\f[
f_{refined} = (k_{peak} + \delta) \cdot \frac{f_s}{N_{FFT}}
\f]

@subsection fft_math_all_maxima ALL_MAXIMA — Blelloch Exclusive Scan

Алгоритм stream compaction из 4 kernels:

| Шаг | Kernel | Описание |
|-----|--------|----------|
| 1 | `detect_all_maxima` | Флаги: `is_max[k] = (\|X[k]\| > \|X[k-1]\|) && (\|X[k]\| > \|X[k+1]\|)` |
| 2 | `block_scan` | Prefix scan (Blelloch) — Up-sweep (reduce) + Down-sweep |
| 3 | `block_add` | Добавление сумм блоков (второй проход scan) |
| 4 | `compact_maxima` | Запись результатов по вычисленным позициям |

Сложность: \f$ O(\log N) \f$ стадий → prefix-sum для компактификации массива максимумов.

@section fft_math_freqbin 6. Частотный бин

\f$ f_k = k \cdot \dfrac{f_s}{nFFT} \quad [\text{Гц}] \f$

@section fft_math_seealso См. также

- @ref fft_func_overview — Обзор модуля
- @ref fft_func_tests — Тесты и бенчмарки
