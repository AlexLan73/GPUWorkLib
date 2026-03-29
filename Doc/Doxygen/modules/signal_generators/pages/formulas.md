@page signal_generators_formulas Signal Generators -- Математические формулы

@tableofcontents

@section sg_math_intro Введение

Данная страница содержит математическое описание всех типов генерируемых сигналов
модуля **signal_generators**. Все генераторы работают с комплексными IQ-данными (`float2`).

---

@section sg_math_cw CW (Continuous Wave) -- Непрерывная несущая

Генерация чистого тонального сигнала на заданной частоте:

\f[
s_{CW}(t) = A \cdot e^{j(2\pi f_0 t + \varphi_0)}
\f]

где:
- \f$ A \f$ -- амплитуда сигнала
- \f$ f_0 \f$ -- несущая частота (Гц)
- \f$ \varphi_0 \f$ -- начальная фаза (рад)
- \f$ t = n / f_s \f$ -- дискретное время, \f$ n = 0, 1, \ldots, N-1 \f$

@subsection sg_math_cw_multi Multi-beam CW

Для многолучевого режима каждый луч имеет свою частоту:

\f[
s_i(t) = A \cdot e^{j(2\pi f_i t + \varphi_0)}, \quad f_i = f_0 + i \cdot \Delta f
\f]

где \f$ i = 0, 1, \ldots, B-1 \f$ -- номер луча, \f$ \Delta f \f$ -- шаг по частоте.

@note На GPU каждый поток вычисляет один отсчет одного луча:
`global_id = beam * N + sample`.

---

@section sg_math_lfm LFM (Linear Frequency Modulation) -- Линейная частотная модуляция

Chirp-сигнал с линейным изменением частоты от \f$ f_{start} \f$ до \f$ f_{end} \f$:

\f[
s_{LFM}(t) = A \cdot e^{j(\pi \mu t^2 + 2\pi f_{start} t)}
\f]

Скорость изменения частоты (chirp rate):

\f[
\mu = \frac{f_{end} - f_{start}}{T}
\f]

где \f$ T \f$ -- длительность импульса (с).

Мгновенная частота:

\f[
f_{inst}(t) = f_{start} + \mu \cdot t
\f]

@subsection sg_math_lfm_conjugate Сопряженный LFM (LfmConjugate)

Для процедуры dechirp генерируется комплексно-сопряженный сигнал:

\f[
s_{conj}(t) = s_{LFM}^*(t) = A \cdot e^{-j(\pi \mu t^2 + 2\pi f_{start} t)}
\f]

Используется в модуле @ref heterodyne_overview для дехирпирования.

---

@section sg_math_noise Noise -- Гауссов шум (Box-Muller + Philox)

Генерация комплексного гауссова шума с заданной дисперсией:

\f[
n(t) = \sigma \sqrt{-2 \ln U_1} \cdot e^{j 2\pi U_2}
\f]

где \f$ U_1, U_2 \sim \mathcal{U}(0,1) \f$ -- равномерно распределенные случайные величины.

@subsection sg_math_noise_components Компоненты шума

В декартовой форме (Re/Im):

\f[
n_{Re} = \sigma \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2)
\f]

\f[
n_{Im} = \sigma \sqrt{-2 \ln U_1} \cdot \sin(2\pi U_2)
\f]

@subsection sg_math_noise_prng Philox-2x32-10 PRNG

Генератор псевдослучайных чисел Philox обеспечивает:
- **Статистическое качество**: проходит тесты BigCrush (TestU01)
- **Параллелизм**: каждый GPU-поток генерирует независимую последовательность
- **Детерминированность**: одинаковый seed --> одинаковый результат

Ключ: `(seed, counter)` --> 2 x uint32 --> нормализация в \f$ [0, 1) \f$:

\f[
U = \frac{\text{philox\_output}}{2^{32}} + \epsilon
\f]

где \f$ \epsilon \f$ -- малая добавка для предотвращения \f$ \ln(0) \f$.

@note Общий PRNG kernel: `modules/signal_generators/kernels/prng.cl`, подключается
через конкатенацию при компиляции основного kernel.

---

@section sg_math_formsignal FormSignal (getX) -- Многоантенный сигнал

Формирование сигнала для антенной решетки с per-channel задержками:

\f[
X(t) = a \cdot \text{norm} \cdot e^{j\varphi(t)} + a_n \cdot \text{norm} \cdot (n_r + j\,n_i)
\f]

где:
- \f$ a \f$ -- амплитуда сигнала
- \f$ a_n \f$ -- амплитуда шума
- \f$ \text{norm} \f$ -- нормировочный коэффициент
- \f$ n_r, n_i \f$ -- гауссов шум Re/Im каналов

@subsection sg_math_formsignal_phase Фаза сигнала

\f[
\varphi(t) = 2\pi f_0 t + \frac{\pi f_{dev}}{t_i} \left(t - \frac{t_i}{2}\right)^2 + \varphi_0
\f]

где:
- \f$ f_0 \f$ -- несущая частота
- \f$ f_{dev} \f$ -- девиация частоты (для LFM-составляющей)
- \f$ t_i \f$ -- длительность импульса

@subsection sg_math_formsignal_delay Задержки между антеннами

Режим LINEAR:

\f[
\tau_{ant} = \tau_{base} + \text{ID} \cdot \tau_{step}
\f]

где:
- \f$ \tau_{base} \f$ -- базовая задержка (мкс)
- \f$ \tau_{step} \f$ -- шаг задержки между антеннами (мкс)
- \f$ \text{ID} \f$ -- номер антенны (0, 1, ..., N_ant - 1)

---

@section sg_math_delayed DelayedFormSignal -- Дробная задержка (Lagrange 48x5)

Реализация дробной задержки через интерполяцию Лагранжа 4-го порядка
с квантованием дробной части на 48 уровней:

\f[
y[n] = \sum_{k=0}^{4} h_k[f] \cdot x[n - D - k + 2]
\f]

@subsection sg_math_delayed_decomposition Разложение задержки

Общая задержка разделяется на целую и дробную части:

\f[
D = \left\lfloor \tau \cdot f_s / 10^6 \right\rfloor
\f]

\f[
f = \text{frac}\left(\tau \cdot f_s / 10^6\right)
\f]

где:
- \f$ D \f$ -- целая часть задержки (в отсчетах)
- \f$ f \f$ -- дробная часть (0...1), квантуется с шагом \f$ 1/48 \f$

@subsection sg_math_delayed_table Таблица Farrow 48x5

Предвычисленная таблица коэффициентов Лагранжа:

\f[
h_k[m] = \prod_{\substack{i=0 \\ i \ne k}}^{4} \frac{m/48 - i}{k - i}, \quad m = 0, 1, \ldots, 47, \quad k = 0, 1, \ldots, 4
\f]

Размер таблицы: **48 строк x 5 столбцов** = 240 float.

@note Квантование дробной части с шагом 1/48 обеспечивает максимальную ошибку
интерполяции < 0.01 sample для полосы до \f$ 0.4 \cdot f_s \f$.

---

@section sg_math_analytical LfmAnalyticalDelay -- Аналитическая задержка LFM

Для LFM-сигнала задержка вычисляется аналитически (без интерполяции):

\f[
s_{delayed}(t) = A \cdot e^{j(\pi \mu t_{local}^2 + 2\pi f_{start} t_{local})}
\f]

где локальное время:

\f[
t_{local} = t - \tau_{ant}
\f]

Граничное условие:

\f[
s_{delayed}(t) = 0 \quad \text{для} \quad t < \tau_{ant}
\f]

@note Аналитическая задержка не вносит артефактов интерполяции и дает идеальный результат
для LFM-сигналов. Для произвольных сигналов используйте `DelayedFormSignalGeneratorROCm`.

---

@section sg_math_script FormScript -- DSL компилятор

`FormScriptGeneratorROCm` принимает текстовое описание сигнала (DSL) и
компилирует его в OpenCL kernel с кешированием на диск.

Пример DSL:

```
signal = cw(f=100e3, amp=1.0) + noise(sigma=0.1)
delay = linear(base=0.5, step=0.1)
```

Компиляция DSL --> OpenCL kernel --> binary cache.

@section sg_math_see_also Смотрите также

- @ref signal_generators_overview -- Обзор модуля и быстрый старт
- @ref signal_generators_tests -- Тесты, бенчмарки, графики
