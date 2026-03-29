@page heterodyne_formulas Heterodyne -- Математические формулы

@tableofcontents

@section het_formulas_intro Введение

Модуль **heterodyne** реализует stretch-processing (dechirp) ЛЧМ сигналов.
Ниже приведены формулы, лежащие в основе каждого этапа обработки:
модель принятого сигнала, dechirp-умножение, вычисление beat-частоты,
расчёт дальности и SNR.

---

@section het_formulas_lfm Модель ЛЧМ сигнала

@subsection het_formulas_lfm_tx Передаваемый (опорный) ЛЧМ

\f[
s_{tx}(t) = \exp\!\big(j[\pi \mu\, t^2 + 2\pi f_0\, t]\big)
\f]

где:
- \f$ f_0 \f$ -- начальная частота (f_start)
- \f$ \mu = B / T \f$ -- скорость перестройки частоты (chirp rate)
- \f$ B = f_{\text{end}} - f_{\text{start}} \f$ -- полоса ЛЧМ (Гц)
- \f$ T = N / f_s \f$ -- длительность импульса (с)

@subsection het_formulas_lfm_rx Принятый ЛЧМ сигнал

Сигнал, отражённый от цели на дальности \f$ R \f$, приходит с задержкой \f$ \tau = 2R/c \f$:

\f[
s_{rx}(t) = A \cdot \exp\!\big(j[\pi \mu (t-\tau)^2 + 2\pi f_0 (t-\tau)]\big)
\f]

где:
- \f$ A \f$ -- амплитуда (определяется ЭПР цели и потерями)
- \f$ \tau = 2R / c \f$ -- задержка двойного распространения (с)
- \f$ c = 3 \times 10^8 \f$ м/с -- скорость света

---

@section het_formulas_dechirp Dechirp (Stretch Processing)

@subsection het_formulas_dechirp_multiply Операция dechirp

Умножение принятого сигнала на **комплексно-сопряжённый** опорный:

\f[
s_{dc}(t) = s_{rx}(t) \cdot s_{tx}^*(t)
\f]

@subsection het_formulas_dechirp_result Результат dechirp

Раскрывая произведение:

\f[
s_{dc}(t) = A \cdot \exp\!\big(-j\,2\pi \mu \tau\, t + j\,\phi_0\big)
\f]

где \f$ \phi_0 = \pi \mu \tau^2 - 2\pi f_0 \tau \f$ -- постоянная фаза (не зависит от \f$ t \f$).

Результат dechirp -- **чистый тон** (синусоида) с частотой:

\f[
f_{\text{beat}} = \mu \cdot \tau = \frac{B}{T} \cdot \frac{2R}{c}
\f]

@note Чем дальше цель, тем выше beat-частота. Это ключевое свойство stretch-processing:
задержка (дальность) преобразуется в частоту, которую легко измерить через FFT.

@subsection het_formulas_dechirp_kernel GPU Kernel: dechirp_multiply

Для каждого отсчёта \f$ n \f$ и антенны \f$ a \f$:

\f[
s_{dc}[n, a] = s_{rx}[n, a] \cdot \overline{s_{tx}[n]}
\f]

где \f$ \overline{z} \f$ -- комплексное сопряжение. В float2:

\f[
\text{out.x} = \text{rx.x} \cdot \text{tx.x} + \text{rx.y} \cdot \text{tx.y}
\f]
\f[
\text{out.y} = \text{rx.y} \cdot \text{tx.x} - \text{rx.x} \cdot \text{tx.y}
\f]

---

@section het_formulas_beat Beat-частота

После dechirp сигнал содержит чистый тон. FFT даёт пик на beat-частоте:

\f[
f_{\text{beat}} = \mu \cdot \tau = \frac{B \cdot 2R}{T \cdot c}
\f]

Пик находится как:

\f[
k_{\text{peak}} = \arg\max_k |X[k]|, \quad f_{\text{beat}} = k_{\text{peak}} \cdot \frac{f_s}{N}
\f]

где \f$ X[k] \f$ -- FFT dechirp-сигнала, \f$ N \f$ -- длина FFT.

---

@section het_formulas_range Вычисление дальности

Из beat-частоты получаем дальность:

\f[
R = \frac{c \cdot T \cdot f_{\text{beat}}}{2 B}
\f]

Подставляя \f$ T = N / f_s \f$:

\f[
R = \frac{c \cdot N \cdot f_{\text{beat}}}{2 \cdot f_s \cdot B}
\f]

@subsection het_formulas_range_resolution Разрешение по дальности

\f[
\Delta R = \frac{c}{2B}
\f]

Определяется только полосой ЛЧМ. Например, при \f$ B = 2 \f$ МГц: \f$ \Delta R = 75 \f$ м.

@subsection het_formulas_range_max Максимальная дальность

\f[
R_{\max} = \frac{c \cdot N}{4 B} \cdot \frac{B}{f_s} = \frac{c \cdot N}{4 f_s}
\f]

Ограничена количеством отсчётов и частотой дискретизации (по теореме Найквиста для beat-частоты).

---

@section het_formulas_snr SNR (Signal-to-Noise Ratio)

Отношение сигнал/шум в дБ вычисляется по спектру:

\f[
\text{SNR} = 20 \cdot \log_{10}\!\left(\frac{|X_{k_{\text{peak}}}|}{(|X_{k_{\text{peak}}-1}| + |X_{k_{\text{peak}}+1}|) / 2}\right) \quad [\text{дБ}]
\f]

где:
- \f$ |X_{k_{\text{peak}}}| \f$ -- амплитуда пика (сигнал)
- \f$ |X_{k_{\text{peak}} \pm 1}| \f$ -- амплитуды соседних бинов (оценка шумового пола)

@note Это упрощённая оценка SNR по соседним бинам. Для более точной оценки
можно использовать среднюю мощность шумовых бинов (исключая область пика).

---

@section het_formulas_correction Коррекция фазы (dechirp_correct)

@subsection het_formulas_correct_dc Удаление DC-составляющей

Перед коррекцией удаляется постоянная составляющая:

\f[
\bar{s} = \frac{1}{N} \sum_{n=0}^{N-1} s_{dc}[n], \qquad s'_{dc}[n] = s_{dc}[n] - \bar{s}
\f]

@subsection het_formulas_correct_phase Фазовая коррекция

Коррекция линейного набега фазы:

\f[
s_{\text{corr}}[n] = s'_{dc}[n] \cdot \exp\!\big(-j \cdot \phi_{\text{step}} \cdot n\big)
\f]

где \f$ \phi_{\text{step}} \f$ -- шаг фазовой коррекции, вычисляемый из параметров ЛЧМ.

На GPU используется `sincosf(phase)` через SFU (Special Function Unit):

\f[
\exp(-j\phi) = \cos(\phi) - j\sin(\phi)
\f]

---

@section het_formulas_multibeam Многоантенный режим

Для \f$ N_a \f$ антенн с различными задержками \f$ \tau_a \f$:

\f[
s_{rx}^{(a)}(t) = A_a \cdot \exp\!\big(j[\pi \mu (t-\tau_a)^2 + 2\pi f_0 (t-\tau_a)]\big)
\f]

Dechirp выполняется **параллельно** для всех антенн в одном kernel launch (2D grid).
Каждая антенна даёт свою beat-частоту \f$ f_{\text{beat}}^{(a)} \f$ и дальность \f$ R_a \f$.

---

@section het_formulas_see_also Смотрите также

- @ref heterodyne_overview -- Обзор модуля, классы и быстрый старт
- @ref heterodyne_tests -- Тесты, бенчмарки и графики
