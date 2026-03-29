@page range_angle_formulas range_angle — Математические формулы

@tableofcontents

@section range_angle_formulas_intro Введение

Модуль `range_angle` реализует классический pipeline обработки ЛЧМ-радара
для 2D антенной решётки: dechirp → range FFT → 2D beam FFT → peak search.
Результат — оценка дальности и двух углов (азимут, угол места) цели.

@section range_angle_formulas_lfm Опорный ЛЧМ сигнал

Генерация опорного линейно-частотно-модулированного (ЛЧМ) сигнала:

\f[
\text{ref\_lfm}[n] = \exp\!\left(j\pi\big(f_{\text{start}} + \tfrac{\mu}{2} t\big) t\right)
\f]

где:
- \f$ f_{\text{start}} \f$ — начальная частота ЛЧМ (Гц)
- \f$ \mu = (f_{\text{end}} - f_{\text{start}}) / T \f$ — скорость девиации частоты (Гц/с)
- \f$ t = n / f_s \f$ — время отсчёта
- \f$ T \f$ — длительность импульса

@section range_angle_formulas_dechirp Dechirp (выделение биений)

Смешение принятого сигнала с комплексно-сопряжённым опорным:

\f[
x_{\text{beat}}[i,n] = x[i,n] \cdot \overline{\text{ref\_lfm}[n]}
\f]

где:
- \f$ x[i,n] \f$ — принятый сигнал антенны \f$ i \f$, отсчёт \f$ n \f$
- \f$ \overline{\text{ref\_lfm}[n]} \f$ — комплексное сопряжение опорного ЛЧМ

Результат — тональный сигнал биений (beat tone), частота которого пропорциональна задержке (дальности).

@note Перед FFT к beat-сигналу применяется окно Хэмминга для снижения боковых лепестков.

@section range_angle_formulas_range_fft Range FFT

Batched FFT по каждой антенне для определения спектра дальности:

\f[
X_{\text{range}}[i,k] = \text{FFT}\{x_{\text{beat}}[i,n]\}, \quad k = 0, \ldots, N-1
\f]

Каждый бин \f$ k \f$ соответствует определённой дальности.

@section range_angle_formulas_range_resolution Разрешение по дальности

Разрешающая способность по дальности определяется полосой ЛЧМ:

\f[
\Delta R = \frac{c}{2B}
\f]

где:
- \f$ c = 3 \times 10^8 \f$ м/с — скорость света
- \f$ B = f_{\text{end}} - f_{\text{start}} \f$ — полоса ЛЧМ (Гц)

Пример: \f$ B = 100 \f$ МГц → \f$ \Delta R = 1.5 \f$ м.

Дальность для бина \f$ k \f$:

\f[
R[k] = \frac{c \cdot k \cdot f_s}{2 B \cdot N}
\f]

@section range_angle_formulas_beam_fft 2D Beam FFT (пространственная обработка)

2D FFT по пространственным координатам (азимут × угол места) для каждого бина дальности:

\f[
X_{\text{beam}}[k, p, q] = \text{FFT2D}\{X_{\text{range}}[i_{az}, i_{el}, k]\}
\f]

где:
- \f$ i_{az} \f$ — индекс антенны по азимуту
- \f$ i_{el} \f$ — индекс антенны по углу места
- \f$ p, q \f$ — индексы угловых бинов после FFT

Результат — **3D power cube** размерности [n_range × n_az × n_el].

После FFT применяется `fftshift` для центрирования нулевой частоты.

@section range_angle_formulas_peak_search Peak Search (обнаружение целей)

Поиск максимумов в 3D power cube:

\f[
\text{TargetInfo} = \{R,\; \theta_{az},\; \theta_{el},\; P_{dB},\; \text{SNR}_{dB}\}
\f]

Алгоритм:
1. **3D GPU max reduction** — поиск глобального максимума (или TOP-N максимумов)
2. Пересчёт индексов бинов в физические координаты (дальность в метрах, углы в градусах)
3. Оценка SNR относительно среднего уровня шума

Режимы:
- **TOP_1** — одна самая сильная цель
- **TOP_N** — N целей по убыванию мощности

@section range_angle_formulas_summary Сводка pipeline

| Стадия | Операция | Результат |
|--------|----------|-----------|
| 1 | Dechirp + Window | \f$ x_{\text{beat}} = x \cdot \overline{\text{ref}} \cdot w \f$ |
| 2 | Range FFT | \f$ X_{\text{range}}[i,k] \f$ — спектр дальности |
| 3 | Transpose | [n_range × n_az × n_el] layout |
| 4 | 2D Beam FFT | 3D power cube [range × az × el] |
| 5 | Peak Search | TargetInfo: \f$ R, \theta_{az}, \theta_{el}, P_{dB} \f$ |

@see range_angle_overview
@see range_angle_tests
