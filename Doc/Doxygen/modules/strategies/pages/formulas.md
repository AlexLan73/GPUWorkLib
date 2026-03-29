@page strategies_formulas strategies — Математические формулы

@tableofcontents

@section strategies_formulas_intro Введение

Модуль `strategies` реализует цифровое формирование диаграммы направленности (DBF)
с тремя сценариями post-FFT анализа. Основная операция — матричное умножение (CGEMM)
весовой матрицы на сигнальную, с последующим спектральным анализом.

@section strategies_formulas_beamforming Beamforming (CGEMM)

Формирование лучей через комплексное матричное умножение:

\f[
X = W \cdot S
\f]

где:
- \f$ W \in \mathbb{C}^{P \times P} \f$ — весовая матрица (\f$ P \f$ = количество антенн)
- \f$ S \in \mathbb{C}^{P \times M} \f$ — сигнальная матрица (\f$ M \f$ = количество отсчётов)
- \f$ X \in \mathbb{C}^{P \times M} \f$ — выходная матрица (сформированные лучи)

Реализация: **hipBLAS CGEMM** — оптимизированное матричное умножение на GPU.

@section strategies_formulas_pipeline Pipeline (7 шагов)

Полный pipeline обработки:

| Шаг | Операция | Формула / Описание |
|-----|----------|-------------------|
| 1 | Statistics(S) | \f$ \mu = \text{mean}(S),\; \sigma = \text{std}(S) \f$ → `pre_input_stats` |
| 2 | CGEMM | \f$ X = W \cdot S \f$ |
| 3 | Statistics(X) | \f$ \mu = \text{mean}(X),\; \sigma = \text{std}(X) \f$ → `post_gemm_stats` |
| 4 | Window + FFT | Hamming → zero-pad → hipFFT |
| 5 | Statistics(\|spectrum\|) | \f$ \mu = \text{mean}(|X_f|),\; \sigma = \text{std}(|X_f|) \f$ → `post_fft_stats` |
| 6 | Post-FFT | OneMax / AllMaxima / MinMax |
| 7 | Sync | Синхронизация 4 HIP streams |

@section strategies_formulas_onemax OneMax — параболическая интерполяция

Нахождение одного максимума спектра с субдискретной точностью через параболическую интерполяцию по 3 точкам:

\f[
\delta = \frac{1}{2} \cdot \frac{m_{k-1} - m_{k+1}}{m_{k-1} - 2m_k + m_{k+1}}
\f]

где:
- \f$ m_k \f$ — амплитуда в бине максимума
- \f$ m_{k-1}, m_{k+1} \f$ — амплитуды в соседних бинах
- \f$ \delta \in [-0.5, +0.5] \f$ — субдискретная поправка

Уточнённая частота:

\f[
f_{\text{peak}} = (k + \delta) \cdot \frac{f_s}{N_{\text{fft}}}
\f]

Уточнённая амплитуда:

\f[
A_{\text{peak}} = m_k - \frac{1}{4}(m_{k-1} - m_{k+1}) \cdot \delta
\f]

@section strategies_formulas_minmax GlobalMinMax — динамический диапазон

Вычисление динамического диапазона спектра:

\f[
DR = 20 \cdot \log_{10}\!\left(\frac{\max}{\max(\min, 10^{-30})}\right) \quad [\text{дБ}]
\f]

где:
- \f$ \max \f$ — глобальный максимум амплитудного спектра
- \f$ \min \f$ — глобальный минимум амплитудного спектра
- \f$ 10^{-30} \f$ — защита от деления на ноль

@note Типичные значения динамического диапазона: 40-120 дБ в зависимости от SNR и типа сигнала.

@section strategies_formulas_statistics Checkpoint-статистика

Три точки контроля статистики в pipeline:

@subsection strategies_formulas_stats_pre Pre-input (Step 1)

\f[
\mu_{\text{in}} = \frac{1}{PM} \sum_{p,m} S[p,m], \quad
\sigma_{\text{in}} = \sqrt{\frac{1}{PM} \sum_{p,m} |S[p,m] - \mu_{\text{in}}|^2}
\f]

@subsection strategies_formulas_stats_gemm Post-CGEMM (Step 3)

\f[
\mu_{\text{gemm}} = \frac{1}{PM} \sum_{p,m} X[p,m], \quad
\sigma_{\text{gemm}} = \sqrt{\frac{1}{PM} \sum_{p,m} |X[p,m] - \mu_{\text{gemm}}|^2}
\f]

@subsection strategies_formulas_stats_fft Post-FFT (Step 5)

\f[
\mu_{\text{fft}} = \frac{1}{PK} \sum_{p,k} |X_f[p,k]|, \quad
\sigma_{\text{fft}} = \sqrt{\frac{1}{PK} \sum_{p,k} (|X_f[p,k]| - \mu_{\text{fft}})^2}
\f]

@section strategies_formulas_hamming Окно Хэмминга

Перед FFT к каждому лучу применяется окно Хэмминга:

\f[
w[n] = 0.54 - 0.46 \cos\!\left(\frac{2\pi n}{N-1}\right), \quad n = 0, \ldots, N-1
\f]

Цель: снижение уровня боковых лепестков в спектре для более точного обнаружения максимумов.

@section strategies_formulas_summary Сводка формул

| Формула | Назначение | GPU библиотека |
|---------|------------|----------------|
| \f$ X = W \cdot S \f$ | Beamforming | hipBLAS CGEMM |
| \f$ X_f = \text{FFT}\{w \cdot X\} \f$ | Спектральный анализ | hipFFT |
| \f$ \delta = \frac{m_{k-1}-m_{k+1}}{2(m_{k-1}-2m_k+m_{k+1})} \f$ | OneMax интерполяция | HIP kernel |
| \f$ DR = 20\log_{10}(\max/\min) \f$ | Dynamic range | HIP kernel |

@see strategies_overview
@see strategies_tests
