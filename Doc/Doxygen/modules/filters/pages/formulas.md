@page filters_formulas Filters — Математика

@tableofcontents

@section filters_math_fir 1. FIR (прямая свёртка)

\f[
y[ch][n] = \sum_{k=0}^{N-1} h[k] \cdot x[ch][n-k]
\f]

где \f$ h[k] \f$ — коэффициенты фильтра, \f$ N \f$ — число тапов, \f$ ch \f$ — канал.

@note GPU: 2D NDRange (channels × samples). Каждый thread вычисляет один выходной отсчёт для одного канала.

@section filters_math_iir 2. IIR (Biquad cascade, DFII-T)

\f[
y[n] = b_0 x[n] + w_1[n-1]
\f]
\f[
w_1[n] = b_1 x[n] - a_1 y[n] + w_2[n-1]
\f]
\f[
w_2[n] = b_2 x[n] - a_2 y[n]
\f]

Каскад из нескольких секций: выход секции \f$ i \f$ — вход секции \f$ i+1 \f$.

@note GPU: 1 thread per channel, последовательная обработка отсчётов (рекуррентная зависимость).

@section filters_math_ma 3. Moving Averages

@subsection filters_math_sma SMA (Simple Moving Average)

\f[
\text{SMA}[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k]
\f]

@subsection filters_math_ema EMA (Exponential Moving Average)

\f[
\text{EMA}[n] = \alpha \cdot x[n] + (1-\alpha) \cdot \text{EMA}[n-1], \quad \alpha = \frac{2}{N+1}
\f]

@subsection filters_math_mma MMA (Modified Moving Average)

\f[
\text{MMA}[n] = \frac{1}{N} \cdot x[n] + \frac{N-1}{N} \cdot \text{MMA}[n-1]
\f]

@subsection filters_math_dema DEMA (Double EMA)

\f[
\text{DEMA}[n] = 2 \cdot \text{EMA}_1[n] - \text{EMA}(\text{EMA}_1)[n]
\f]

@subsection filters_math_tema TEMA (Triple EMA)

\f[
\text{TEMA}[n] = 3\text{EMA}_1 - 3\text{EMA}_2 + \text{EMA}_3
\f]

@section filters_math_kalman 4. Kalman (1D scalar)

**Predict**:
\f[
\hat{x}^-[n] = \hat{x}[n-1], \quad P^-[n] = P[n-1] + Q
\f]

**Update**:
\f[
K[n] = \frac{P^-[n]}{P^-[n] + R}
\f]
\f[
\hat{x}[n] = \hat{x}^-[n] + K[n](z[n] - \hat{x}^-[n])
\f]
\f[
P[n] = (1 - K[n]) P^-[n]
\f]

Re и Im обрабатываются **независимо** с одинаковыми Q и R.

@section filters_math_kama 5. KAMA (Kaufman Adaptive Moving Average)

**Efficiency Ratio** (ER):
\f[
ER[n] = \frac{|x[n] - x[n-N]|}{\sum_{k=1}^{N} |\Delta x[k]|}
\f]

**Smoothing Constant** (SC):
\f[
SC[n] = \big(ER[n] \cdot (\alpha_{\text{fast}} - \alpha_{\text{slow}}) + \alpha_{\text{slow}}\big)^2
\f]

где \f$ \alpha_{\text{fast}} = \frac{2}{3} \f$, \f$ \alpha_{\text{slow}} = \frac{2}{31} \f$.

**KAMA update**:
\f[
\text{KAMA}[n] = \text{KAMA}[n-1] + SC[n] \cdot (x[n] - \text{KAMA}[n-1])
\f]

@section filters_math_seealso См. также

- @ref filters_overview — Обзор модуля
- @ref filters_tests — Тесты и бенчмарки
