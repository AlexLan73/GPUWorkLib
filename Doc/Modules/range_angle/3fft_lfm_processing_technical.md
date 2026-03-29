# 🔬 2FFT для обработки ЛЧМ — Инженерно-технический анализ

> **Версия**: 1.0 | **Дата**: 2026-03-17 | **Автор**: Кодо
> **Уровень**: Senior DSP Engineer / GPU-разработчик
> **Связь с проектом**: HeterodyneDechirp → LFMRangeProcessor → RangeDopplerProcessor

---

## 1. Математическая модель ЛЧМ-сигнала

Линейно-частотно-модулированный (LFM/ЛЧМ) сигнал:

```
s(t) = A · exp(j · (2π·f₀·t + π·μ·t²)),   0 ≤ t ≤ T
```

| Параметр | Обозначение | Единица |
|----------|-------------|---------|
| Начальная частота | f₀ | Гц |
| Скорость девиации (chirp rate) | μ = B/T | Гц/с |
| Полоса девиации | B | Гц |
| Длительность импульса | T | с |
| Мгновенная частота | f(t) = f₀ + μ·t | Гц |
| **Время-полосовой продукт (TBP)** | **D = B·T >> 1** | безразм. |

Спектр LFM-сигнала: квадратичная фаза → FFT даёт **широкий прямоугольный** спектр шириной B Гц.
**Ключевое свойство**: квадратичная фаза позволяет использовать два последовательных FFT для сжатия.

---

## 2. Метод A — Matched Filter via FFT (Согласованная фильтрация)

### Принцип

Выход согласованного фильтра = кросс-корреляция принятого сигнала с опорным:
```
y(t) = r(t) ⋆ s(-t) = IFFT[ FFT(r) · FFT*(s) ]
```

Теорема о свёртке → два FFT вместо прямой свёртки O(N²).

### Алгоритм

```
1. R(f) = FFT(r(t))              ← спектр принятого
2. Y(f) = R(f) · S*(f)           ← умножение на сопряжённый опорный спектр
3. y(t) = IFFT(Y(f))             ← сжатый импульс
```

### Вычислительная сложность

| Метод | Сложность | N=1024 операций |
|-------|-----------|-----------------|
| Прямая свёртка | O(N²) | ~1 000 000 |
| FFT-метод | O(N log N) | ~10 240 |
| **Выигрыш** | **≈ 100×** | — |

### Математика: почему работает

Принятый сигнал от точечной цели (задержка τ = 2R/c):
```
r(t) = A · s(t - τ)
```

Спектры:
```
R(f) = A · S(f) · exp(-j2πfτ)

Y(f) = R(f) · S*(f) = A · |S(f)|² · exp(-j2πfτ)
```

После IFFT — **импульс sinc(t-τ) с шириной ~1/B**, т.е. дальностное разрешение:
```
ΔR = c / (2B)  [метры]
```

### Python-реализация

```python
import numpy as np

def matched_filter_fft(received: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Согласованная фильтрация LFM через FFT/IFFT."""
    N = max(len(received) + len(reference) - 1, 1)
    Nfft = 1 << int(np.ceil(np.log2(N)))   # следующая степень 2
    R = np.fft.fft(received,   n=Nfft)
    S = np.fft.fft(reference,  n=Nfft)
    y = np.fft.ifft(R * np.conj(S))
    return y[:len(received)]
```

### GPU-реализация (hipFFT)

```cpp
// Предварительно вычислить S*(f) один раз:
hipfftHandle plan;
hipfftPlan1d(&plan, Nfft, HIPFFT_C2C, batch_count);

// Для каждого принятого сигнала:
hipfftExecC2C(plan, d_received,  d_R, HIPFFT_FORWARD);   // FFT(r)
// Kernel: d_Y[i] = d_R[i] * d_S_conj[i]
hipfftExecC2C(plan, d_Y, d_output, HIPFFT_BACKWARD);      // IFFT
// Normalize: /Nfft
```

### Преимущества / Ограничения

✅ Оптимальный SNR (теорема Шварца)
✅ Универсален для любой формы опорного сигнала
✅ Простая batch-обработка на GPU
❌ Требует полосы АЦП = B Гц
❌ Движущиеся цели: доплеровский сдвиг снижает выход коррелятора

---

## 3. Метод B — Stretch Processing (Dechirp + FFT)

### Принцип

**Ключевая идея**: умножение принятого LFM на комплексно-сопряжённый опорный LFM переводит временную задержку τ в **постоянную частоту** биений (beat frequency).

### Математический вывод

Принятый сигнал: `r(t) = A · exp(j(2πf₀(t-τ) + πμ(t-τ)²))`
Опорный: `s_ref(t) = exp(j(2πf₀t + πμt²))`

Dechirp-произведение:
```
d(t) = r(t) · s_ref*(t)
     = A · exp(-j(2πf₀τ + 2πμτt - πμτ²))
     ≈ A · exp(-j2π·f_beat·t) · const_phase
```

где `f_beat = μ·τ = μ·2R/c` — частота биений, линейно пропорциональная дальности R.

**FFT от d(t) → пик на частоте f_beat → дальность:**
```
R = f_beat · c / (2μ)
```

### Алгоритм (2 шага вместо 3)

```
1. d(t) = r(t) · conj(s_ref(t))    ← dechirp (умножение поэлементно)
2. D(f) = FFT(d(t))                 ← спектр → пик на f_beat → R
```

### Критическое преимущество: полоса АЦП

После dechirp-а сигнал узкополосный: его полоса = `μ·Δτ_max` (где Δτ_max — максимальный разброс задержек в интересующем окне дальностей).

```
B_АЦП_stretch = μ · Δτ_max << B_LFM
```

Для типичного ШПС-радара: **снижение частоты дискретизации в 10–1000 раз!**

### Сравнение с методом A

| Критерий | Matched Filter FFT | Stretch Processing |
|----------|-------------------|--------------------|
| Полоса АЦП | = B | << B (в 10–1000 раз) |
| Число FFT | 2 (FFT + IFFT) | 1 (FFT) |
| Дальностный интервал | Весь | "Окно" Δτ_max |
| Применение | Универсально | ШПС, SAR, FMCW |
| Связь с GPUWorkLib | — | **Использует HeterodyneDechirp** |

### Связь с модулем HeterodyneDechirp

Существующий модуль уже реализует Шаг 1! Цепочка:
```
HeterodyneDechirp → [dechirped buffer] → FFTProcessor → [range profile]
```

---

## 4. Метод C — Range-Doppler 2D FFT

### Структура данных: Radar Data Cube

Для пачки из P импульсов — матрица `[P × N]`:
```
Data[p, n]:  p = 0..P-1  (slow time, межимпульсный интервал)
             n = 0..N-1  (fast time, внутри импульса)
```

### Алгоритм 2D FFT

```
Шаг 1: Range FFT (вдоль fast time, axis=1)
   Для каждой строки p: Range[p, :] = FFT(Data[p, :] · conj(ref))
   → P параллельных FFT длиной N

Шаг 2: Doppler FFT (вдоль slow time, axis=0)
   Для каждого столбца k: RD_map[:, k] = FFT(Range[:, k])
   → N параллельных FFT длиной P
```

### Физика Доплера

Фаза от цели со скоростью v между импульсами (интервал T_PRI):
```
φ_p = 2π · (2·v·p·T_PRI) / λ = 2π · f_d · p·T_PRI
```
где `f_d = 2v/λ = 2v·f₀/c` — доплеровская частота.

FFT по slow time → пик на f_d → скорость:
```
v = f_d · c / (2·f₀)
```

### Оси Range-Doppler карты

```
Дальность:   R  = f_beat   · c / (2μ)          [м]
Скорость:    v  = f_doppler · c / (2·f₀)        [м/с]
Разрешение:  ΔR = c / (2B),  Δv = λ / (2·P·T_PRI)
```

### Вычислительная сложность

```
Всего: O(P·N·log N + N·P·log P) = O(P·N·(log N + log P))
```

Пример: P=256, N=1024 → ~4.8M ops, vs O((P·N)²) ≈ 70 триллионов для прямой обработки.

### GPU-реализация (hipFFT batch)

```cpp
// Batch Range FFT: P независимых FFT по N точек
hipfftHandle plan_range;
hipfftPlan1d(&plan_range, N, HIPFFT_C2C, P);   // batch = P

// Transpose [P×N] → [N×P] (HIP kernel, coalesced access)
launch_transpose_kernel(d_range_fft, d_transposed, P, N, stream);

// Batch Doppler FFT: N независимых FFT по P точек
hipfftHandle plan_doppler;
hipfftPlan1d(&plan_doppler, P, HIPFFT_C2C, N);  // batch = N

hipfftExecC2C(plan_range,   d_data,       d_range_fft,  HIPFFT_FORWARD);
hipfftExecC2C(plan_doppler, d_transposed, d_rd_map,     HIPFFT_FORWARD);
```

**Критично**: transpose между FFT-ами нужен для coalesced memory access при Doppler FFT!

---

## 5. Метод D — Fractional Fourier Transform (FrFT)

### Математическое определение

FrFT порядка α — поворот на угол φ = α·π/2 в плоскости время-частота (Wigner distribution):

```
F_α{x}(u) = ∫ x(t) · K_α(t,u) dt
K_α(t,u) = √(1 - j·cot φ) · exp(jπ(t²+u²)·cot φ - j2π·t·u / sin φ)
```

При α=1 → стандартный FFT. При α=0 → тождественное преобразование.

### LFM — собственная функция FrFT

LFM с chirp rate μ концентрируется в **точечный пик** при оптимальном угле:
```
φ_opt = -arccot(μ / f_s²)
α_opt = 2·φ_opt / π
```

### Реализация через алгоритм Bluestein (3×FFT, O(N log N))

```
FrFT(x, α):
  1. pre_chirp[n]  = x[n] · exp(-jπ·cot(φ)·n²/N)
  2. chirp_conv = IFFT( FFT(pre_chirp) · FFT(h) )     ← chirp z-transform
  3. result[n]     = chirp_conv[n] · exp(-jπ·cot(φ)·n²/N)
```

### Применение в GPUWorkLib

| Задача | Метод |
|--------|-------|
| Оценка chirp rate μ неизвестного сигнала | FrFT (поиск α_opt) |
| Детектирование слабого LFM в шуме | FrFT (SNR +10..15 dB vs FFT) |
| Разделение 2 LFM разных chirp rates | FrFT (разные α_opt) |
| Оценка начальной частоты f₀ | FrFT пик + пересчёт |

---

## 6. Сравнительный анализ для GPUWorkLib

### Таблица: что выбрать

| Задача | Метод | Сложность | Связь с модулями |
|--------|-------|-----------|-----------------|
| Дальность, стационарная цель | Stretch (B) | O(N log N) | HeterodyneDechirp + FFTProcessor |
| Дальность, произвольный сигнал | Matched Filter (A) | O(N log N) | FFTProcessor |
| Дальность + скорость | 2D Range-Doppler (C) | O(PN log N) | Новый модуль |
| Оценка параметров LFM | FrFT (D) | O(3N log N) | Новый модуль |
| Максимальный SNR при слабых сигналах | FrFT (D) | O(3N log N) | Новый модуль |

### GPU-эффективность: приоритет

```
Stretch Processing (B):  Самый GPU-эффективный
  → Dechirp: trivial kernel, Mem-bound
  → FFT: один batch hipfftExecC2C
  → Latency: минимальная (без transpose, без IFFT)

2D Range-Doppler (C):    Высокая параллельность
  → Два batch FFT + transpose
  → Весь pipeline на GPU без CPU roundtrip
  → Масштабируется с P (больше импульсов → лучше SNR)

Matched Filter (A):      Базовый, всегда работает
  → Три операции: FFT + element-wise mul + IFFT
  → GPU-эффективен, прост в реализации
```

---

## 7. Потенциальные новые модули (Ref03-совместимые)

### LFMRangeProcessor

```
Фасад: LFMRangeProcessor
├── DechirpOp        ← из HeterodyneDechirp
├── WindowOp         ← Hamming/Taylor/Chebyshev
└── RangeFFTOp       ← hipfftExecC2C batch
Output: range_profile[N/2+1], range_axis[N/2+1]
```

### RangeDopplerProcessor

```
Фасад: RangeDopplerProcessor
├── BatchDechirpOp   ← для P импульсов
├── WindowOp (2D)    ← range и doppler окна
├── RangeFFTOp       ← batch P×N
├── TransposeOp      ← HIP kernel (coalesced)
├── DopplerFFTOp     ← batch N×P
└── MagnitudeOp      ← |·|² + dB
Output: rd_map[P × N/2+1], ranges[N/2+1], velocities[P]
```

---

## 8. Источники

| Тема | Источник |
|------|----------|
| Stretch Processing теория | [IEEE RC18](https://www.ittc.ku.edu/~sdblunt/papers/IEEERC18-StretchComp.pdf) |
| Chirp compression | [Wikipedia](https://en.wikipedia.org/wiki/Chirp_compression) |
| FrFT для LFM детектирования | [EURASIP 2010](https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/876282) |
| GPU Range-Doppler (CUDA) | [NiclasEsser1, GitHub](https://github.com/NiclasEsser1/CUDARangeDopplerProcessing) |
| hipFFT API | [ROCm/hipFFT, GitHub](https://github.com/ROCm/hipFFT) |
| VkFFT (кросс-платформ.) | [DTolm/VkFFT, GitHub](https://github.com/DTolm/VkFFT) |
| FMCW Range-Doppler | [WirelessPi](https://wirelesspi.com/fmcw-radar-part-2-velocity-angle-and-radar-data-cube/) |
| Radar Pulse Compression | [MathWorks](https://www.mathworks.com/help/signal/ug/radar-pulse-compression.html) |
