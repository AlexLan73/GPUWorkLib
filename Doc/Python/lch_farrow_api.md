# LchFarrow — Python API

> **Модуль**: `gpuworklib.LchFarrow`
> **C++ namespace**: `lch_farrow`
> **Обновлено**: 2026-02-18

## Обзор

Standalone GPU-процессор дробной задержки на основе Lagrange 48x5 интерполяции.

Независим от генераторов сигналов — работает с любым входным сигналом.

**Алгоритм:**
- `delay_samples = delay_us * 1e-6 * sample_rate`
- `D = floor(delay_samples)` — целый сдвиг
- `mu = delay_samples - D` — дробная часть [0, 1)
- `row = int(mu * 48) % 48` — строка матрицы Lagrange
- `output[n] = sum(L[row][k] * input[n - D - 1 + k], k=0..4)`

**Матрица:** 48 строк (bins дробной задержки) × 5 столбцов (коэффициенты интерполяции).

---

## Быстрый старт

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.GPUContext(0)

# Генерируем CW сигнал
t = np.arange(4096) / 1e6
signal = np.exp(1j * 2 * np.pi * 50000 * t).astype(np.complex64)

# Применяем дробную задержку
proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([2.7])  # 2.7 мкс = 2.7 сэмплов
delayed = proc.process(signal)

print(f"Input shape: {signal.shape}")   # (4096,)
print(f"Output shape: {delayed.shape}") # (4096,)
```

---

## Конструктор

```python
proc = gpuworklib.LchFarrow(ctx)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `GPUContext` | GPU-контекст (обязательный) |

---

## Методы

### set_sample_rate(sample_rate)

```python
proc.set_sample_rate(1e6)  # 1 МГц
```

| Аргумент | Тип | Описание |
|----------|-----|----------|
| `sample_rate` | `float` | Частота дискретизации (Гц). По умолчанию 1 МГц |

### set_delays(delay_us)

```python
proc.set_delays([0.0, 2.7, 5.0, 7.5])  # 4 антенны
```

| Аргумент | Тип | Описание |
|----------|-----|----------|
| `delay_us` | `list[float]` | Задержки per-antenna в **микросекундах** |

### set_noise(noise_amplitude, norm_val=0.7071, noise_seed=0)

```python
proc.set_noise(0.1, noise_seed=42)  # шум ПОСЛЕ задержки
```

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `noise_amplitude` | — | Амплитуда шума (0 = без шума) |
| `norm_val` | `1/sqrt(2)` | Нормировка |
| `noise_seed` | `0` | Seed для Philox PRNG (0 = auto) |

### load_matrix(json_path)

```python
proc.load_matrix("path/to/lagrange_matrix_48x5.json")
```

Опционально. Встроенная матрица 48×5 используется по умолчанию.

### process(input)

```python
delayed = proc.process(signal)
```

| Аргумент | Тип | Описание |
|----------|-----|----------|
| `input` | `np.ndarray complex64` | `(points,)` или `(antennas, points)` |

| Возврат | Условие |
|---------|---------|
| `np.ndarray (points,) complex64` | 1 антенна |
| `np.ndarray (antennas, points) complex64` | N антенн |

---

## Свойства (read-only)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `sample_rate` | `float` | Частота дискретизации |
| `delays` | `list[float]` | Текущие задержки (мкс) |

---

## Примеры

### Multi-channel delay

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.GPUContext(0)

# 8-канальный сигнал
fs = 1e6
t = np.arange(4096) / fs
single = np.exp(1j * 2 * np.pi * 50000 * t).astype(np.complex64)
signal = np.tile(single, (8, 1))  # (8, 4096)

proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(fs)
proc.set_delays([i * 1.5 for i in range(8)])  # 0..10.5 мкс

delayed = proc.process(signal)
print(f"Shape: {delayed.shape}")  # (8, 4096)
```

### LFM + LchFarrow vs Analytical

```python
# Стандартный LFM + Farrow delay
sig = gpuworklib.SignalGenerator(ctx)
lfm = sig.generate_lfm(f_start=1e6, f_end=2e6, fs=12e6, length=4096)

proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(12e6)
proc.set_delays([0.5])
farrow = proc.process(lfm)

# Аналитический delay (идеальный, без интерполяции)
gen = gpuworklib.LfmAnalyticalDelay(ctx, f_start=1e6, f_end=2e6)
gen.set_sampling(fs=12e6, length=4096)
gen.set_delays([0.5])
analytical = gen.generate_gpu()

err = np.max(np.abs(farrow.ravel() - analytical.ravel()))
print(f"Farrow vs Analytical: {err:.2e}")
```

---

## Тесты

| Файл | Описание |
|------|----------|
| `Python_test/test_lch_farrow.py` | 5 тестов: zero delay, integer, fractional, multi-antenna, vs analytical |

```bash
python Python_test/test_lch_farrow.py
```
