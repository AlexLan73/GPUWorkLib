# GPUWorkLib Python API — ROCm Classes

> **Версия**: 1.0
> **Дата**: 2026-02-24
> **Модуль**: `gpuworklib` (собрать с `-DBUILD_PYTHON=ON -DENABLE_ROCM=ON`)
> **Требует**: Linux + AMD GPU (ROCm/HIP)

---

## Сборка

```bash
cmake -B build/debian-radeon9070 -DBUILD_PYTHON=ON -DENABLE_ROCM=ON
cmake --build build/debian-radeon9070 -j4
```

`.so` файл: `build/debian-radeon9070/python/gpuworklib.cpython-313-x86_64-linux-gnu.so`

## Использование в Python

```python
import sys
sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np
```

**Запуск с GPU** (нужна группа `render`):
```bash
sg render -c "python3 my_script.py"
```

---

## ROCmGPUContext

Контекст ROCm GPU. Все ROCm-классы принимают `ROCmGPUContext` в конструктор.

```python
# Конструктор
ctx = gpuworklib.ROCmGPUContext(0)        # device_index=0

# Свойства (readonly)
ctx.device_name     # str  — имя GPU, напр. "gfx1201"
ctx.device_index    # int  — индекс устройства

# repr
str(ctx)            # '<ROCmGPUContext gfx1201>'
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `device_index` | `int` | `0` | Индекс AMD GPU |

---

## FirFilterROCm

GPU FIR фильтр (direct-form convolution, ROCm).

```python
# Конструктор
fir = gpuworklib.FirFilterROCm(ctx)

# Настройка
fir.set_coefficients([0.25, 0.5, 0.25])           # список float
# или из scipy:
from scipy.signal import firwin
fir.set_coefficients(firwin(64, 0.1).tolist())

# Загрузка из JSON-файла
fir.load_config('path/to/fir_config.json')

# Обработка (1D — один канал)
signal = np.ones(1024, dtype=np.complex64)
result = fir.process(signal)             # → np.complex64 (1024,)

# Обработка (2D — несколько каналов)
multi = np.ones((4, 1024), dtype=np.complex64)
result = fir.process(multi)              # → np.complex64 (4, 1024)

# Свойства (readonly)
fir.num_taps        # int  — число коэффициентов
fir.coefficients    # list[float]  — текущие коэффициенты
```

### Методы

| Метод | Аргументы | Возврат | Описание |
|-------|-----------|---------|----------|
| `set_coefficients(c)` | `list[float]` | `None` | Задать коэффициенты FIR |
| `load_config(path)` | `str` | `None` | Загрузить коэффициенты из JSON |
| `process(input)` | `np.complex64` `(N,)` или `(C,N)` | `np.complex64` | Применить фильтр на GPU |

### Пример: LP фильтр

```python
import numpy as np
from scipy.signal import firwin
import sys; sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib

ctx = gpuworklib.ROCmGPUContext(0)
fir = gpuworklib.FirFilterROCm(ctx)

coeffs = firwin(64, 0.1)
fir.set_coefficients(coeffs.tolist())

signal = np.random.randn(1024).astype(np.complex64)
filtered = fir.process(signal)
print(f"FIR: {fir.num_taps} taps, output shape: {filtered.shape}")
```

---

## IirFilterROCm

GPU IIR биквад-каскадный фильтр (DFII-Transposed, ROCm).

> ⚠️ GPU IIR эффективен ТОЛЬКО при большом числе каналов (≥ 8). Для одного канала быстрее CPU.

```python
# Конструктор
iir = gpuworklib.IirFilterROCm(ctx)

# Настройка через список секций (biquad)
iir.set_sections([
    {'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.56, 'a2': 0.64},
    {'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.49, 'a2': 0.56},
])

# или из scipy:
from scipy.signal import butter, zpk2sos
z, p, k = butter(4, 0.1, output='zpk')
sos = zpk2sos(z, p, k)
sections = [{'b0': s[0], 'b1': s[1], 'b2': s[2],
             'a1': s[4], 'a2': s[5]} for s in sos]
iir.set_sections(sections)

# Загрузка из JSON-файла
iir.load_config('path/to/iir_config.json')

# Обработка
result = iir.process(signal)             # np.complex64 (N,) или (C,N)

# Свойства (readonly)
iir.num_sections    # int  — число биквад-секций
iir.sections        # list[dict]  — текущие секции
```

### Методы

| Метод | Аргументы | Возврат | Описание |
|-------|-----------|---------|----------|
| `set_sections(sections)` | `list[dict]` | `None` | Задать биквад-секции |
| `load_config(path)` | `str` | `None` | Загрузить секции из JSON |
| `process(input)` | `np.complex64` `(N,)` или `(C,N)` | `np.complex64` | Применить фильтр на GPU |

### Формат секции (dict)

```python
# Каждая секция — передаточная функция 2-го порядка:
# H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
section = {'b0': ..., 'b1': ..., 'b2': ..., 'a1': ..., 'a2': ...}
```

---

## LchFarrowROCm

Дробно-задерживающий процессор (Lagrange 48×5, ROCm).
Реализует точную дробную задержку через матрицу Лагранжа 48×5.

```python
# Конструктор
proc = gpuworklib.LchFarrowROCm(ctx)

# Настройка
proc.set_sample_rate(1e6)                  # Гц
proc.set_delays([0.0, 1.5, 3.3, 5.25])    # мкс, одна задержка на антенну

# Добавить шум после задержки (опционально)
proc.set_noise(
    noise_amplitude=0.01,
    norm_val=0.7071067811865476,           # 1/sqrt(2)
    noise_seed=42
)

# Загрузить матрицу из JSON (опционально, по умолчанию встроенная)
proc.load_matrix('path/to/lagrange_48x5.json')

# Обработка
signal = np.ones(1024, dtype=np.complex64)
delayed = proc.process(signal)             # → (1024,)

# Несколько антенн (2D)
multi = np.ones((4, 1024), dtype=np.complex64)
delayed = proc.process(multi)             # → (4, 1024)

# Свойства (readonly)
proc.sample_rate    # float — частота дискретизации (Гц)
proc.delays         # list[float] — задержки (мкс)
```

### ⚠️ Известное ограничение

**НЕ использовать целочисленные задержки** при целой частоте дискретизации (напр. 1.0, 2.0, 3.0 мкс при 1 МГц).
Из-за float32 boundary issue (`frac ≈ 0.9999 → row=47` матрицы Лагранжа) возникает огромная ошибка.

```python
# ❌ Плохо: 3.0 мкс × 1 МГц = ровно 3 сэмпла
proc.set_delays([0.0, 1.5, 3.0, 5.25])

# ✅ Хорошо: дробные задержки
proc.set_delays([0.0, 1.5, 3.3, 5.25])
```

### Алгоритм

```
delay_samples = delay_us × 1e-6 × sample_rate
D   = floor(delay_samples)       # целая часть
mu  = frac(delay_samples)        # дробная часть
row = int(mu × 48) % 48          # строка матрицы Лагранжа
output[n] = Σ L[row][k] × input[n - D - 1 + k],  k = 0..4
```

---

## HeterodyneROCm

LFM гетеродинный процессор (ROCm).
Предоставляет операции Dechirp и Correct для stretch-processing.

```python
# Конструктор
het = gpuworklib.HeterodyneROCm(ctx)

# Настройка параметров ЛЧМ
het.set_params(
    f_start=0,          # Гц — начальная частота
    f_end=2e6,          # Гц — конечная частота (B = f_end - f_start)
    sample_rate=12e6,   # Гц — частота дискретизации
    num_samples=8000,   # N — число сэмплов на антенну
    num_antennas=5      # K — число антенн/каналов
)

# Dechirp: s_dc = rx × conj(ref)  — на GPU
rx  = signal.astype(np.complex64)   # (K×N,)
ref = reference.astype(np.complex64) # (N,) или (K×N,)
dc = het.dechirp(rx, ref)           # → np.complex64 (K×N,)

# Correct: частотная коррекция exp(j × phase_step × n)
f_beat = [123.0, 456.0, 78.9, 234.5, 567.8]   # Гц, по одной на антенну
out = het.correct(dc, f_beat)       # → np.complex64 (K×N,)

# Параметры (readonly dict)
p = het.params
# p['f_start'], p['f_end'], p['sample_rate'], p['num_samples'],
# p['num_antennas'], p['bandwidth'], p['duration'], p['chirp_rate']
```

### Методы

| Метод | Аргументы | Возврат | Описание |
|-------|-----------|---------|----------|
| `set_params(f_start, f_end, sample_rate, num_samples, num_antennas)` | float×5 | `None` | Задать параметры ЛЧМ |
| `dechirp(rx, ref)` | `np.complex64`, `np.complex64` | `np.complex64` | Деширпинг на GPU: rx × conj(ref) |
| `correct(dc, f_beat_hz)` | `np.complex64`, `list[float]` | `np.complex64` | Частотная коррекция |

### Вычисляемые параметры

```python
p = het.params
print(f"Bandwidth:   {p['bandwidth']:.0f} Hz")
print(f"Duration:    {p['duration']*1e6:.1f} μs")
print(f"Chirp rate:  {p['chirp_rate']:.3e} Hz/s")
```

---

## StatisticsProcessor

GPU статистика комплексных сигналов (ROCm, один проход Уэлфорда + radix sort).

```python
# Конструктор
proc = gpuworklib.StatisticsProcessor(ctx)

# Данные: numpy complex64 (beam_count × n_point)
data = np.random.randn(4096).astype(np.complex64)  # 4 луча × 1024 т.

# Полная статистика (mean + variance + std + mean_magnitude)
results = proc.compute_statistics(data, beam_count=4)
for r in results:
    print(f"Beam {r['beam_id']}: mean={r['mean_real']:.4f}+{r['mean_imag']:.4f}j, "
          f"std={r['std_dev']:.4f}, |mean|={r['mean_magnitude']:.4f}")

# Только среднее
means = proc.compute_mean(data, beam_count=4)
# [{'beam_id': 0, 'mean_real': ..., 'mean_imag': ...}, ...]

# Только медиана (GPU radix sort по модулю)
medians = proc.compute_median(data, beam_count=4)
# [{'beam_id': 0, 'median_magnitude': ...}, ...]
```

### Методы

| Метод | Аргументы | Возврат | Описание |
|-------|-----------|---------|----------|
| `compute_all(data, beam_count)` | `np.complex64`, `int` | `list[dict]` | **Статистика + медиана** (1 GPU-вызов) ⭐ |
| `compute_all_float(data, beam_count)` | `np.float32`, `int` | `list[dict]` | Статистика + медиана для float магнитуд |
| `compute_statistics(data, beam_count)` | `np.complex64`, `int` | `list[dict]` | Статистика: mean + variance + std |
| `compute_mean(data, beam_count)` | `np.complex64`, `int` | `list[dict]` | Только комплексное среднее |
| `compute_median(data, beam_count)` | `np.complex64`, `int` | `list[dict]` | Только медиана модулей (GPU radix sort) |
| `compute_statistics_float(data, beam_count)` | `np.float32`, `int` | `list[dict]` | Статистика для float магнитуд |
| `compute_median_float(data, beam_count)` | `np.float32`, `int` | `list[dict]` | Медиана для float магнитуд |

> 📄 Полное описание: [Doc/Python/statistics_api.md](statistics_api.md)

### Формат результата compute_statistics

```python
result = {
    'beam_id':        int,    # индекс луча
    'mean_real':      float,  # Re(среднее)
    'mean_imag':      float,  # Im(среднее)
    'variance':       float,  # дисперсия
    'std_dev':        float,  # среднеквадратичное отклонение
    'mean_magnitude': float,  # среднее модулей |x|
}
```

### Производительность

GPU speedup при 4 лучах × 131072 точек: **18.7×** быстрее CPU (numpy).

---

## Совместное использование

```python
import numpy as np
import sys; sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib

# Один контекст на все объекты
ctx = gpuworklib.ROCmGPUContext(0)
print(f"GPU: {ctx.device_name}")

# Пайплайн: фильтр → статистика
fir  = gpuworklib.FirFilterROCm(ctx)
stat = gpuworklib.StatisticsProcessor(ctx)

from scipy.signal import firwin
fir.set_coefficients(firwin(64, 0.1).tolist())

# Мультиканальный сигнал: 4 луча × 1024 точки
beams = np.random.randn(4, 1024).astype(np.complex64)

filtered = fir.process(beams)            # (4, 1024)
results  = stat.compute_statistics(filtered.flatten(), beam_count=4)

for r in results:
    print(f"  beam {r['beam_id']}: std={r['std_dev']:.4f}")
```

---

## Таблица всех ROCm-классов

| Класс | Конструктор | Ключевые методы | Данные |
|-------|-------------|-----------------|--------|
| `ROCmGPUContext` | `(device_index=0)` | `.device_name`, `.device_index` | — |
| `FirFilterROCm` | `(ctx)` | `set_coefficients()`, `process()` | `complex64 (N,)/(C,N)` |
| `IirFilterROCm` | `(ctx)` | `set_sections()`, `process()` | `complex64 (N,)/(C,N)` |
| `LchFarrowROCm` | `(ctx)` | `set_delays()`, `set_sample_rate()`, `set_noise()`, `process()` | `complex64 (N,)/(A,N)` |
| `HeterodyneROCm` | `(ctx)` | `set_params()`, `dechirp()`, `correct()` | `complex64 (K×N,)` |
| `StatisticsProcessor` | `(ctx)` | `compute_all()` ⭐, `compute_statistics()`, `compute_mean()`, `compute_median()`, `compute_all_float()` | `complex64`/`float32 (B×N,)` |

---

*Создано: 2026-02-24, Кодо (AI Assistant)*
