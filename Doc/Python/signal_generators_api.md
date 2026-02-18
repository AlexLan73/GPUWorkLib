# Signal Generators — Python API

> **Модуль**: `gpuworklib.SignalGenerator`, `gpuworklib.FormSignalGenerator`, `gpuworklib.FormScriptGenerator`, `gpuworklib.DelayedFormSignalGenerator`
> **C++ namespace**: `signal_gen`
> **Обновлено**: 2026-02-17

## Обзор

GPU-ускоренные генераторы сигналов для антенных систем и ЦОС.

| Класс | Назначение |
|-------|-----------|
| `SignalGenerator` | CW, LFM, Noise — базовые одноканальные/многолучевые сигналы |
| `ScriptGenerator` | DSL → OpenCL kernel (text скрипт → GPU) |
| `FormSignalGenerator` | Мультиканальный генератор по формуле getX (Philox+Box-Muller) |
| `FormScriptGenerator` | DSL + on-disk kernel cache для FormSignal |
| `DelayedFormSignalGenerator` | Дробная задержка Farrow 48×5 (Lagrange interpolation) |
| `LfmAnalyticalDelay` | ЛЧМ с аналитической per-antenna задержкой (идеальный эталон) |
| `GPUBuffer` | Handle GPU-буфера (при `generate(output='gpu')`) |

---

## FormSignalGenerator

Мультиканальный генератор комплексных сигналов по формуле:

```
X = a * norm * exp(j * (2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
  + an * norm * (randn + j*randn)
X = 0  при t < 0 или t > ti - dt
```

**Поддержка:**
- Мультиканальная генерация (N антенн параллельно на GPU)
- Per-channel задержка: FIXED / LINEAR (tau_step) / RANDOM (tau_min..tau_max)
- Шум: Philox-2x32-10 + Box-Muller (встроен в kernel)
- Chirp: fdev != 0 дает ЛЧМ-модуляцию

### Быстрый старт

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormSignalGenerator(ctx)

# CW сигнал 1 МГц, 8 каналов
gen.set_params(
    fs=12e6,        # частота дискретизации
    f0=1e6,         # несущая частота
    antennas=8,     # количество каналов
    points=4096,    # отсчётов на канал
    amplitude=1.0,
    noise_amplitude=0.1,
    tau_step=1e-5   # 10 мкс шаг задержки между каналами
)

data = gen.generate()
print(f"Shape: {data.shape}")   # (8, 4096) complex64
print(f"Max: {np.abs(data).max():.4f}")
```

### Конструктор

```python
gen = gpuworklib.FormSignalGenerator(ctx)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `GPUContext` | GPU-контекст (обязательный) |

### Методы

#### set_params()

```python
gen.set_params(
    fs=12e6,             # float — частота дискретизации, Гц
    antennas=1,          # int — количество каналов
    points=4096,         # int — отсчётов на канал
    f0=0.0,              # float — несущая частота, Гц
    amplitude=1.0,       # float — амплитуда сигнала
    noise_amplitude=0.0, # float — амплитуда шума
    phase=0.0,           # float — начальная фаза, рад
    fdev=0.0,            # float — девиация частоты (chirp), Гц
    norm=0.7071,         # float — коэффициент нормировки (1/sqrt(2))
    tau_base=0.0,        # float — базовая задержка, с
    tau_step=0.0,        # float — шаг задержки между каналами, с
    tau_min=0.0,         # float — мин задержка (random mode), с
    tau_max=0.0,         # float — макс задержка (random mode), с
    tau_seed=12345,      # int — seed для random tau
    noise_seed=0         # int — seed для Philox PRNG
)
```

**Режимы задержки:**
- `tau_step > 0`: LINEAR — `tau[ch] = tau_base + ch * tau_step`
- `tau_min != tau_max`: RANDOM — `tau[ch] = uniform(tau_min, tau_max)` (Philox)
- Иначе: FIXED — `tau[ch] = tau_base`

#### set_params_from_string()

```python
gen.set_params_from_string("f0=1e6,a=1.0,an=0.1,antennas=8,points=4096,fs=12e6")
```

| Ключ | Параметр | Ключ | Параметр |
|------|----------|------|----------|
| `fs` | sample rate | `f0` | frequency |
| `a` | amplitude | `an` | noise_amplitude |
| `phi` | phase | `fdev` | freq deviation |
| `norm` | normalization | `tau` | tau_base |
| `tau_step` | delay step | `tau_min` | min delay |
| `tau_max` | max delay | `tau_seed` | tau PRNG seed |
| `noise_seed` | noise seed | `antennas` | channels |
| `points` | samples | | |

#### generate(output="cpu")

```python
data = gen.generate()           # default: numpy
buf = gen.generate(output='gpu')  # GPUBuffer
data = buf.read()               # explicit readback
```

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `output` | `"cpu"` | `"cpu"` — readback → numpy, `"gpu"` — возвращает `GPUBuffer` |

| Возврат (output="cpu") | Условие |
|------------------------|---------|
| `np.ndarray (points,) complex64` | 1 антенна |
| `np.ndarray (antennas, points) complex64` | N антенн |

| Возврат (output="gpu") | Методы `GPUBuffer` |
|------------------------|--------------------|
| `GPUBuffer` | `.read()` → numpy, `.shape`, `.antenna_count`, `.n_point`, `.release()` |

### Свойства (read-only)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `antennas` | `int` | Количество каналов |
| `points` | `int` | Отсчётов на канал |
| `fs` | `float` | Частота дискретизации |

---

## FormScriptGenerator

DSL-обёртка над FormSignalGenerator с:
- Генерация OpenCL kernel source с `#define` параметрами (оптимизация: 1 аргумент вместо 18)
- On-disk кэш скомпилированных кернелов (.cl + binary)
- Человекочитаемый DSL скрипт

### Быстрый старт

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormScriptGenerator(ctx)

# Вариант 1: из параметров
gen.set_params(fs=10e6, f0=500000, antennas=4, points=8192,
               noise_amplitude=0.05, tau_step=0.0001)
gen.compile()
data = gen.generate()

# Сохранить скомпилированный kernel
gen.save_kernel("my_cw_500k", "CW 500kHz 4ch")

# Вариант 2: загрузка из кэша (быстро!)
gen2 = gpuworklib.FormScriptGenerator(ctx)
gen2.set_params(fs=10e6, f0=500000, antennas=4, points=8192,
                noise_amplitude=0.05, tau_step=0.0001)
gen2.load_kernel("my_cw_500k")    # binary → instant
data2 = gen2.generate()
```

### Конструктор

```python
gen = gpuworklib.FormScriptGenerator(ctx)
```

### Методы

#### set_params() / set_params_from_string()

Те же параметры, что у `FormSignalGenerator`.

#### compile()

```python
gen.compile()
```
Генерирует OpenCL kernel source с `#define` параметрами и компилирует.

#### generate(output="cpu")

```python
data = gen.generate()            # np.ndarray complex64 (default)
buf = gen.generate(output='gpu') # GPUBuffer — без readback
data = buf.read()                # явный readback при необходимости
```
Требует предварительного `compile()` или `load_kernel()`.

#### generate_script()

```python
script = gen.generate_script()  # str — DSL текст
print(script)
```

Пример вывода:
```
[Params]
fs       = 10000000.000000
f0       = 500000.000000
amplitude = 1.000000
...

[Defs]
delay_mode = LINEAR
tau_base   = 0.000000
tau_step   = 0.000100

[Signal]
formula = getX(a, norm, f0, fdev, ti, t, phi) + noise(an, norm, seed)
window  = rectangular: X=0 if t<0 or t>ti-dt
```

#### generate_kernel_source()

```python
source = gen.generate_kernel_source()  # str — OpenCL C
```
Полный исходник кернела с `#define` константами и встроенным PRNG.

#### save_kernel()

```python
gen.save_kernel("name", "optional comment")
```
Сохраняет на диск:
- `kernels/name.cl` — OpenCL source
- `kernels/bin/name_opencl.bin` — скомпилированный binary
- `kernels/manifest.json` — метаданные

При коллизии: старые файлы переименовываются в `name_00.cl`, `name_01.cl`, ...

#### load_kernel()

```python
gen.load_kernel("name")
```
Загружает с приоритетом: binary (мгновенно) → source (перекомпиляция).

#### list_kernels()

```python
names = gen.list_kernels()  # list[str]
print(names)  # ['my_cw_500k', 'chirp_20k']
```

### Свойства (read-only)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `antennas` | `int` | Количество каналов |
| `points` | `int` | Отсчётов на канал |
| `fs` | `float` | Частота дискретизации |
| `is_ready` | `bool` | True если kernel скомпилирован |
| `kernel_source` | `str` | Текущий OpenCL source |

### Статические методы

```python
gpuworklib.FormScriptGenerator.get_kernels_dir()      # путь к .cl файлам
gpuworklib.FormScriptGenerator.get_kernels_bin_dir()   # путь к бинарникам
```

---

## DelayedFormSignalGenerator

Мультиканальный генератор с **дробной задержкой** (Farrow 48×5 Lagrange interpolation).

**Алгоритм:**
1. Генерация чистого сигнала (getX, без шума) через FormSignalGenerator
2. Применение дробной задержки: целый сдвиг D + 5-точечная Lagrange интерполяция
3. Добавление шума (Philox + Box-Muller) — **после** задержки

**Задержка:** в **микросекундах** (float) на каждую антенну.

**Матрица:** 48 строк (дробные задержки 0/48..47/48) × 5 столбцов (коэффициенты интерполяции).

### Быстрый старт

```python
import gpuworklib
import numpy as np

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.DelayedFormSignalGenerator(ctx)

# CW 50 kHz, 8 каналов, нарастающая задержка
gen.set_params(
    fs=1e6,          # 1 МГц
    f0=50000.0,      # 50 кГц
    antennas=8,
    points=4096,
    amplitude=1.0,
    noise_amplitude=0.1  # шум добавляется ПОСЛЕ задержки
)

# Задержки: 0, 1.5, 3.0, ..., 10.5 мкс
gen.set_delays([i * 1.5 for i in range(8)])

data = gen.generate()
print(f"Shape: {data.shape}")  # (8, 4096) complex64
```

### Конструктор

```python
gen = gpuworklib.DelayedFormSignalGenerator(ctx)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `GPUContext` | GPU-контекст (обязательный) |

### Методы

#### set_params()

```python
gen.set_params(
    fs=12e6,             # float — частота дискретизации, Гц
    antennas=1,          # int — количество каналов
    points=4096,         # int — отсчётов на канал
    f0=0.0,              # float — несущая частота, Гц
    amplitude=1.0,       # float — амплитуда сигнала
    noise_amplitude=0.0, # float — шум (добавляется ПОСЛЕ задержки)
    phase=0.0,           # float — начальная фаза, рад
    fdev=0.0,            # float — девиация частоты (chirp), Гц
    norm=0.7071,         # float — нормировка (1/sqrt(2))
    noise_seed=0         # int — seed для Philox PRNG (0 = auto)
)
```

> **Важно:** `noise_amplitude` добавляется **после** задержки (в отличие от FormSignalGenerator, где шум встроен в getX kernel).

#### set_delays(delay_us)

```python
gen.set_delays([0.0, 1.5, 3.0, 4.5])  # 4 антенны
```

| Аргумент | Тип | Описание |
|----------|-----|----------|
| `delay_us` | `list[float]` | Задержки per-antenna в **микросекундах**. `len(delay_us) == antennas` |

**Как работает:**
- `delay_us → delay_samples = delay_us × 1e-6 × fs`
- `D = floor(delay_samples)` — целый сдвиг
- `μ = delay_samples - D` — дробная часть [0, 1)
- `row = int(μ × 48) % 48` — строка матрицы Lagrange
- Для каждого отсчёта: 5-точечная свёртка с коэффициентами из строки row

#### load_matrix(json_path)

```python
gen.load_matrix("path/to/lagrange_matrix_48x5.json")
```

Опционально. Встроенная матрица 48×5 используется по умолчанию.

**Формат JSON:**
```json
{
  "data": [[0.0, 1.0, 0.0, 0.0, 0.0], ...]
}
```

#### generate(output="cpu")

```python
data = gen.generate()             # numpy (default)
buf = gen.generate(output='gpu')  # GPUBuffer
```

| Возврат (output="cpu") | Условие |
|------------------------|---------|
| `np.ndarray (points,) complex64` | 1 антенна |
| `np.ndarray (antennas, points) complex64` | N антенн |

### Свойства (read-only)

| Свойство | Тип | Описание |
|----------|-----|----------|
| `antennas` | `int` | Количество каналов |
| `points` | `int` | Отсчётов на канал |
| `fs` | `float` | Частота дискретизации |
| `delays` | `list[float]` | Текущие задержки (мкс) |

### Пример: Multi-channel с визуализацией

```python
import numpy as np
import matplotlib.pyplot as plt
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.DelayedFormSignalGenerator(ctx)

gen.set_params(fs=1e6, f0=50000, antennas=8, points=4096, amplitude=1.0)
gen.set_delays([i * 2.0 for i in range(8)])

data = gen.generate()

# Waterfall: видно нарастающую задержку
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(np.abs(data[:, :300]), aspect='auto', cmap='inferno')
ax.set_xlabel('Sample')
ax.set_ylabel('Antenna')
ax.set_title('Fractional Delay Waterfall')
plt.savefig('delay_waterfall.png')
```

---

## LfmAnalyticalDelay

ЛЧМ-генератор с **аналитической** (идеальной) per-antenna задержкой.

**Формула:**
- `t_local = t - tau` (tau — задержка в секундах)
- `phase = pi * chirp_rate * t_local^2 + 2*pi * f_start * t_local`
- `output = amplitude * exp(j * phase)` если `t >= tau`, иначе 0

Нет интерполяционных артефактов — идеальный эталон для проверки LchFarrow.

### Быстрый старт

```python
import gpuworklib
import numpy as np

ctx = gpuworklib.GPUContext(0)

gen = gpuworklib.LfmAnalyticalDelay(ctx, f_start=1e6, f_end=2e6)
gen.set_sampling(fs=12e6, length=4096)
gen.set_delays([0.0, 0.1, 0.2, 0.5])  # 4 антенны, мкс
data = gen.generate_gpu()  # (4, 4096) complex64
```

### Конструктор

```python
gen = gpuworklib.LfmAnalyticalDelay(ctx, f_start, f_end,
                                     amplitude=1.0, complex_iq=True)
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `ctx` | — | GPUContext (обязательный) |
| `f_start` | — | Начальная частота (Гц) |
| `f_end` | — | Конечная частота (Гц) |
| `amplitude` | `1.0` | Амплитуда |
| `complex_iq` | `True` | True = IQ, False = real-only |

### Методы

#### set_sampling(fs, length)
```python
gen.set_sampling(fs=12e6, length=4096)
```

#### set_delays(delay_us)
```python
gen.set_delays([0.0, 0.1, 0.2])  # мкс
```

#### generate_gpu() / generate_cpu()
```python
gpu_data = gen.generate_gpu()  # GPU (float32)
cpu_data = gen.generate_cpu()  # CPU (double precision reference)
```

### Свойства

| Свойство | Тип | Описание |
|----------|-----|----------|
| `antennas` | `int` | Количество антенн (= len(delays)) |
| `delays` | `list[float]` | Задержки (мкс) |

### Тесты

| Файл | Описание |
|------|----------|
| `Python_test/test_lfm_analytical_delay.py` | 5 тестов: zero delay, boundary, GPU vs CPU, multi-antenna, vs NumPy |

---

## SignalGenerator

Базовые одноканальные/многолучевые генераторы (CW, LFM, Noise).

### Быстрый старт

```python
ctx = gpuworklib.GPUContext(0)
sig = gpuworklib.SignalGenerator(ctx)

# CW сигнал
cw = sig.generate_cw(freq=1000, fs=48000, length=4096)

# LFM (chirp)
lfm = sig.generate_lfm(freq=1000, fs=48000, length=4096, bandwidth=5000)

# Белый шум
noise = sig.generate_noise(fs=48000, length=4096)

# Многолучевой CW
multi = sig.generate_cw(freq=100, fs=4000, length=4096,
                         beam_count=8, freq_step=100)
```

---

## ScriptGenerator

DSL → OpenCL для произвольных сигналов (без PRNG/noise).

```python
ctx = gpuworklib.GPUContext(0)
sg = gpuworklib.ScriptGenerator(ctx)

script = """
[Params]
f0 = 1000.0
fs = 48000.0
amplitude = 1.0

[Signal]
X = amplitude * sin(2*PI*f0*t)
"""

sg.compile(script)
data = sg.generate(length=4096)
```

---

## Примеры

### Пример 1: Chirp + FFT + поиск пиков

```python
ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormSignalGenerator(ctx)
fft = gpuworklib.FFTProcessor(ctx)

gen.set_params(fs=100000, f0=5000, fdev=20000,
               antennas=1, points=8192, noise_amplitude=0.1, noise_seed=42)

signal = gen.generate()
spectrum = fft.process_complex(signal, sample_rate=100000)

mag = np.abs(spectrum.ravel())
freq = np.fft.fftfreq(len(mag), d=1/100000)
peak = freq[np.argmax(mag[:len(mag)//2])]
print(f"Peak: {peak:.0f} Hz")
```

### Пример 2: Kernel cache workflow

```python
# Первый запуск: компиляция + сохранение
gen = gpuworklib.FormScriptGenerator(ctx)
gen.set_params(fs=10e6, f0=1e6, antennas=16, points=8192)
gen.compile()                          # ~50 мс (OpenCL compile)
gen.save_kernel("radar_16ch", "16-канальный РЛС 1 МГц")

# Повторные запуски: загрузка binary (~1 мс)
gen2 = gpuworklib.FormScriptGenerator(ctx)
gen2.set_params(fs=10e6, f0=1e6, antennas=16, points=8192)
gen2.load_kernel("radar_16ch")         # мгновенно
data = gen2.generate()                 # генерация
```

### Пример 3: GPU vs NumPy reference

```python
gen = gpuworklib.FormSignalGenerator(ctx)
gen.set_params(fs=12e6, f0=1e6, antennas=1, points=4096,
               amplitude=1.0, phase=0.3, fdev=2000)

gpu = gen.generate().ravel()

# NumPy reference
dt = 1 / 12e6
t = np.arange(4096) * dt
ti = 4096 * dt
t_c = t - ti / 2
norm = 1 / np.sqrt(2)
ph = 2*np.pi*1e6*t + np.pi*2000/ti*(t_c**2) + 0.3
ref = 1.0 * norm * np.exp(1j * ph)

err = np.max(np.abs(gpu - ref.astype(np.complex64)))
print(f"Error: {err:.2e}")  # < 1e-6
```

## Тесты

| Файл | Описание |
|------|----------|
| `Python_test/test_form_signal.py` | FormSignalGenerator: 7 тестов + 6 графиков |
| `Python_test/test_delayed_form_signal.py` | DelayedFormSignalGenerator: 5 тестов + 4 графика |
| `Python_test/test_lfm_analytical_delay.py` | LfmAnalyticalDelay: 5 тестов (zero, boundary, GPU vs CPU/NumPy, multi-antenna) |
| `Python_test/test_lch_farrow.py` | LchFarrow: 5 тестов (zero, integer, fractional, multi-antenna, vs analytical) |
| `Python_test/example_form_signal.py` | Демо: 5 сценариев + 5 презентационных графиков |

```bash
python Python_test/test_form_signal.py
python Python_test/test_delayed_form_signal.py
python Python_test/example_form_signal.py
```

Графики:
- `Results/Plots/FormSignal/` — FormSignalGenerator
- `Results/Plots/DelayedFormSignal/` — DelayedFormSignalGenerator (Farrow 48×5)
