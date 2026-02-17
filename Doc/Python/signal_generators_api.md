# Signal Generators — Python API

> **Модуль**: `gpuworklib.SignalGenerator`, `gpuworklib.FormSignalGenerator`, `gpuworklib.FormScriptGenerator`
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

#### generate()

```python
data = gen.generate()
```

| Возврат | Условие |
|---------|---------|
| `np.ndarray (points,) complex64` | 1 антенна |
| `np.ndarray (antennas, points) complex64` | N антенн |

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

#### generate()

```python
data = gen.generate()  # np.ndarray complex64
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
| `Python_test/example_form_signal.py` | Демо: 5 сценариев + 5 презентационных графиков |

```bash
python Python_test/test_form_signal.py
python Python_test/example_form_signal.py
```

Графики: `Results/Plots/FormSignal/`
