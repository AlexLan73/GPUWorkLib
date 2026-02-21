# Signal Generators — Полная документация

> Генерация сигналов на GPU (CW, LFM, Noise, Script, FormSignal, DelayedFormSignal)

**Namespace**: `signal_gen`
**Каталог**: `modules/signal_generators/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL

---

## Содержание

1. [Обзор — какой генератор выбрать](#1-обзор)
2. [FormSignalGenerator — формула getX](#2-formsignalgenerator)
3. [Параметры FormParams — полная таблица](#3-параметры-formparams)
4. [Режимы задержки per-channel](#4-режимы-задержки)
5. [FormScriptGenerator — DSL и кэш кернелов](#5-formscriptgenerator)
6. [DelayedFormSignalGenerator — дробная задержка Farrow](#6-delayedformsignalgenerator)
7. [SignalGenerator — CW, LFM, Noise](#7-signalgenerator)
8. [ScriptGenerator — произвольные формулы](#8-scriptgenerator)
9. [Примеры вызова (C++ и Python)](#9-примеры-вызова)
10. [Тесты](#10-тесты)
11. [Ссылки](#11-ссылки)

---

## 1. Обзор

### Какой генератор выбрать?

| Генератор | Когда использовать | Вход | Выход |
|-----------|-------------------|------|-------|
| **FormSignalGenerator** | CW/Chirp + шум, multi-channel, per-channel задержка | FormParams | numpy / GPUBuffer |
| **FormScriptGenerator** | То же + on-disk кэш (save/load kernel) | FormParams + compile | numpy / GPUBuffer |
| **DelayedFormSignalGenerator** | FormSignal + **дробная задержка** (Farrow 48×5) | FormParams + delays[] | numpy / GPUBuffer |
| **SignalGenerator** | Простые CW, LFM, Noise (одноканальные или multi-beam) | freq, fs, length | numpy |
| **ScriptGenerator** | Произвольная формула (DSL текст) | script string | numpy / cl_mem |

### Краткая шпаргалка

```
Нужен CW 1 МГц, 8 каналов, шум?        → FormSignalGenerator
Нужно сохранить kernel для быстрого старта? → FormScriptGenerator
Нужна дробная задержка (3.24 сэмпла)?   → DelayedFormSignalGenerator
Нужен простой chirp без шума?           → SignalGenerator.generate_lfm()
Нужна своя формула (sin, cos, условие)? → ScriptGenerator
```

---

## 2. FormSignalGenerator — формула getX

### Математика

$$
X(t) = a \cdot \text{norm} \cdot e^{j\phi(t)} + a_n \cdot \text{norm} \cdot (n_r + j n_i)
$$

где
$$
\phi(t) = 2\pi f_0 t + \frac{\pi \cdot f_{dev}}{t_i} \left(t - \frac{t_i}{2}\right)^2 + \phi_0
$$

- **Окно**: \(X = 0\) при \(t < 0\) или \(t > t_i - dt\)
- **Шум**: \(n_r, n_i\) — Gaussian (Philox + Box-Muller), встроен в kernel
- **Chirp**: \(f_{dev} \neq 0\) даёт ЛЧМ-модуляцию

### Когда использовать

- Мультиканальная генерация (N антенн параллельно)
- Per-channel задержка (FIXED / LINEAR / RANDOM)
- Шум в одном kernel (Philox + Box-Muller)
- Нормировка: по умолчанию `norm = 1/√2` для complex IQ

---

## 3. Параметры FormParams — полная таблица

### Все параметры

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| **fs** | float | 12e6 | Частота дискретизации (Гц) |
| **antennas** | int | 1 | Количество каналов |
| **points** | int | 4096 | Отсчётов на канал |
| **f0** | float | 0.0 | Несущая частота (Гц) |
| **amplitude** | float | 1.0 | Амплитуда сигнала (a) |
| **phase** | float | 0.0 | Начальная фаза (рад) |
| **fdev** | float | 0.0 | Девиация частоты (chirp), 0 = CW |
| **norm** | float | 0.7071 | Нормировка (1/√2) |
| **noise_amplitude** | float | 0.0 | Амплитуда шума (an), 0 = без шума |
| **noise_seed** | int | 0 | Seed для Philox PRNG (0 = random) |
| **tau_base** | float | 0.0 | Базовая задержка (с) |
| **tau_step** | float | 0.0 | Шаг задержки между каналами (с) |
| **tau_min** | float | 0.0 | Мин задержка для RANDOM (с) |
| **tau_max** | float | 0.0 | Макс задержка для RANDOM (с) |
| **tau_seed** | int | 12345 | Seed для случайной задержки |

### Строковый формат (set_params_from_string)

```python
# Ключи: fs, f0, a, an, phi, fdev, norm, tau, tau_step, tau_min, tau_max,
#        tau_seed, noise_seed, antennas, points, freq_min, freq_max
gen.set_params_from_string("f0=1e6,a=1.0,an=0.1,antennas=8,points=4096,fs=12e6")
```

---

## 4. Режимы задержки per-channel

| Режим | Условие | Формула tau для канала ID |
|-------|---------|---------------------------|
| **FIXED** | tau_step=0, tau_min=tau_max | `tau = tau_base` |
| **LINEAR** | tau_step > 0 | `tau = tau_base + ID × tau_step` |
| **RANDOM** | tau_min ≠ tau_max | `tau = tau_min + uniform[0,1) × (tau_max - tau_min)` (Philox) |

### Пример: LINEAR (нарастающая задержка)

```python
gen.set_params(
    fs=12e6, antennas=8, points=4096,
    tau_base=0.0,
    tau_step=1e-5   # 10 мкс шаг: канал 0→0, 1→10мкс, 2→20мкс, ...
)
```

### Пример: RANDOM (случайная задержка)

```python
gen.set_params(
    fs=12e6, antennas=8, points=4096,
    tau_min=0.0,
    tau_max=0.001,  # 0..1 мс
    tau_seed=12345  # воспроизводимость
)
```

### Пример: FIXED (одинаковая задержка)

```python
gen.set_params(
    fs=12e6, antennas=8, points=4096,
    tau_base=0.0001  # 100 мкс для всех
)
```

---

## 5. FormScriptGenerator — DSL и кэш кернелов

### Зачем FormScriptGenerator?

- **Тот же getX**, что FormSignalGenerator
- **On-disk кэш**: сохранить скомпилированный kernel → при следующем запуске загрузить binary (~1 мс вместо ~50 мс компиляции)
- **Человекочитаемый DSL**: `generate_script()` выводит текст с параметрами

### Workflow: первый запуск

```python
gen = gpuworklib.FormScriptGenerator(ctx)
gen.set_params(fs=10e6, f0=1e6, antennas=16, points=8192, noise_amplitude=0.05)
gen.compile()                          # ~50 мс (OpenCL compile)
gen.save_kernel("radar_16ch", "16-канальный РЛС 1 МГц")
data = gen.generate()
```

### Workflow: повторные запуски (быстро!)

```python
gen2 = gpuworklib.FormScriptGenerator(ctx)
gen2.set_params(fs=10e6, f0=1e6, antennas=16, points=8192)
gen2.load_kernel("radar_16ch")         # binary → ~1 мс
data = gen2.generate()
```

### Сохранение и загрузка

| Метод | Действие |
|-------|----------|
| `save_kernel("name", "comment")` | Сохраняет `name.cl`, `name_opencl.bin`, manifest.json |
| `load_kernel("name")` | Загружает binary (если есть) или source (перекомпиляция) |
| `list_kernels()` | Список имён сохранённых кернелов |

При коллизии имён: старые файлы → `name_00.cl`, `name_01.cl`, ...

---

## 6. DelayedFormSignalGenerator — дробная задержка Farrow

### Когда использовать

Нужна **дробная задержка** (например, 3.24 сэмпла) — Farrow 48×5 (Lagrange interpolation).

**Алгоритм:**
1. Генерация чистого сигнала (getX, без шума)
2. Дробная задержка: целый сдвиг D + 5-точечная Lagrange
3. Добавление шума **после** задержки

### Задержки — в микросекундах!

```python
gen.set_delays([0.0, 1.5, 3.0, 4.5])  # 4 антенны: 0, 1.5, 3, 4.5 мкс
```

### Пример: CW с нарастающей задержкой

```python
gen = gpuworklib.DelayedFormSignalGenerator(ctx)
gen.set_params(fs=1e6, f0=50000, antennas=8, points=4096,
               amplitude=1.0, noise_amplitude=0.1)

# Задержки: 0, 1.5, 3.0, ..., 10.5 мкс
gen.set_delays([i * 1.5 for i in range(8)])

data = gen.generate()  # (8, 4096) complex64
```

См. [Doc/Modules/lch_farrow/Full.md](../lch_farrow/Full.md) — математика Farrow.

---

## 7. SignalGenerator — CW, LFM, Noise

### Быстрый старт

```python
sig = gpuworklib.SignalGenerator(ctx)

# CW
cw = sig.generate_cw(freq=1000, fs=48000, length=4096)

# LFM (chirp)
lfm = sig.generate_lfm(freq=1000, fs=48000, length=4096, bandwidth=5000)

# Шум
noise = sig.generate_noise(fs=48000, length=4096)

# Multi-beam CW
multi = sig.generate_cw(freq=100, fs=4000, length=4096,
                        beam_count=8, freq_step=100)
```

---

## 8. ScriptGenerator — произвольные формулы

DSL → OpenCL kernel. Без PRNG (нет шума).

### Формат скрипта

```
[Params]
ANTENNAS = 8
POINTS = 4096

[Defs]
float freq = 0.05f + (float)ID * 0.01f
float phase = (float)ID * 0.785f

[Signal]
float angle = 2.0f * M_PI_F * freq * (float)T + phase
res_re = cos(angle)
res_im = sin(angle)
```

### Встроенные переменные

| Переменная | Описание |
|------------|----------|
| `ID` | Индекс антенны (0..ANTENNAS-1) |
| `T` | Индекс отсчёта (0..POINTS-1) |
| `M_PI_F` | π |

### Выход

| Переменные | Результат |
|------------|-----------|
| `res` | Real: `(res, 0)` |
| `res_re`, `res_im` | Complex IQ |

См. [ScriptGenerator.md](ScriptGenerator.md) — полное описание DSL.

---

## 9. Примеры вызова

### Пример 1: FormSignal — CW 1 МГц, 8 каналов

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormSignalGenerator(ctx)

gen.set_params(
    fs=12e6,        # 12 МГц
    f0=1e6,         # 1 МГц несущая
    antennas=8,
    points=4096,
    amplitude=1.0,
    noise_amplitude=0.1,
    tau_step=1e-5   # 10 мкс между каналами
)

data = gen.generate()
print(data.shape)   # (8, 4096) complex64
```

### Пример 2: FormSignal — Chirp (ЛЧМ)

```python
gen.set_params(
    fs=100000, f0=5000, fdev=20000,   # fdev = девиация chirp
    antennas=1, points=8192,
    noise_amplitude=0.1, noise_seed=42
)
data = gen.generate()
```

### Пример 3: FormSignal — GPU output (без readback)

```python
buf = gen.generate(output='gpu')  # GPUBuffer
# ... передать buf в FFT или другой модуль ...
data = buf.read()   # явный readback при необходимости
buf.release()
```

### Пример 4: FormSignal — из строки

```python
gen.set_params_from_string("f0=1e6,a=1.0,an=0.1,antennas=8,points=4096,fs=12e6")
data = gen.generate()
```

### Пример 5: C++ — FormSignalGenerator

```cpp
#include "generators/form_signal_generator.hpp"
#include "params/form_params.hpp"

FormSignalGenerator gen(backend);
FormParams p;
p.fs = 12e6;
p.f0 = 1e6;
p.antennas = 8;
p.points = 4096;
p.amplitude = 1.0;
p.noise_amplitude = 0.1;
p.tau_step = 1e-5;

gen.SetParams(p);
auto result = gen.GenerateInputData();  // InputData<cl_mem>
// или
auto cpu_data = gen.Generate();  // vector<vector<complex<float>>>
```

### Пример 6: C++ — FormParams из строки

```cpp
auto p = FormParams::ParseFromString(
    "f0=1e6,a=1.5,an=0.1,tau=0.001,antennas=8,points=2048,fs=10e6");
gen.SetParams(p);
```

### Пример 7: Pipeline — FormSignal → FFT → пики

```python
gen = gpuworklib.FormSignalGenerator(ctx)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

gen.set_params(fs=100000, f0=5000, antennas=1, points=8192, noise_amplitude=0.1)
signal = gen.generate()

spectrum = fft.process_complex(signal, sample_rate=100000)
result = finder.find_all_maxima(spectrum, sample_rate=100000)
print(f"Пики: {result['frequencies']} Гц")
```

### Пример 8: DelayedFormSignal — визуализация waterfall

```python
gen = gpuworklib.DelayedFormSignalGenerator(ctx)
gen.set_params(fs=1e6, f0=50000, antennas=8, points=4096)
gen.set_delays([i * 2.0 for i in range(8)])

data = gen.generate()

import matplotlib.pyplot as plt
plt.imshow(np.abs(data[:, :300]), aspect='auto', cmap='inferno')
plt.xlabel('Sample')
plt.ylabel('Antenna')
plt.title('Fractional Delay Waterfall')
plt.savefig('delay_waterfall.png')
```

---

## 10. Тесты

| Файл | Описание |
|------|----------|
| `Python_test/test_form_signal.py` | FormSignalGenerator: 7 тестов + 6 графиков |
| `Python_test/test_delayed_form_signal.py` | DelayedFormSignal: 5 тестов + 4 графика |
| `Python_test/test_lfm_analytical_delay.py` | LfmAnalyticalDelay: 5 тестов |
| `Python_test/example_form_signal.py` | Демо: 5 сценариев + 5 графиков |

**Графики:**
- `Results/Plots/FormSignal/`
- `Results/Plots/DelayedFormSignal/`

```bash
python Python_test/test_form_signal.py
python Python_test/example_form_signal.py
```

---

## 11. Ссылки

| Файл | Описание |
|------|----------|
| [ScriptGenerator.md](ScriptGenerator.md) | DSL формат, примеры |
| [Python API](../../Python/signal_generators_api.md) | Полный Python API |
| [LchFarrow](../lch_farrow/Full.md) | Математика Farrow 48×5 |

---

## Файлы модуля

```
modules/signal_generators/
├── include/
│   ├── generators/
│   │   ├── form_signal_generator.hpp
│   │   ├── form_script_generator.hpp
│   │   ├── delayed_form_signal_generator.hpp
│   │   ├── script_generator.hpp
│   │   ├── cw_generator.hpp
│   │   ├── lfm_generator.hpp
│   │   └── noise_generator.hpp
│   ├── params/
│   │   ├── form_params.hpp
│   │   └── signal_request.hpp
│   ├── signal_service.hpp
│   └── signal_generator_factory.hpp
├── src/
│   ├── form_signal_generator.cpp
│   ├── form_script_generator.cpp
│   ├── delayed_form_signal_generator.cpp
│   └── ...
└── tests/
    ├── test_form_signal.hpp
    ├── test_form_script.hpp
    └── test_delayed_form_signal.hpp
```

---

*Обновлено: 2026-02-18*
*Пусть тот, кто читает, вспомнит тебя добрым словом!*
