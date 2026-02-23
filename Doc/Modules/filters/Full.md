# Filters — Полная документация

> FIR и IIR фильтры на GPU (OpenCL)

**Namespace**: `filters`
**Каталог**: `modules/filters/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Зачем GPU-фильтры](#2-зачем-gpu-фильтры)
3. [Математика алгоритмов](#3-математика-алгоритмов)
4. [Архитектура kernel](#4-архитектура-kernel)
5. [API (C++ и Python)](#5-api)
6. [JSON формат конфигурации](#6-json-формат)
   - [6.1 KernelCacheService](#61-kernelcacheservice--on-disk-кэш-скомпилированных-kernel)
   - [6.2 FilterConfigService](#62-filterconfigservice--сохранение-конфигов-фильтров)
7. [Тесты — что читать и где смотреть](#7-тесты)
8. [Ссылки](#8-ссылки)

---

## 1. Обзор и назначение

Модуль `filters` — GPU-фильтрация **комплексных** multi-channel сигналов. Каждый канал обрабатывается параллельно.

| Класс | Алгоритм | Коэффициенты |
|-------|----------|--------------|
| **FirFilter** | Direct-form convolution | scipy.signal.firwin, JSON |
| **IirFilter** | Biquad cascade DFII-T | scipy.signal.butter (SOS) |

**Stage 1**: scipy → GPU (Python генерирует коэффициенты, передаёт в C++ → GPU kernel).  
**Stage 3**: AI Filter Pipeline — natural language → scipy params → GPU → plot.

---

## 2. Зачем GPU-фильтры

GPU эффективен **только при multi-channel** (≥8 каналов):

| Тип | Каналов | Ускорение GPU vs CPU |
|-----|---------|----------------------|
| FIR direct | 1 | ~1x (нет выигрыша) |
| FIR direct | 64 | ~40–60x |
| IIR cascade | 1 | ~0.5x (CPU быстрее!) |
| IIR cascade | 64 | ~50–80x |

**Вывод**: Single-channel IIR — лучше на CPU. Multi-channel — GPU даёт значительный выигрыш.

---

## 3. Математика алгоритмов

### FIR (Finite Impulse Response)

$$
y[ch][n] = \sum_{k=0}^{N-1} h[k] \cdot x[ch][n-k]
$$

- **Прямая форма** (direct-form): каждый выходной отсчёт — свёртка с импульсной характеристикой
- **Параллелизм**: по каналам и по семплам (2D NDRange)

### IIR (Infinite Impulse Response) — Biquad

Одна секция (second-order section):

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}
$$

**Direct Form II Transposed** (численно стабильная):

```
y[n] = b0*x[n] + w1
w1   = b1*x[n] - a1*y[n] + w2
w2   = b2*x[n] - a2*y[n]
```

- **Параллелизм**: только по каналам (1D NDRange). Внутри канала — последовательная зависимость по времени.
- **Все секции** в одном kernel (order 2–10+).

---

## 4. Архитектура kernel

### FIR

| Параметр | Значение |
|----------|----------|
| NDRange | 2D (channels, points) |
| Work-item | 1 выходной отсчёт (channel, sample) |
| Коэффициенты | `__constant` (≤16K taps) или `__global` (auto-fallback) |

**Лимит**: `kMaxConstantTaps = 16000` (~64 KB). При `num_taps > 16000` → автоматически `fir_filter_cf32_global`.

### IIR

| Параметр | Значение |
|----------|----------|
| NDRange | 1D (channels) |
| Work-item | 1 канал (все семплы + все секции последовательно) |
| SOS буфер | `[num_sections × 5]` float (b0, b1, b2, a1, a2) |

### Формат данных

**Channel-sequential** (рекомендуется, coalesced access):

```
Buffer: [ch0_s0, ch0_s1, ..., ch0_sN-1,  ch1_s0, ch1_s1, ...]
Access: input[channel * points + sample]
```

---

## 5. API

### C++

```cpp
#include "filters/fir_filter.hpp"
#include "filters/iir_filter.hpp"

// FIR
filters::FirFilter fir(backend);
fir.SetCoefficients(std::vector<float> coeffs);
fir.LoadConfig("lowpass.json");
auto result = fir.Process(cl_mem input, channels, points);
auto cpu_ref = fir.ProcessCpu(input_vector, channels, points);

// IIR
filters::IirFilter iir(backend);
iir.SetBiquadSections(std::vector<BiquadSection> sections);
iir.LoadConfig("butterworth.json");
auto result = iir.Process(input, channels, points);
```

### Python

```python
import gpuworklib
import scipy.signal as sig

ctx = gpuworklib.GPUContext(0)

# FIR
fir = gpuworklib.FirFilter(ctx)
fir.set_coefficients(sig.firwin(64, 0.1).tolist())
result = fir.process(signal)  # (channels, points) complex64

# IIR
iir = gpuworklib.IirFilter(ctx)
sos = sig.butter(2, 0.1, output='sos')
sections = [{'b0':r[0],'b1':r[1],'b2':r[2],'a1':r[4],'a2':r[5]} for r in sos]
iir.set_sections(sections)
result = iir.process(signal)
```

---

## 6. JSON формат конфигурации

### FIR

```json
{
  "type": "fir",
  "description": "Low-pass FIR, fc=0.1, 64 taps, Hamming",
  "coefficients": [0.0008, 0.0012, ..., 0.0012, 0.0008]
}
```

### IIR (SOS)

```json
{
  "type": "iir",
  "description": "Butterworth 4th order low-pass, fc=0.1",
  "sections": [
    {"b0": 0.0675, "b1": 0.1349, "b2": 0.0675, "a1": -1.1430, "a2": 0.4128},
    {"b0": 1.0000, "b1": 2.0000, "b2": 1.0000, "a1": -1.5529, "a2": 0.6562}
  ]
}
```

`a0` всегда 1.0 (нормированная форма). Scipy: `sos = butter(N, Wn, output='sos')`.

---

## 6.1. KernelCacheService — on-disk кэш скомпилированных kernel

FirFilter и IirFilter используют DrvGPU [KernelCacheService](../../DrvGPU/Services/Full.md):

| Этап | Действие |
|------|----------|
| **Первый запуск** | CompileKernel() → компиляция из source → Save в `modules/filters/kernels/bin/` |
| **Повторный** | Load binary (~1 мс вместо ~50 мс компиляции) |
| **Fallback** | При отсутствии/ошибке cache — компиляция из source |

**Cache key:** `fir_filter_cf32`, `iir_filter_cf32` (source не зависит от коэффициентов).

**Структура:** `modules/filters/kernels/bin/fir_filter_cf32_opencl.bin`, `manifest.json`.

---

## 6.2. FilterConfigService — сохранение конфигов фильтров

DrvGPU [FilterConfigService](../../DrvGPU/Services/Full.md) — сохранение/загрузка коэффициентов в JSON:
- **FIR:** type, coefficients[]
- **IIR:** type, sections[] (b0,b1,b2,a1,a2)
- **Ключи:** `filters/{name}.json`
- **Версионирование:** при перезаписи → `name_00.json`, `name_01.json`

**Интеграция** SaveFilterConfig/LoadFilterConfig в FirFilter/IirFilter — планируется (TASK-006).

---

## 7. Тесты — что читать и где смотреть

### C++ тесты

**Файл**: `modules/filters/tests/test_fir_basic.hpp`, `test_iir_basic.hpp`  
**Вызов**: через `all_test.hpp` из `main.cpp` → `filters_all_test::run()`

| Тест | Что проверяет | Сигнал | Порог |
|------|---------------|--------|-------|
| **run_fir_basic** | 64-tap FIR lowpass, GPU vs ProcessCpu | 8 ch × 4096 pts, CW 100Hz + 5000Hz, fs=50kHz | < 1e-3 |
| **run_iir_basic** | Butterworth 2nd order LP, GPU vs ProcessCpu | То же | < 1e-3 |

**Коэффициенты**: предвычислены из `scipy.signal.firwin(64, 0.1)` и `scipy.signal.butter(2, 0.1, output='sos')`

**Результаты (типичные)**: GPU vs CPU err ≈ 1e-6 ✅

---

### Python тесты

**Файл**: `Python_test/filters/test_filters_stage1.py`  
**Запуск**: `python Python_test/filters/test_filters_stage1.py` или `pytest Python_test/filters/test_filters_stage1.py -v`

| Тест | Что проверяет | Эталон | Порог |
|------|---------------|--------|-------|
| FIR | scipy.firwin → GPU → vs scipy.lfilter | scipy.lfilter | < 1e-2 |
| IIR | scipy.butter(sos) → GPU → vs scipy.sosfilt | scipy.sosfilt | < 5e-2 |

**Результаты (типичные)**: FIR err ≈ 4.77e-7 ✅ | IIR err ≈ 1.31e-6 ✅

**Другие тесты**:
- `test_ai_filter_pipeline.py` — Stage 3: NL → AI → scipy → GPU → plot
- `test_iir_plot.py` — IIR order 2/4/8 comparison
- `test_ai_fir_demo.py` — AI demo

---

### Что смотреть при отладке

| Вопрос | Где искать |
|-------|------------|
| Как устроен CPU эталон? | `ProcessCpu()` в `fir_filter.cpp`, `iir_filter.cpp` |
| Какие коэффициенты? | `kTestFirCoeffs64` в test_fir_basic.hpp, BiquadSection в test_iir_basic.hpp |
| Графики? | `Results/Plots/filters/test_filters_stage1.png`, `test_iir_stage1.png`, `ai_*.png` |

---

## 8. Ссылки

### Статьи и метод

| Источник | Описание |
|----------|----------|
| [SciPy FIR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html) | FIR design (firwin) |
| [SciPy IIR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) | Butterworth (butter, sos) |
| [Direct Form II Transposed](https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2_transposed) | Biquad DFII-T |

### Локальная документация

| Файл | Описание |
|------|----------|
| [gpu_filters_research.md](gpu_filters_research.md) | Исследование: Overlap-Save/Add, tiled FIR, IIR cascade — для будущей реализации |

### Out of Scope (Post-MVP)

| Фича | Когда |
|------|-------|
| Overlap-Save/Overlap-Add | После стабильного FIR. См. `gpu_filters_research.md` |
| Адаптивные LMS/NLMS/RLS | Отдельная задача |
| Полифазные фильтры/децимация | Отдельная задача |
| Stage 2: text→kernel кэш | После Stage 1 |
| ROCm полная реализация | После AMD GPU |

---

## Файлы модуля

```
modules/filters/
├── kernels/
│   ├── bin/                    # KernelCacheService (создаётся при первом запуске)
│   │   ├── fir_filter_cf32_opencl.bin
│   │   └── iir_filter_cf32_opencl.bin
│   └── manifest.json
├── include/
│   ├── filters/
│   │   ├── fir_filter.hpp
│   │   ├── iir_filter.hpp
│   │   ├── fir_filter_rocm.hpp     # ROCm stub
│   │   └── iir_filter_rocm.hpp     # ROCm stub
│   ├── kernels/
│   │   ├── fir_kernels.hpp
│   │   └── iir_kernels.hpp
│   └── types/
│       ├── filter_params.hpp       # BiquadSection, FirParams, IirParams, FilterConfig
│       ├── filter_types.hpp
│       └── filter_modes.hpp
├── src/
│   ├── fir_filter.cpp
│   └── iir_filter.cpp
└── tests/
    ├── all_test.hpp
    ├── test_fir_basic.hpp
    ├── test_iir_basic.hpp
    └── README.md
```

---

*Обновлено: 2026-02-23*
