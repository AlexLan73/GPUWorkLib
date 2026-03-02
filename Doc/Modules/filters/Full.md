# Filters — Полная документация

> FIR и IIR фильтры на GPU (OpenCL + ROCm/HIP) для комплексных multi-channel сигналов

**Namespace**: `filters`
**Каталог**: `modules/filters/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, ROCm/HIP (опционально, `ENABLE_ROCM=1`)

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Когда GPU-фильтры выгодны](#2-когда-gpu-фильтры-выгодны)
3. [Математика алгоритмов](#3-математика-алгоритмов)
4. [Архитектура kernel](#4-архитектура-kernel)
5. [Pipeline](#5-pipeline)
6. [C4 — Архитектурные диаграммы](#6-c4--архитектурные-диаграммы)
7. [API (C++ и Python)](#7-api)
8. [JSON формат конфигурации](#8-json-формат-конфигурации)
9. [KernelCacheService и FilterConfigService](#9-kernelcacheservice-и-filterconfigservice)
10. [Тесты — описание и ратionale](#10-тесты)
11. [Профилирование (бенчмарки)](#11-профилирование)
12. [Файловое дерево модуля](#12-файловое-дерево)
13. [Важные нюансы](#13-важные-нюансы)
14. [Ссылки](#14-ссылки)

---

## 1. Обзор и назначение

Модуль `filters` — GPU-фильтрация **комплексных** multi-channel сигналов в формате `float2` (complex64). Каждый канал обрабатывается параллельно.

| Класс | Backend | Алгоритм | Коэффициенты |
|-------|---------|----------|--------------|
| **FirFilter** | OpenCL | Direct-form convolution | `SetCoefficients()`, JSON |
| **IirFilter** | OpenCL | Biquad cascade DFII-T | `SetBiquadSections()`, JSON |
| **FirFilterROCm** | ROCm/HIP | Direct-form + hiprtc | `SetCoefficients()` |
| **IirFilterROCm** | ROCm/HIP | Biquad cascade DFII-T | `SetBiquadSections()` |

**Workflow Stage 1**: scipy → коэффициенты → GPU (Python генерирует, передаёт в C++).
**Workflow Stage 3**: Natural language → AI → scipy params → GPU → plot.

---

## 2. Когда GPU-фильтры выгодны

GPU эффективен **только при multi-channel** (≥ 8 каналов):

| Тип | Каналов | Ускорение GPU vs CPU |
|-----|---------|----------------------|
| FIR direct | 1 | ~1× (нет выигрыша) |
| FIR direct | 64 | ~40–60× |
| IIR cascade | 1 | ~0.5× (CPU быстрее!) |
| IIR cascade | 64 | ~50–80× |

**Вывод**: Single-channel IIR — лучше на CPU. Multi-channel — GPU даёт значительный выигрыш.

---

## 3. Математика алгоритмов

### 3.1 FIR (Finite Impulse Response)

$$
y[ch][n] = \sum_{k=0}^{N-1} h[k] \cdot x[ch][n-k]
$$

- **Прямая форма** (direct-form convolution): каждый выходной отсчёт — свёртка с импульсной характеристикой $h[k]$.
- **Тип**: линейно-фазовый (симметричные коэффициенты → линейная фаза).
- **Параллелизм**: по каналам и по семплам (2D NDRange).
- **Нулевые граничные условия**: при $n - k < 0$ отсчёт считается нулевым (causal filtering).

```
FIR kernel (OpenCL):
  work-item (ch, n) вычисляет y[ch][n]
  for k = 0..N-1:
    if (n - k >= 0): acc += h[k] * x[ch, n-k]
  output[ch * P + n] = acc
```

### 3.2 IIR (Infinite Impulse Response) — Biquad cascade

Передаточная функция одной секции (second-order section, SOS):

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}
$$

**Direct Form II Transposed** (численно стабильная форма):

$$
y[n] = b_0 \cdot x[n] + w_1[n-1]
$$
$$
w_1[n] = b_1 \cdot x[n] - a_1 \cdot y[n] + w_2[n-1]
$$
$$
w_2[n] = b_2 \cdot x[n] - a_2 \cdot y[n]
$$

- **Параллелизм**: только по каналам (1D NDRange, 1 work-item = 1 канал).
- **Последовательность**: внутри канала зависимость по времени — $y[n]$ зависит от $y[n-1]$.
- **Cascade**: несколько секций обрабатываются последовательно в одном kernel.
  - Секция 0 читает из `input`, пишет в `output`.
  - Секции 1..N читают из `output` (переиспользование буфера).

**SOS матрица** (буфер на GPU): `[num_sections × 5]` float:

```
sos[sec * 5 + 0] = b0
sos[sec * 5 + 1] = b1
sos[sec * 5 + 2] = b2
sos[sec * 5 + 3] = a1
sos[sec * 5 + 4] = a2
```

---

## 4. Архитектура kernel

### FIR kernel

| Параметр | Значение |
|----------|----------|
| NDRange | 2D `(channels, ceil(points/256)×256)` |
| Local size | `(1, 256)` |
| Work-item | 1 выходной отсчёт `(ch, n)` |
| Коэффициенты | `__constant float*` (≤ 16 000 тапов) или `__global float*` (авто-fallback) |
| Флаги компиляции | `-cl-fast-relaxed-math` |

**Лимит**: `kMaxConstantTaps = 16000` (~64 KB). При `num_taps > 16000` → автоматически `fir_filter_cf32_global` (коэффициенты в `__global`).

**Два kernel в одном `.cl` файле**:
- `fir_filter_cf32` — константная память
- `fir_filter_cf32_global` — глобальная память (при большом количестве тапов)

### IIR kernel

| Параметр | Значение |
|----------|----------|
| NDRange | 1D `(channels,)` |
| Work-item | 1 канал (все семплы + все секции последовательно) |
| SOS буфер | `__constant float*` `[num_sections × 5]` |
| Флаги компиляции | `-cl-fast-relaxed-math` |

### Формат данных (layout)

**Channel-sequential** (рекомендуется, coalesced access):

```
Buffer: [ch0_s0, ch0_s1, ..., ch0_sN-1,  ch1_s0, ch1_s1, ...]
Access: input[channel * points + sample]
```

---

## 5. Pipeline

### FIR Pipeline (OpenCL)

```
    ┌─────────────────────────────────────────────────────────────────┐
    │ Host (C++ / Python)                                             │
    │                                                                 │
    │  SetCoefficients(h)                                             │
    │    └─► UploadCoefficients() → coeff_buf_ (cl_mem, __constant) │
    │                                                                 │
    │  Process(input_buf, ch, pts)                                    │
    │    ├─► clCreateBuffer(output_buf)                               │
    │    ├─► clCreateKernel("fir_filter_cf32" | "_global")            │
    │    ├─► clSetKernelArg(0..4)                                     │
    │    ├─► clEnqueueNDRangeKernel [channels, ⌈pts/256⌉×256]        │
    │    ├─► clFinish()                                               │
    │    └─► CollectOrRelease(kernel_event, "Kernel", pe)             │
    │                                                                 │
    │  result.data = output_buf  ← caller clReleaseMemObject!         │
    └─────────────────────────────────────────────────────────────────┘
```

### IIR Pipeline (OpenCL)

```
    ┌─────────────────────────────────────────────────────────────────┐
    │ Host (C++ / Python)                                             │
    │                                                                 │
    │  SetBiquadSections(sections)                                    │
    │    └─► UploadSosMatrix() → sos_buf_ [S×5 float]                │
    │                                                                 │
    │  Process(input_buf, ch, pts)                                    │
    │    ├─► clCreateBuffer(output_buf)                               │
    │    ├─► clCreateKernel("iir_biquad_cascade_cf32")                │
    │    ├─► clSetKernelArg(0..4)                                     │
    │    ├─► clEnqueueNDRangeKernel [channels]  ← 1D!                │
    │    ├─► clFinish()                                               │
    │    └─► CollectOrRelease(kernel_event, "Kernel", pe)             │
    │                                                                 │
    │  result.data = output_buf  ← caller clReleaseMemObject!         │
    └─────────────────────────────────────────────────────────────────┘
```

### Mermaid (полный pipeline FIR)

```mermaid
flowchart LR
  A[Python/C++ caller] -->|SetCoefficients| B[UploadCoefficients]
  B -->|cl_mem coeff_buf| C[GPU __constant buffer]
  A -->|Process input_buf| D[clCreateBuffer output]
  D --> E[clCreateKernel]
  E --> F[clSetKernelArg]
  F --> G[clEnqueueNDRangeKernel 2D]
  C --> G
  G --> H[clFinish]
  H --> I[result.data cl_mem]
  I -->|caller owns| A
```

---

## 6. C4 — Архитектурные диаграммы

### C1 — System Context

```
┌─────────────────────────────────────────────────────────────────┐
│ GPUWorkLib System                                               │
│                                                                 │
│  ┌─────────────────────────────┐                               │
│  │  filters module             │                               │
│  │  FIR/IIR GPU filtering      │◄──── Python / C++ App         │
│  └─────────────────────────────┘                               │
│              │                                                  │
│              ▼                                                  │
│  ┌─────────────────────────────┐                               │
│  │  DrvGPU                     │                               │
│  │  OpenCL / ROCm backend      │                               │
│  └─────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### C2 — Container

```
┌─────────────────── filters module ─────────────────────────────┐
│                                                                 │
│  ┌────────────────────────┐   ┌────────────────────────────┐   │
│  │  FirFilter (OpenCL)    │   │  IirFilter (OpenCL)        │   │
│  │  fir_filter.hpp/.cpp   │   │  iir_filter.hpp/.cpp       │   │
│  └────────────┬───────────┘   └────────────┬───────────────┘   │
│               │                            │                   │
│  ┌────────────┴────────────────────────────┴───────────────┐   │
│  │  FirFilterROCm / IirFilterROCm (ROCm/HIP, ENABLE_ROCM) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  types: BiquadSection, FirParams, IirParams,           │     │
│  │         FilterConfig (JSON), filter_modes.hpp          │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  kernels: fir_filter_cf32.cl, iir_filter_cf32.cl       │     │
│  │  + kernels/bin/ (KernelCacheService)                   │     │
│  └────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  DrvGPU                    │
            │  IBackend / OpenCLBackend  │
            │  ROCmBackend               │
            │  KernelCacheService        │
            └────────────────────────────┘
```

### C3 — Component (FirFilter, OpenCL)

```
FirFilter
├── Constructor(IBackend*)
│     ├── GetNativeContext/Queue/Device
│     ├── KernelCacheService::Load("fir_filter_cf32")
│     └── CompileKernel() ← clCreateProgramWithSource + clBuildProgram
│
├── SetCoefficients(h[])
│     ├── coefficients_ = h
│     ├── use_global_coeffs_ = (size > 16000)
│     └── UploadCoefficients() → coeff_buf_ (cl_mem)
│
├── Process(input_buf, ch, pts, prof_events)
│     ├── clCreateBuffer(output_buf)
│     ├── clCreateKernel("fir_filter_cf32" | "_global")
│     ├── clSetKernelArg(input, output, coeffs, num_taps, points)
│     ├── clEnqueueNDRangeKernel [ch, ⌈pts/256⌉×256] local=[1,256]
│     ├── clFinish()
│     └── CollectOrRelease(ev, "Kernel", pe)
│
└── ProcessCpu(input, ch, pts) → CPU reference (validation)
```

### C4 — Code (Kernel)

```opencl
// fir_filter_cf32.cl (упрощённо)
__kernel void fir_filter_cf32(
    __global const float2* restrict input,   // [ch * pts + n]
    __global       float2* restrict output,
    __constant     float*  coeffs,           // h[0..N-1]
    const uint num_taps,
    const uint points)
{
    const uint ch = get_global_id(0);  // канал
    const uint n  = get_global_id(1);  // семпл
    if (n >= points) return;

    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            float2 x = input[ch * points + (uint)idx];
            acc.x += coeffs[k] * x.x;
            acc.y += coeffs[k] * x.y;
        }
    }
    output[ch * points + n] = acc;
}
```

---

## 7. API

### 7.1 C++ — OpenCL

```cpp
#include "filters/fir_filter.hpp"
#include "filters/iir_filter.hpp"

// ─── FirFilter ───────────────────────────────────────────────────
filters::FirFilter fir(backend);

// Из кода:
fir.SetCoefficients(std::vector<float>{ 0.1f, 0.2f, 0.4f, 0.2f, 0.1f });

// Из JSON:
fir.LoadConfig("modules/filters/configs/lowpass_64tap.json");

// GPU processing
drv_gpu_lib::InputData<cl_mem> result = fir.Process(input_buf, channels, points);
// result.data — cl_mem, caller must clReleaseMemObject(result.data)

// CPU reference (validation)
auto cpu_ref = fir.ProcessCpu(input_vector, channels, points);

// Getters
uint32_t n_taps = fir.GetNumTaps();
const auto& coeffs = fir.GetCoefficients();
bool ready = fir.IsReady();

// ─── IirFilter ───────────────────────────────────────────────────
filters::IirFilter iir(backend);

// Из кода:
filters::BiquadSection sec;
sec.b0 = 0.02008337f;  sec.b1 = 0.04016673f;  sec.b2 = 0.02008337f;
sec.a1 = -1.56101808f; sec.a2 = 0.64135154f;
iir.SetBiquadSections({ sec });

// Из JSON:
iir.LoadConfig("modules/filters/configs/butterworth4.json");

// GPU processing
auto result = iir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);  // caller owns!

// CPU reference
auto cpu_ref = iir.ProcessCpu(input_vector, channels, points);

uint32_t n_sec = iir.GetNumSections();
```

### 7.2 C++ — ROCm/HIP (`ENABLE_ROCM=1`, Linux)

```cpp
#include "filters/fir_filter_rocm.hpp"
#include "filters/iir_filter_rocm.hpp"

// ─── FirFilterROCm ───────────────────────────────────────────────
filters::FirFilterROCm fir_rocm(rocm_backend);
fir_rocm.SetCoefficients(coeffs);

// Из GPU-указателя (void* device ptr)
drv_gpu_lib::InputData<void*> res = fir_rocm.Process(
    gpu_input_ptr, channels, points);
hipFree(res.data);  // caller owns!

// Из CPU данных (upload + process)
auto res2 = fir_rocm.ProcessFromCPU(cpu_data, channels, points);
hipFree(res2.data);

// CPU reference
auto cpu_ref = fir_rocm.ProcessCpu(cpu_data, channels, points);

// ─── IirFilterROCm ───────────────────────────────────────────────
filters::IirFilterROCm iir_rocm(rocm_backend);
iir_rocm.SetBiquadSections({ sec0, sec1 });

auto res = iir_rocm.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);
```

### 7.3 Python — OpenCL

```python
import gpuworklib as gw
import scipy.signal as sig
import numpy as np

ctx = gw.GPUContext(0)

# ─── FirFilter ───────────────────────────────────────────────────
fir = gw.FirFilter(ctx)

# Дизайн через scipy
taps = sig.firwin(64, 0.1).astype(np.float32)
fir.set_coefficients(taps.tolist())

# Фильтрация: signal.shape = (channels, points) complex64
result = fir.process(signal)   # возвращает (channels, points) complex64

# Или 1D:
result_1d = fir.process(signal[0])   # ndarray shape (points,)

# Properties
print(fir.num_taps)              # int
print(fir.coefficients)          # list[float]
print(repr(fir))                 # "FirFilter(num_taps=64)"

# ─── IirFilter ───────────────────────────────────────────────────
iir = gw.IirFilter(ctx)

sos = sig.butter(2, 0.1, output='sos').astype(np.float64)
sections = [
    {'b0': float(r[0]), 'b1': float(r[1]), 'b2': float(r[2]),
     'a1': float(r[4]), 'a2': float(r[5])}
    for r in sos
]
iir.set_sections(sections)

result = iir.process(signal)  # (channels, points) complex64

print(iir.num_sections)        # int
print(iir.sections)            # list[dict]
```

### 7.4 Python — ROCm

```python
import gpuworklib as gw
import scipy.signal as sig

ctx = gw.ROCmGPUContext(0)

fir = gw.FirFilterROCm(ctx)
coeffs = sig.firwin(64, 0.1).tolist()
fir.set_coefficients(coeffs)

result = fir.process(data)   # data: np.ndarray complex64 1D или 2D

print(fir.num_taps)
print(repr(fir))   # "FirFilterROCm(num_taps=64)"
```

---

## 8. JSON формат конфигурации

### FIR

```json
{
  "type": "fir",
  "description": "Low-pass FIR, fc=0.1 (normalized), 64 taps, Hamming window",
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

**`a0` всегда 1.0** (нормированная форма SciPy: `sos = butter(N, Wn, output='sos')`).

Парсинг реализован без внешних зависимостей (`FilterConfig::LoadJson` — minimal parser, no nlohmann).

---

## 9. KernelCacheService и FilterConfigService

### KernelCacheService — on-disk кэш скомпилированных kernel

FirFilter и IirFilter используют DrvGPU `KernelCacheService`:

| Этап | Действие |
|------|----------|
| **Первый запуск** | `CompileKernel()` → JIT из source → `Save()` в `modules/filters/kernels/bin/` |
| **Повторный** | `Load()` binary (~1 мс вместо ~50 мс компиляции) |
| **Fallback** | При отсутствии/ошибке cache — компиляция из source |

**Cache key:** `fir_filter_cf32` / `iir_filter_cf32`.
**Бинари:** `kernels/bin/fir_filter_cf32_opencl.bin`, `kernels/bin/iir_filter_cf32_opencl.bin`.
**Примечание:** ROCm (hiprtc) использует другой механизм кэширования — HSACO в `kernels/bin/`.

### FilterConfigService — сохранение конфигов фильтров

DrvGPU `FilterConfigService` — сохранение/загрузка коэффициентов в JSON:
- **FIR:** type, coefficients[]
- **IIR:** type, sections[] (b0,b1,b2,a1,a2)
- **Ключи:** `filters/{name}.json`
- **Версионирование:** при перезаписи → `name_00.json`, `name_01.json`

**⚠️ Интеграция** `SaveFilterConfig`/`LoadFilterConfig` в FirFilter/IirFilter — планируется (TASK-006). Пока используется `FilterConfig::LoadJson` напрямую.

---

## 10. Тесты

### 10.1 C++ тесты (OpenCL)

Вызов: `filters_all_test::run()` из `main.cpp` через `modules/filters/tests/all_test.hpp`

| # | Файл | Функция | Сигнал | Порог | Что проверяет и почему |
|---|------|---------|--------|-------|------------------------|
| 1 | `test_fir_basic.hpp` | `run_fir_basic()` | 8 ch × 4096 pts, CW 100 Hz + CW 5000 Hz, fs=50 kHz, 64-tap LP FIR (Hamming) | < 1e-3 | **GPU ≈ CPU reference.** Два тона: 100 Hz должен пройти, 5000 Hz — подавиться. Выбран Hamming window (good stopband attenuation). Порог 1e-3 учитывает `-cl-fast-relaxed-math` (float32 precision ~1e-6, т.к. суммирование 64 членов). Ловит ошибки индексации в kernel, неверную раскладку буфера. |
| 2 | `test_iir_basic.hpp` | `run_iir_basic()` | 8 ch × 4096 pts, то же CW-сигнал, Butterworth 2nd order LP, fc=0.1, 1 секция | < 1e-3 | **GPU biquad ≈ CPU DFII-T reference.** Butterworth — минимально-пульсирующий АЧХ. 1 секция проверяет базовый цикл state-machine. Порог 1e-3 — граница float32 precision при каскадном накоплении ошибки. Ловит баги в state переменных w1/w2, ошибку в порядке операций DFII-T. |

**Коэффициенты теста FIR**: `kTestFirCoeffs64` — предвычислены из `scipy.signal.firwin(64, 0.1, window='hamming')`.

### 10.2 C++ тесты (ROCm, `test_filters_rocm.hpp`)

Запуск: `test_filters_rocm::run()`. На Windows — compile-only (ENABLE_ROCM не определён). На Linux + AMD GPU — 6 тестов:

| # | Функция | Сигнал | Порог | Что проверяет и почему |
|---|---------|--------|-------|------------------------|
| 1 | `test_fir_basic` | 8 ch × 4096 pts, 64-tap LP, hipMemcpy | < 1e-3 | ROCm FIR basic: GPU (HIP) ≈ CPU reference. Аналог OpenCL теста #1, верифицирует hiprtc-компиляцию и HIP NDRange. |
| 2 | `test_fir_large` | 16 ch × 8192 pts, 256-tap LP (sinc×Hamming, синтезируется в тесте) | < 1e-3 | **Масштабируемость**: 256 тапов — проверяет производительность при больших фильтрах. Коэффициенты синтезируются в тесте (sinc × Hamming window), что исключает зависимость от scipy. |
| 3 | `test_fir_gpu_ptr` | 4 ch × 2048 pts, ручной hipMalloc+hipMemcpyHtoDAsync | < 1e-3 | **`Process(void* gpu_ptr, ...)` overload** — GPU pipeline без лишнего upload. Ловит ошибки в overload, принимающем уже-на-GPU данные (без пере-upload). |
| 4 | `test_iir_basic` | 8 ch × 4096 pts, Butterworth 2nd order (1 секция) | < 1e-3 | ROCm IIR basic: GPU (HIP) biquad ≈ CPU reference. |
| 5 | `test_iir_multi_section` | 8 ch × 4096 pts, Butterworth 4th order (2 секции) | < 1e-3 | **Cascade**: 2 секции → Butterworth 4th order. Проверяет корректность цикла по секциям, правильный re-read из output при sec>0. |
| 6 | `test_iir_gpu_ptr` | 4 ch × 2048 pts, ручной hipMalloc | < 1e-3 | **`Process(void*, ...)` overload для IIR** — аналог теста #3. |

### 10.3 C++ бенчмарки (GpuBenchmarkBase)

| Файл | Класс | Стейджи | Результаты |
|------|-------|---------|------------|
| `filters_benchmark.hpp` | `FirFilterBenchmark` | `Kernel` | `Results/Profiler/GPU_00_FirFilter/` |
| `filters_benchmark.hpp` | `IirFilterBenchmark` | `Kernel` | `Results/Profiler/GPU_00_IirFilter/` |
| `filters_benchmark_rocm.hpp` | `FirFilterROCmBenchmark` | `Upload + Kernel` | `Results/Profiler/GPU_00_FirFilter_ROCm/` |
| `filters_benchmark_rocm.hpp` | `IirFilterROCmBenchmark` | `Upload + Kernel` | `Results/Profiler/GPU_00_IirFilter_ROCm/` |

Вызов benchmark: `test_filters_benchmark::run()` (закомментирован в `all_test.hpp`).

### 10.4 Python тесты (OpenCL)

**Файл**: `Python_test/filters/test_filters_stage1.py`
**Запуск**: `pytest Python_test/filters/test_filters_stage1.py -v`

| # | Функция | Сигнал | Порог | Что проверяет и почему |
|---|---------|--------|-------|------------------------|
| 1 | `test_fir_gpu_vs_scipy` | 8 ch × 4096 pts, CW 100+5000 Hz, firwin(64, 0.1) | < 1e-2 | **GPU FIR ≈ scipy.lfilter**. Внешний эталон (не CPU-reference класса). Порог 1e-2 шире 1e-3 из-за разницы в boundary conditions (`lfilter` vs GPU causal). |
| 2 | `test_fir_basic_properties` | — | `num_taps == 64` | Проверяет Python API: `fir.num_taps`, `fir.coefficients`, `repr(fir)`. |
| 3 | `test_fir_single_channel` | 1D input (4096 pts), firwin(32, 0.2) | `ndim==1` | 1D input — выход тоже 1D. Ловит ошибки binding'а при одноканальном случае. |
| 4 | `test_iir_gpu_vs_scipy` | 8 ch × 4096 pts, butter(2, 0.1, sos) | < 5e-2 | **GPU IIR ≈ scipy.sosfilt**. Порог 5e-2 — IIR больше накапливает ошибку из-за рекурсии. |
| 5 | `test_iir_basic_properties` | 1 секция | `num_sections == 1` | Проверяет `iir.num_sections`, `iir.sections`, `repr(iir)`. |

**Результаты (типичные)**: FIR err ≈ 4.77e-7 ✅ | IIR err ≈ 1.31e-6 ✅

### 10.5 Python тесты (ROCm, Linux)

**Файл**: `Python_test/filters/test_fir_filter_rocm.py` и `test_iir_filter_rocm.py`
**Context**: `gw.ROCmGPUContext(0)`, класс `gw.FirFilterROCm` / `gw.IirFilterROCm`

**FIR ROCm тесты** (`test_fir_filter_rocm.py`):

| # | Функция | Что проверяет | Порог |
|---|---------|---------------|-------|
| 1 | `test_fir_single_channel_basic` | 1D complex, GPU vs scipy.lfilter | atol=1e-4 |
| 2 | `test_fir_multi_channel` | 2D (8ch × 4096pts), per-channel vs scipy | atol=1e-4 |
| 3 | `test_fir_all_pass` | Delta-filter [1.0] → output == input | atol=1e-4 |
| 4 | `test_fir_lowpass_attenuation` | Two-tone: power ratio < 0.9 после LP | energy |
| 5 | `test_fir_properties` | `num_taps`, `coefficients`, `repr` | exact |

**Ключевые идеи ROCm Python тестов**:
- Тест 3 (delta): `h=[1.0]` → `y[n] = x[n]` — идентичное преобразование. Ловит ошибки инициализации kernel (case h→0).
- Тест 4 (attenuation): физическая проверка без точного эталона: высокочастотная компонента должна ослабляться, `ratio < 0.9`.
- Порог `atol=1e-4` (строже 1e-2 OpenCL) — ROCm не использует `-cl-fast-relaxed-math`, точность выше.

**Другие Python тесты**:

| Файл | Назначение |
|------|------------|
| `test_iir_filter_rocm.py` | IIR ROCm: multi-section, GPU ptr, properties |
| `test_iir_plot.py` | IIR order 2/4/8: сравнение АЧХ, сохраняет граф |
| `test_ai_filter_pipeline.py` | Stage 3: natural language → AI → scipy → GPU → plot |
| `test_ai_fir_demo.py` | AI demo: описание фильтра в тексте → GPU |

---

## 11. Профилирование

### OpenCL стейджи

| Бенчмарк | Класс | Стейджи | Описание |
|----------|-------|---------|----------|
| FirFilter | `FirFilterBenchmark` | `Kernel` | Время выполнения NDRange kernel |
| IirFilter | `IirFilterBenchmark` | `Kernel` | Время выполнения biquad cascade kernel |

### ROCm стейджи

| Бенчмарк | Класс | Стейджи | Описание |
|----------|-------|---------|----------|
| FirFilterROCm | `FirFilterROCmBenchmark` | `Upload`, `Kernel` | H2D + kernel через `ProcessFromCPU` |
| IirFilterROCm | `IirFilterROCmBenchmark` | `Upload`, `Kernel` | H2D + kernel через `ProcessFromCPU` |

### Запуск бенчмарка

```cpp
// ⚠️ OpenCL queue нужен с CL_QUEUE_PROFILING_ENABLE!
cl_command_queue queue = clCreateCommandQueue(
    context, device, CL_QUEUE_PROFILING_ENABLE, &err);

auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
backend->InitializeFromExternalContext(context, device, queue);

filters::FirFilter fir(backend.get());
fir.SetCoefficients(kTestFirCoeffs64);

test_filters::FirFilterBenchmark bench(
    backend.get(), fir, input_buf, 8, 4096,
    {.n_warmup = 5, .n_runs = 20,
     .output_dir = "Results/Profiler/GPU_00_FirFilter"});

bench.Run();
bench.Report();
```

---

## 12. Файловое дерево

```
modules/filters/
├── CMakeLists.txt
├── include/
│   ├── filters/
│   │   ├── fir_filter.hpp          # OpenCL FIR filter class
│   │   ├── iir_filter.hpp          # OpenCL IIR biquad cascade class
│   │   ├── fir_filter_rocm.hpp     # ROCm FIR (hiprtc), stub на Windows
│   │   └── iir_filter_rocm.hpp     # ROCm IIR (hiprtc), stub на Windows
│   ├── kernels/
│   │   ├── fir_kernels.hpp         # GetFirDirectSource_opencl()
│   │   ├── fir_kernels_rocm.hpp    # GetFirDirectSource_rocm()
│   │   ├── iir_kernels.hpp         # GetIirBiquadSource_opencl()
│   │   └── iir_kernels_rocm.hpp    # GetIirBiquadSource_rocm()
│   └── types/
│       ├── filter_params.hpp       # BiquadSection, FirParams, IirParams, FilterConfig (JSON)
│       ├── filter_types.hpp        # ProfEvents, ROCmProfEvents
│       └── filter_modes.hpp        # FilterMode enum
├── kernels/
│   ├── fir_filter_cf32.cl          # FIR OpenCL kernel (2 kernels: __constant + __global)
│   ├── iir_filter_cf32.cl          # IIR biquad cascade OpenCL kernel
│   ├── manifest.json               # KernelCacheService manifest
│   └── bin/                        # On-disk binary cache (создаётся при первом запуске)
│       ├── fir_filter_cf32_opencl.bin
│       └── iir_filter_cf32_opencl.bin
├── src/
│   ├── fir_filter.cpp              # FirFilter implementation
│   ├── iir_filter.cpp              # IirFilter implementation
│   ├── fir_filter_rocm.cpp         # FirFilterROCm (hiprtc, Linux only)
│   └── iir_filter_rocm.cpp         # IirFilterROCm (hiprtc, Linux only)
└── tests/
    ├── all_test.hpp                # Entry point: filters_all_test::run()
    ├── test_fir_basic.hpp          # OpenCL FIR test (kTestFirCoeffs64)
    ├── test_iir_basic.hpp          # OpenCL IIR test (Butterworth 2nd order)
    ├── test_filters_rocm.hpp       # ROCm: 6 тестов (Linux + AMD GPU)
    ├── filters_benchmark.hpp       # OpenCL benchmark classes (FirFilterBenchmark, IirFilterBenchmark)
    ├── test_filters_benchmark.hpp  # OpenCL benchmark runner (test_filters_benchmark::run())
    ├── filters_benchmark_rocm.hpp  # ROCm benchmark classes (FirFilterROCmBenchmark, IirFilterROCmBenchmark)
    ├── test_filters_benchmark_rocm.hpp  # ROCm benchmark runner
    └── README.md                   # Tests overview

Python_test/filters/
├── test_filters_stage1.py          # FIR + IIR vs scipy (5 тестов)
├── test_fir_filter_rocm.py         # FirFilterROCm: 5 тестов (Linux)
├── test_iir_filter_rocm.py         # IirFilterROCm: multi-section, GPU ptr
├── test_iir_plot.py                # IIR order 2/4/8 сравнение (графики)
├── test_ai_filter_pipeline.py      # Stage 3: NL → AI → scipy → GPU → plot
└── test_ai_fir_demo.py             # AI demo

Doc/Modules/filters/
├── Full.md                         # Этот файл
├── Quick.md                        # Краткий справочник
├── README.md                       # Overview
└── gpu_filters_research.md         # Исследование: Overlap-Save/Add, tiled FIR, будущие алгоритмы
```

---

## 13. Важные нюансы

| # | Нюанс |
|---|-------|
| ⚠️ | **`result.data` — caller owns**: `clReleaseMemObject(result.data)` (OpenCL) или `hipFree(result.data)` (ROCm). Не забыть! |
| ⚠️ | **GPU IIR одноканальный — медленнее CPU**: Single-channel IIR вычисляется лучше на CPU. GPU оправдан только при ≥8 каналах. |
| ⚠️ | **`-cl-fast-relaxed-math`**: точность GPU ~1e-4..1e-6, а не IEEE 754. Если нужна точность — убрать флаг. |
| ⚠️ | **`SetCoefficients` vs `SetBiquadSections`**: вызов загружает буфер на GPU немедленно. При повторном вызове предыдущий буфер освобождается и создаётся новый. |
| ⚠️ | **`kMaxConstantTaps = 16000`**: при num_taps > 16000 автоматически переключается на `fir_filter_cf32_global` (глобальная память — медленнее). |
| ⚠️ | **IIR boundary conditions**: нулевые начальные условия (w1=w2=0). Первые `num_taps/order` отсчётов — переходный процесс. |
| ⚠️ | **SOS формат scipy**: `sos = butter(N, Wn, output='sos')`. Row: `[b0, b1, b2, a0, a1, a2]`, но `a0=1` пропускается. Передавать `a1=row[4], a2=row[5]`. |
| ⚠️ | **ROCm FirFilter**: компиляция hiprtc занимает ~100–500 мс при первом запуске. На Windows — compile-only stub (throws). |
| ⚠️ | **CL_QUEUE_PROFILING_ENABLE**: для бенчмарков OpenCL обязателен флаг при создании queue. Иначе `cl_event` timing вернёт 0. |
| ⚠️ | **Granichnye условия FIR**: sample `n-k < 0` считается 0 (causal). Результат первых `num_taps-1` семплов отличается от `scipy.lfilter` не из-за бага, а из-за метода (`lfilter` тоже нулевые IC). Расхождение мало (< 1e-3). |

---

## 14. Ссылки

### Статьи и стандарты

| Источник | Описание |
|----------|----------|
| [SciPy firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html) | FIR design (Parks-McClellan, windowed sinc) |
| [SciPy butter + SOS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html) | Butterworth IIR, SOS output |
| [Direct Form II Transposed](https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2_transposed) | Biquad DFII-T — численно стабильная форма |
| [OpenCL __constant memory](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html) | Limits: device-dependent (~64 KB типично) |

### Локальная документация

| Файл | Описание |
|------|----------|
| [Quick.md](Quick.md) | Краткий справочник (шпаргалка) |
| [gpu_filters_research.md](gpu_filters_research.md) | Overlap-Save/Add, tiled FIR, будущие алгоритмы |
| [Doc/DrvGPU/Services/Full.md](../../DrvGPU/Services/Full.md) | KernelCacheService, FilterConfigService |

### Out of Scope (Post-MVP)

| Фича | Когда |
|------|-------|
| Overlap-Save/Overlap-Add | После стабильного FIR. См. `gpu_filters_research.md` |
| Адаптивные LMS/NLMS/RLS | Отдельная задача |
| Полифазные фильтры / децимация | Отдельная задача |
| Stage 2: text→kernel кэш | После Stage 1 |
| ROCm HSACO disk cache (для FIR/IIR) | После AMD GPU тестирования |

---

*Обновлено: 2026-03-02*
