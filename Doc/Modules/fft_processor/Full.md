# FFT Processor — Полная документация

> GPU FFT с вариантами вывода (Complex / MagPhase / MagPhaseFreq)

**Namespace**: `fft_processor`
**Каталог**: `modules/fft_processor/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Pipeline и zero-padding](#2-pipeline-и-zero-padding)
3. [Режимы вывода](#3-режимы-вывода)
4. [API (C++ и Python)](#4-api)
5. [Тесты](#5-тесты)
6. [Важные замечания](#6-важные-замечания)
7. [Ссылки](#7-ссылки)

---

## 1. Обзор и назначение

`FFTProcessor` возвращает **полный спектр** в нужном формате. Отдельно: `SpectrumMaximaFinder` (модуль fft_maxima) ищет максимумы.

| Класс | Назначение |
|-------|------------|
| **FFTProcessor** | FFT → полный спектр (complex / mag+phase / mag+phase+freq) |
| **SpectrumMaximaFinder** | FFT → поиск пиков (1, 2 или все максимумы) |

### Реализовано

- [x] FFT для размеров 2^n (n = 8..24)
- [x] IFFT (через clFFT)
- [x] Zero-padding (pre-callback)
- [x] Режимы: COMPLEX, MAGNITUDE_PHASE, MAGNITUDE_PHASE_FREQ
- [x] Batch (несколько лучей за вызов)
- [x] Профилирование через GPUProfiler

### Планируется

- [ ] Оконные функции (Hann, Hamming, Blackman, Kaiser)
- [ ] Real-to-Complex FFT (R2C)
- [ ] ROCm (hipFFT)

---

## 2. Pipeline и zero-padding

```
Input → [32B header][data] → Pre-Callback (zero-pad) → clFFT → [Post-Kernel: mag+phase] → Output
```

### Zero-padding

Входные данные `n_point` приводятся к `nFFT`:

$$
nFFT = \text{nextPowerOf2}(n\_point) \times repeat\_count
$$

Пример: `n_point=1000`, `repeat_count=2` → `nFFT = 1024 × 2 = 2048`

Pre-callback копирует данные и дополняет нулями до `nFFT`. Заголовок 32 байта: `beam_count`, `count_points`, `nFFT`.

### Архитектура

```
Input (host или cl_mem)
      │
      ▼
┌─────────────────────┐
│  Upload / Copy GPU  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Pre-Callback        │  ← zero-pad, 32B header
│ (clFFT user data)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   clFFT (1D C2C)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ MagPhase Kernel     │  ← если output_mode ≠ COMPLEX
│ (|FFT|, phase, freq)│
└──────────┬──────────┘
           │
           ▼
    Results (host)
```

---

## 3. Режимы вывода

| Режим | Выход | Использование |
|-------|-------|---------------|
| **COMPLEX** | `vector<complex<float>>` | Сырой спектр, дальнейшая обработка |
| **MAGNITUDE_PHASE** | `magnitude[]`, `phase[]` | Амплитуда и фаза |
| **MAGNITUDE_PHASE_FREQ** | `magnitude[]`, `phase[]`, `frequency[]` | + частоты в Hz: `freq[k] = k * fs / nFFT` |

---

## 4. API

### C++

```cpp
#include "fft_processor.hpp"

FFTProcessor fft(backend);
FFTProcessorParams params;
params.beam_count = 256;
params.n_point = 1024;
params.sample_rate = 1000.0f;
params.output_mode = FFTOutputMode::MAGNITUDE_PHASE_FREQ;

// CPU данные
auto results = fft.ProcessComplex(data, params);
auto results2 = fft.ProcessMagPhase(data, params);

// GPU данные (cl_mem)
auto results3 = fft.ProcessComplex(gpu_buf, params, gpu_bytes);
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)

# process_complex(data, beam_count, n_point, sample_rate)
results = fft.process_complex(signal, beam_count=8, n_point=1024, sample_rate=50000.0)

# process_mag_phase(data, beam_count, n_point, sample_rate, include_freq=True)
results = fft.process_mag_phase(signal, 8, 1024, 50000.0, include_freq=True)
# results[i].magnitude, .phase, .frequency (если include_freq)
```

### FFTProcessorParams

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `beam_count` | uint32_t | 1 | Количество лучей (каналов) |
| `n_point` | uint32_t | 0 | Входных точек на луч |
| `sample_rate` | float | 1000.0f | Частота дискретизации (Hz) |
| `output_mode` | FFTOutputMode | COMPLEX | Режим вывода |
| `repeat_count` | uint32_t | 1 | Множитель nFFT |
| `memory_limit` | float | 0.80f | Лимит GPU памяти для batch (0.0–1.0) |

---

## 5. Тесты

**Файлы**: `modules/fft_processor/tests/test_fft_processor.hpp`, `test_fft_vs_cpu.hpp`  
**Вызов**: через `all_test.hpp` из `main.cpp` → `fft_processor_all_test::run()`

| Тест | Что проверяет |
|------|---------------|
| **test_fft_processor** | Режимы Complex, MagPhase |
| **test_fft_vs_cpu** | GPU vs CPU reference (pocketfft) |

**Примечание**: тесты могут быть закомментированы в `all_test.hpp`.

---

## 6. Важные замечания

### FFT Plan Caching

FFTProcessor кеширует clFFT plan для текущего размера. **При изменении `n_point` — создать новый экземпляр**.

```cpp
// ПРАВИЛЬНО
FFTProcessor fft_4096(backend);
FFTProcessor fft_8192(backend);

// НЕПРАВИЛЬНО: один экземпляр для разных размеров
fft.ProcessComplex(data_4096, params_4096);  // OK
fft.ProcessComplex(data_8192, params_8192);  // CL_INVALID_VALUE
```

### GPU Memory

- CPU данные копируются на GPU внутри `ProcessXxx`
- GPU данные (`cl_mem`) используются напрямую, ownership не передаётся
- Результаты всегда возвращаются на CPU

### Метрики (ориентир)

| Операция | Размер | GPU | Время |
|----------|--------|-----|-------|
| FFT | 1M points | RX 7900 XTX | ~0.5 ms |
| FFT | 4M points | RX 7900 XTX | ~2 ms |

---

## 7. Ссылки

| Источник | Описание |
|----------|----------|
| [clFFT GitHub](https://github.com/clMathLibraries/clFFT) | OpenCL FFT библиотека |
| [rocFFT Docs](https://rocm.docs.amd.com/projects/rocFFT/) | ROCm FFT (planned) |
| [API.md](API.md) | Детальный API reference |

---

## Файлы модуля

```
modules/fft_processor/
├── include/
│   ├── fft_processor.hpp
│   ├── fft_processor_types.hpp
│   ├── kernels/
│   │   └── fft_processor_kernels.hpp
│   └── types/
│       ├── fft_modes.hpp
│       ├── fft_params.hpp
│       ├── fft_results.hpp
│       └── fft_types.hpp
├── src/
│   └── fft_processor.cpp
└── tests/
    ├── all_test.hpp
    ├── test_fft_processor.hpp
    └── test_fft_vs_cpu.hpp
```

---

*Обновлено: 2026-02-18*
