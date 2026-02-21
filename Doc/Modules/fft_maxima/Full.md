# FFT Maxima — Полная документация

> Поиск максимумов спектра FFT (SpectrumMaximaFinder)

**Namespace**: `antenna_fft`
**Каталог**: `modules/fft_maxima/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Содержание

1. [Обзор и назначение](#1-обзор-и-назначение)
2. [Три режима работы](#2-три-режима-работы)
3. [InputData и типы данных](#3-inputdata-и-типы-данных)
4. [API (C++ и Python)](#4-api)
5. [Pipeline и алгоритмы](#5-pipeline-и-алгоритмы)
6. [Тесты](#6-тесты)
7. [Ссылки](#7-ссылки)

---

## 1. Обзор и назначение

`SpectrumMaximaFinder` ищет **максимумы** (пики) в FFT-спектре для массива антенн/лучей. Отдельно от `FFTProcessor`, который возвращает полный спектр.

| Класс | Назначение |
|-------|------------|
| **FFTProcessor** | FFT → полный спектр (complex / mag+phase) |
| **SpectrumMaximaFinder** | FFT → поиск пиков (1, 2 или все максимумы) |

### Реализовано

- [x] **Process** — один/два пика (ONE_PEAK, TWO_PEAKS) с параболической интерполяцией
- [x] **FindAllMaxima** — полный pipeline: сырой сигнал → FFT → все локальные максимумы
- [x] **AllMaxima** — поиск в готовом FFT (без FFT)
- [x] **InputData\<T\>** — CPU (vector), GPU (cl_mem)
- [x] Batch processing через BatchManager
- [x] Профилирование через GPUProfiler

### Планируется

- [ ] InputData\<void*\> — SVM (zero-copy)
- [ ] ROCm backend (hipFFT)

---

## 2. Три режима работы

### Process (ONE_PEAK / TWO_PEAKS)

Поиск 1 или 2 пиков с параболической интерполяцией. Результат: `SpectrumResult` с `refined_frequency`, `magnitude`.

```
Input → Zero-Pad → clFFT → Post-Kernel (max + parabolic) → Results
```

### FindAllMaxima

Полный pipeline: **сырой сигнал** → FFT → Detect → Scan → Compact → все локальные максимумы.

```
Input → Zero-Pad → clFFT → |FFT| → Detect → Prefix Sum → Compact → AllMaximaResult
```

### AllMaxima

Только detect+scan+compact. **Данные уже FFT!** (input.n_point = nFFT)

```
Input (FFT spectrum) → |FFT| → Detect → Scan → Compact → AllMaximaResult
```

### Когда что использовать

| Метод | Вход | Когда |
|-------|------|-------|
| **FindAllMaxima** | Сырой сигнал | Нужно и FFT, и поиск пиков в одном вызове |
| **AllMaxima** | FFT-спектр | FFT уже сделан (FFTProcessor, другой код), нужен только поиск пиков |

#### Пример: FindAllMaxima — сырой сигнал

```cpp
// Сырой сигнал (time domain) — FFT делается внутри
std::vector<std::complex<float>> raw_signal = GenerateSignal();  // 64 × 512 точек

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 64,
    .n_point = 512,
    .data = raw_signal,   // ← Сырые данные!
    .sample_rate = 1000.0f
};

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
// Pipeline: Upload → FFT → Detect → Scan → Compact
```

#### Пример: AllMaxima — спектр уже есть

```cpp
// FFT уже посчитан (FFTProcessor или другой источник)
cl_mem fft_spectrum = fft.ProcessComplex(...);  // или от CwGenerator + FFT

InputData<cl_mem> input{
    .antenna_count = 5,
    .n_point = 1024,       // ← Размер спектра (nFFT)!
    .data = fft_spectrum,  // ← Комплексный спектр!
    .sample_rate = 1000.0f
};

auto result = finder.AllMaxima(input, OutputDestination::CPU);
// Pipeline: |FFT| → Detect → Scan → Compact (без FFT!)
```

---

## 3. InputData и типы данных

### Структура

```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;     // Количество лучей
    uint32_t n_point = 0;           // Точек на луч (или nFFT для AllMaxima!)
    T data{};                       // Данные
    size_t gpu_memory_bytes = 0;   // Размер GPU буфера (для cl_mem)

    uint32_t repeat_count = 2;      // nFFT = nextPow2(n_point) × repeat_count
    float sample_rate = 1000.0f;    // Гц
    uint32_t search_range = 0;      // 0 = auto = nFFT/4
    float memory_limit = 0.80f;     // Доля GPU памяти для batch (0.0-1.0)
    size_t max_maxima_per_beam = 1000;  // Лимит максимумов на луч (FindAllMaxima)
};
```

### Типы T

| T | Источник | Когда использовать |
|---|----------|-------------------|
| `std::vector<std::complex<float>>` | CPU | Upload → GPU |
| `cl_mem` | GPU | Данные уже на GPU (zero-copy) |
| `void*` | SVM | TODO |

---

## 4. API

### C++ — Process (1/2 пика)

```cpp
#include "spectrum_maxima_finder.h"
#include "interface/spectrum_input_data.hpp"

SpectrumMaximaFinder finder(backend);
finder.Initialize();  // Автоматически при первом Process

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1024,
    .data = my_signal,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
// results[i].interpolated.refined_frequency, .magnitude
```

### C++ — FindAllMaxima (все пики)

```cpp
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 64,
    .n_point = 512,
    .data = raw_signal,   // Сырой сигнал!
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f
};

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
// result.beams[i].maxima — vector<MaxValue> (index, real, imag, magnitude, phase, refined_frequency)
```

### C++ — AllMaxima (FFT уже посчитан)

```cpp
InputData<cl_mem> fft_input{
    .antenna_count = 5,
    .n_point = 1024,       // = nFFT!
    .data = gpu_fft_result,
    .sample_rate = 1000.0f
};

auto result = finder.AllMaxima(fft_input, OutputDestination::CPU);
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# FFT → поиск всех максимумов
spectrum = fft.process_complex(signal, sample_rate=fs)
result = finder.find_all_maxima(spectrum, sample_rate=fs)
# result['frequencies'], result['magnitudes'], result['positions']
```

См. [Doc/Python/spectrum_maxima_api.md](../../Python/spectrum_maxima_api.md)

---

## 5. Pipeline и алгоритмы

### Process (ONE_PEAK / TWO_PEAKS)

1. PrepareParams() — nFFT, base_fft
2. Upload (CPU) или Copy (GPU)
3. Pre-callback: zero-pad n_point → nFFT
4. clFFT batched
5. Post-kernel: max(|FFT|²) + параболическая интерполяция
6. ReadResults() → vector\<SpectrumResult\>

### FindAllMaxima (полный pipeline)

1. PrepareParams() с max_maxima_per_beam
2. Upload/Copy → FFT (pre+post callback: |FFT|)
3. **Detect**: локальные максимумы (mag[i] > mag[i-1] && mag[i] > mag[i+1])
4. **Prefix Sum** (Blelloch Scan): beam-aware
5. **Compact**: stream compaction → позиции и амплитуды
6. Output: CPU (beams[].maxima) или GPU (gpu_maxima, gpu_counts)

### Batch-обработка

- BatchManager разбивает при нехватке памяти (memory_limit)
- Dest=GPU: общие буферы gpu_maxima, gpu_counts; caller обязан clReleaseMemObject
- Профилирование: GPUProfiler по всем batch

### MaxValue (формат результата)

```cpp
struct MaxValue {
    uint32_t index;             // Бин в спектре
    float real, imag;           // Re/Im FFT
    float magnitude, phase;     // |FFT|, arg(FFT)
    float refined_frequency;    // Частота в Hz (параболическая интерполяция)
};
```

Подробнее: [FindAllMaxima_MaxValue_Guide.md](FindAllMaxima_MaxValue_Guide.md)

---

## 6. Тесты

**Файлы**: `modules/fft_maxima/tests/`  
**Вызов**: через `all_test.hpp` из `main.cpp`

| Тест | Что проверяет |
|------|---------------|
| **test_spectrum_maxima** | ONE_PEAK, TWO_PEAKS, параболическая интерполяция |
| **test_find_all_maxima** | FindAllMaxima, AllMaxima (CPU/GPU, 7 тестов) |
| **test_batch_all_maxima** | Batch: CPU/GPU input, Dest CPU/GPU |
| **test_large_batch** | 256 × 1.3M точек, автоматическая разбивка |
| **test_gpu_generator_integration** | CwGenerator → SpectrumMaximaFinder |
| **test_benchmark_all_maxima** | 10 лучей × 500k, GPUProfiler |
| **test_fft_maxima** | Legacy (AntennaFFTRelease) |

См. [modules/fft_maxima/tests/README.md](../../../modules/fft_maxima/tests/README.md)

### Производительность (RTX 2080 Ti)

| Конфигурация | Время |
|-------------|-------|
| 1 луч × 1024 FFT | ~0.03 мс |
| 5 лучей × 512 FFT | ~0.09 мс |
| 256 лучей × 4096 FFT | ~56 мс |
| 256 лучей FindAllMax (10 GPU) | ~50–75 мс |

---

## 7. Ссылки

### Локальная документация

| Файл | Описание |
|------|----------|
| [FindAllMaxima_MaxValue_Guide.md](FindAllMaxima_MaxValue_Guide.md) | Формат MaxValue, CPU/GPU пути |
| [API.md](API.md) | Детальный API reference |
| [README.md](README.md) | Краткий обзор |

### Внешние

| Источник | Описание |
|----------|----------|
| [FFT Processor](../fft_processor/Full.md) | FFTProcessor (полный спектр) |
| [Python API](../../Python/spectrum_maxima_api.md) | gpuworklib.SpectrumMaximaFinder |
| [clFFT](https://github.com/clMathLibraries/clFFT) | OpenCL FFT |

### Исследование (MemoryBank)

| Файл | Описание |
|------|----------|
| `DiscussionPlan/~1. FFT_FindAllMax/` | Custom Kernel + Scan-based Compaction |

---

## Файлы модуля

```
modules/fft_maxima/
├── include/
│   ├── spectrum_maxima_finder.h
│   ├── interface/
│   │   ├── spectrum_input_data.hpp
│   │   ├── spectrum_maxima_types.h
│   │   ├── i_spectrum_processor.hpp
│   │   └── i_all_maxima_pipeline.hpp
│   ├── processors/
│   │   ├── spectrum_processor_opencl.hpp
│   │   └── spectrum_processor_rocm.hpp
│   ├── pipelines/
│   │   └── all_maxima_pipeline_opencl.hpp
│   ├── kernels/
│   │   ├── fft_kernel_sources.hpp
│   │   └── all_maxima_kernel_sources.hpp
│   ├── fft_plan_cache.hpp
│   ├── fft_batch_adapter.hpp
│   └── antenna_fft_core.h, antenna_fft_release.h  # Legacy
├── src/
│   ├── spectrum_maxima_finder.cpp
│   ├── spectrum_maxima_finder_process.cpp
│   ├── spectrum_maxima_finder_all_maxima.cpp
│   ├── spectrum_processor_opencl.cpp
│   ├── all_maxima_pipeline_opencl.cpp
│   └── antenna_fft_core.cpp, antenna_fft_release.cpp
├── kernels/
│   └── fft_kernels.cl
└── tests/
    ├── all_test.hpp
    ├── test_spectrum_maxima.hpp
    ├── test_find_all_maxima.hpp
    ├── test_batch_all_maxima.hpp
    └── ...
```

---

*Обновлено: 2026-02-18*
