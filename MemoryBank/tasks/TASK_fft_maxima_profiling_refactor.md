# TASK: Рефакторинг профилирования fft_maxima → GpuBenchmarkBase

> **Статус**: 📋 PLAN
> **Дата создания**: 2026-03-01
> **Приоритет**: Высокий
> **Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`
> **Референс**: `modules/fft_processor/` (OpenCL)

---

## Проблема

Текущий `modules/fft_maxima` нарушает принцип "production-класс чистый":

| Файл | Нарушение |
|------|-----------|
| `src/spectrum_maxima_finder.cpp` | `RecordProfilingEvent` в `ProcessBatch()` — 3 вызова |
| `src/spectrum_maxima_finder_process.cpp` | `RecordProfilingEvent` в `ProcessFromGPU()`, `ProcessBatchFromGPU()` — 4 вызова |
| `src/spectrum_maxima_finder_all_maxima.cpp` | `RecordProfilingEvent` в AllMaxima pipeline — **8 вызовов** |
| `tests/test_benchmark_all_maxima.hpp` | СТАРЫЙ API: прямой GPUProfiler.Start/Stop, sleep_for, не GpuBenchmarkBase |

**Всего RecordProfilingEvent в production**: 15 вызовов — все нужно убрать.

---

## Цель

Привести `fft_maxima` к стандарту `fft_processor`:

```
Production-класс (SpectrumMaximaFinder)   Benchmark-класс (тест)
────────────────────────────────────       ─────────────────────────────
Process()        → prof_events=nullptr     SpectrumMaximaFinderBenchmark
FindAllMaxima()  → prof_events=nullptr       : GpuBenchmarkBase
                 → prof_events=&vec           ExecuteKernel() / ExecuteKernelTimed()
```

---

## Архитектура изменений

### Публичные методы с prof_events (после рефакторинга):

```cpp
// --- Process (ONE_PEAK) ---
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::ROCm,
    ProfEvents* prof_events = nullptr);   // ← НОВЫЙ параметр

// --- FindAllMaxima (ALL_MAXIMA pipeline) ---
template<typename T>
AllMaximaResult FindAllMaxima(
    const InputData<T>& input,
    OutputDestination dest = OutputDestination::CPU,
    DriverType driver = DriverType::OPENCL,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    ProfEvents* prof_events = nullptr);   // ← НОВЫЙ параметр

// --- FindAllMaxima (cl_mem) ---
AllMaximaResult FindAllMaxima(
    cl_mem fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 0, uint32_t search_end = 0,
    uint32_t beam_offset = 0,
    cl_mem external_out_maxima = nullptr, cl_mem external_out_counts = nullptr,
    ProfEvents* prof_events = nullptr);   // ← НОВЫЙ параметр
```

### Тип ProfEvents:
```cpp
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;
```

### Паттерн CollectOrRelease (как в FFTProcessor):
```cpp
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) prof_events->push_back({name, ev});
    else clReleaseEvent(ev);
}
```

---

## ТАСКИ (по порядку выполнения)

---

### TASK 1 — Модифицировать заголовок `spectrum_maxima_finder.h`

**Файл**: `modules/fft_maxima/include/spectrum_maxima_finder.h`

#### 1.1 Добавить ProfEvents и CollectOrRelease

В начале класса (после include-ов) добавить:
```cpp
// Тип для сбора cl_event'ов (используется только тестами через prof_events*)
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;
```

В private секции добавить static helper:
```cpp
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) {
        prof_events->push_back({name, ev});
    } else {
        clReleaseEvent(ev);
    }
}
```

#### 1.2 Добавить prof_events к Process<T> (шаблон в заголовке)

```cpp
// В декларации:
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::ROCm,
    ProfEvents* prof_events = nullptr);   // ← добавить
```

В реализации шаблона (также в .h файле) — передать prof_events дальше:
```cpp
// В Process<T>():
if constexpr (is_cpu_vector_v<T>) {
    if (!initialized_) Initialize();
    return ProcessFromCPU(input.data, prof_events);  // ← передать
}
else if constexpr (std::is_same_v<T, cl_mem>) {
    return ProcessFromGPU(input.data, input.antenna_count, input.n_point,
                          input.ActualGpuMemory(), prof_events);  // ← передать
}
```

#### 1.3 Добавить prof_events к FindAllMaxima<T> (шаблон в заголовке)

```cpp
template<typename T>
AllMaximaResult FindAllMaxima(
    const InputData<T>& input,
    OutputDestination dest = OutputDestination::CPU,
    DriverType driver = DriverType::OPENCL,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    ProfEvents* prof_events = nullptr);  // ← добавить
```

В реализации шаблона:
```cpp
if constexpr (is_cpu_vector_v<T>) {
    if (!initialized_) Initialize();
    return FindAllMaximaFromCPU(input.data, dest, search_start, search_end, prof_events);
}
else if constexpr (std::is_same_v<T, cl_mem>) {
    return FindAllMaximaFromGPUPipeline(input.data, ..., dest, search_start, search_end, prof_events);
}
```

#### 1.4 Добавить prof_events к FindAllMaxima(cl_mem, ...) — публичный нешаблонный

```cpp
AllMaximaResult FindAllMaxima(
    cl_mem fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 0, uint32_t search_end = 0,
    uint32_t beam_offset = 0,
    cl_mem external_out_maxima = nullptr, cl_mem external_out_counts = nullptr,
    ProfEvents* prof_events = nullptr);  // ← добавить
```

#### 1.5 Обновить объявления приватных методов

Добавить `ProfEvents*` к:
- `ProcessFromCPU(data, prof_events*)`
- `ProcessFromGPU(gpu_data, count, n_point, bytes, prof_events*)`
- `ProcessBatchFromGPU(gpu_data, offset, start, count, prof_events*)`
- `ProcessBatch(data, start, count, prof_events*)`
- `FindAllMaximaFromCPU(data, dest, search_start, search_end, prof_events*)`
- `FindAllMaximaFromGPUPipeline(data, count, n_point, bytes, dest, search_start, search_end, prof_events*)`
- `AllMaximaFromCPU(data, count, nFFT, rate, dest, search_start, search_end, prof_events*)`

---

### TASK 2 — Очистить `spectrum_maxima_finder.cpp` (Process path)

**Файл**: `src/spectrum_maxima_finder.cpp`

Затронутые методы: `ProcessBatch()`

#### 2.1 Изменить сигнатуру ProcessBatch

```cpp
// БЫЛО:
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatch(
    const std::vector<std::complex<float>>& data,
    size_t start_antenna, size_t batch_antenna_count);

// СТАЛО:
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatch(
    const std::vector<std::complex<float>>& data,
    size_t start_antenna, size_t batch_antenna_count,
    ProfEvents* prof_events);
```

#### 2.2 Заменить RecordProfilingEvent → CollectOrRelease в ProcessBatch

```cpp
// БЫЛО (3 места):
drv_gpu_lib::RecordProfilingEvent(upload_event, gpu_id, "SpectrumMaxima", "Upload");
clReleaseEvent(upload_event);
// ...
drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");
clReleaseEvent(fft_event);
// ...
drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");
clReleaseEvent(post_event);

// СТАЛО:
CollectOrRelease(upload_event, "Upload", prof_events);
// clReleaseEvent убрать — CollectOrRelease сам решает
CollectOrRelease(fft_event, "FFT", prof_events);
CollectOrRelease(post_event, "PostKernel", prof_events);
```

⚠️ **Важно**: `CollectOrRelease` сам вызывает `clReleaseEvent` если `prof_events = nullptr`.
Поэтому отдельный `clReleaseEvent` после нужно УБРАТЬ.

#### 2.3 Удалить include gpu_profiler.hpp из .cpp (если больше не нужен)

Проверить после чистки: если `RecordProfilingEvent` больше не используется — удалить:
```cpp
// Удалить эту строку из .cpp если больше не используется:
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
```

---

### TASK 3 — Очистить `spectrum_maxima_finder_process.cpp` (ProcessFromGPU path)

**Файл**: `src/spectrum_maxima_finder_process.cpp`

#### 3.1 Изменить сигнатуры ProcessFromCPU, ProcessFromGPU, ProcessBatchFromGPU

```cpp
// ProcessFromCPU — добавить prof_events (передаётся дальше в ProcessBatch)
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    ProfEvents* prof_events);

// ProcessFromGPU — добавить prof_events
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromGPU(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    ProfEvents* prof_events);

// ProcessBatchFromGPU — добавить prof_events
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatchFromGPU(
    cl_mem gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count,
    ProfEvents* prof_events);
```

#### 3.2 Заменить RecordProfilingEvent → CollectOrRelease

В `ProcessFromGPU()` (single batch, non-batch path):
```cpp
// БЫЛО:
drv_gpu_lib::RecordProfilingEvent(copy_event, gpu_id, "SpectrumMaxima", "GPU→GPU Copy");
// ...
drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");
// ...
drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");

// СТАЛО:
CollectOrRelease(copy_event, "GPU→GPU Copy", prof_events);
CollectOrRelease(fft_event, "FFT", prof_events);
CollectOrRelease(post_event, "PostKernel", prof_events);
```

В `ProcessBatchFromGPU()` — аналогично (3 вызова).

В `ProcessFromCPU()` — передать prof_events в `ProcessBatch(data, batch.start, batch.count, prof_events)`.

#### 3.3 Удалить неиспользуемые include-ы gpu_profiler.hpp

---

### TASK 4 — Очистить `spectrum_maxima_finder_all_maxima.cpp` (AllMaxima path)

**Файл**: `src/spectrum_maxima_finder_all_maxima.cpp`

Это самый сложный файл — **8 RecordProfilingEvent вызовов** в 4 методах.

#### Схема propagation prof_events через AllMaxima pipeline:

```
FindAllMaxima<T>()
  → FindAllMaximaFromCPU(data, dest, start, end, prof_events*)
      UploadData() → CollectOrRelease("Upload", prof_events)
      ExecuteAllMaximaFFT() → CollectOrRelease("FFT+PostCallback", prof_events)
      FindAllMaxima(fft_data, count, nFFT, rate, dest, start, end, 0, null, null, prof_events*)
          → CollectOrRelease("Detect", prof_events)
          → CollectOrRelease("Scan", prof_events)
          → CollectOrRelease("Compact", prof_events)

  → FindAllMaximaFromGPUPipeline(data, count, n_point, bytes, dest, start, end, prof_events*)
      clEnqueueCopyBuffer() → CollectOrRelease("GPU→GPU Copy", prof_events)
      ExecuteAllMaximaFFT() → CollectOrRelease("FFT+PostCallback", prof_events)
      FindAllMaxima(fft_data, ..., prof_events*)
```

#### 4.1 Изменить сигнатуры

```cpp
void SpectrumMaximaFinder::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events);   // ← добавить

AllMaximaResult SpectrumMaximaFinder::FindAllMaximaFromGPUPipeline(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events);   // ← добавить

// Нешаблонный публичный FindAllMaxima(cl_mem, ...)
AllMaximaResult SpectrumMaximaFinder::FindAllMaxima(
    cl_mem fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest,
    uint32_t search_start, uint32_t search_end,
    uint32_t beam_offset,
    cl_mem external_out_maxima, cl_mem external_out_counts,
    ProfEvents* prof_events);   // ← добавить
```

#### 4.2 Заменить все RecordProfilingEvent → CollectOrRelease (8 мест)

```
"AllMaxima/Upload"          → CollectOrRelease("Upload", prof_events)
"AllMaxima/FFT+PostCallback"→ CollectOrRelease("FFT+PostCallback", prof_events)
"AllMaxima/Upload_Batch"    → CollectOrRelease("Upload", prof_events)  (batch path)
"AllMaxima/FFT_Batch"       → CollectOrRelease("FFT", prof_events)     (batch path)
"AllMaxima/GPU→GPU Copy"    → CollectOrRelease("GPU→GPU Copy", prof_events)
"AllMaxima/GPU→GPU Copy_Batch" → CollectOrRelease("GPU→GPU Copy", prof_events)
"AllMaxima/ComputeMagnitudes" → CollectOrRelease("ComputeMagnitudes", prof_events)
"AllMaxima/Detect"          → CollectOrRelease("Detect", prof_events)
"AllMaxima/Scan"            → CollectOrRelease("Scan", prof_events)
"AllMaxima/Compact"         → CollectOrRelease("Compact", prof_events)
```

---

### TASK 5 — Создать `tests/fft_maxima_benchmark.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/fft_maxima_benchmark.hpp`

Два класса в одном файле:

#### 5.1 SpectrumMaximaFinderBenchmark (Process / ONE_PEAK)

```cpp
#pragma once
#include "spectrum_maxima_finder.h"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <complex>
#include <vector>

namespace test_fft_maxima {

/// Бенчмарк для Process<T> (ONE_PEAK path): Upload → FFT → PostKernel → Download
class SpectrumMaximaFinderBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumMaximaFinderBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumMaximaFinder& proc,
      const antenna_fft::InputData<std::vector<std::complex<float>>>& input,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_Process"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_Process", cfg),
      proc_(proc), input_(input) {}

protected:
  void ExecuteKernel() override {
    // Warmup без timing
    proc_.Process(input_);
  }

  void ExecuteKernelTimed() override {
    antenna_fft::SpectrumMaximaFinder::ProfEvents events;
    proc_.Process(input_,
                  antenna_fft::PeakSearchMode::ONE_PEAK,
                  antenna_fft::DriverType::OPENCL,
                  &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  antenna_fft::SpectrumMaximaFinder& proc_;
  antenna_fft::InputData<std::vector<std::complex<float>>> input_;
};

}  // namespace test_fft_maxima
```

#### 5.2 SpectrumMaximaAllMaximaBenchmark (FindAllMaxima pipeline)

```cpp
namespace test_fft_maxima {

/// Бенчмарк для FindAllMaxima (full pipeline): Upload→FFT→Detect→Scan→Compact
class SpectrumMaximaAllMaximaBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumMaximaAllMaximaBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumMaximaFinder& proc,
      const antenna_fft::InputData<std::vector<std::complex<float>>>& input,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_AllMaxima", cfg),
      proc_(proc), input_(input) {}

protected:
  void ExecuteKernel() override {
    proc_.FindAllMaxima(input_,
                        antenna_fft::OutputDestination::CPU,
                        antenna_fft::DriverType::OPENCL);
  }

  void ExecuteKernelTimed() override {
    antenna_fft::SpectrumMaximaFinder::ProfEvents events;
    proc_.FindAllMaxima(input_,
                        antenna_fft::OutputDestination::CPU,
                        antenna_fft::DriverType::OPENCL,
                        0, 0,
                        &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  antenna_fft::SpectrumMaximaFinder& proc_;
  antenna_fft::InputData<std::vector<std::complex<float>>> input_;
};

}  // namespace test_fft_maxima
```

---

### TASK 6 — Создать `tests/test_fft_maxima_benchmark.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/test_fft_maxima_benchmark.hpp`

```cpp
#pragma once
/**
 * @file test_fft_maxima_benchmark.hpp
 * @brief Бенчмарк SpectrumMaximaFinder через GpuBenchmarkBase
 *
 * Запускает:
 *  - Process (ONE_PEAK):      5 warmup + 20 runs → Upload/FFT/PostKernel
 *  - FindAllMaxima (pipeline): 5 warmup + 20 runs → Upload/FFT/Detect/Scan/Compact
 *
 * Результаты: Results/Profiler/GPU_00_SpectrumMaxima_Process/
 *             Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/
 */
#include "fft_maxima_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include <CL/cl.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_maxima_benchmark {

// Генерация тестовых данных (beam_count × n_point)
inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t beam_count, uint32_t n_point, float sample_rate)
{
    std::vector<std::complex<float>> data(
        static_cast<size_t>(beam_count) * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
        float freq = 50.0f + b * 30.0f;
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI)
                                 * freq * t / sample_rate);
            data[static_cast<size_t>(b) * n_point + t] = {val, 0.0f};
        }
    }
    return data;
}

inline int run() {
    std::cout << "\n"
              << "============================================================\n"
              << "  SpectrumMaximaFinder Benchmark (GpuBenchmarkBase)\n"
              << "============================================================\n";

    try {
        // ── OpenCL init ────────────────────────────────────────────────
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clGetPlatformIDs failed: " + std::to_string(err));

        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clGetDeviceIDs failed: " + std::to_string(err));

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

        // ⚠️ CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
        cl_command_queue queue = clCreateCommandQueue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));
        }

        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // ── Параметры ─────────────────────────────────────────────────
        const uint32_t BEAM_COUNT   = 10;
        const uint32_t N_POINT      = 8192;      // < 500k для быстрого бенчмарка
        const float    SAMPLE_RATE  = 100000.0f;

        auto signal = GenerateSignal(BEAM_COUNT, N_POINT, SAMPLE_RATE);

        antenna_fft::InputData<std::vector<std::complex<float>>> input;
        input.antenna_count = BEAM_COUNT;
        input.n_point       = N_POINT;
        input.sample_rate   = SAMPLE_RATE;
        input.repeat_count  = 1;
        input.data          = signal;

        antenna_fft::SpectrumMaximaFinder proc(backend.get());

        // ── Benchmark 1: Process (ONE_PEAK) ───────────────────────────
        std::cout << "\n--- Benchmark 1: Process (ONE_PEAK) ---\n";
        {
            test_fft_maxima::SpectrumMaximaFinderBenchmark bench(
                backend.get(), proc, input,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_Process"});

            if (!bench.IsProfEnabled()) {
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            } else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] Process benchmark complete\n";
            }
        }

        // ── Benchmark 2: FindAllMaxima (full pipeline) ─────────────────
        std::cout << "\n--- Benchmark 2: FindAllMaxima (pipeline) ---\n";
        {
            test_fft_maxima::SpectrumMaximaAllMaximaBenchmark bench(
                backend.get(), proc, input,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima"});

            if (!bench.IsProfEnabled()) {
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            } else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] AllMaxima benchmark complete\n";
            }
        }

        // ── Cleanup ────────────────────────────────────────────────────
        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "  FATAL: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace test_fft_maxima_benchmark
```

---

### TASK 7 — Обновить `tests/all_test.hpp`

**Файл**: `modules/fft_maxima/tests/all_test.hpp`

Добавить include и вызов:
```cpp
#include "test_fft_maxima_benchmark.hpp"   // ← добавить

namespace fft_maxima_all_test {
inline void run() {
    // ...существующие тесты...

    // BENCHMARK (старый стиль — deprecated):
    // test_benchmark_all_maxima::run();   // ← закомментировать

    // BENCHMARK (GpuBenchmarkBase — новый стиль):
    // test_fft_maxima_benchmark::run();   // ← раскомментировать для запуска
}
}
```

---

### TASK 8 — Обновить `tests/README.md`

**Файл**: `modules/fft_maxima/tests/README.md`

Добавить раздел о новых benchmark файлах:

```markdown
## Benchmark (GpuBenchmarkBase)

| Файл | Что тестирует |
|------|---------------|
| `fft_maxima_benchmark.hpp` | Benchmark-классы: SpectrumMaximaFinderBenchmark, SpectrumMaximaAllMaximaBenchmark |
| `test_fft_maxima_benchmark.hpp` | Test runner: 2 бенчмарка (Process + FindAllMaxima) |

Старый `test_benchmark_all_maxima.hpp` — DEPRECATED, заменён новым.
```

---

---

## ══════════════════════════════════════════════════
## ЧАСТЬ 2: ROCm (SpectrumProcessorROCm)
## ══════════════════════════════════════════════════

### Текущее состояние ROCm

**Хорошая новость**: `SpectrumProcessorROCm` и `AllMaximaPipelineROCm` уже ЧИСТЫЕ:
- `spectrum_processor_rocm.cpp` — **ноль** `RecordProfilingEvent` (в отличие от OpenCL!)
- `all_maxima_pipeline_rocm.cpp` — **ноль** `RecordProfilingEvent`

**Плохая новость**: prof_events поддержки нет вообще — добавить с нуля.

**Ключевое отличие от OpenCL**:
- OpenCL: `SpectrumMaximaFinder` (фасад) — используется напрямую
- ROCm: `SpectrumProcessorROCm` (ISpectrumProcessor) — используется через фабрику или напрямую
- ROCm внутренние методы: `UploadData()`, `ExecutePadKernel()`, `ExecuteFFT()`, `ExecutePostKernel()`, `ReadResults()` — НЕ возвращают события (void)
- `AllMaximaPipelineROCm::Execute()` — "stream-ordered, no explicit events" — нет hipEvent вообще!

### Типы для ROCm профилирования

```cpp
// Из DrvGPU/services/profiling_types.hpp (уже существует):
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

// Helpers из fft_processor_rocm.cpp (паттерн для копирования):
// MakeROCmDataFromEvents(ev_start, ev_end, kind, op_string)  → для async hipEvent
// MakeROCmDataFromClock(t_start, t_end, kind, op_string)     → для sync wall-clock
```

---

### TASK 9 — Добавить ROCmProfEvents в `spectrum_processor_rocm.hpp`

**Файл**: `modules/fft_maxima/include/processors/spectrum_processor_rocm.hpp`

#### 9.1 Добавить include и type alias

В блоке `#if ENABLE_ROCM`:
```cpp
#include "DrvGPU/services/profiling_types.hpp"  // ROCmProfilingData

namespace antenna_fft {

// Тип для ROCm prof_events — только в #if ENABLE_ROCM блоке
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

#### 9.2 Добавить prof_events к публичным методам

Все публичные методы override → добавить `ROCmProfEvents* prof_events = nullptr`:

```cpp
// БЫЛО:
std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data) override;

// СТАЛО:
std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    ROCmProfEvents* prof_events = nullptr);
    // ⚠️ НЕ override если интерфейс не менялся — или выделить в отдельный override
```

> **Вопрос про интерфейс**: Метод объявлен в `ISpectrumProcessor`.
> Если добавить prof_events только в `SpectrumProcessorROCm` — ломается override.
> **Решение**: Добавить `prof_events*` также в `ISpectrumProcessor` с `= nullptr` (default).
> Это backward-compatible — OpenCL реализация просто игнорирует параметр.

Полный список изменений сигнатур:
```cpp
// Process path:
std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    ROCmProfEvents* prof_events = nullptr) override;

std::vector<SpectrumResult> ProcessBatch(
    const std::vector<std::complex<float>>& batch_data,
    size_t start_antenna, size_t batch_antenna_count,
    ROCmProfEvents* prof_events = nullptr) override;

std::vector<SpectrumResult> ProcessFromGPU(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes = 0,
    ROCmProfEvents* prof_events = nullptr) override;

std::vector<SpectrumResult> ProcessBatchFromGPU(
    void* gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count,
    ROCmProfEvents* prof_events = nullptr) override;

// AllMaxima path:
AllMaximaResult FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ROCmProfEvents* prof_events = nullptr) override;

AllMaximaResult FindAllMaximaFromGPUPipeline(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ROCmProfEvents* prof_events = nullptr) override;
```

---

### TASK 10 — Инструментировать `spectrum_processor_rocm.cpp`

**Файл**: `modules/fft_maxima/src/spectrum_processor_rocm.cpp`

Инструментация делается **в теле ProcessBatch / ProcessFromCPU** (не в приватных методах).
Паттерн такой же как в `fft_processor_rocm.cpp`.

#### 10.1 Добавить helper-функции MakeROCmDataFrom*

В начало файла (static, как в fft_processor_rocm):

```cpp
#include <chrono>
#include "DrvGPU/services/profiling_types.hpp"

// Helper A: async GPU операции (hipEvent → elapsed)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind, const char* op_string = "")
{
    hipEventSynchronize(ev_end);
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_end);

    drv_gpu_lib::ROCmProfilingData d{};
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
    d.start_ns = 0; d.end_ns = elapsed_ns; d.complete_ns = elapsed_ns;
    d.kind = kind; d.op_string = op_string;
    return d;
}

// Helper B: sync операции (wall-clock)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t_start,
    std::chrono::high_resolution_clock::time_point t_end,
    uint32_t kind, const char* op_string = "")
{
    uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    drv_gpu_lib::ROCmProfilingData d{};
    d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
    d.kind = kind; d.op_string = op_string;
    return d;
}
```

#### 10.2 Инструментировать ProcessBatch (Process path)

Стадии: Upload → PadKernel → FFT → PostKernel → ReadResults

```cpp
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(
    const std::vector<std::complex<float>>& batch_data,
    size_t start_antenna, size_t batch_antenna_count,
    ROCmProfEvents* prof_events)
{
    // ── Upload (H2D) ──────────────────────────────────────────────────
    hipEvent_t ev_up_start = nullptr, ev_up_end = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_up_start); hipEventCreate(&ev_up_end);
        hipEventRecord(ev_up_start, stream_);
    }

    UploadData(batch_data.data() + start_antenna * params_.n_point,
               batch_antenna_count * params_.n_point);

    if (prof_events) hipEventRecord(ev_up_end, stream_);

    // ── PadKernel ─────────────────────────────────────────────────────
    hipEvent_t ev_pad_start = nullptr, ev_pad_end = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pad_start); hipEventCreate(&ev_pad_end);
        hipEventRecord(ev_pad_start, stream_);
    }

    ExecutePadKernel(batch_antenna_count);

    if (prof_events) hipEventRecord(ev_pad_end, stream_);

    // ── FFT ───────────────────────────────────────────────────────────
    hipEvent_t ev_fft_start = nullptr, ev_fft_end = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_fft_start); hipEventCreate(&ev_fft_end);
        hipEventRecord(ev_fft_start, stream_);
    }

    ExecuteFFT();

    if (prof_events) hipEventRecord(ev_fft_end, stream_);

    // ── PostKernel ────────────────────────────────────────────────────
    hipEvent_t ev_post_start = nullptr, ev_post_end = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_post_start); hipEventCreate(&ev_post_end);
        hipEventRecord(ev_post_start, stream_);
    }

    ExecutePostKernel(batch_antenna_count);

    if (prof_events) hipEventRecord(ev_post_end, stream_);

    // ── ReadResults (D2H sync) ─────────────────────────────────────────
    auto t_dl_start = std::chrono::high_resolution_clock::now();
    auto results = ReadResults(batch_antenna_count, start_antenna);
    auto t_dl_end = std::chrono::high_resolution_clock::now();

    // ── Собрать prof_events ───────────────────────────────────────────
    if (prof_events) {
        prof_events->push_back({"Upload",    MakeROCmDataFromEvents(ev_up_start,   ev_up_end,   1, "H2D")});
        prof_events->push_back({"PadKernel", MakeROCmDataFromEvents(ev_pad_start,  ev_pad_end,  0, "pad_kernel")});
        prof_events->push_back({"FFT",       MakeROCmDataFromEvents(ev_fft_start,  ev_fft_end,  0, "hipfftExecC2C")});
        prof_events->push_back({"PostKernel",MakeROCmDataFromEvents(ev_post_start, ev_post_end, 0, "post_kernel")});
        prof_events->push_back({"Download",  MakeROCmDataFromClock(t_dl_start,    t_dl_end,    1, "D2H")});
    }

    return results;
}
```

#### 10.3 Инструментировать FindAllMaximaFromCPU (AllMaxima path)

Стадии: Upload → PadKernel → FFT(allmax) → ComputeMagnitudes → Pipeline (Detect+Scan+Compact) → Download

```cpp
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ROCmProfEvents* prof_events)
{
    // ── Upload ────────────────────────────────────────────────────────
    hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_up_s); hipEventCreate(&ev_up_e);
        hipEventRecord(ev_up_s, stream_);
    }
    UploadData(data.data(), data.size());
    if (prof_events) hipEventRecord(ev_up_e, stream_);

    // ── PadKernel ─────────────────────────────────────────────────────
    hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
        hipEventRecord(ev_pad_s, stream_);
    }
    ExecutePadKernel(params_.antenna_count);
    if (prof_events) hipEventRecord(ev_pad_e, stream_);

    // ── FFT (allmax plan) ─────────────────────────────────────────────
    hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
        hipEventRecord(ev_fft_s, stream_);
    }
    CreateAllMaximaFFTPlan(params_.antenna_count);  // reuse если создан
    // hipfftExecC2C... (внутренняя логика)
    if (prof_events) hipEventRecord(ev_fft_e, stream_);

    // ── ComputeMagnitudes ─────────────────────────────────────────────
    hipEvent_t ev_mag_s = nullptr, ev_mag_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_mag_s); hipEventCreate(&ev_mag_e);
        hipEventRecord(ev_mag_s, stream_);
    }
    ExecuteComputeMagnitudes(params_.antenna_count * params_.nFFT);
    if (prof_events) hipEventRecord(ev_mag_e, stream_);

    // ── Pipeline (Detect + Scan + Compact) ────────────────────────────
    hipEvent_t ev_pipe_s = nullptr, ev_pipe_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pipe_s); hipEventCreate(&ev_pipe_e);
        hipEventRecord(ev_pipe_s, stream_);
    }
    auto result = pipeline_->Execute(magnitudes_buffer_, fft_output_,
                                     params_.antenna_count, params_.nFFT,
                                     params_.sample_rate, dest, search_start, search_end);
    if (prof_events) hipEventRecord(ev_pipe_e, stream_);

    // ── Собрать prof_events ───────────────────────────────────────────
    if (prof_events) {
        prof_events->push_back({"Upload",            MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
        prof_events->push_back({"PadKernel",         MakeROCmDataFromEvents(ev_pad_s,  ev_pad_e,  0, "pad")});
        prof_events->push_back({"FFT",               MakeROCmDataFromEvents(ev_fft_s,  ev_fft_e,  0, "hipFFT")});
        prof_events->push_back({"ComputeMagnitudes", MakeROCmDataFromEvents(ev_mag_s,  ev_mag_e,  0, "mag")});
        prof_events->push_back({"Pipeline",          MakeROCmDataFromEvents(ev_pipe_s, ev_pipe_e, 0, "detect+scan+compact")});
    }

    return result;
}
```

> ⚠️ **Примечание**: Если нужна более гранулярная разбивка pipeline на Detect/Scan/Compact —
> это делается в **TASK 11** (AllMaximaPipelineROCm с prof_events).

---

### TASK 11 — Добавить ROCmProfEvents к `AllMaximaPipelineROCm` (опционально)

**Файл**: `modules/fft_maxima/include/pipelines/all_maxima_pipeline_rocm.hpp`
**Файл**: `modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp`

> **Степень необходимости**: ОПЦИОНАЛЬНО.
> TASK 10 уже профилирует pipeline как единый блок "Pipeline" (Detect+Scan+Compact вместе).
> Если нужна детализация по стадиям — выполнить TASK 11.

#### 11.1 Обновить IAllMaximaPipeline интерфейс

**Файл**: `modules/fft_maxima/include/interface/i_all_maxima_pipeline.hpp`

```cpp
#include "DrvGPU/services/profiling_types.hpp"

class IAllMaximaPipeline {
public:
    using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

    virtual AllMaximaResult Execute(
        void* magnitudes_gpu,
        void* fft_data_gpu,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 1,
        uint32_t search_end = 0,
        size_t max_maxima_per_beam = 1000,
        ROCmProfEvents* prof_events = nullptr) = 0;   // ← добавить
};
```

#### 11.2 Обновить AllMaximaPipelineROCm::Execute()

В `all_maxima_pipeline_rocm.cpp` добавить hipEvent обёртки вокруг каждого кернела:

```cpp
// Стадия 1: Detect
hipEvent_t ev_det_s = nullptr, ev_det_e = nullptr;
if (prof_events) { hipEventCreate(&ev_det_s); hipEventCreate(&ev_det_e); hipEventRecord(ev_det_s, stream_); }
// ... hipModuleLaunchKernel(detect_kernel_, ...) ...
if (prof_events) hipEventRecord(ev_det_e, stream_);

// Стадия 2: Scan (PrefixSum)
hipEvent_t ev_scan_s = nullptr, ev_scan_e = nullptr;
if (prof_events) { hipEventCreate(&ev_scan_s); hipEventCreate(&ev_scan_e); hipEventRecord(ev_scan_s, stream_); }
ExecutePrefixSum(flags_buf, scan_buf, nFFT, beam_count);
if (prof_events) hipEventRecord(ev_scan_e, stream_);

// Стадия 3: Compact
hipEvent_t ev_comp_s = nullptr, ev_comp_e = nullptr;
if (prof_events) { hipEventCreate(&ev_comp_s); hipEventCreate(&ev_comp_e); hipEventRecord(ev_comp_s, stream_); }
// ... hipModuleLaunchKernel(compact_kernel_, ...) ...
if (prof_events) hipEventRecord(ev_comp_e, stream_);

// Собрать события
if (prof_events) {
    prof_events->push_back({"Detect",  MakeROCmDataFromEvents(ev_det_s,  ev_det_e,  0, "detect")});
    prof_events->push_back({"Scan",    MakeROCmDataFromEvents(ev_scan_s, ev_scan_e, 0, "prefix_sum")});
    prof_events->push_back({"Compact", MakeROCmDataFromEvents(ev_comp_s, ev_comp_e, 0, "compact")});
}
```

> **Обратите внимание**: Если TASK 11 выполнен — в TASK 10 (FindAllMaximaFromCPU) заменить:
> ```cpp
> // Вместо одного "Pipeline" события:
> pipeline_->Execute(..., prof_events);   // события добавятся внутри
> // Убрать ev_pipe_s/ev_pipe_e обёртку
> ```

---

### TASK 12 — Создать `tests/fft_maxima_benchmark_rocm.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/fft_maxima_benchmark_rocm.hpp`

Паттерн: точно как `fft_processor_benchmark_rocm.hpp`, но для `SpectrumProcessorROCm`.

```cpp
#pragma once

#if ENABLE_ROCM

#include "processors/spectrum_processor_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <complex>
#include <vector>

namespace test_fft_maxima_rocm {

// ─────────────────────────────────────────────────────────────────────────
// Benchmark 1: ProcessBatch (ONE_PEAK) — Upload+Pad+FFT+PostKernel+Download
// ─────────────────────────────────────────────────────────────────────────

class SpectrumProcessorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumProcessorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const antenna_fft::SpectrumParams& params,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_Process", cfg),
      proc_(proc), params_(params), input_data_(input_data) {}

protected:
  void ExecuteKernel() override {
    // Warmup: ProcessFromCPU без prof_events
    proc_.ProcessFromCPU(input_data_);
  }

  void ExecuteKernelTimed() override {
    // Замер: с prof_events → hipEvent timing
    antenna_fft::ROCmProfEvents events;
    proc_.ProcessFromCPU(input_data_, &events);

    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  antenna_fft::SpectrumProcessorROCm& proc_;
  antenna_fft::SpectrumParams         params_;
  std::vector<std::complex<float>>    input_data_;
};

// ─────────────────────────────────────────────────────────────────────────
// Benchmark 2: FindAllMaximaFromCPU — Upload+Pad+FFT+Mag+Pipeline(Detect+Scan+Compact)
// ─────────────────────────────────────────────────────────────────────────

class SpectrumProcessorROCmAllMaximaBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumProcessorROCmAllMaximaBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const antenna_fft::SpectrumParams& params,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_AllMaxima", cfg),
      proc_(proc), params_(params), input_data_(input_data) {}

protected:
  void ExecuteKernel() override {
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU,
                               1, 0);
  }

  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU,
                               1, 0, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  antenna_fft::SpectrumProcessorROCm& proc_;
  antenna_fft::SpectrumParams         params_;
  std::vector<std::complex<float>>    input_data_;
};

}  // namespace test_fft_maxima_rocm

#endif  // ENABLE_ROCM
```

---

### TASK 13 — Создать `tests/test_fft_maxima_benchmark_rocm.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/test_fft_maxima_benchmark_rocm.hpp`

Паттерн: точно как `test_fft_benchmark_rocm.hpp` из fft_processor.

```cpp
#pragma once

#if ENABLE_ROCM

/**
 * @file test_fft_maxima_benchmark_rocm.hpp
 * @brief ROCm Benchmark: SpectrumProcessorROCm через GpuBenchmarkBase
 *
 * Запускает:
 *  - Process (ONE_PEAK): 5 warmup + 20 runs → Upload/Pad/FFT/PostKernel/Download
 *  - FindAllMaxima:      5 warmup + 20 runs → Upload/Pad/FFT/Mag/Pipeline
 *
 * Результаты: Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/
 *             Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/
 *
 * Запускается ТОЛЬКО на Linux с AMD GPU (ENABLE_ROCM=1).
 */

#include "fft_maxima_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "processors/spectrum_processor_rocm.hpp"
#include "factory/spectrum_processor_factory.hpp"

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_maxima_benchmark_rocm {

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t beam_count, uint32_t n_point, float sample_rate)
{
    std::vector<std::complex<float>> data(
        static_cast<size_t>(beam_count) * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
        float freq = 50.0f + b * 30.0f;
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI)
                                 * freq * t / sample_rate);
            data[static_cast<size_t>(b) * n_point + t] = {val, 0.0f};
        }
    }
    return data;
}

inline int run() {
    std::cout << "\n"
              << "============================================================\n"
              << "  SpectrumProcessorROCm Benchmark (GpuBenchmarkBase)\n"
              << "============================================================\n";

    // Проверка наличия ROCm-устройств
    // (pattern from test_fft_benchmark_rocm.hpp)
    try {
        drv_gpu_lib::ROCmBackend backend;
        backend.Initialize(0);
        drv_gpu_lib::IBackend* b = &backend;

        // ── Параметры ─────────────────────────────────────────────────
        const uint32_t BEAM_COUNT  = 10;
        const uint32_t N_POINT     = 8192;
        const float    SAMPLE_RATE = 100000.0f;

        auto signal = GenerateSignal(BEAM_COUNT, N_POINT, SAMPLE_RATE);

        antenna_fft::SpectrumParams params;
        params.antenna_count = BEAM_COUNT;
        params.n_point       = N_POINT;
        params.sample_rate   = SAMPLE_RATE;
        params.repeat_count  = 1;
        params.peak_mode     = antenna_fft::PeakSearchMode::ONE_PEAK;
        params.memory_limit  = 0.8f;

        // ── Создать SpectrumProcessorROCm напрямую ─────────────────────
        antenna_fft::SpectrumProcessorROCm proc(b);
        proc.Initialize(params);

        // ── Benchmark 1: Process (ONE_PEAK) ───────────────────────────
        std::cout << "\n--- ROCm Benchmark 1: Process (ONE_PEAK) ---\n";
        {
            test_fft_maxima_rocm::SpectrumProcessorROCmBenchmark bench(
                b, proc, params, signal,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"});

            if (!bench.IsProfEnabled()) {
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            } else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] ROCm Process benchmark complete\n";
            }
        }

        // ── Benchmark 2: FindAllMaxima ─────────────────────────────────
        std::cout << "\n--- ROCm Benchmark 2: FindAllMaxima (pipeline) ---\n";
        {
            test_fft_maxima_rocm::SpectrumProcessorROCmAllMaximaBenchmark bench(
                b, proc, params, signal,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"});

            if (!bench.IsProfEnabled()) {
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            } else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] ROCm AllMaxima benchmark complete\n";
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cout << "  [SKIP] ROCm not available: " << e.what() << "\n";
        return 0;  // Не ошибка — просто нет AMD GPU
    }
}

}  // namespace test_fft_maxima_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

### TASK 14 — Обновить `tests/all_test.hpp` (ROCm часть)

**Файл**: `modules/fft_maxima/tests/all_test.hpp`

```cpp
#if ENABLE_ROCM
#include "test_fft_maxima_benchmark_rocm.hpp"   // ← добавить
#endif

namespace fft_maxima_all_test {
inline void run() {
    // ...существующие тесты...

    // ROCm: SpectrumProcessorROCm тесты (корректность)
#if ENABLE_ROCM
    test_spectrum_maxima_rocm::run();   // существующий тест — без изменений
#endif

    // ROCm: Benchmark (GpuBenchmarkBase)
#if ENABLE_ROCM
//  test_fft_maxima_benchmark_rocm::run();  // ← раскомментировать для запуска
#endif
}
}
```

---

## Итоговая структура файлов (после ПОЛНОГО рефакторинга)

```
modules/fft_maxima/
├── include/
│   ├── spectrum_maxima_finder.h          ← ИЗМЕНЁН: ProfEvents + prof_events* (OpenCL)
│   ├── interface/
│   │   └── i_all_maxima_pipeline.hpp     ← ИЗМЕНЁН (TASK 11): ROCmProfEvents в Execute()
│   └── processors/
│       └── spectrum_processor_rocm.hpp   ← ИЗМЕНЁН: ROCmProfEvents + prof_events*
├── src/
│   ├── spectrum_maxima_finder.cpp        ← ИЗМЕНЁН: нет RecordProfilingEvent (OpenCL)
│   ├── spectrum_maxima_finder_process.cpp← ИЗМЕНЁН: нет RecordProfilingEvent (OpenCL)
│   ├── spectrum_maxima_finder_all_maxima.cpp ← ИЗМЕНЁН: нет RecordProfilingEvent (OpenCL)
│   ├── spectrum_processor_rocm.cpp       ← ИЗМЕНЁН: hipEvent timing (ROCm)
│   └── all_maxima_pipeline_rocm.cpp      ← ИЗМЕНЁН (TASK 11): hipEvent по стадиям (ROCm)
└── tests/
    ├── all_test.hpp                      ← ИЗМЕНЁН: добавлены все benchmark includes
    ├── fft_maxima_benchmark.hpp          ← НОВЫЙ: OpenCL benchmark классы
    ├── test_fft_maxima_benchmark.hpp     ← НОВЫЙ: OpenCL test runner
    ├── fft_maxima_benchmark_rocm.hpp     ← НОВЫЙ: ROCm benchmark классы
    ├── test_fft_maxima_benchmark_rocm.hpp← НОВЫЙ: ROCm test runner
    ├── test_benchmark_all_maxima.hpp     ← DEPRECATED (оставить, не удалять)
    └── README.md                        ← ИЗМЕНЁН: доки по обоим бенчмаркам
```

---

## Проверка правильности (чеклист)

### OpenCL (TASK 1-8)

- [ ] Production код (`src/spectrum_maxima_finder*.cpp`) — **ноль** `RecordProfilingEvent`
- [ ] Production код — **ноль** `#include "services/gpu_profiler.hpp"` (убрать если остался)
- [ ] `Process(input)` без prof_events — работает как раньше (no-op)
- [ ] `FindAllMaxima(input)` без prof_events — работает как раньше
- [ ] OpenCL очередь создана с `CL_QUEUE_PROFILING_ENABLE` в test runner
- [ ] `IsProfEnabled()` → `false` → бенчмарк пропускается без ошибки
- [ ] Файлы в `Results/Profiler/GPU_00_SpectrumMaxima_Process/` после запуска
- [ ] Файлы в `Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/` после запуска

### ROCm (TASK 9-14)

- [ ] `SpectrumProcessorROCm::ProcessFromCPU(data, &events)` — заполняет events (5 стадий)
- [ ] `SpectrumProcessorROCm::FindAllMaximaFromCPU(data, ..., &events)` — заполняет events
- [ ] `ProcessFromCPU(data)` без prof_events — работает как раньше (ноль overhead hipEvent)
- [ ] ROCm benchmark под `#if ENABLE_ROCM` (не компилируется на Windows)
- [ ] ROCm test runner обрабатывает отсутствие AMD GPU (catch → skip, не error)
- [ ] Файлы в `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/` после запуска на Linux
- [ ] Файлы в `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/` после запуска на Linux

---

## Примечания

### Про старые тесты

Старый `test_benchmark_all_maxima.hpp` НЕ удалять — только закомментировать вызов в `all_test.hpp`.
Он может быть полезен как референс или для быстрой проверки AllMaxima с большими данными.

### Про `const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;`

Эту строку в src/*.cpp можно убрать из методов, которые больше не используют `RecordProfilingEvent`.
Если gpu_id больше нигде не нужен — удалить.

### Про AllMaxima Download

В отличие от FFTProcessor, `FindAllMaxima` читает результаты через `clEnqueueReadBuffer` в конце.
Если это отдельная операция — добавить event "Download" туда же через CollectOrRelease.
Если это синхронная операция — можно не профилировать (Download обычно мал по сравнению с FFT).

### Про SpectrumProcessorOpenCL (параллельная архитектура)

В файле `include/processors/spectrum_processor_opencl.hpp` есть класс `SpectrumProcessorOpenCL`.
Это ДОПОЛНИТЕЛЬНЫЙ backend (Strategy pattern). Для него prof_events добавлять ОТДЕЛЬНОЙ задачей.
В данной задаче фокус на `SpectrumMaximaFinder` (основной фасад).

### Про интерфейс ISpectrumProcessor и prof_events

TASK 9 добавляет `prof_events*` к `SpectrumProcessorROCm`.
Если добавляем как `override` — нужно также обновить `ISpectrumProcessor` (и OpenCL-реализацию как stub).
Альтернатива: НЕ использовать `override` для prof_events-методов, а добавить их как отдельные перегрузки.

**Рекомендация**: Добавить в `ISpectrumProcessor` с `= nullptr` (все реализации backward-compatible):
```cpp
virtual std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    void* prof_events = nullptr) = 0;  // void* как type-erased placeholder
```
Или проще — добавить в интерфейс напрямую `ROCmProfEvents*` под `#if ENABLE_ROCM`.

### Таблица TaskID

| ID | Компонент | Тип | Сложность |
|----|-----------|-----|-----------|
| TASK 1 | spectrum_maxima_finder.h (header) | OpenCL | Средняя |
| TASK 2 | spectrum_maxima_finder.cpp (ProcessBatch) | OpenCL | Низкая |
| TASK 3 | spectrum_maxima_finder_process.cpp | OpenCL | Низкая |
| TASK 4 | spectrum_maxima_finder_all_maxima.cpp | OpenCL | Средняя |
| TASK 5 | fft_maxima_benchmark.hpp (НОВЫЙ) | OpenCL | Низкая |
| TASK 6 | test_fft_maxima_benchmark.hpp (НОВЫЙ) | OpenCL | Низкая |
| TASK 7 | all_test.hpp (OpenCL часть) | OpenCL | Минимальная |
| TASK 8 | README.md | Доки | Минимальная |
| TASK 9 | spectrum_processor_rocm.hpp | ROCm | Средняя |
| TASK 10 | spectrum_processor_rocm.cpp (инструментация) | ROCm | Средняя |
| TASK 11 | all_maxima_pipeline_rocm (опционально) | ROCm | Высокая |
| TASK 12 | fft_maxima_benchmark_rocm.hpp (НОВЫЙ) | ROCm | Низкая |
| TASK 13 | test_fft_maxima_benchmark_rocm.hpp (НОВЫЙ) | ROCm | Низкая |
| TASK 14 | all_test.hpp (ROCm часть) | ROCm | Минимальная |

---

*Создано: 2026-03-01*
*Обновлено: 2026-03-01 (добавлена ROCm часть)*
*Автор плана: Кодо (AI Assistant)*
