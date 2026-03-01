# TASK: Профилирование fft_maxima — OpenCL (GpuBenchmarkBase)

> **Статус**: 📋 PLAN
> **Дата**: 2026-03-01
> **Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`
> **Референс**: `modules/fft_processor/tests/fft_processor_benchmark.hpp`
> **Связанный таск (ROCm)**: `TASK_fft_maxima_profiling_rocm.md`

---

## Проблема

`SpectrumMaximaFinder` нарушает принцип "production-класс чистый":

| Файл | Нарушение |
|------|-----------|
| `src/spectrum_maxima_finder.cpp` | `RecordProfilingEvent` в `ProcessBatch()` — 3 вызова |
| `src/spectrum_maxima_finder_process.cpp` | `RecordProfilingEvent` в `ProcessFromGPU()`, `ProcessBatchFromGPU()` — 4 вызова |
| `src/spectrum_maxima_finder_all_maxima.cpp` | `RecordProfilingEvent` в AllMaxima pipeline — **8 вызовов** |
| `tests/test_benchmark_all_maxima.hpp` | СТАРЫЙ API: прямой `GPUProfiler.Start/Stop`, `sleep_for`, без `GpuBenchmarkBase` |

**Итого**: 15 `RecordProfilingEvent` в production — все убрать.

---

## Цель

```
SpectrumMaximaFinder (production)     Benchmark-класс (тест)
─────────────────────────────────     ──────────────────────────────────
Process()       prof_events=nullptr   SpectrumMaximaFinderBenchmark
FindAllMaxima() prof_events=nullptr     : GpuBenchmarkBase
                prof_events=&vec          ExecuteKernel()       ← warmup
                                          ExecuteKernelTimed()  ← timing
```

---

## Новые типы и helpers

```cpp
// В spectrum_maxima_finder.h:
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) prof_events->push_back({name, ev});
    else             clReleaseEvent(ev);
}
```

---

## TASK 1 — Заголовок `spectrum_maxima_finder.h`

**Файл**: `modules/fft_maxima/include/spectrum_maxima_finder.h`

### 1.1 Добавить ProfEvents + CollectOrRelease

```cpp
// После #include-ов, в теле класса:
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

// В private секции:
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) { prof_events->push_back({name, ev}); }
    else             { clReleaseEvent(ev); }
}
```

### 1.2 Process\<T\> — добавить prof_events

```cpp
// Декларация:
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::ROCm,
    ProfEvents* prof_events = nullptr);   // ← добавить

// Реализация шаблона в том же .h:
if constexpr (is_cpu_vector_v<T>) {
    if (!initialized_) Initialize();
    return ProcessFromCPU(input.data, prof_events);      // ← передать
}
else if constexpr (std::is_same_v<T, cl_mem>) {
    return ProcessFromGPU(input.data, input.antenna_count, input.n_point,
                          input.ActualGpuMemory(), prof_events);  // ← передать
}
```

### 1.3 FindAllMaxima\<T\> — добавить prof_events

```cpp
template<typename T>
AllMaximaResult FindAllMaxima(
    const InputData<T>& input,
    OutputDestination dest = OutputDestination::CPU,
    DriverType driver = DriverType::OPENCL,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    ProfEvents* prof_events = nullptr);  // ← добавить

// Реализация шаблона:
if constexpr (is_cpu_vector_v<T>) {
    if (!initialized_) Initialize();
    return FindAllMaximaFromCPU(input.data, dest, search_start, search_end, prof_events);
}
else if constexpr (std::is_same_v<T, cl_mem>) {
    return FindAllMaximaFromGPUPipeline(input.data, input.antenna_count, input.n_point,
                                        input.ActualGpuMemory(), dest,
                                        search_start, search_end, prof_events);
}
```

### 1.4 FindAllMaxima(cl_mem, ...) — добавить prof_events

```cpp
AllMaximaResult FindAllMaxima(
    cl_mem fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 0, uint32_t search_end = 0,
    uint32_t beam_offset = 0,
    cl_mem external_out_maxima = nullptr, cl_mem external_out_counts = nullptr,
    ProfEvents* prof_events = nullptr);  // ← добавить
```

### 1.5 Приватные методы — обновить объявления

Добавить `ProfEvents*` к:

```cpp
std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data, ProfEvents* pe = nullptr);

std::vector<SpectrumResult> ProcessBatch(
    const std::vector<std::complex<float>>& data,
    size_t start_antenna, size_t batch_antenna_count, ProfEvents* pe = nullptr);

std::vector<SpectrumResult> ProcessFromGPU(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes, ProfEvents* pe = nullptr);

std::vector<SpectrumResult> ProcessBatchFromGPU(
    cl_mem gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count, ProfEvents* pe = nullptr);

AllMaximaResult FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* pe = nullptr);

AllMaximaResult FindAllMaximaFromGPUPipeline(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* pe = nullptr);

AllMaximaResult AllMaximaFromCPU(
    const std::vector<std::complex<float>>& fft_data,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* pe = nullptr);
```

---

## TASK 2 — `spectrum_maxima_finder.cpp` (ProcessBatch)

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`

### Изменить сигнатуру ProcessBatch

```cpp
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatch(
    const std::vector<std::complex<float>>& data,
    size_t start_antenna, size_t batch_antenna_count,
    ProfEvents* prof_events)   // ← добавить
```

### Заменить RecordProfilingEvent → CollectOrRelease (3 места)

```cpp
// БЫЛО:
cl_event upload_event = UploadData(batch_data);
drv_gpu_lib::RecordProfilingEvent(upload_event, gpu_id, "SpectrumMaxima", "Upload");

cl_event fft_event = ExecuteFFT(upload_event);
clReleaseEvent(upload_event);
drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");

cl_event post_event = ExecutePostKernel(fft_event);
clReleaseEvent(fft_event);
drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");
clReleaseEvent(post_event);

// СТАЛО:
cl_event upload_event = UploadData(batch_data);
CollectOrRelease(upload_event, "Upload", prof_events);   // ← сам освобождает если nullptr

cl_event fft_event = ExecuteFFT(upload_event);
// Примечание: wait_event upload_event уже освобождён через CollectOrRelease выше
// → нужно сохранить event ПЕРЕД CollectOrRelease или скорректировать порядок
CollectOrRelease(fft_event, "FFT", prof_events);

cl_event post_event = ExecutePostKernel(fft_event);
CollectOrRelease(post_event, "PostKernel", prof_events);
```

> ⚠️ **Важно про wait_event**: `ExecuteFFT(upload_event)` использует `upload_event` как wait.
> Если `CollectOrRelease` вызван ДО `ExecuteFFT`, событие уже освобождено.
> **Порядок**: сначала передать event дальше (как wait), потом вызвать CollectOrRelease.
>
> ```cpp
> cl_event upload = UploadData(batch_data);
> cl_event fft    = ExecuteFFT(upload);         // ← сначала использовать upload
> CollectOrRelease(upload, "Upload", pe);       // ← потом освободить/сохранить
>
> cl_event post   = ExecutePostKernel(fft);
> CollectOrRelease(fft, "FFT", pe);
>
> auto results = ReadResults(post);
> CollectOrRelease(post, "PostKernel", pe);
> ```

### Удалить gpu_id если больше не используется

```cpp
// Удалить эту строку (если RecordProfilingEvent убран):
const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
```

### Удалить include gpu_profiler.hpp если больше не нужен

```cpp
// Проверить и убрать если нет других использований:
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
```

---

## TASK 3 — `spectrum_maxima_finder_process.cpp`

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder_process.cpp`

### Изменить сигнатуры

```cpp
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromCPU(
    const std::vector<std::complex<float>>& data, ProfEvents* prof_events);

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromGPU(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes, ProfEvents* prof_events);

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatchFromGPU(
    cl_mem gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count, ProfEvents* prof_events);
```

### ProcessFromCPU — передать prof_events в ProcessBatch

```cpp
// БЫЛО:
return ProcessBatch(data, batch.start, batch.count);

// СТАЛО:
return ProcessBatch(data, batch.start, batch.count, prof_events);
```

### ProcessFromGPU (single-batch path) — 3 замены

```cpp
// БЫЛО:
drv_gpu_lib::RecordProfilingEvent(copy_event, gpu_id, "SpectrumMaxima", "GPU→GPU Copy");
// ... fft, post ...
drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");
drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");

// СТАЛО (с правильным порядком wait/release):
cl_event copy = ...;
cl_event fft  = ExecuteFFT(copy);
CollectOrRelease(copy, "GPU→GPU Copy", prof_events);

cl_event post = ExecutePostKernel(fft);
CollectOrRelease(fft, "FFT", prof_events);

auto results = ReadResults(post);
CollectOrRelease(post, "PostKernel", prof_events);
```

### ProcessBatchFromGPU — аналогично (3 замены)

Тот же паттерн: GPU→GPU Copy → FFT → PostKernel.

---

## TASK 4 — `spectrum_maxima_finder_all_maxima.cpp`

**Файл**: `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`

8 вызовов `RecordProfilingEvent` в 4 методах. Паттерн замены везде одинаковый.

### Изменить сигнатуры

```cpp
AllMaximaResult SpectrumMaximaFinder::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events);

AllMaximaResult SpectrumMaximaFinder::FindAllMaximaFromGPUPipeline(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events);

AllMaximaResult SpectrumMaximaFinder::FindAllMaxima(
    cl_mem fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest,
    uint32_t search_start, uint32_t search_end,
    uint32_t beam_offset,
    cl_mem external_out_maxima, cl_mem external_out_counts,
    ProfEvents* prof_events);
```

### Таблица замен (8 вызовов)

| Был (RecordProfilingEvent) | Стал (CollectOrRelease) |
|----------------------------|-------------------------|
| `"AllMaxima"`, `"Upload"` | `CollectOrRelease(ev, "Upload", prof_events)` |
| `"AllMaxima"`, `"FFT+PostCallback"` | `CollectOrRelease(ev, "FFT+PostCallback", prof_events)` |
| `"AllMaxima"`, `"Upload_Batch"` | `CollectOrRelease(ev, "Upload", prof_events)` |
| `"AllMaxima"`, `"FFT_Batch"` | `CollectOrRelease(ev, "FFT", prof_events)` |
| `"AllMaxima"`, `"GPU→GPU Copy"` | `CollectOrRelease(ev, "GPU→GPU Copy", prof_events)` |
| `"AllMaxima"`, `"GPU→GPU Copy_Batch"` | `CollectOrRelease(ev, "GPU→GPU Copy", prof_events)` |
| `"AllMaxima"`, `"ComputeMagnitudes"` | `CollectOrRelease(ev, "ComputeMagnitudes", prof_events)` |
| `"AllMaxima"`, `"Detect"` | `CollectOrRelease(ev, "Detect", prof_events)` |
| `"AllMaxima"`, `"Scan"` | `CollectOrRelease(ev, "Scan", prof_events)` |
| `"AllMaxima"`, `"Compact"` | `CollectOrRelease(ev, "Compact", prof_events)` |

> ⚠️ Паттерн wait_event такой же — сначала передать event как wait следующей операции,
> потом вызвать `CollectOrRelease`. Не освобождать event вручную (`clReleaseEvent`) —
> `CollectOrRelease` делает это сам когда `prof_events = nullptr`.

---

## TASK 5 — Создать `tests/fft_maxima_benchmark.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/fft_maxima_benchmark.hpp`

```cpp
#pragma once
/**
 * @file fft_maxima_benchmark.hpp
 * @brief Benchmark-классы для SpectrumMaximaFinder (OpenCL, GpuBenchmarkBase)
 *
 * SpectrumMaximaFinderBenchmark   → Process (ONE_PEAK): Upload+FFT+PostKernel
 * SpectrumMaximaAllMaximaBenchmark → FindAllMaxima: Upload+FFT+Detect+Scan+Compact
 */

#include "spectrum_maxima_finder.h"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <complex>
#include <vector>

namespace test_fft_maxima {

// ─── Benchmark 1: Process (ONE_PEAK) ─────────────────────────────────────

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
    proc_.Process(input_);   // warmup: prof_events = nullptr
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

// ─── Benchmark 2: FindAllMaxima (full pipeline) ───────────────────────────

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
                        0, 0, &events);
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

## TASK 6 — Создать `tests/test_fft_maxima_benchmark.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/test_fft_maxima_benchmark.hpp`

```cpp
#pragma once
/**
 * @file test_fft_maxima_benchmark.hpp
 * @brief Test runner: SpectrumMaximaFinder OpenCL benchmark (GpuBenchmarkBase)
 *
 * Benchmark 1: Process (ONE_PEAK)  → Results/Profiler/GPU_00_SpectrumMaxima_Process/
 * Benchmark 2: FindAllMaxima       → Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/
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
            throw std::runtime_error("clGetPlatformIDs: " + std::to_string(err));

        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clGetDeviceIDs: " + std::to_string(err));

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateContext: " + std::to_string(err));

        // ⚠️ CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
        cl_command_queue queue = clCreateCommandQueue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("clCreateCommandQueue: " + std::to_string(err));
        }

        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // ── Данные ────────────────────────────────────────────────────
        const uint32_t BEAM_COUNT  = 10;
        const uint32_t N_POINT     = 8192;
        const float    SAMPLE_RATE = 100000.0f;

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

            if (!bench.IsProfEnabled())
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] Process benchmark complete\n";
            }
        }

        // ── Benchmark 2: FindAllMaxima (pipeline) ─────────────────────
        std::cout << "\n--- Benchmark 2: FindAllMaxima (pipeline) ---\n";
        {
            test_fft_maxima::SpectrumMaximaAllMaximaBenchmark bench(
                backend.get(), proc, input,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima"});

            if (!bench.IsProfEnabled())
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            else {
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

## TASK 7 — Обновить `tests/all_test.hpp`

**Файл**: `modules/fft_maxima/tests/all_test.hpp`

```cpp
// Добавить include:
#include "test_fft_maxima_benchmark.hpp"

namespace fft_maxima_all_test {
inline void run() {
    // ...существующие тесты...

    // DEPRECATED (старый стиль):
    // test_benchmark_all_maxima::run();

    // НОВЫЙ (GpuBenchmarkBase):
    // test_fft_maxima_benchmark::run();   // раскомментировать для запуска
}
}
```

---

## TASK 8 — Обновить `tests/README.md`

Добавить раздел:

```markdown
## Benchmark (GpuBenchmarkBase, OpenCL)

| Файл | Классы / Назначение |
|------|---------------------|
| `fft_maxima_benchmark.hpp` | `SpectrumMaximaFinderBenchmark` (Process), `SpectrumMaximaAllMaximaBenchmark` (FindAllMaxima) |
| `test_fft_maxima_benchmark.hpp` | Test runner: 2 бенчмарка (Process + AllMaxima), OpenCL init |
| `test_benchmark_all_maxima.hpp` | **DEPRECATED** — заменён test_fft_maxima_benchmark.hpp |
```

---

## Итог: изменённые файлы

```
modules/fft_maxima/
├── include/
│   └── spectrum_maxima_finder.h              ← ИЗМЕНЁН (TASK 1)
├── src/
│   ├── spectrum_maxima_finder.cpp            ← ИЗМЕНЁН (TASK 2)
│   ├── spectrum_maxima_finder_process.cpp    ← ИЗМЕНЁН (TASK 3)
│   └── spectrum_maxima_finder_all_maxima.cpp ← ИЗМЕНЁН (TASK 4)
└── tests/
    ├── fft_maxima_benchmark.hpp              ← НОВЫЙ   (TASK 5)
    ├── test_fft_maxima_benchmark.hpp         ← НОВЫЙ   (TASK 6)
    ├── all_test.hpp                          ← ИЗМЕНЁН (TASK 7)
    └── README.md                            ← ИЗМЕНЁН (TASK 8)
```

---

## Чеклист

- [ ] `src/spectrum_maxima_finder*.cpp` — **ноль** `RecordProfilingEvent`
- [ ] `src/spectrum_maxima_finder*.cpp` — **ноль** `#include "services/gpu_profiler.hpp"`
- [ ] `Process(input)` без prof_events — работает как раньше
- [ ] `FindAllMaxima(input)` без prof_events — работает как раньше
- [ ] OpenCL queue создана с `CL_QUEUE_PROFILING_ENABLE`
- [ ] `IsProfEnabled() = false` → бенчмарк выводит `[SKIP]`, не падает
- [ ] После запуска: файлы в `Results/Profiler/GPU_00_SpectrumMaxima_Process/`
- [ ] После запуска: файлы в `Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/`

---

*Создано: 2026-03-01*
*Автор: Кодо (AI Assistant)*
