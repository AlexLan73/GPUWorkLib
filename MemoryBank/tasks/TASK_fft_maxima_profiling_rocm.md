# TASK: Профилирование fft_maxima — ROCm (GpuBenchmarkBase)

> **Статус**: 📋 PLAN
> **Дата**: 2026-03-01
> **Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`
> **Референс**: `modules/fft_processor/tests/fft_processor_benchmark_rocm.hpp`
> **Связанный таск (OpenCL)**: `TASK_fft_maxima_profiling_opencl.md`
> **Запускается**: только на Linux + AMD GPU (ENABLE_ROCM=1)

---

## Текущее состояние

| | OpenCL | ROCm |
|--|--------|------|
| Production чистый? | ❌ 15 RecordProfilingEvent | ✅ **Уже чистый** |
| prof_events поддержка? | ❌ Нет | ❌ Нет — добавить с нуля |
| Benchmark есть? | ❌ Только deprecated | ❌ Нет совсем |

**Хорошая новость**: `spectrum_processor_rocm.cpp` и `all_maxima_pipeline_rocm.cpp` — ноль `RecordProfilingEvent`. Чистить не нужно, только добавить timing.

---

## Ключевые отличия от OpenCL

| Аспект | OpenCL | ROCm |
|--------|--------|------|
| Класс для бенчмарка | `SpectrumMaximaFinder` (фасад) | `SpectrumProcessorROCm` (напрямую) |
| Тип prof_events | `ProfEvents` (cl_event пары) | `ROCmProfEvents` (ROCmProfilingData) |
| Timing async | `cl_event` → clGetEventProfilingInfo | `hipEvent_t` → hipEventElapsedTime |
| Timing sync (D2H) | встроен в cl_event | wall-clock (chrono) |
| Record в benchmark | `RecordEvent(name, ev)` | `RecordROCmEvent(name, data)` |
| Внутренние методы | возвращают `cl_event` | возвращают `void` |
| AllMaxima pipeline | события через callback | stream-ordered (нет hipEvent) |

---

## Типы и helpers

```cpp
// Тип (из DrvGPU/services/profiling_types.hpp):
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

// Helper A — для async GPU операций (hipEvent → elapsed):
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end, uint32_t kind, const char* op = "")
{
    hipEventSynchronize(ev_end);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev_start, ev_end);
    hipEventDestroy(ev_start); hipEventDestroy(ev_end);
    uint64_t ns = static_cast<uint64_t>(ms * 1e6f);
    return {.start_ns=0, .end_ns=ns, .complete_ns=ns, .kind=kind, .op_string=op};
}

// Helper B — для sync операций (wall-clock):
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1,
    uint32_t kind, const char* op = "")
{
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return {.start_ns=0, .end_ns=ns, .complete_ns=ns, .kind=kind, .op_string=op};
}
```

**Когда что использовать:**

| Операция | Helper | Причина |
|----------|--------|---------|
| `UploadData` (hipMemcpyAsync H2D) | `MakeROCmDataFromEvents` | async → hipEvent на stream |
| `ExecutePadKernel` | `MakeROCmDataFromEvents` | GPU kernel → hipEvent |
| `ExecuteFFT` (hipfftExecC2C) | `MakeROCmDataFromEvents` | GPU op → hipEvent |
| `ExecutePostKernel` | `MakeROCmDataFromEvents` | GPU kernel → hipEvent |
| `ReadResults` (hipMemcpy D2H sync) | `MakeROCmDataFromClock` | sync → wall-clock |
| `ExecuteComputeMagnitudes` | `MakeROCmDataFromEvents` | GPU kernel → hipEvent |
| `Pipeline::Execute` (целиком) | `MakeROCmDataFromEvents` | GPU ops → hipEvent |

---

## TASK 9 — Заголовок `spectrum_processor_rocm.hpp`

**Файл**: `modules/fft_maxima/include/processors/spectrum_processor_rocm.hpp`

### 9.1 Добавить include и type alias (в блоке #if ENABLE_ROCM)

```cpp
#include "DrvGPU/services/profiling_types.hpp"

namespace antenna_fft {
// ROCm prof_events тип — только для ENABLE_ROCM
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

### 9.2 Добавить prof_events к публичным методам

> **Про интерфейс**: Методы объявлены как `override` из `ISpectrumProcessor`.
> **Решение**: Добавить `prof_events*` также в `ISpectrumProcessor` с `= nullptr` (backward-compatible).
> OpenCL реализация (`SpectrumProcessorOpenCL`) просто добавляет `(void)prof_events;` в тело.

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

## TASK 10 — Инструментировать `spectrum_processor_rocm.cpp`

**Файл**: `modules/fft_maxima/src/spectrum_processor_rocm.cpp`

Инструментация только в публичных методах — **НЕ** в приватных (`UploadData`, `ExecuteFFT` и т.д.).
Приватные методы остаются void, timing оборачивается снаружи.

### 10.1 Добавить helpers в начало файла

```cpp
#include <chrono>
#include "DrvGPU/services/profiling_types.hpp"

static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end, uint32_t kind, const char* op = "")
{
    hipEventSynchronize(ev_end);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev_start, ev_end);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_end);
    drv_gpu_lib::ROCmProfilingData d{};
    uint64_t ns = static_cast<uint64_t>(ms * 1e6f);
    d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
    d.kind = kind; d.op_string = op;
    return d;
}

static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1,
    uint32_t kind, const char* op = "")
{
    uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    drv_gpu_lib::ROCmProfilingData d{};
    d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
    d.kind = kind; d.op_string = op;
    return d;
}
```

### 10.2 ProcessBatch — 5 стадий

Стадии: Upload → PadKernel → FFT → PostKernel → Download(sync)

```cpp
// Паттерн для каждой GPU стадии:
hipEvent_t ev_NAME_s = nullptr, ev_NAME_e = nullptr;
if (prof_events) {
    hipEventCreate(&ev_NAME_s);
    hipEventCreate(&ev_NAME_e);
    hipEventRecord(ev_NAME_s, stream_);
}

/* ... вызов операции (UploadData / ExecutePadKernel / ...) ... */

if (prof_events) hipEventRecord(ev_NAME_e, stream_);

// --- Для D2H (синхронная) — wall-clock вместо hipEvent:
auto t_dl_s = std::chrono::high_resolution_clock::now();
auto results = ReadResults(batch_antenna_count, start_antenna);
auto t_dl_e = std::chrono::high_resolution_clock::now();

// --- Собрать в конце метода:
if (prof_events) {
    prof_events->push_back({"Upload",     MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
    prof_events->push_back({"PadKernel",  MakeROCmDataFromEvents(ev_pad_s,  ev_pad_e,  0, "pad_kernel")});
    prof_events->push_back({"FFT",        MakeROCmDataFromEvents(ev_fft_s,  ev_fft_e,  0, "hipfftExecC2C")});
    prof_events->push_back({"PostKernel", MakeROCmDataFromEvents(ev_post_s, ev_post_e, 0, "post_kernel")});
    prof_events->push_back({"Download",   MakeROCmDataFromClock(t_dl_s,    t_dl_e,    1, "D2H")});
}
```

### 10.3 FindAllMaximaFromCPU — 5 стадий

Стадии: Upload → PadKernel → FFT(allmax) → ComputeMagnitudes → Pipeline

```cpp
// Аналогичный паттерн для каждой стадии...

if (prof_events) {
    prof_events->push_back({"Upload",            MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
    prof_events->push_back({"PadKernel",         MakeROCmDataFromEvents(ev_pad_s,  ev_pad_e,  0, "pad")});
    prof_events->push_back({"FFT",               MakeROCmDataFromEvents(ev_fft_s,  ev_fft_e,  0, "hipFFT")});
    prof_events->push_back({"ComputeMagnitudes", MakeROCmDataFromEvents(ev_mag_s,  ev_mag_e,  0, "mag")});
    prof_events->push_back({"Pipeline",          MakeROCmDataFromEvents(ev_pipe_s, ev_pipe_e, 0, "detect+scan+compact")});
}
```

> **Примечание**: Если нужна детализация Pipeline → Detect/Scan/Compact — выполнить TASK 11.

---

## TASK 11 — AllMaximaPipelineROCm с детальным timing (ОПЦИОНАЛЬНО)

**Файлы**:
- `modules/fft_maxima/include/interface/i_all_maxima_pipeline.hpp`
- `modules/fft_maxima/include/pipelines/all_maxima_pipeline_rocm.hpp`
- `modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp`

> **Нужен ли?** TASK 10 профилирует pipeline как единый блок `"Pipeline"`.
> TASK 11 добавляет раздельные `"Detect"`, `"Scan"`, `"Compact"`.
> Выполнять только если нужна такая детализация.

### 11.1 Обновить IAllMaximaPipeline

```cpp
#if ENABLE_ROCM
#include "DrvGPU/services/profiling_types.hpp"
using ROCmProfEventsPtr = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>*;
#endif

virtual AllMaximaResult Execute(
    void* magnitudes_gpu, void* fft_data_gpu,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 1, uint32_t search_end = 0,
    size_t max_maxima_per_beam = 1000
#if ENABLE_ROCM
    , ROCmProfEventsPtr prof_events = nullptr
#endif
    ) = 0;
```

### 11.2 Добавить hipEvent timing в AllMaximaPipelineROCm::Execute()

```cpp
// Detect:
hipEvent_t ev_det_s=nullptr, ev_det_e=nullptr;
if (prof_events) { hipEventCreate(&ev_det_s); hipEventCreate(&ev_det_e); hipEventRecord(ev_det_s,stream_); }
/* hipModuleLaunchKernel(detect_kernel_, ...) */
if (prof_events) hipEventRecord(ev_det_e, stream_);

// Scan (PrefixSum):
hipEvent_t ev_scan_s=nullptr, ev_scan_e=nullptr;
if (prof_events) { hipEventCreate(&ev_scan_s); hipEventCreate(&ev_scan_e); hipEventRecord(ev_scan_s,stream_); }
ExecutePrefixSum(flags_buf, scan_buf, nFFT, beam_count);
if (prof_events) hipEventRecord(ev_scan_e, stream_);

// Compact:
hipEvent_t ev_comp_s=nullptr, ev_comp_e=nullptr;
if (prof_events) { hipEventCreate(&ev_comp_s); hipEventCreate(&ev_comp_e); hipEventRecord(ev_comp_s,stream_); }
/* hipModuleLaunchKernel(compact_kernel_, ...) */
if (prof_events) hipEventRecord(ev_comp_e, stream_);

// Собрать:
if (prof_events) {
    prof_events->push_back({"Detect",  MakeROCmDataFromEvents(ev_det_s,  ev_det_e,  0, "detect")});
    prof_events->push_back({"Scan",    MakeROCmDataFromEvents(ev_scan_s, ev_scan_e, 0, "prefix_sum")});
    prof_events->push_back({"Compact", MakeROCmDataFromEvents(ev_comp_s, ev_comp_e, 0, "compact")});
}
```

> Если TASK 11 выполнен → в TASK 10 заменить обёртку вокруг pipeline:
> ```cpp
> // Вместо отдельных ev_pipe_s/ev_pipe_e:
> pipeline_->Execute(..., prof_events);   // события уже добавятся внутри
> ```

---

## TASK 12 — Создать `tests/fft_maxima_benchmark_rocm.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/fft_maxima_benchmark_rocm.hpp`

```cpp
#pragma once

#if ENABLE_ROCM

/**
 * @file fft_maxima_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для SpectrumProcessorROCm (GpuBenchmarkBase)
 *
 * SpectrumProcessorROCmBenchmark        → ProcessFromCPU: Upload+Pad+FFT+PostKernel+Download
 * SpectrumProcessorROCmAllMaximaBenchmark → FindAllMaximaFromCPU: Upload+Pad+FFT+Mag+Pipeline
 */

#include "processors/spectrum_processor_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <complex>
#include <vector>

namespace test_fft_maxima_rocm {

// ─── Benchmark 1: ProcessFromCPU (ONE_PEAK) ───────────────────────────────

class SpectrumProcessorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumProcessorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_Process", cfg),
      proc_(proc), input_data_(input_data) {}

protected:
  void ExecuteKernel() override {
    proc_.ProcessFromCPU(input_data_);   // warmup: prof_events = nullptr
  }

  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.ProcessFromCPU(input_data_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  antenna_fft::SpectrumProcessorROCm& proc_;
  std::vector<std::complex<float>>    input_data_;
};

// ─── Benchmark 2: FindAllMaximaFromCPU ────────────────────────────────────

class SpectrumProcessorROCmAllMaximaBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SpectrumProcessorROCmAllMaximaBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_AllMaxima", cfg),
      proc_(proc), input_data_(input_data) {}

protected:
  void ExecuteKernel() override {
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU, 1, 0);
  }

  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU, 1, 0,
                               &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  antenna_fft::SpectrumProcessorROCm& proc_;
  std::vector<std::complex<float>>    input_data_;
};

}  // namespace test_fft_maxima_rocm

#endif  // ENABLE_ROCM
```

---

## TASK 13 — Создать `tests/test_fft_maxima_benchmark_rocm.hpp`

**Файл НОВЫЙ**: `modules/fft_maxima/tests/test_fft_maxima_benchmark_rocm.hpp`

```cpp
#pragma once

#if ENABLE_ROCM

/**
 * @file test_fft_maxima_benchmark_rocm.hpp
 * @brief ROCm test runner: SpectrumProcessorROCm benchmark (GpuBenchmarkBase)
 *
 * Benchmark 1: ProcessFromCPU   → Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/
 * Benchmark 2: FindAllMaximaFromCPU → Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/
 *
 * Запускается ТОЛЬКО на Linux + AMD GPU (ENABLE_ROCM=1).
 * На Windows/без AMD: catch → [SKIP], не ошибка.
 */

#include "fft_maxima_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "processors/spectrum_processor_rocm.hpp"

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

    try {
        // ── ROCm backend ──────────────────────────────────────────────
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

        antenna_fft::SpectrumProcessorROCm proc(b);
        proc.Initialize(params);

        // ── Benchmark 1: Process (ONE_PEAK) ───────────────────────────
        std::cout << "\n--- ROCm Benchmark 1: ProcessFromCPU (ONE_PEAK) ---\n";
        {
            test_fft_maxima_rocm::SpectrumProcessorROCmBenchmark bench(
                b, proc, signal,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"});

            if (!bench.IsProfEnabled())
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] ROCm Process benchmark complete\n";
            }
        }

        // ── Benchmark 2: FindAllMaxima ─────────────────────────────────
        std::cout << "\n--- ROCm Benchmark 2: FindAllMaximaFromCPU (pipeline) ---\n";
        {
            test_fft_maxima_rocm::SpectrumProcessorROCmAllMaximaBenchmark bench(
                b, proc, signal,
                {.n_warmup   = 5,
                 .n_runs     = 20,
                 .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"});

            if (!bench.IsProfEnabled())
                std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
            else {
                bench.Run();
                bench.Report();
                std::cout << "  [OK] ROCm AllMaxima benchmark complete\n";
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cout << "  [SKIP] ROCm not available: " << e.what() << "\n";
        return 0;  // ← не ошибка: просто нет AMD GPU
    }
}

}  // namespace test_fft_maxima_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

## TASK 14 — Обновить `tests/all_test.hpp` (ROCm)

**Файл**: `modules/fft_maxima/tests/all_test.hpp`

```cpp
// Добавить в начало файла:
#if ENABLE_ROCM
#include "test_fft_maxima_benchmark_rocm.hpp"
#endif

namespace fft_maxima_all_test {
inline void run() {
    // ...

    // ROCm: тесты корректности (уже был):
#if ENABLE_ROCM
    test_spectrum_maxima_rocm::run();
#endif

    // ROCm: Benchmark (GpuBenchmarkBase) — раскомментировать для запуска:
#if ENABLE_ROCM
//  test_fft_maxima_benchmark_rocm::run();
#endif
}
}
```

---

## Итог: изменённые файлы

```
modules/fft_maxima/
├── include/
│   ├── interface/
│   │   └── i_all_maxima_pipeline.hpp     ← ИЗМЕНЁН (TASK 11, опционально)
│   └── processors/
│       └── spectrum_processor_rocm.hpp   ← ИЗМЕНЁН (TASK 9)
├── src/
│   ├── spectrum_processor_rocm.cpp       ← ИЗМЕНЁН (TASK 10)
│   └── all_maxima_pipeline_rocm.cpp      ← ИЗМЕНЁН (TASK 11, опционально)
└── tests/
    ├── fft_maxima_benchmark_rocm.hpp     ← НОВЫЙ   (TASK 12)
    ├── test_fft_maxima_benchmark_rocm.hpp← НОВЫЙ   (TASK 13)
    └── all_test.hpp                      ← ИЗМЕНЁН (TASK 14)
```

---

## Чеклист

- [ ] `ROCmProfEvents` тип доступен в `antenna_fft` namespace (TASK 9)
- [ ] `ProcessFromCPU(data, &events)` — заполняет 5 событий (Upload/Pad/FFT/Post/Download)
- [ ] `FindAllMaximaFromCPU(data,..., &events)` — заполняет 5 событий (Upload/Pad/FFT/Mag/Pipeline)
- [ ] `ProcessFromCPU(data)` без prof_events — **ноль overhead** (нет hipEventCreate)
- [ ] ROCm benchmark под `#if ENABLE_ROCM` (compile-only на Windows)
- [ ] `ROCmBackend::Initialize` fail → `[SKIP]`, не ошибка
- [ ] `IsProfEnabled() = false` → `[SKIP]`, не ошибка
- [ ] После запуска: `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/`
- [ ] После запуска: `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/`

---

*Создано: 2026-03-01*
*Автор: Кодо (AI Assistant)*
