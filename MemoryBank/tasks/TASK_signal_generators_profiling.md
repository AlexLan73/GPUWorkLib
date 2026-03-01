# TASK: Рефакторинг профилирования modules/signal_generators → GpuBenchmarkBase (OpenCL + ROCm)

> **Ветка**: `Profiller01`
> **Создана**: 2026-03-01
> **Статус**: 📋 PLAN (не начата)
> **Референс-реализация**: `modules/fft_processor/` (полный эталон)
> **Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`

---

## Цель

Привести профилирование `modules/signal_generators` в соответствие с паттерном из `modules/fft_processor`.

**Принцип**: production-класс — **чистый** (ноль кода профилирования по умолчанию).
Весь код профилирования изолирован в тест-файлах (`/tests/`).

---

## Анализ модуля

### Генераторы OpenCL (8 штук)

| Класс | Файлы | Основные методы | Kernel |
|-------|-------|-----------------|--------|
| `CwGenerator` | cw_generator.hpp/cpp | `GenerateToGpu(system, beam_count)`, `GenerateToCpu(system, out, size)` | cw_kernel.cl |
| `LfmGenerator` | lfm_generator.hpp/cpp | `GenerateToGpu(system, beam_count)`, `GenerateToCpu(system, out, size)` | lfm_kernel.cl |
| `NoiseGenerator` | noise_generator.hpp/cpp | `GenerateToGpu(system, beam_count)`, `GenerateToCpu(system, out, size)` | noise_kernel.cl |
| `LfmConjugateGenerator` | lfm_conjugate_generator.hpp/cpp | `GenerateToGpu()`, `GenerateToCpu()` | lfm_conjugate.cl |
| `LfmGeneratorAnalyticalDelay` | lfm_generator_analytical_delay.hpp/cpp | `GenerateToGpu()` → `InputData<cl_mem>`, `GenerateToCpu()` | lfm_analytical_delay.cl |
| `FormSignalGenerator` | form_signal_generator.hpp/cpp | `GenerateInputData()` → `InputData<cl_mem>`, `GenerateToCpu()` | form_signal.cl |
| `DelayedFormSignalGenerator` | delayed_form_signal_generator.hpp/cpp | `GenerateInputData()` → `InputData<cl_mem>`, `GenerateToCpu()` | form_signal.cl + delayed_form_signal.cl (2 kernel прохода!) |
| `FormScriptGenerator` | form_script_generator.hpp/cpp | `GenerateInputData()` → `InputData<cl_mem>`, `GenerateToCpu()` | DSL kernel (runtime) |

### Генератор ROCm (1 штука)

| Класс | Файлы | Основные методы | Kernel |
|-------|-------|-----------------|--------|
| `FormSignalGeneratorROCm` | form_signal_generator_rocm.hpp/cpp | `GenerateInputData()` → `InputData<void*>`, `GenerateToCpu()` | form_signal.hip |

### Интерфейс ISignalGenerator (НЕ ТРОГАТЬ!)

`ISignalGenerator` — базовый интерфейс для CW/LFM/Noise.
**Правило**: prof_events добавляются только к конкретным классам (не к интерфейсу),
потому что benchmark классы работают с конкретными типами, а не через полиморфизм.

### Текущее состояние

- ❌ Профилирование ОТСУТСТВУЕТ полностью
- ❌ Нет benchmark файлов
- ❌ Нет `*_benchmark.hpp`, нет `test_*_benchmark.hpp`
- ✅ Все генераторы рабочие (27+ тестов)

---

## Новые файлы (создать)

```
modules/signal_generators/tests/
├── signal_generators_benchmark.hpp          ← C1: OpenCL benchmark (CW + LFM + LfmConj + Noise)
├── test_signal_generators_benchmark.hpp     ← D1: OpenCL test runner (CW/LFM/Noise)
├── form_signal_benchmark.hpp                ← C2: OpenCL benchmark (Form + Delayed + Script + LfmDelay)
├── test_form_signal_benchmark.hpp           ← D2: OpenCL test runner (Form генераторы)
├── signal_generators_benchmark_rocm.hpp     ← E1: ROCm benchmark (FormSignalROCm)
└── test_signal_generators_benchmark_rocm.hpp← E2: ROCm test runner
```

## Изменения в существующих файлах

```
modules/signal_generators/
├── include/generators/cw_generator.hpp                    ← A1: добавить prof_events
├── include/generators/lfm_generator.hpp                   ← A2: добавить prof_events
├── include/generators/noise_generator.hpp                  ← A3: добавить prof_events
├── include/generators/lfm_conjugate_generator.hpp          ← A4: добавить prof_events
├── include/generators/lfm_generator_analytical_delay.hpp   ← A5: добавить prof_events
├── include/generators/form_signal_generator.hpp             ← A6: добавить prof_events
├── include/generators/delayed_form_signal_generator.hpp     ← A7: добавить prof_events
├── include/generators/form_script_generator.hpp             ← A8: добавить prof_events
├── include/generators/form_signal_generator_rocm.hpp        ← B1: добавить ROCmProfEvents
├── src/cw_generator.cpp                                    ← A1: impl + CollectOrRelease
├── src/lfm_generator.cpp                                   ← A2: impl + CollectOrRelease
├── src/noise_generator.cpp                                  ← A3: impl + CollectOrRelease
├── src/lfm_conjugate_generator.cpp                          ← A4: impl + CollectOrRelease
├── src/lfm_generator_analytical_delay.cpp                   ← A5: impl + CollectOrRelease
├── src/form_signal_generator.cpp                            ← A6: impl + CollectOrRelease
├── src/delayed_form_signal_generator.cpp                    ← A7: impl (2 kernel stages)
├── src/form_script_generator.cpp                            ← A8: impl + CollectOrRelease
├── src/form_signal_generator_rocm.cpp                       ← B1: impl + MakeROCmData*
└── tests/
    ├── all_test.hpp                                         ← F1: добавить includes + вызовы
    └── README.md                                            ← F2: описание новых тестов
```

---

## ДЕТАЛЬНЫЙ ПЛАН ТАСКОВ

---

### ════ ГРУППА A: Production code — OpenCL prof_events ════

---

#### TASK A1: CwGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/cw_generator.hpp`, `src/cw_generator.cpp`

**В заголовке** — добавить перегрузки с prof_events:
```cpp
#include <utility>  // pair

// Тип для prof_events (OpenCL)
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

// Перегрузки с профилированием
cl_mem GenerateToGpu(const SystemSampling& system,
                     size_t beam_count = 1,
                     ProfEvents* prof_events = nullptr);

void GenerateToCpu(const SystemSampling& system,
                   std::complex<float>* out,
                   size_t out_size,
                   ProfEvents* prof_events = nullptr);
```

**В реализации** — добавить static helper + использовать в методах:
```cpp
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) prof_events->push_back({name, ev});
    else clReleaseEvent(ev);
}

// В GenerateToGpu():
//   clEnqueueNDRangeKernel(..., &ev_kernel)  — сохранить event
//   CollectOrRelease(ev_kernel, "Kernel", prof_events);

// В GenerateToCpu():
//   CollectOrRelease(ev_kernel, "Kernel", prof_events);
//   clEnqueueReadBuffer(..., &ev_download)
//   CollectOrRelease(ev_download, "Download", prof_events);
```

**Этапы профилирования**:
- `GenerateToGpu()` → `"Kernel"` (cw_kernel)
- `GenerateToCpu()` → `"Kernel"` + `"Download"` (D2H)

---

#### TASK A2: LfmGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/lfm_generator.hpp`, `src/lfm_generator.cpp`

Аналогично A1 — тот же паттерн, тот же API.

**Этапы профилирования**:
- `GenerateToGpu()` → `"Kernel"` (lfm_kernel)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

---

#### TASK A3: NoiseGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/noise_generator.hpp`, `src/noise_generator.cpp`

Аналогично A1.

**Этапы профилирования**:
- `GenerateToGpu()` → `"Kernel"` (noise_kernel / Philox+BoxMuller)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

---

#### TASK A4: LfmConjugateGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/lfm_conjugate_generator.hpp`, `src/lfm_conjugate_generator.cpp`

У этого класса другой API (не ISignalGenerator):
```cpp
cl_mem GenerateToGpu(ProfEvents* prof_events = nullptr);
std::vector<std::complex<float>> GenerateToCpu(ProfEvents* prof_events = nullptr);
```

**Этапы профилирования**:
- `GenerateToGpu()` → `"Kernel"` (lfm_conjugate.cl)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

---

#### TASK A5: LfmGeneratorAnalyticalDelay — добавить OpenCL prof_events

**Файлы**: `include/generators/lfm_generator_analytical_delay.hpp`, `src/lfm_generator_analytical_delay.cpp`

```cpp
drv_gpu_lib::InputData<cl_mem> GenerateToGpu(ProfEvents* prof_events = nullptr);
std::vector<std::vector<std::complex<float>>> GenerateToCpu(ProfEvents* prof_events = nullptr);
```

**Этапы профилирования**:
- `GenerateToGpu()` → `"Kernel"` + (опционально `"Download"` — если есть D2H внутри)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

⚠️ Нужно внимательно изучить `.cpp` чтобы понять структуру внутренних вызовов.

---

#### TASK A6: FormSignalGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/form_signal_generator.hpp`, `src/form_signal_generator.cpp`

```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events = nullptr);
std::vector<std::vector<std::complex<float>>> GenerateToCpu(ProfEvents* prof_events = nullptr);
```

**Этапы профилирования**:
- `GenerateInputData()` → `"Kernel"` (form_signal.cl — мультиканальный)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

---

#### TASK A7: DelayedFormSignalGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/delayed_form_signal_generator.hpp`, `src/delayed_form_signal_generator.cpp`

⚠️ **Сложнее других**: внутри 2 kernel прохода:
1. `FormSignalGenerator::GenerateInputData()` — генерация чистого сигнала
2. Farrow delay kernel — применение задержки + шум

```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events = nullptr);
std::vector<std::vector<std::complex<float>>> GenerateToCpu(ProfEvents* prof_events = nullptr);
```

**Этапы профилирования**:
- `GenerateInputData()` → `"FormSignal"` (form_signal.cl) + `"FarrowDelay"` (delayed_form_signal.cl)
- `GenerateToCpu()` → `"FormSignal"` + `"FarrowDelay"` + `"Download"`

**Техническая задача**: `FormSignalGenerator` (внутренний `signal_gen_`) тоже нужно вызвать
с prof_events, а потом добавить FarrowDelay event. Т.е. нужно:
```cpp
ProfEvents internal_events;
signal_gen_.GenerateInputData(&internal_events);  // → "Kernel" event внутри
// переименовать event: "Kernel" → "FormSignal"
CollectOrRelease(farrow_ev, "FarrowDelay", prof_events);
```
Или лучше передать prof_events напрямую во внутренний FormSignalGenerator и
добавить farrow event с другим именем.

---

#### TASK A8: FormScriptGenerator — добавить OpenCL prof_events

**Файлы**: `include/generators/form_script_generator.hpp`, `src/form_script_generator.cpp`

```cpp
drv_gpu_lib::InputData<cl_mem> GenerateInputData(ProfEvents* prof_events = nullptr);
std::vector<std::vector<std::complex<float>>> GenerateToCpu(ProfEvents* prof_events = nullptr);
```

**Этапы профилирования** (аналогично FormSignalGenerator):
- `GenerateInputData()` → `"Kernel"` (DSL-скомпилированный kernel)
- `GenerateToCpu()` → `"Kernel"` + `"Download"`

---

### ════ ГРУППА B: Production code — ROCm prof_events ════

---

#### TASK B1: FormSignalGeneratorROCm — добавить ROCmProfEvents

**Файлы**: `include/generators/form_signal_generator_rocm.hpp`, `src/form_signal_generator_rocm.cpp`

**Только под ENABLE_ROCM** (stub остаётся без изменений).

**В заголовке** (внутри `#if ENABLE_ROCM`):
```cpp
#include "DrvGPU/services/profiling_types.hpp"

// Тип для ROCm events
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

drv_gpu_lib::InputData<void*> GenerateInputData(ROCmProfEvents* prof_events = nullptr);
std::vector<std::vector<std::complex<float>>> GenerateToCpu(ROCmProfEvents* prof_events = nullptr);
```

**В реализации** — добавить helpers:
```cpp
// Helper A: для async GPU операций (hipEvent_t → hipEventElapsedTime)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end, uint32_t kind, const char* op_string = "");

// Helper B: для sync CPU/GPU операций (wall-clock через std::chrono)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t_start,
    std::chrono::high_resolution_clock::time_point t_end,
    uint32_t kind, const char* op_string = "");
```

**Этапы профилирования**:
- `GenerateInputData()` → `"Kernel"` (HIP kernel, hipEvent_t, MakeROCmDataFromEvents)
- `GenerateToCpu()` → `"Kernel"` (hipEvent) + `"Download"` (sync D2H → MakeROCmDataFromClock)

**Паттерн в GenerateInputData()**:
```cpp
hipEvent_t ev_k_start = nullptr, ev_k_end = nullptr;
if (prof_events) {
    hipEventCreate(&ev_k_start);
    hipEventCreate(&ev_k_end);
    hipEventRecord(ev_k_start, stream_);
}

// ... hipLaunchKernelGGL(form_signal_kernel, ...) ...

if (prof_events) {
    hipEventRecord(ev_k_end, stream_);
    hipStreamSynchronize(stream_);
    prof_events->push_back({"Kernel", MakeROCmDataFromEvents(ev_k_start, ev_k_end, 0, "form_signal")});
}
```

---

### ════ ГРУППА C: OpenCL Benchmark классы ════

---

#### TASK C1: Создать `tests/signal_generators_benchmark.hpp`

Файл содержит **4 benchmark класса** для CW, LFM, LfmConjugate, Noise.

**Структура файла:**
```cpp
#pragma once

#include "../include/generators/cw_generator.hpp"
#include "../include/generators/lfm_generator.hpp"
#include "../include/generators/lfm_conjugate_generator.hpp"
#include "../include/generators/noise_generator.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <CL/cl.h>
#include <vector>
#include <complex>

namespace test_signal_generators {

// ══════════════════════════════════════════════════════
// CwGeneratorBenchmark
// ══════════════════════════════════════════════════════
class CwGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  CwGeneratorBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::CwGenerator& gen,
      const signal_gen::SystemSampling& system,
      size_t beam_count = 1,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_CwGenerator"})
    : GpuBenchmarkBase(backend, "CwGenerator", cfg),
      gen_(gen), system_(system), beam_count_(beam_count) {}

protected:
  void ExecuteKernel() override {
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_);
    clReleaseMemObject(buf);
  }
  void ExecuteKernelTimed() override {
    signal_gen::CwGenerator::ProfEvents events;
    cl_mem buf = gen_.GenerateToGpu(system_, beam_count_, &events);
    clReleaseMemObject(buf);
    for (auto& [name, ev] : events) RecordEvent(name, ev);
  }

private:
  signal_gen::CwGenerator& gen_;
  signal_gen::SystemSampling system_;
  size_t beam_count_;
};

// ══════════════════════════════════════════════════════
// LfmGeneratorBenchmark
// ══════════════════════════════════════════════════════
// ... аналогично CwGeneratorBenchmark ...
// output_dir = "Results/Profiler/GPU_00_LfmGenerator"

// ══════════════════════════════════════════════════════
// LfmConjugateGeneratorBenchmark
// ══════════════════════════════════════════════════════
// ... аналогично, но GenerateToGpu() без system/beam_count ...
// output_dir = "Results/Profiler/GPU_00_LfmConjugateGenerator"

// ══════════════════════════════════════════════════════
// NoiseGeneratorBenchmark
// ══════════════════════════════════════════════════════
// ... аналогично CwGeneratorBenchmark ...
// output_dir = "Results/Profiler/GPU_00_NoiseGenerator"

}  // namespace test_signal_generators
```

---

#### TASK C2: Создать `tests/form_signal_benchmark.hpp`

Файл содержит **4 benchmark класса** для Form генераторов.

**Классы:**
- `FormSignalGeneratorBenchmark` — `GenerateInputData()` → output_dir `Results/Profiler/GPU_00_FormSignal`
- `DelayedFormSignalGeneratorBenchmark` — `GenerateInputData()` → output_dir `Results/Profiler/GPU_00_DelayedFormSignal`
- `LfmAnalyticalDelayBenchmark` — `GenerateToGpu()` → output_dir `Results/Profiler/GPU_00_LfmAnalyticalDelay`
- `FormScriptGeneratorBenchmark` — `GenerateInputData()` → output_dir `Results/Profiler/GPU_00_FormScriptGenerator`

**Паттерн для FormSignalGeneratorBenchmark:**
```cpp
class FormSignalGeneratorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
protected:
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();    // prof_events = nullptr → warmup
    clReleaseMemObject(input.data);
  }
  void ExecuteKernelTimed() override {
    signal_gen::FormSignalGenerator::ProfEvents events;
    auto input = gen_.GenerateInputData(&events);
    clReleaseMemObject(input.data);
    for (auto& [name, ev] : events) RecordEvent(name, ev);
  }
};
```

---

### ════ ГРУППА D: OpenCL Test Runners ════

---

#### TASK D1: Создать `tests/test_signal_generators_benchmark.hpp`

Test runner для CW, LFM, LfmConjugate, Noise.

**Структура:**
```cpp
#pragma once

#include "signal_generators_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>

namespace test_signal_generators_benchmark {

inline int run() {
  std::cout << "\n============================================================\n";
  std::cout << "  Signal Generators Benchmark (CW / LFM / Noise)\n";
  std::cout << "============================================================\n";

  try {
    // ── OpenCL init с CL_QUEUE_PROFILING_ENABLE ────────────────────────
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->InitializeFromExternalContext(ctx, device, queue);

    // ── Параметры ─────────────────────────────────────────────────────
    signal_gen::SystemSampling system;
    system.fs     = 12e6;
    system.length = 4096;

    // ── CW Benchmark ──────────────────────────────────────────────────
    {
      signal_gen::CwParams cw_params;
      cw_params.frequency  = 1e6;
      cw_params.amplitude  = 1.0;
      signal_gen::CwGenerator cw_gen(backend.get(), cw_params);

      test_signal_generators::CwGeneratorBenchmark bench(
          backend.get(), cw_gen, system, /*beam_count=*/1,
          {.n_warmup=5, .n_runs=20, .output_dir="Results/Profiler/GPU_00_CwGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] CW: is_prof=false\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] CW Benchmark complete\n";
      }
    }

    // ── LFM Benchmark ─────────────────────────────────────────────────
    { /* ... аналогично ... */ }

    // ── Noise Benchmark ───────────────────────────────────────────────
    { /* ... аналогично ... */ }

    // ── Cleanup ───────────────────────────────────────────────────────
    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_signal_generators_benchmark
```

---

#### TASK D2: Создать `tests/test_form_signal_benchmark.hpp`

Test runner для FormSignalGenerator, DelayedFormSignal, LfmAnalyticalDelay, FormScriptGenerator.

Аналогично D1 по структуре. Запускает по одному benchmark для каждого генератора.

---

### ════ ГРУППА E: ROCm Benchmark ════

---

#### TASK E1: Создать `tests/signal_generators_benchmark_rocm.hpp`

```cpp
#pragma once
#if ENABLE_ROCM

#include "../include/generators/form_signal_generator_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

namespace test_signal_generators_rocm {

class FormSignalGeneratorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  FormSignalGeneratorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      signal_gen::FormSignalGeneratorROCm& gen,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_FormSignalROCm"})
    : GpuBenchmarkBase(backend, "FormSignalROCm", cfg), gen_(gen) {}

protected:
  void ExecuteKernel() override {
    auto input = gen_.GenerateInputData();    // prof_events = nullptr → warmup
    hipFree(input.data);
  }
  void ExecuteKernelTimed() override {
    signal_gen::FormSignalGeneratorROCm::ROCmProfEvents events;
    auto input = gen_.GenerateInputData(&events);
    hipFree(input.data);
    for (auto& [name, data] : events) RecordROCmEvent(name, data);
  }

private:
  signal_gen::FormSignalGeneratorROCm& gen_;
};

}  // namespace test_signal_generators_rocm

#endif  // ENABLE_ROCM
```

---

#### TASK E2: Создать `tests/test_signal_generators_benchmark_rocm.hpp`

```cpp
#pragma once
#if ENABLE_ROCM

#include "signal_generators_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include <iostream>
#include <stdexcept>

namespace test_signal_generators_benchmark_rocm {

inline int run() {
  std::cout << "\n============================================================\n";
  std::cout << "  FormSignalGeneratorROCm Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  std::cout << "  Available ROCm devices: " << device_count << "\n";
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices found\n";
    return 0;
  }

  try {
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    signal_gen::FormParams params;
    params.fs       = 12e6;
    params.f0       = 1e6;
    params.antennas = 8;
    params.points   = 4096;

    signal_gen::FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(params);

    test_signal_generators_rocm::FormSignalGeneratorROCmBenchmark bench(
        &backend, gen,
        {.n_warmup=5, .n_runs=20, .output_dir="Results/Profiler/GPU_00_FormSignalROCm"});

    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false\n";
    } else {
      bench.Run();
      bench.Report();
      std::cout << "  [OK] Benchmark complete\n";
    }
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_signal_generators_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

### ════ ГРУППА F: Интеграция ════

---

#### TASK F1: Обновить `tests/all_test.hpp`

Добавить новые includes и вызовы бенчмарков:

```cpp
// Добавить после существующих includes:
#include "test_signal_generators_benchmark.hpp"
#include "test_form_signal_benchmark.hpp"
#if ENABLE_ROCM
#include "test_signal_generators_benchmark_rocm.hpp"
#endif

// В теле run():
// Signal Generators Benchmarks (CW, LFM, LfmConjugate, Noise)
//   test_signal_generators_benchmark::run();  // раскомментировать для запуска

// Form Signal Benchmarks (Form, Delayed, Script, LfmAnalyticalDelay)
//   test_form_signal_benchmark::run();  // раскомментировать для запуска

#if ENABLE_ROCM
// FormSignalROCm Benchmark
//   test_signal_generators_benchmark_rocm::run();  // раскомментировать для запуска
#endif
```

---

#### TASK F2: Обновить `tests/README.md`

Добавить описание новых benchmark тестов в секцию README:
- Описание всех 6 новых файлов
- Как запустить (раскомментировать в all_test.hpp)
- Где смотреть результаты (Results/Profiler/GPU_00_*)

---

## Сводка тасков

| Таск | Файл(ы) | Тип | Статус |
|------|---------|-----|--------|
| A1 | cw_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| A2 | lfm_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| A3 | noise_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| A4 | lfm_conjugate_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| A5 | lfm_generator_analytical_delay.hpp/cpp | Production OpenCL | ⬜ TODO |
| A6 | form_signal_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| A7 | delayed_form_signal_generator.hpp/cpp | Production OpenCL (2 kernels!) | ⬜ TODO |
| A8 | form_script_generator.hpp/cpp | Production OpenCL | ⬜ TODO |
| B1 | form_signal_generator_rocm.hpp/cpp | Production ROCm | ⬜ TODO |
| C1 | tests/signal_generators_benchmark.hpp | Benchmark CW+LFM+Noise | ⬜ TODO |
| C2 | tests/form_signal_benchmark.hpp | Benchmark Form+Delayed+Script+LfmDelay | ⬜ TODO |
| D1 | tests/test_signal_generators_benchmark.hpp | Test runner OpenCL | ⬜ TODO |
| D2 | tests/test_form_signal_benchmark.hpp | Test runner OpenCL | ⬜ TODO |
| E1 | tests/signal_generators_benchmark_rocm.hpp | Benchmark ROCm | ⬜ TODO |
| E2 | tests/test_signal_generators_benchmark_rocm.hpp | Test runner ROCm | ⬜ TODO |
| F1 | tests/all_test.hpp | Интеграция | ⬜ TODO |
| F2 | tests/README.md | Документация | ⬜ TODO |

**Итого**: 17 тасков (9 production + 2 benchmark + 2 runners + 2 ROCm + 2 integration)

---

## Правила выполнения

### Порядок выполнения
1. **Сначала** группа A + B (production code) — без них не скомпилируются benchmarks
2. **Потом** группа C + D (OpenCL benchmark классы и runners)
3. **Затем** группа E (ROCm — только на Linux с AMD GPU)
4. **В конце** группа F (интеграция — all_test.hpp + README.md)

### Ключевые правила (из инструкции)
- ✅ `CollectOrRelease` — static helper внутри .cpp (не в .hpp!)
- ✅ `prof_events = nullptr` — **обязательно дефолтное значение** (zero overhead в production)
- ✅ `CL_QUEUE_PROFILING_ENABLE` — обязательный флаг в test runner (иначе timing = 0)
- ✅ `bench.IsProfEnabled()` — проверка перед `Run()`
- ✅ `bench.Report()` — единственный способ вывода результатов
- ❌ ЗАПРЕЩЕНО: `GetStats()` + цикл + `std::cout` напрямую
- ❌ ЗАПРЕЩЕНО: менять интерфейс `ISignalGenerator`

### Особые случаи

**DelayedFormSignalGenerator (A7)**:
Внутри есть `FormSignalGenerator signal_gen_`. При передаче prof_events нужно:
1. Вызвать `signal_gen_.GenerateInputData(&events)` → event называется "Kernel"
2. Переименовать его в "FormSignal" (для читаемости в отчёте)
3. Добавить FarrowDelay event с именем "FarrowDelay"

**FormScriptGenerator (A8)**:
DSL kernel компилируется при `Compile()`. Профилируется только `GenerateInputData()`,
не компиляция. API аналогичен FormSignalGenerator.

**FormSignalGeneratorROCm (B1)**:
- Только в `#if ENABLE_ROCM` блоке
- `GenerateToCpu()` использует sync `hipMemcpyDtoH` → `MakeROCmDataFromClock`
- `GenerateInputData()` — async HIP kernel → `MakeROCmDataFromEvents`

---

## Результаты (куда записывает GPUProfiler)

```
Results/Profiler/
├── GPU_00_CwGenerator/          ← CW benchmark
├── GPU_00_LfmGenerator/         ← LFM benchmark
├── GPU_00_LfmConjugateGenerator/← LfmConjugate benchmark
├── GPU_00_NoiseGenerator/       ← Noise benchmark
├── GPU_00_FormSignal/           ← FormSignal benchmark
├── GPU_00_DelayedFormSignal/    ← DelayedFormSignal benchmark (2 events)
├── GPU_00_LfmAnalyticalDelay/   ← LfmAnalyticalDelay benchmark
├── GPU_00_FormScriptGenerator/  ← FormScript benchmark
└── GPU_00_FormSignalROCm/       ← ROCm benchmark
```
