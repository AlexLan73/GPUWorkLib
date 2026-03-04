# Механизм профилирования GPU-модулей: OpenCL & ROCm

> **Версия**: 1.0 | **Дата**: 2026-03-01
> **Автор**: Кодо (AI Assistant)
> **Референс-реализация**: `modules/fft_processor/` (OpenCL), `modules/fft_processor/` (ROCm)

---

## Содержание

1. [Принцип работы](#1-принцип-работы)
2. [Архитектура потока](#2-архитектура-потока)
3. [OpenCL: шаг за шагом](#3-opencl-шаг-за-шагом)
   - [Шаг 1 — production-класс: добавить prof_events](#шаг-1--production-класс-добавить-prof_events)
   - [Шаг 2 — benchmark-класс: наследник GpuBenchmarkBase](#шаг-2--benchmark-класс-наследник-gpubenchmarkbase)
   - [Шаг 3 — test runner](#шаг-3--test-runner)
   - [Шаг 4 — all_test.hpp](#шаг-4--all_testhpp)
4. [ROCm: шаг за шагом](#4-rocm-шаг-за-шагом)
   - [Шаг 1 — production-класс: добавить ROCmProfEvents](#шаг-1--production-класс-добавить-rocmprofevents)
   - [Шаг 2 — helper функции для timing](#шаг-2--helper-функции-для-timing)
   - [Шаг 3 — benchmark-класс](#шаг-3--benchmark-класс)
   - [Шаг 4 — test runner](#шаг-4--test-runner-1)
   - [Шаг 5 — all_test.hpp](#шаг-5--all_testhpp)
5. [Структура файлов модуля](#5-структура-файлов-модуля)
6. [Чеклист добавления профилирования](#6-чеклист-добавления-профилирования)
7. [Частые ошибки](#7-частые-ошибки)

---

## 1. Принцип работы

**Ключевой принцип**: production-класс модуля — **чистый** (ноль кода профилирования).
Весь код профилирования изолирован в тест-файлах (`/tests/`).

```
Production-класс            Benchmark-класс (тест)
─────────────────           ──────────────────────────────
MyModule::Process()         ExecuteKernel()      → warmup (без timing)
  - делает полезную работу  ExecuteKernelTimed() → Record() → GPUProfiler
  - prof_events = nullptr
    → ноль overhead
  - prof_events = &vec      GpuBenchmarkBase::Run()
    → собирает события        1. warmup × n_warmup
                              2. Reset profiler
                              3. measure × n_runs
```

**Профилирование включается в `configGPU.json`** (поле `is_prof: true`).
Если `is_prof: false` — `Run()` и `Report()` — полные no-op, ноль overhead в production.

---

## 2. Архитектура потока

```
main.cpp
  └── all_test.hpp
        └── test_my_module_benchmark::run()           [test runner]
              ├── InitBackend()                        [OpenCL/ROCm init]
              ├── MyModule proc(backend)               [production объект]
              └── MyModuleBenchmark bench(backend, proc, ...)
                    │
                    ├── bench.Run()
                    │     ├── warmup × n_warmup
                    │     │     └── ExecuteKernel()
                    │     │           └── proc.Process(data)   ← без prof_events
                    │     │
                    │     ├── GPUProfiler::Reset()
                    │     │
                    │     └── measure × n_runs
                    │           └── ExecuteKernelTimed()
                    │                 ├── proc.Process(data, &events)
                    │                 └── RecordEvent(name, ev)  ← → GPUProfiler
                    │
                    └── bench.Report()
                          ├── profiler.WaitEmpty()       ← дождаться async очереди
                          ├── profiler.PrintReport()
                          ├── profiler.ExportJSON(...)
                          └── profiler.ExportMarkdown(...)
```

---

## 3. OpenCL: шаг за шагом

### Шаг 1 — production-класс: добавить prof_events

В заголовке (`include/my_module.hpp`) добавить параметр к публичным методам:

```cpp
// ДО: без профилирования
std::vector<Result> Process(const std::vector<float>& data,
                             const Params& params);

// ПОСЛЕ: prof_events опциональный, nullptr по умолчанию
std::vector<Result> Process(const std::vector<float>& data,
                             const Params& params,
                             std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

В реализации (`src/my_module.cpp`) — добавить helper и использовать его:

```cpp
// Helper — сохранить cl_event или освободить (не дублировать на каждый модуль!)
// Уже есть в FFTProcessor, можно скопировать паттерн:
static void CollectOrRelease(cl_event ev, const char* name,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    if (!ev) return;
    if (prof_events) {
        prof_events->push_back({name, ev});
    } else {
        clReleaseEvent(ev);
    }
}

// В теле Process():
std::vector<Result> MyModule::Process(const std::vector<float>& data,
                                       const Params& params,
                                       std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    cl_event ev_upload = nullptr;
    cl_event ev_kernel = nullptr;
    cl_event ev_download = nullptr;

    // ... ваш OpenCL код ...
    // H2D
    clEnqueueWriteBuffer(..., &ev_upload);
    CollectOrRelease(ev_upload, "Upload", prof_events);

    // Запуск ядра
    clEnqueueNDRangeKernel(..., &ev_kernel);
    CollectOrRelease(ev_kernel, "Kernel", prof_events);

    // D2H
    clEnqueueReadBuffer(..., &ev_download);
    CollectOrRelease(ev_download, "Download", prof_events);

    // ...
}
```

> ⚠️ **Очередь должна быть создана с `CL_QUEUE_PROFILING_ENABLE`** — иначе
> `clGetEventProfilingInfo` вернёт `CL_PROFILING_INFO_NOT_AVAILABLE`.
> В test runner передавать флаг при создании очереди (см. Шаг 3).

---

### Шаг 2 — benchmark-класс: наследник GpuBenchmarkBase

Создать файл `tests/my_module_benchmark.hpp`:

```cpp
#pragma once

#include "my_module.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <CL/cl.h>
#include <vector>

namespace test_my_module {

class MyModuleBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  MyModuleBenchmark(
      drv_gpu_lib::IBackend* backend,
      MyModule& proc,
      const Params& params,
      const std::vector<float>& input_data,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_MyModule"})
    : GpuBenchmarkBase(backend, "MyModule", cfg),
      proc_(proc),
      params_(params),
      input_data_(input_data) {}

protected:
  // Warmup — без timing. Просто запуск (прогрев GPU: JIT, clock ramp-up).
  void ExecuteKernel() override {
    proc_.Process(input_data_, params_);    // prof_events = nullptr → ноль overhead
  }

  // Замер — с timing. Собирает cl_event'ы → RecordEvent → GPUProfiler.
  void ExecuteKernelTimed() override {
    std::vector<std::pair<const char*, cl_event>> events;
    proc_.Process(input_data_, params_, &events);

    for (auto& [name, ev] : events) {
      RecordEvent(name, ev);   // clWaitForEvents + timing + profiler.Record + clReleaseEvent
    }
  }

private:
  MyModule&         proc_;
  Params            params_;
  std::vector<float> input_data_;
};

}  // namespace test_my_module
```

---

### Шаг 3 — test runner

Создать файл `tests/test_my_module_benchmark.hpp`:

```cpp
#pragma once

#include "my_module_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace test_my_module_benchmark {

inline int run() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  MyModule Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  try {
    // ── OpenCL init ───────────────────────────────────────────────────
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // ⚠️ CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->InitializeFromExternalContext(context, device, queue);

    // ── Параметры бенчмарка ───────────────────────────────────────────
    Params params;
    params.n_samples = 1024;
    // ... заполнить параметры ...

    std::vector<float> input_data(params.n_samples, 1.0f);
    // ... заполнить данные ...

    // ── Создать модуль и бенчмарк ─────────────────────────────────────
    MyModule proc(backend.get());

    test_my_module::MyModuleBenchmark bench(
        backend.get(), proc, params, input_data,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "Results/Profiler/GPU_00_MyModule"});

    // ── Запуск ────────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
    } else {
      bench.Run();     // warmup(5) + measure(20) → GPUProfiler
      bench.Report();  // PrintReport + ExportJSON + ExportMarkdown
      std::cout << "  [OK] Benchmark complete\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────
    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_my_module_benchmark
```

---

### Шаг 4 — all_test.hpp

```cpp
#include "test_my_module_benchmark.hpp"

namespace my_module_all_test {
inline void run() {
    // ...другие тесты...

    // MyModule Benchmark
    test_my_module_benchmark::run();
}
}
```

---

## 4. ROCm: шаг за шагом

### Шаг 1 — production-класс: добавить ROCmProfEvents

В заголовке (`include/my_module_rocm.hpp`) объявить тип и добавить параметр:

```cpp
#include "DrvGPU/services/profiling_types.hpp"  // ROCmProfilingData
#include <vector>
#include <utility>

namespace my_module {

// Тип для сбора ROCm событий (имя → данные)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

class MyModuleROCm {
public:
  // prof_events = nullptr → production, ноль overhead
  std::vector<Result> Process(const std::vector<float>& data,
                               const Params& params,
                               ROCmProfEvents* prof_events = nullptr);
};

}  // namespace my_module
```

---

### Шаг 2 — helper функции для timing

В реализации (`src/my_module_rocm.cpp`) добавить два helper'а:

```cpp
#include <chrono>
#include "hip/hip_runtime.h"

// Helper A: для async GPU операций (hipEvent_t → hipEventElapsedTime)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind,           // 0=kernel, 1=copy, 2=barrier
    const char* op_string = "")
{
    hipEventSynchronize(ev_end);
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_end);

    drv_gpu_lib::ROCmProfilingData d{};
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
    d.start_ns    = 0;
    d.end_ns      = elapsed_ns;
    d.complete_ns = elapsed_ns;
    d.kind        = kind;
    d.op_string   = op_string;
    return d;
}

// Helper B: для sync CPU/GPU операций (wall-clock через std::chrono)
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t_start,
    std::chrono::high_resolution_clock::time_point t_end,
    uint32_t kind,
    const char* op_string = "")
{
    uint64_t elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());

    drv_gpu_lib::ROCmProfilingData d{};
    d.start_ns    = 0;
    d.end_ns      = elapsed_ns;
    d.complete_ns = elapsed_ns;
    d.kind        = kind;
    d.op_string   = op_string;
    return d;
}
```

**Когда что использовать:**

| Операция | Helper | Причина |
|----------|--------|---------|
| `hipMemcpyHtoD` (async + stream) | `MakeROCmDataFromEvents` | hipEvent захватывает GPU timeline |
| Запуск ядра (`hipLaunchKernelGGL`) | `MakeROCmDataFromEvents` | hipEvent на GPU |
| `hipMemcpyDtoH` (sync) | `MakeROCmDataFromClock` | hipEvent для синхронной DtoH ненадёжен |
| любая синхронная операция | `MakeROCmDataFromClock` | wall-clock точен для sync |

---

**Паттерн в теле метода Process():**

```cpp
std::vector<Result> MyModuleROCm::Process(const std::vector<float>& data,
                                           const Params& params,
                                           ROCmProfEvents* prof_events)
{
    // ── Upload (H2D, async) ───────────────────────────────────────────
    hipEvent_t ev_up_start, ev_up_end;
    if (prof_events) {
        hipEventCreate(&ev_up_start);
        hipEventCreate(&ev_up_end);
        hipEventRecord(ev_up_start, stream_);
    }

    hipMemcpyAsync(d_buf_, data.data(), data.size() * sizeof(float),
                   hipMemcpyHostToDevice, stream_);

    if (prof_events) {
        hipEventRecord(ev_up_end, stream_);
    }

    // ── Kernel ────────────────────────────────────────────────────────
    hipEvent_t ev_k_start, ev_k_end;
    if (prof_events) {
        hipEventCreate(&ev_k_start);
        hipEventCreate(&ev_k_end);
        hipEventRecord(ev_k_start, stream_);
    }

    hipLaunchKernelGGL(my_kernel, grid, block, 0, stream_, d_buf_, params.n_samples);

    if (prof_events) {
        hipEventRecord(ev_k_end, stream_);
    }

    // ── Download (D2H, sync) ──────────────────────────────────────────
    auto t_dl_start = std::chrono::high_resolution_clock::now();

    hipMemcpy(result.data(), d_buf_, result.size() * sizeof(float),
              hipMemcpyDeviceToHost);   // ← синхронная!

    auto t_dl_end = std::chrono::high_resolution_clock::now();

    // ── Собрать prof_events ───────────────────────────────────────────
    if (prof_events) {
        prof_events->push_back({"Upload",   MakeROCmDataFromEvents(ev_up_start, ev_up_end, 1, "H2D")});
        prof_events->push_back({"Kernel",   MakeROCmDataFromEvents(ev_k_start,  ev_k_end,  0, "my_kernel")});
        prof_events->push_back({"Download", MakeROCmDataFromClock(t_dl_start,  t_dl_end,  1, "D2H")});
    }

    return result;
}
```

---

### Шаг 3 — benchmark-класс

Создать файл `tests/my_module_benchmark_rocm.hpp`:

```cpp
#pragma once

#if ENABLE_ROCM

#include "my_module_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

namespace test_my_module_rocm {

class MyModuleBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  MyModuleBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      my_module::MyModuleROCm& proc,
      const my_module::Params& params,
      const std::vector<float>& input_data,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_MyModule_ROCm"})
    : GpuBenchmarkBase(backend, "MyModuleROCm", cfg),
      proc_(proc),
      params_(params),
      input_data_(input_data) {}

protected:
  void ExecuteKernel() override {
    proc_.Process(input_data_, params_);   // prof_events = nullptr → warmup
  }

  void ExecuteKernelTimed() override {
    my_module::ROCmProfEvents events;
    proc_.Process(input_data_, params_, &events);

    for (auto& [name, data] : events) {
      RecordROCmEvent(name, data);   // → GPUProfiler
    }
  }

private:
  my_module::MyModuleROCm&  proc_;
  my_module::Params         params_;
  std::vector<float>        input_data_;
};

}  // namespace test_my_module_rocm

#endif  // ENABLE_ROCM
```

---

### Шаг 4 — test runner

Создать файл `tests/test_my_module_benchmark_rocm.hpp`:

```cpp
#pragma once

#if ENABLE_ROCM

#include "my_module_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>

namespace test_my_module_benchmark_rocm {

inline int run() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  MyModuleROCm Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  // Проверка наличия ROCm-устройств
  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  std::cout << "  Available ROCm devices: " << device_count << "\n";
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices found\n";
    return 0;
  }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    // ── Параметры ─────────────────────────────────────────────────────
    my_module::Params params;
    params.n_samples = 1024;
    // ... заполнить параметры ...

    std::vector<float> input_data(params.n_samples, 1.0f);
    // ... заполнить данные ...

    // ── Создать модуль и бенчмарк ─────────────────────────────────────
    my_module::MyModuleROCm proc(&backend);

    test_my_module_rocm::MyModuleBenchmarkROCm bench(
        &backend, proc, params, input_data,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "Results/Profiler/GPU_00_MyModule_ROCm"});

    // ── Запуск ────────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
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

}  // namespace test_my_module_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

### Шаг 5 — all_test.hpp

```cpp
#if ENABLE_ROCM
#include "test_my_module_benchmark_rocm.hpp"
#endif

namespace my_module_all_test {
inline void run() {
    // ...другие тесты...

    // MyModuleROCm Benchmark (hipEvent timing)
#if ENABLE_ROCM
//    test_my_module_benchmark_rocm::run();  // раскомментировать для запуска
#endif
}
}
```

---

## 5. Структура файлов модуля

```
modules/my_module/
├── include/
│   ├── my_module.hpp            ← prod OpenCL: Process(..., prof_events*)
│   └── my_module_rocm.hpp       ← prod ROCm:   Process(..., prof_events*)  [ENABLE_ROCM]
├── src/
│   ├── my_module.cpp            ← impl + CollectOrRelease()
│   └── my_module_rocm.cpp       ← impl + MakeROCmDataFromEvents/Clock()
└── tests/
    ├── all_test.hpp             ← точка входа, вызывается из main
    ├── my_module_benchmark.hpp          ← класс OpenCL (: GpuBenchmarkBase)
    ├── test_my_module_benchmark.hpp     ← test runner OpenCL
    ├── my_module_benchmark_rocm.hpp     ← класс ROCm  (: GpuBenchmarkBase) [ENABLE_ROCM]
    ├── test_my_module_benchmark_rocm.hpp← test runner ROCm                  [ENABLE_ROCM]
    └── README.md                ← описание всех тестов
```

---

## 6. Чеклист добавления профилирования

### Production-класс

- [ ] Добавить `prof_events*` параметр к каждому публичному методу (default = `nullptr`)
- [ ] **OpenCL**: использовать `CollectOrRelease(ev, "Name", prof_events)` вместо прямого `clReleaseEvent`
- [ ] **ROCm**: обернуть `hipEventCreate/Record` в `if (prof_events)` блоки
- [ ] Убедиться: при `prof_events = nullptr` — **ноль дополнительного overhead**

### Benchmark-класс (: GpuBenchmarkBase)

- [ ] Унаследовать от `drv_gpu_lib::GpuBenchmarkBase`
- [ ] Реализовать `ExecuteKernel()` — вызов без prof_events (warmup)
- [ ] Реализовать `ExecuteKernelTimed()` — вызов с prof_events + цикл `RecordEvent()` / `RecordROCmEvent()`
- [ ] Задать `output_dir = "Results/Profiler/GPU_NN_ModuleName"`

### Test runner

- [ ] **OpenCL**: создать очередь с флагом `CL_QUEUE_PROFILING_ENABLE`
- [ ] Проверить `bench.IsProfEnabled()` перед вызовом `Run()`
- [ ] Wrapped в `#if ENABLE_ROCM` для ROCm-специфичного кода

### configGPU.json

- [ ] Убедиться что `"is_prof": true` для нужного GPU
  ```json
  {
    "gpus": [
      { "gpu_id": 0, "is_prof": true, ... }
    ]
  }
  ```

### Результаты

- [ ] Вывод только через `bench.Report()` → `GPUProfiler::PrintReport()`
- [ ] **ЗАПРЕЩЕНО**: `GetStats()` + цикл + `con.Print` / `std::cout` напрямую

### Автоматическая проверка

```bash
python scripts/check_profiling.py [module]   # один модуль
python scripts/check_profiling.py --all       # все модули
```

Skill-агент: `.cursor/skills/check-profiling-implementation/` — описание чеклиста и триггеров.

---

## 7. Частые ошибки

### ❌ `is_prof = false` → бенчмарк — no-op, нет вывода

**Причина A**: `GPUConfig` не загружен (тест обходит стандартный DrvGPU init).
**Причина B**: `backend->GetDeviceIndex()` возвращает `-1` (external context) → `IsProfilingEnabled(-1) = false`.

**Исправление**: `GpuBenchmarkBase` автоматически:
1. Маппит `gpu_id = -1` → `gpu_id = 0`
2. Загружает `configGPU.json` если ещё не загружен

```cpp
// Конструктор GpuBenchmarkBase (уже реализовано):
if (gpu_id_ < 0) gpu_id_ = 0;
if (!GPUConfig::GetInstance().IsLoaded())
    GPUConfig::GetInstance().LoadOrCreate("configGPU.json");
is_prof_ = GPUConfig::GetInstance().IsProfilingEnabled(gpu_id_);
```

---

### ❌ N = 19 вместо 20 в отчёте

**Причина**: `Report()` был вызван до того как последнее сообщение обработала async очередь GPUProfiler.

**Исправление**: `GpuBenchmarkBase::Report()` вызывает `profiler.WaitEmpty()` перед `PrintReport()` (уже реализовано).

---

### ❌ ИТОГО N = 57 (неправильно), Всего = 2.18 ms (слишком большое)

**Причина**: Старая логика суммировала `total_calls` по всем событиям (`3 events × 20 runs = 60`) и `total_time_ms` (raw sum вместо avg per run).

**Правильные значения**:
- `N` = `ModuleStats::GetRunCount()` — число прогонов одного бенчмарка (= 20)
- `Всего` = `ModuleStats::GetAvgRunTimeMs()` — среднее время одного полного прогона (сумма avg по всем этапам)

---

### ❌ OpenCL: cl_event timing возвращает 0 или ошибку

**Причина**: Очередь создана **без** `CL_QUEUE_PROFILING_ENABLE`.

```cpp
// ❌ НЕПРАВИЛЬНО:
clCreateCommandQueue(context, device, 0, &err);

// ✅ ПРАВИЛЬНО:
clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
```

---

### ❌ ROCm: hipEventElapsedTime для синхронной hipMemcpyDtoH

**Причина**: `hipMemcpyDeviceToHost` (синхронный) возвращает управление только после завершения — `hipEventRecord` после него не имеет смысла (event сразу в состоянии "complete").

**Решение**: Для синхронных операций использовать `MakeROCmDataFromClock()` (wall-clock).

---

### ❌ Множественные экземпляры GPUProfiler пишут в один файл

`GPUProfiler` — singleton. Если запускать несколько бенчмарков подряд без `Reset()` между ними — данные смешаются. `GpuBenchmarkBase::Run()` вызывает `profiler.Reset()` в начале (в `InitProfiler()`), поэтому **каждый бенчмарк начинает с чистого листа**.

---

## Референс-реализации

| Что | Файл |
|-----|------|
| OpenCL benchmark-класс | `modules/fft_processor/tests/fft_processor_benchmark.hpp` |
| OpenCL test runner | `modules/fft_processor/tests/test_fft_benchmark.hpp` |
| ROCm benchmark-класс | `modules/fft_processor/tests/fft_processor_benchmark_rocm.hpp` |
| ROCm test runner | `modules/fft_processor/tests/test_fft_benchmark_rocm.hpp` |
| GpuBenchmarkBase | `DrvGPU/services/gpu_benchmark_base.hpp` |
| GPUProfiler | `DrvGPU/services/gpu_profiler.hpp` |
| Profiling types | `DrvGPU/services/profiling_types.hpp` |
| SetGPUInfo паттерн | `Examples/GPUProfiler_SetGPUInfo.md` |
| Проверка соответствия | `python scripts/check_profiling.py [module]` |
