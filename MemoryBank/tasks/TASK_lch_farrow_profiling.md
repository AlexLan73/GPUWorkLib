# TASK: Рефакторинг профилирования modules/lch_farrow → GpuBenchmarkBase (OpenCL + ROCm)

> **Статус**: ✅ COMPLETED
> **Ветка**: `Profiller01`
> **Дата создания**: 2026-03-01
> **Автор**: Кодо (AI Assistant)
> **Приоритет**: HIGH

---

## Цель

Переделать профилирование модуля `modules/lch_farrow` по образцу `modules/fft_processor`:
- Production-классы (`LchFarrow`, `LchFarrowROCm`) — **чистые**, ноль кода профилирования
- Профилирование — изолировано в `/tests/*.hpp` через `GpuBenchmarkBase`
- OpenCL: `CollectOrRelease` + `cl_event` → `RecordEvent`
- ROCm: `MakeROCmDataFromEvents/Clock` + `ROCmProfEvents` → `RecordROCmEvent`

**Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`
**Референс**: `modules/fft_processor/tests/` (benchmark/test runner OpenCL + ROCm)

---

## Анализ текущего состояния

### Проблемы в `lch_farrow.cpp` (OpenCL Production)

1. `Process()` вызывает `SetGPUInfo(...)` **внутри каждого вызова** — нарушение принципа (production = чистый)
2. `Process()` вызывает `RecordProfilingEvent(upload_event, ...)` напрямую — нарушение (не через GpuBenchmarkBase)
3. Нет `prof_events*` параметра → нельзя изолировать профилирование в тесты

### Проблемы в `lch_farrow_rocm.cpp` (ROCm Production)

1. Нет профилирования вообще (ни hipEvent, ни ROCmProfEvents)
2. Нет `prof_events*` параметра

### Проблемы в `tests/test_lch_farrow.hpp`

1. Использует старое профилирование: `profiler.Start()`, `profiler.Stop()`, `PrintReport` напрямую
2. После рефакторинга `Process()` профилирование там перестанет работать (события не будут собираться)
3. Нет отдельного бенчмарк-файла

### Проблемы в `tests/all_test.hpp`

1. Нет вызовов бенчмарков (только функциональные тесты)
2. Нет `README.md`

---

## Архитектура после рефакторинга

```
modules/lch_farrow/
├── include/
│   ├── lch_farrow.hpp             ← A1: добавить ProfEvents + prof_events* в Process()
│   └── lch_farrow_rocm.hpp        ← C1: добавить ROCmProfEvents + prof_events* в Process()/ProcessFromCPU()
├── src/
│   ├── lch_farrow.cpp             ← A2: CollectOrRelease, убрать SetGPUInfo + RecordProfilingEvent
│   └── lch_farrow_rocm.cpp        ← C2: MakeROCmDataFromEvents/Clock + профилирование Process+ProcessFromCPU
└── tests/
    ├── all_test.hpp               ← F2: добавить бенчмарк инклюды + закомментированные вызовы
    ├── test_lch_farrow.hpp        ← F1: убрать старый код профилирования (оставить только функциональные тесты)
    ├── test_lch_farrow_rocm.hpp   (БЕЗ ИЗМЕНЕНИЙ — только функциональные тесты)
    ├── lch_farrow_benchmark.hpp           ← B1: НОВЫЙ (OpenCL benchmark класс : GpuBenchmarkBase)
    ├── test_lch_farrow_benchmark.hpp      ← B2: НОВЫЙ (OpenCL test runner)
    ├── lch_farrow_benchmark_rocm.hpp      ← D1: НОВЫЙ (ROCm benchmark класс : GpuBenchmarkBase)
    ├── test_lch_farrow_benchmark_rocm.hpp ← D2: НОВЫЙ (ROCm test runner)
    └── README.md                          ← G1: НОВЫЙ
```

---

## Детальный план задач

---

### A1: `include/lch_farrow.hpp` — Добавить ProfEvents + prof_events*

**Файл**: `modules/lch_farrow/include/lch_farrow.hpp`
**Статус**: ⬜ TODO

**Что изменить**:

1. Добавить `#include <utility>` (для `std::pair`)
2. Добавить type alias перед классом:
   ```cpp
   /// Events collected during Process() for external profiling (optional)
   using ProfEvents = std::vector<std::pair<const char*, cl_event>>;
   ```
3. Изменить сигнатуру `Process()`:
   ```cpp
   // ДО:
   drv_gpu_lib::InputData<cl_mem> Process(
       cl_mem input_buf, uint32_t antennas, uint32_t points);

   // ПОСЛЕ:
   drv_gpu_lib::InputData<cl_mem> Process(
       cl_mem input_buf, uint32_t antennas, uint32_t points,
       ProfEvents* prof_events = nullptr);
   ```

**Принцип**: `prof_events = nullptr` → production path, ноль overhead.
При `prof_events != nullptr` → бенчмарк собирает события.

---

### A2: `src/lch_farrow.cpp` — Рефакторинг Process()

**Файл**: `modules/lch_farrow/src/lch_farrow.cpp`
**Статус**: ⬜ TODO
**Зависит от**: A1

**Что изменить**:

#### 1. Добавить статический helper `CollectOrRelease` (перед `Process`)

```cpp
// Helper: сохранить cl_event в prof_events или освободить
// Ключевое правило: передать event как wait ПЕРЕД CollectOrRelease
static void CollectOrRelease(
    cl_event ev, const char* name,
    lch_farrow::ProfEvents* prof_events)
{
  if (!ev) return;
  if (prof_events) {
    prof_events->push_back({name, ev});
  } else {
    clReleaseEvent(ev);
  }
}
```

#### 2. Убрать из `Process()`:

- Блок `SetGPUInfo` (строки ~419–435 в текущем lch_farrow.cpp):
  ```cpp
  // УДАЛИТЬ весь блок:
  int gpu_id = backend_->GetDeviceIndex();
  if (gpu_id < 0) gpu_id = 0;
  auto device_info = backend_->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo gpu_info;
  ...
  drv_gpu_lib::GPUProfiler::GetInstance().SetGPUInfo(gpu_id, gpu_info);
  ```

- Прямые вызовы `RecordProfilingEvent`:
  ```cpp
  // УДАЛИТЬ:
  drv_gpu_lib::RecordProfilingEvent(upload_event, gpu_id, "LchFarrow", "Upload_delay_us");
  drv_gpu_lib::RecordProfilingEvent(kernel_event, gpu_id, "LchFarrow", "lch_farrow_delay");
  if (upload_event) clReleaseEvent(upload_event);
  if (kernel_event) clReleaseEvent(kernel_event);
  ```

#### 3. Добавить в `Process()` вместо удалённого:

```cpp
// После запуска ядра:
CollectOrRelease(upload_event, "Upload_delay", prof_events);
CollectOrRelease(kernel_event, "Kernel", prof_events);
clFinish(queue_);
```

> ⚠️ Порядок важен: upload_event передаётся как wait в `clEnqueueNDRangeKernel`,
> поэтому `CollectOrRelease` вызывается ПОСЛЕ постановки в очередь,
> а `clFinish` — после всех `CollectOrRelease`.

#### 4. Удалить неиспользуемые includes:

```cpp
// УДАЛИТЬ если больше не используется:
#include "services/gpu_profiler.hpp"
#include "backends/opencl/opencl_profiling.hpp"
```

#### 5. Обновить сигнатуру функции:

```cpp
// ДО:
drv_gpu_lib::InputData<cl_mem>
LchFarrow::Process(cl_mem input_buf, uint32_t antennas, uint32_t points)

// ПОСЛЕ:
drv_gpu_lib::InputData<cl_mem>
LchFarrow::Process(cl_mem input_buf, uint32_t antennas, uint32_t points,
                   ProfEvents* prof_events)
```

---

### B1: `tests/lch_farrow_benchmark.hpp` — OpenCL Benchmark класс (НОВЫЙ)

**Файл**: `modules/lch_farrow/tests/lch_farrow_benchmark.hpp`
**Статус**: ⬜ TODO
**Зависит от**: A1, A2

**Структура**:

```cpp
#pragma once

/**
 * @file lch_farrow_benchmark.hpp
 * @brief LchFarrowBenchmark — наследник GpuBenchmarkBase для LchFarrow (OpenCL)
 *
 * LchFarrow — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование через опциональный prof_events:
 *  - ExecuteKernel()      → Process(input_buf_, ...) — без событий (warmup)
 *  - ExecuteKernelTimed() → Process(input_buf_, ..., &events) — с cl_event
 *    → RecordEvent() для Upload_delay + Kernel → GPUProfiler
 *
 * Stages:
 *  - Upload_delay: clEnqueueWriteBuffer (delay_us массив на GPU, каждый вызов)
 *  - Kernel:       lch_farrow_delay (Lagrange 48x5 интерполяция)
 *  (input_buf уже на GPU — загружается один раз в test runner)
 *
 * @date 2026-03-01
 */

#include "lch_farrow.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <utility>

namespace test_lch_farrow {

class LchFarrowBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @param backend   IBackend для GPUProfiler
   * @param proc      Ссылка на LchFarrow (не владеет)
   * @param input_buf cl_mem с входным сигналом на GPU (не освобождается здесь)
   * @param antennas  Число антенн
   * @param points    Число отсчётов на антенну
   * @param cfg       Параметры бенчмарка
   */
  LchFarrowBenchmark(
      drv_gpu_lib::IBackend* backend,
      lch_farrow::LchFarrow& proc,
      cl_mem input_buf,
      uint32_t antennas,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_LchFarrow"})
    : GpuBenchmarkBase(backend, "LchFarrow", cfg),
      proc_(proc),
      input_buf_(input_buf),
      antennas_(antennas),
      points_(points) {}

protected:
  // Warmup — без timing. Прогрев GPU (JIT, clock ramp-up).
  // result.data освобождается сразу (result остаётся на GPU — ненужен)
  void ExecuteKernel() override {
    auto result = proc_.Process(input_buf_, antennas_, points_);
    // Освободить GPU буфер результата
    if (result.data) clReleaseMemObject(result.data);
  }

  // Замер — с timing. Собирает cl_event'ы → RecordEvent → GPUProfiler.
  void ExecuteKernelTimed() override {
    lch_farrow::ProfEvents events;
    auto result = proc_.Process(input_buf_, antennas_, points_, &events);
    if (result.data) clReleaseMemObject(result.data);

    for (auto& [name, ev] : events) {
      RecordEvent(name, ev);
    }
  }

private:
  lch_farrow::LchFarrow& proc_;
  cl_mem    input_buf_;
  uint32_t  antennas_;
  uint32_t  points_;
};

}  // namespace test_lch_farrow
```

---

### B2: `tests/test_lch_farrow_benchmark.hpp` — OpenCL Test Runner (НОВЫЙ)

**Файл**: `modules/lch_farrow/tests/test_lch_farrow_benchmark.hpp`
**Статус**: ⬜ TODO
**Зависит от**: B1

**Параметры бенчмарка**:
- `antennas = 8` (реальный сценарий)
- `points = 4096`
- `sample_rate = 1e6f`
- `delays = {0.3f, 1.7f, 2.1f, 3.5f, 4.0f, 5.3f, 6.7f, 7.9f}` (разные задержки)
- `n_warmup = 5`, `n_runs = 20`
- `output_dir = "Results/Profiler/GPU_00_LchFarrow"`

**Структура**:

```cpp
namespace test_lch_farrow_benchmark {

// Утилита генерации CW сигнала
inline std::vector<std::complex<float>> GenerateBenchmarkData(
    uint32_t antennas, uint32_t points, float fs, float freq)
{
  std::vector<std::complex<float>> data(antennas * points);
  for (uint32_t a = 0; a < antennas; ++a) {
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / fs;
      float phase = 2.0f * M_PI * freq * t;
      data[a * points + n] = std::complex<float>(std::cos(phase), std::sin(phase));
    }
  }
  return data;
}

inline int run() {
  // Вывод заголовка
  // ── OpenCL init с CL_QUEUE_PROFILING_ENABLE ───────────────────────
  // ── SetGPUInfo для profiler ───────────────────────────────────────
  //    (до создания бенчмарка, чтобы в отчёте было имя GPU + драйвер)
  // ── Подготовить данные (GenerateBenchmarkData) ────────────────────
  // ── Загрузить на GPU: cl_mem input_buf = clCreateBuffer(... CL_MEM_COPY_HOST_PTR ...) ─
  // ── Создать LchFarrow + SetDelays + SetSampleRate ─────────────────
  // ── Создать LchFarrowBenchmark ────────────────────────────────────
  // ── bench.IsProfEnabled() → Run() → Report() ─────────────────────
  // ── Cleanup: clReleaseMemObject(input_buf) + clReleaseCommandQueue + clReleaseContext ─
}

}  // namespace test_lch_farrow_benchmark
```

> ⚠️ **Ключевые моменты**:
> 1. `CL_QUEUE_PROFILING_ENABLE` обязателен для cl_event timing
> 2. `SetGPUInfo` вызвать ДО `bench.Run()` (чтобы в отчёте был GPU name + драйвер)
> 3. `input_buf` создаётся ОДИН РАЗ (загрузка входных данных — вне замера)
> 4. `LchFarrow` создаётся с `backend->InitializeFromExternalContext(context, device, queue)` бэкендом

---

### C1: `include/lch_farrow_rocm.hpp` — Добавить ROCmProfEvents + prof_events*

**Файл**: `modules/lch_farrow/include/lch_farrow_rocm.hpp`
**Статус**: ⬜ TODO

**Что изменить** (в части `#if ENABLE_ROCM`):

1. Добавить include:
   ```cpp
   #include "DrvGPU/services/profiling_types.hpp"  // ROCmProfilingData
   ```

2. Добавить type alias в namespace `lch_farrow`:
   ```cpp
   /// Events collected during ROCm Process() for external profiling (optional)
   using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
   ```

3. Изменить сигнатуру `Process()`:
   ```cpp
   // ДО:
   drv_gpu_lib::InputData<void*> Process(
       void* input_ptr, uint32_t antennas, uint32_t points);

   // ПОСЛЕ:
   drv_gpu_lib::InputData<void*> Process(
       void* input_ptr, uint32_t antennas, uint32_t points,
       ROCmProfEvents* prof_events = nullptr);
   ```

4. Изменить сигнатуру `ProcessFromCPU()`:
   ```cpp
   // ДО:
   drv_gpu_lib::InputData<void*> ProcessFromCPU(
       const std::vector<std::complex<float>>& data,
       uint32_t antennas, uint32_t points);

   // ПОСЛЕ:
   drv_gpu_lib::InputData<void*> ProcessFromCPU(
       const std::vector<std::complex<float>>& data,
       uint32_t antennas, uint32_t points,
       ROCmProfEvents* prof_events = nullptr);
   ```

5. Обновить **stub** (часть `#else // !ENABLE_ROCM`):
   ```cpp
   drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t,
       void* = nullptr) { ... }
   drv_gpu_lib::InputData<void*> ProcessFromCPU(
       const std::vector<std::complex<float>>&, uint32_t, uint32_t,
       void* = nullptr) { ... }
   ```

---

### C2: `src/lch_farrow_rocm.cpp` — ROCm Profiling

**Файл**: `modules/lch_farrow/src/lch_farrow_rocm.cpp`
**Статус**: ⬜ TODO
**Зависит от**: C1

#### 1. Добавить includes

```cpp
#include "DrvGPU/services/profiling_types.hpp"
#include <chrono>
```

#### 2. Добавить два static helper'а (перед `Process`)

```cpp
// Helper A: для async GPU операций (hipEvent → hipEventElapsedTime)
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
    uint32_t kind, const char* op_string = "")
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

#### 3. Модифицировать `Process()` — добавить hipEvent timing

**Операции для профилирования**:
| Операция | Метод | Пояснение |
|----------|-------|-----------|
| `Upload_delay` | `hipMemcpyHtoDAsync(delay_buf, ...)` | async → hipEvent |
| `Kernel` | `hipModuleLaunchKernel(...)` | async → hipEvent |

**Паттерн**:
```cpp
drv_gpu_lib::InputData<void*>
LchFarrowROCm::Process(void* input_ptr, uint32_t antennas, uint32_t points,
                        ROCmProfEvents* prof_events)
{
  // ... (existing code: allocate output_ptr, delay_buf) ...

  // ── Upload_delay ─────────────────────────────────────────────────
  hipEvent_t ev_up_start = nullptr, ev_up_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_up_start);
    hipEventCreate(&ev_up_end);
    hipEventRecord(ev_up_start, stream_);
  }

  // hipMemcpyHtoDAsync(delay_buf, ..., stream_)  ← уже есть

  if (prof_events) {
    hipEventRecord(ev_up_end, stream_);
  }

  // ── Kernel ────────────────────────────────────────────────────────
  hipEvent_t ev_k_start = nullptr, ev_k_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_start);
    hipEventCreate(&ev_k_end);
    hipEventRecord(ev_k_start, stream_);
  }

  // hipModuleLaunchKernel(...)  ← уже есть

  if (prof_events) {
    hipEventRecord(ev_k_end, stream_);
  }

  // ── Sync ──────────────────────────────────────────────────────────
  hipStreamSynchronize(stream_);
  hipFree(delay_buf);

  // ── Собрать prof_events ───────────────────────────────────────────
  if (prof_events) {
    prof_events->push_back({"Upload_delay",
        MakeROCmDataFromEvents(ev_up_start, ev_up_end, 1, "H2D_delay")});
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_start, ev_k_end, 0, "lch_farrow_delay")});
  }

  // ... (existing: return result) ...
}
```

#### 4. Модифицировать `ProcessFromCPU()` — добавить Upload_input timing

**Операции для профилирования**:
| Операция | Метод | Пояснение |
|----------|-------|-----------|
| `Upload_input` | `hipMemcpyHtoDAsync(input_ptr, ...)` | async → hipEvent |
| (делегат) | `Process(input_ptr, ..., prof_events)` | Upload_delay + Kernel |

```cpp
drv_gpu_lib::InputData<void*>
LchFarrowROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events)
{
  // ... (existing: validate, allocate input_ptr) ...

  // ── Upload_input ──────────────────────────────────────────────────
  hipEvent_t ev_in_start = nullptr, ev_in_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_in_start);
    hipEventCreate(&ev_in_end);
    hipEventRecord(ev_in_start, stream_);
  }

  // hipMemcpyHtoDAsync(input_ptr, data.data(), ..., stream_)  ← уже есть

  if (prof_events) {
    hipEventRecord(ev_in_end, stream_);
  }

  hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Upload_input",
        MakeROCmDataFromEvents(ev_in_start, ev_in_end, 1, "H2D_input")});
  }

  // Делегировать в Process() с prof_events
  auto result = Process(input_ptr, antennas, points, prof_events);

  hipFree(input_ptr);
  return result;
}
```

> 📌 **Итоговые stages в ProcessFromCPU()**: `Upload_input`, `Upload_delay`, `Kernel`

---

### D1: `tests/lch_farrow_benchmark_rocm.hpp` — ROCm Benchmark класс (НОВЫЙ)

**Файл**: `modules/lch_farrow/tests/lch_farrow_benchmark_rocm.hpp`
**Статус**: ⬜ TODO
**Зависит от**: C1, C2

```cpp
#pragma once

/**
 * @file lch_farrow_benchmark_rocm.hpp
 * @brief LchFarrowBenchmarkROCm — наследник GpuBenchmarkBase для LchFarrowROCm
 *
 * Stages: Upload_input (H2D), Upload_delay (H2D), Kernel (lch_farrow_delay)
 * Бенчмаркирует ProcessFromCPU() — полный путь от CPU данных до GPU результата.
 *
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "lch_farrow_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <vector>
#include <complex>

namespace test_lch_farrow_rocm {

class LchFarrowBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  LchFarrowBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      lch_farrow::LchFarrowROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      uint32_t antennas,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_LchFarrow_ROCm"})
    : GpuBenchmarkBase(backend, "LchFarrowROCm", cfg),
      proc_(proc),
      input_data_(input_data),
      antennas_(antennas),
      points_(points) {}

protected:
  // Warmup — без timing (прогрев GPU, JIT)
  void ExecuteKernel() override {
    auto result = proc_.ProcessFromCPU(input_data_, antennas_, points_);
    // Освободить GPU буфер результата
    if (result.data) hipFree(result.data);
  }

  // Замер — с timing → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    lch_farrow::ROCmProfEvents events;
    auto result = proc_.ProcessFromCPU(input_data_, antennas_, points_, &events);
    if (result.data) hipFree(result.data);

    for (auto& [name, data] : events) {
      RecordROCmEvent(name, data);
    }
  }

private:
  lch_farrow::LchFarrowROCm&             proc_;
  std::vector<std::complex<float>>       input_data_;
  uint32_t                               antennas_;
  uint32_t                               points_;
};

}  // namespace test_lch_farrow_rocm

#endif  // ENABLE_ROCM
```

---

### D2: `tests/test_lch_farrow_benchmark_rocm.hpp` — ROCm Test Runner (НОВЫЙ)

**Файл**: `modules/lch_farrow/tests/test_lch_farrow_benchmark_rocm.hpp`
**Статус**: ⬜ TODO
**Зависит от**: D1

**Параметры**:
- `antennas = 8`, `points = 4096`, `sample_rate = 1e6f`
- `delays = {0.3f, 1.7f, 2.1f, 3.5f, 4.0f, 5.3f, 6.7f, 7.9f}`
- `n_warmup = 5`, `n_runs = 20`
- `output_dir = "Results/Profiler/GPU_00_LchFarrow_ROCm"`

**Структура**:

```cpp
#pragma once

#if ENABLE_ROCM

#include "lch_farrow_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

namespace test_lch_farrow_benchmark_rocm {

inline int run() {
  // Проверка наличия ROCm-устройств
  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) { /* SKIP */ return 0; }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    // ── Параметры ─────────────────────────────────────────────────────
    const uint32_t antennas = 8;
    const uint32_t points   = 4096;
    const float    fs       = 1e6f;
    std::vector<float> delays = {0.3f, 1.7f, 2.1f, 3.5f, 4.0f, 5.3f, 6.7f, 7.9f};

    // Генерация данных (CW синусоида)
    // ...

    // ── Создать LchFarrowROCm ─────────────────────────────────────────
    lch_farrow::LchFarrowROCm proc(&backend);
    proc.SetDelays(delays);
    proc.SetSampleRate(fs);

    // ── Создать бенчмарк ──────────────────────────────────────────────
    test_lch_farrow_rocm::LchFarrowBenchmarkROCm bench(...);

    // ── Запуск ────────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) { /* SKIP */ } else {
      bench.Run();
      bench.Report();
    }
    return 0;
  } catch (...) { return 1; }
}

}  // namespace test_lch_farrow_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

### F1: `tests/test_lch_farrow.hpp` — Убрать старое профилирование

**Файл**: `modules/lch_farrow/tests/test_lch_farrow.hpp`
**Статус**: ⬜ TODO

**Что удалить**:
- `#include "DrvGPU/services/gpu_profiler.hpp"` ← убрать (profiler больше не используется)
- `auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();` ← убрать
- `profiler.Start();` ← убрать
- `profiler.Stop();` ← убрать
- Весь блок экспорта:
  ```cpp
  // УДАЛИТЬ:
  profiler.PrintReport();
  std::string profiler_dir = "Results/Profiler";
  std::filesystem::create_directories(profiler_dir);
  ...
  profiler.ExportMarkdown(base_path + ".md");
  profiler.ExportJSON(base_path + ".json");
  ```
- `#include <filesystem>` ← убрать (если только для profiler_dir)

**Что оставить**: все три функциональных теста (Test 1, Test 2, Test 3) без изменений.

---

### F2: `tests/all_test.hpp` — Обновить

**Файл**: `modules/lch_farrow/tests/all_test.hpp`
**Статус**: ⬜ TODO

**Что добавить**:

```cpp
#pragma once

#include "test_lch_farrow.hpp"
#include "test_lch_farrow_rocm.hpp"
#include "test_lch_farrow_benchmark.hpp"         // ← НОВЫЙ
#if ENABLE_ROCM
#include "test_lch_farrow_benchmark_rocm.hpp"    // ← НОВЫЙ
#endif

namespace lch_farrow_all_test {

inline void run() {
  test_lch_farrow::run();
  test_lch_farrow_rocm::run();

  // LchFarrow Benchmark (OpenCL, GpuBenchmarkBase)
  // test_lch_farrow_benchmark::run();

  // LchFarrowROCm Benchmark (hipEvent timing, GpuBenchmarkBase)
#if ENABLE_ROCM
  // test_lch_farrow_benchmark_rocm::run();
#endif
}

}  // namespace lch_farrow_all_test
```

---

### G1: `tests/README.md` — Документация тестов (НОВЫЙ)

**Файл**: `modules/lch_farrow/tests/README.md`
**Статус**: ⬜ TODO

**Содержание**:

```markdown
# lch_farrow — Tests

## Функциональные тесты

| Файл | Namespace | Описание |
|------|-----------|----------|
| test_lch_farrow.hpp | test_lch_farrow | OpenCL: 3 теста (zero/integer/fractional delay) |
| test_lch_farrow_rocm.hpp | test_lch_farrow_rocm | ROCm: 4 теста (zero/integer/fractional/multi-antenna) |

## Бенчмарки

| Файл | Namespace | Описание |
|------|-----------|----------|
| lch_farrow_benchmark.hpp | test_lch_farrow | OpenCL benchmark класс : GpuBenchmarkBase |
| test_lch_farrow_benchmark.hpp | test_lch_farrow_benchmark | OpenCL test runner (warmup=5, runs=20) |
| lch_farrow_benchmark_rocm.hpp | test_lch_farrow_rocm | ROCm benchmark класс : GpuBenchmarkBase |
| test_lch_farrow_benchmark_rocm.hpp | test_lch_farrow_benchmark_rocm | ROCm test runner (warmup=5, runs=20) |

## Профилируемые stages

### OpenCL
- `Upload_delay` — clEnqueueWriteBuffer (delay_us на GPU)
- `Kernel` — lch_farrow_delay (Lagrange 48x5 интерполяция)

### ROCm (ProcessFromCPU)
- `Upload_input` — hipMemcpyHtoDAsync (входной сигнал на GPU)
- `Upload_delay` — hipMemcpyHtoDAsync (delay_us на GPU)
- `Kernel` — lch_farrow_delay (hipModuleLaunchKernel)

## Результаты профилирования

- OpenCL: `Results/Profiler/GPU_00_LchFarrow/`
- ROCm:   `Results/Profiler/GPU_00_LchFarrow_ROCm/`

## Как запустить

В `all_test.hpp` раскомментировать нужные строки:
- `test_lch_farrow_benchmark::run();`     ← OpenCL
- `test_lch_farrow_benchmark_rocm::run();` ← ROCm (только Linux + AMD GPU)

Требование: `configGPU.json` → `"is_prof": true` для нужного GPU.
```

---

## Чеклист выполнения

### OpenCL (10 пунктов)

- [x] **A1** `include/lch_farrow.hpp` — добавить `ProfEvents` alias + `prof_events*` в `Process()`
- [x] **A2** `src/lch_farrow.cpp` — `CollectOrRelease` + убрать `SetGPUInfo` + убрать `RecordProfilingEvent` + обновить сигнатуру
- [x] **B1** `tests/lch_farrow_benchmark.hpp` — СОЗДАН (OpenCL benchmark класс)
- [x] **B2** `tests/test_lch_farrow_benchmark.hpp` — СОЗДАН (OpenCL test runner)
- [x] **F1** `tests/test_lch_farrow.hpp` — убрат старый profiler код

### ROCm (5 пунктов)

- [x] **C1** `include/lch_farrow_rocm.hpp` — добавить `ROCmProfEvents` alias + `prof_events*` в `Process()` и `ProcessFromCPU()`
- [x] **C2** `src/lch_farrow_rocm.cpp` — `MakeROCmDataFromEvents/Clock` + hipEvent в `Process()` + hipEvent в `ProcessFromCPU()`
- [x] **D1** `tests/lch_farrow_benchmark_rocm.hpp` — СОЗДАН (ROCm benchmark класс)
- [x] **D2** `tests/test_lch_farrow_benchmark_rocm.hpp` — СОЗДАН (ROCm test runner)

### Инфраструктура (2 пункта)

- [x] **F2** `tests/all_test.hpp` — добавлены includes + закомментированные вызовы
- [x] **G1** `tests/README.md` — СОЗДАН

**Итого: 12 задач**

---

## Важные нюансы реализации

### 1. CollectOrRelease — порядок вызовов

```
upload_event → используется как wait в clEnqueueNDRangeKernel
             → затем CollectOrRelease(upload_event, ...)
kernel_event → CollectOrRelease(kernel_event, ...)
clFinish()   → ждём завершения
```

### 2. OpenCL benchmark — input_buf на GPU

`Process()` принимает `cl_mem input_buf` (уже на GPU).
В test runner нужно ОДИН РАЗ загрузить данные:
```cpp
cl_mem input_buf = clCreateBuffer(context,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    data_size, cpu_data.data(), &err);
// ... benchmark ...
clReleaseMemObject(input_buf);  // после завершения
```

### 3. ROCm — освобождение результата в ExecuteKernel/ExecuteKernelTimed

`Process()`/`ProcessFromCPU()` возвращает `InputData<void*>` где `result.data` — новый GPU буфер.
**Его нужно освобождать после каждого вызова**:
```cpp
auto result = proc_.ProcessFromCPU(...);
if (result.data) hipFree(result.data);
```

### 4. SetGPUInfo — в test runner, не в production

Вызывать в test runner перед `bench.Run()`:
```cpp
auto device_info = backend->GetDeviceInfo();
drv_gpu_lib::GPUReportInfo gpu_info;
gpu_info.gpu_name = device_info.name;
gpu_info.backend_type = drv_gpu_lib::BackendType::OPENCL;
gpu_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);
// ...
drv_gpu_lib::GPUProfiler::GetInstance().SetGPUInfo(0, gpu_info);
```

### 5. ROCm stub в lch_farrow_rocm.hpp

Stub (`!ENABLE_ROCM`) тоже нужно обновить — добавить `prof_events = nullptr` к сигнатурам заглушек.

---

## Ожидаемые результаты

После выполнения:
1. `lch_farrow.cpp` — нет ни `SetGPUInfo`, ни `RecordProfilingEvent` — **production чистый**
2. `lch_farrow_rocm.cpp` — нет кода профилирования в production путях (только в `if (prof_events)`)
3. Бенчмарки запускаются через `test_lch_farrow_benchmark::run()` / `test_lch_farrow_benchmark_rocm::run()`
4. Отчёты в `Results/Profiler/GPU_00_LchFarrow/` + `Results/Profiler/GPU_00_LchFarrow_ROCm/`

---

*Задача создана: 2026-03-01*
*Ветка: Profiller01*
