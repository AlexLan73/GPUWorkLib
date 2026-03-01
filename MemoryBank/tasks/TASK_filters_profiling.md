# TASK: Рефакторинг профилирования modules/filters

> **Статус**: BACKLOG
> **Приоритет**: Medium
> **Ветка**: Profiller01
> **Дата**: 2026-03-01
> **Инструкция**: `Doc_Addition/GPU_Profiling_Mechanism.md`
> **Эталон**: `modules/fft_processor` (полностью выполнено)

---

## Цель

Привести `modules/filters` к архитектуре "чистый production-код + опциональное профилирование":

1. **OpenCL** (`FirFilter`, `IirFilter`):
   - Удалить `RecordProfilingEvent()` из `.cpp` (нарушение принципа чистого production-кода)
   - Добавить `ProfEvents` + `CollectOrRelease` + параметр `prof_events = nullptr` к `Process()`
   - Создать benchmark-классы + test runner

2. **ROCm** (`FirFilterROCm`, `IirFilterROCm`):
   - Добавить `ROCmProfEvents` + `MakeROCmDataFromEvents/Clock` helpers + параметр к методам
   - Добавить hipEvent-инструментацию в `Process()` и `ProcessFromCPU()`
   - Создать ROCm benchmark-классы + test runner

---

## Анализ текущего состояния

### OpenCL — НАРУШЕНИЯ (требуют исправления)

#### `fir_filter.cpp` — строки 192–197
```cpp
// ❌ СТАРЫЙ API (нарушение: прямой вызов GPUProfiler из production-кода)
int gpu_id = backend_->GetDeviceIndex();
if (gpu_id < 0) gpu_id = 0;
drv_gpu_lib::RecordProfilingEvent(
    kernel_event, gpu_id, "FirFilter", kernel_name);
if (kernel_event) clReleaseEvent(kernel_event);
```
Заголовки для удаления из `fir_filter.cpp` строки 16–17:
```cpp
#include "services/gpu_profiler.hpp"           // ❌ удалить
#include "backends/opencl/opencl_profiling.hpp" // ❌ удалить
```

#### `iir_filter.cpp` — строки 181–186
```cpp
// ❌ СТАРЫЙ API
int gpu_id = backend_->GetDeviceIndex();
if (gpu_id < 0) gpu_id = 0;
drv_gpu_lib::RecordProfilingEvent(
    kernel_event, gpu_id, "IirFilter", "iir_biquad_cascade_cf32");
if (kernel_event) clReleaseEvent(kernel_event);
```
Аналогично — удалить `#include "services/gpu_profiler.hpp"` и `opencl_profiling.hpp`.

### OpenCL — Стадии профилирования

| Метод | Стадия | Событие |
|-------|--------|---------|
| `FirFilter::Process()` | `"Kernel"` | `clEnqueueNDRangeKernel` → `cl_event kernel_event` |
| `IirFilter::Process()` | `"Kernel"` | `clEnqueueNDRangeKernel` → `cl_event kernel_event` |

> ⚠️ Нет стадии Upload — входные данные уже `cl_mem` на GPU.

### ROCm — ЧИСТЫЕ (только добавить профилирование)

#### `fir_filter_rocm.cpp` — `Process()`:
```
hipModuleLaunchKernel → hipStreamSynchronize
```

#### `fir_filter_rocm.cpp` — `ProcessFromCPU()`:
```
hipMalloc → hipMemcpyHtoDAsync → hipStreamSynchronize → Process() → hipFree(input)
```

#### `iir_filter_rocm.cpp` — идентичная структура.

### ROCm — Стадии профилирования

| Метод | Стадии |
|-------|--------|
| `Process()` | `"Kernel"` (hipEvent на `hipModuleLaunchKernel`) |
| `ProcessFromCPU()` | `"Upload"` (H2D, hipEvent) + передаёт `prof_events` в `Process()` |

---

## Задачи

---

### TASK 1: `fir_filter.hpp` — ProfEvents + CollectOrRelease + сигнатура

**Файл**: `modules/filters/include/filters/fir_filter.hpp`

#### Что добавить

После `#include <CL/cl.h>` (или в начале public-секции) добавить в namespace `filters`:

```cpp
/// Список событий профилирования OpenCL
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;
```

Добавить статический helper в класс `FirFilter` (private или protected):
```cpp
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* pe) {
    if (pe) {
        pe->push_back({name, ev});
    } else {
        if (ev) clReleaseEvent(ev);
    }
}
```

Изменить сигнатуру `Process()`:
```cpp
// БЫЛО:
drv_gpu_lib::InputData<cl_mem>
Process(cl_mem input_buf, uint32_t channels, uint32_t points);

// СТАЛО:
drv_gpu_lib::InputData<cl_mem>
Process(cl_mem input_buf, uint32_t channels, uint32_t points,
        ProfEvents* prof_events = nullptr);
```

---

### TASK 2: `fir_filter.cpp` — Замена RecordProfilingEvent на CollectOrRelease

**Файл**: `modules/filters/src/fir_filter.cpp`

#### 2.1 Удалить старые #include (строки 16–17)
```cpp
// ❌ УДАЛИТЬ:
#include "services/gpu_profiler.hpp"
#include "backends/opencl/opencl_profiling.hpp"
```

#### 2.2 Обновить сигнатуру метода
```cpp
// БЫЛО:
drv_gpu_lib::InputData<cl_mem>
FirFilter::Process(cl_mem input_buf, uint32_t channels, uint32_t points) {

// СТАЛО:
drv_gpu_lib::InputData<cl_mem>
FirFilter::Process(cl_mem input_buf, uint32_t channels, uint32_t points,
                   ProfEvents* prof_events) {
```

#### 2.3 Заменить блок профилирования (строки 192–197)

```cpp
// ❌ БЫЛО (строки 192–197):
int gpu_id = backend_->GetDeviceIndex();
if (gpu_id < 0) gpu_id = 0;
drv_gpu_lib::RecordProfilingEvent(
    kernel_event, gpu_id, "FirFilter", kernel_name);
if (kernel_event) clReleaseEvent(kernel_event);

// ✅ СТАЛО:
CollectOrRelease(kernel_event, "Kernel", prof_events);
```

> **Важно**: `CollectOrRelease` либо сохраняет event в `pe` (если `prof_events != nullptr`),
> либо вызывает `clReleaseEvent(ev)` (если `nullptr`). Отдельный `clReleaseEvent` больше
> не нужен — он внутри `CollectOrRelease`.

#### Итоговый вид блока (после `clFinish`):
```cpp
clFinish(queue_);

CollectOrRelease(kernel_event, "Kernel", prof_events);

// Build result
drv_gpu_lib::InputData<cl_mem> result;
result.antenna_count    = channels;
result.n_point          = points;
result.data             = output_buf;
result.gpu_memory_bytes = buffer_size;
return result;
```

---

### TASK 3: `iir_filter.hpp` — ProfEvents + CollectOrRelease + сигнатура

**Файл**: `modules/filters/include/filters/iir_filter.hpp`

Идентично TASK 1, но для `IirFilter`. Добавить:
- `using ProfEvents = std::vector<std::pair<const char*, cl_event>>;` в namespace `filters`
  (или переиспользовать из `fir_filter.hpp`, если они в одном namespace — но лучше
  определить в каждом классе отдельно для независимости)
- `static void CollectOrRelease(...)` в класс `IirFilter`
- Параметр `ProfEvents* prof_events = nullptr` к `Process()`

```cpp
// СТАЛО:
drv_gpu_lib::InputData<cl_mem>
Process(cl_mem input_buf, uint32_t channels, uint32_t points,
        ProfEvents* prof_events = nullptr);
```

---

### TASK 4: `iir_filter.cpp` — Замена RecordProfilingEvent на CollectOrRelease

**Файл**: `modules/filters/src/iir_filter.cpp`

Идентично TASK 2:

#### 4.1 Удалить старые #include (аналогично fir_filter.cpp)

#### 4.2 Обновить сигнатуру метода `IirFilter::Process()`

#### 4.3 Заменить блок профилирования (строки 181–186):
```cpp
// ❌ БЫЛО:
int gpu_id = backend_->GetDeviceIndex();
if (gpu_id < 0) gpu_id = 0;
drv_gpu_lib::RecordProfilingEvent(
    kernel_event, gpu_id, "IirFilter", "iir_biquad_cascade_cf32");
if (kernel_event) clReleaseEvent(kernel_event);

// ✅ СТАЛО:
CollectOrRelease(kernel_event, "Kernel", prof_events);
```

---

### TASK 5: `fir_filter_rocm.hpp` — ROCmProfEvents + сигнатуры

**Файл**: `modules/filters/include/filters/fir_filter_rocm.hpp`
**Условие**: внутри `#if ENABLE_ROCM`

#### 5.1 Добавить #include
```cpp
#include "DrvGPU/services/profiling_types.hpp"
```

#### 5.2 Добавить type alias в namespace `filters` (внутри `#if ENABLE_ROCM`):
```cpp
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

#### 5.3 Обновить сигнатуры методов класса `FirFilterROCm`:
```cpp
// БЫЛО:
drv_gpu_lib::InputData<void*>
Process(void* input_ptr, uint32_t channels, uint32_t points);

drv_gpu_lib::InputData<void*>
ProcessFromCPU(const std::vector<std::complex<float>>& data,
               uint32_t channels, uint32_t points);

// СТАЛО:
drv_gpu_lib::InputData<void*>
Process(void* input_ptr, uint32_t channels, uint32_t points,
        ROCmProfEvents* prof_events = nullptr);

drv_gpu_lib::InputData<void*>
ProcessFromCPU(const std::vector<std::complex<float>>& data,
               uint32_t channels, uint32_t points,
               ROCmProfEvents* prof_events = nullptr);
```

---

### TASK 6: `fir_filter_rocm.cpp` — hipEvent-инструментация

**Файл**: `modules/filters/src/fir_filter_rocm.cpp`
**Условие**: внутри `#if ENABLE_ROCM`

#### 6.1 Добавить #include
```cpp
#include "DrvGPU/services/profiling_types.hpp"
#include <chrono>
```

#### 6.2 Добавить анонимный namespace с helpers (после всех #include, перед `namespace filters`)

```cpp
namespace {

/// Helper для GPU-событий (async: hipEvent → elapsed → destroy)
drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind = 0, const char* op = "")
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

}  // namespace
```

> **Примечание**: `MakeROCmDataFromClock` (для chrono) не нужен — H2D Upload
> удобнее мерять hipEvent-ами, как и kernel. `hipStreamSynchronize` уже есть
> в `ProcessFromCPU`, поэтому hipEvent будет синхронизирован.

#### 6.3 Обновить сигнатуру `FirFilterROCm::Process()`

```cpp
// БЫЛО:
drv_gpu_lib::InputData<void*>
FirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {

// СТАЛО:
drv_gpu_lib::InputData<void*>
FirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points,
                       ROCmProfEvents* prof_events) {
```

#### 6.4 Добавить hipEvent-инструментацию в `Process()` вокруг `hipModuleLaunchKernel`

```cpp
// Перед запуском ядра — создать events если нужно профилирование:
hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
}

err = hipModuleLaunchKernel(
    kernel_,
    grid_size, 1, 1,
    kBlockSize, 1, 1,
    0, stream_,
    args, nullptr);

if (prof_events) {
    hipEventRecord(ev_k_e, stream_);
}

if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(...);
}

(void)hipStreamSynchronize(stream_);

// Собрать profiling data после синхронизации:
if (prof_events) {
    prof_events->push_back({"Kernel", MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "fir_filter")});
}
```

#### 6.5 Обновить сигнатуру `FirFilterROCm::ProcessFromCPU()`

```cpp
// БЫЛО:
drv_gpu_lib::InputData<void*>
FirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {

// СТАЛО:
drv_gpu_lib::InputData<void*>
FirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points,
    ROCmProfEvents* prof_events) {
```

#### 6.6 Добавить hipEvent для H2D Upload в `ProcessFromCPU()`

```cpp
// Перед hipMemcpyHtoDAsync:
hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
if (prof_events) {
    hipEventCreate(&ev_up_s);
    hipEventCreate(&ev_up_e);
    hipEventRecord(ev_up_s, stream_);
}

err = hipMemcpyHtoDAsync(input_ptr,
                          const_cast<std::complex<float>*>(data.data()),
                          data_size, stream_);

if (prof_events) {
    hipEventRecord(ev_up_e, stream_);
}

if (err != hipSuccess) {
    if (ev_up_s) { hipEventDestroy(ev_up_s); hipEventDestroy(ev_up_e); }
    (void)hipFree(input_ptr);
    throw std::runtime_error("FirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
}
(void)hipStreamSynchronize(stream_);

if (prof_events) {
    prof_events->push_back({"Upload", MakeROCmDataFromEvents(ev_up_s, ev_up_e, 0, "H2D")});
}

// Передать prof_events дальше — Process() добавит "Kernel":
auto result = Process(input_ptr, channels, points, prof_events);
```

> ⚠️ Обратить внимание: `hipStreamSynchronize` вызывается ПОСЛЕ `hipEventRecord(ev_up_e)`.
> После `hipStreamSynchronize` можно вызывать `hipEventElapsedTime` (внутри `MakeROCmDataFromEvents`).

---

### TASK 7: `iir_filter_rocm.hpp` — ROCmProfEvents + сигнатуры

**Файл**: `modules/filters/include/filters/iir_filter_rocm.hpp`

Идентично TASK 5. Добавить:
- `#include "DrvGPU/services/profiling_types.hpp"`
- `using ROCmProfEvents = ...;`
- `ROCmProfEvents* prof_events = nullptr` к `Process()` и `ProcessFromCPU()`

---

### TASK 8: `iir_filter_rocm.cpp` — hipEvent-инструментация

**Файл**: `modules/filters/src/iir_filter_rocm.cpp`

Идентично TASK 6:

#### 8.1 Добавить #include
```cpp
#include "DrvGPU/services/profiling_types.hpp"
#include <chrono>
```

#### 8.2 Добавить анонимный namespace с `MakeROCmDataFromEvents` (идентично TASK 6.2)

#### 8.3 Обновить сигнатуру `IirFilterROCm::Process()`
```cpp
drv_gpu_lib::InputData<void*>
IirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points,
                       ROCmProfEvents* prof_events) {
```

#### 8.4 Инструментация `Process()` — вокруг `hipModuleLaunchKernel`

Структура `Process()` в `iir_filter_rocm.cpp`:
```
hipMalloc(output_ptr) → hipModuleLaunchKernel → hipStreamSynchronize
```

Добавить hipEvent вокруг `hipModuleLaunchKernel` (идентично TASK 6.4):
```cpp
hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
}

err = hipModuleLaunchKernel(kernel_, ...);

if (prof_events) {
    hipEventRecord(ev_k_e, stream_);
}

if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(...);
}

(void)hipStreamSynchronize(stream_);

if (prof_events) {
    prof_events->push_back({"Kernel", MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "iir_filter")});
}
```

#### 8.5 Обновить сигнатуру `IirFilterROCm::ProcessFromCPU()` + H2D timing

Идентично TASK 6.5 и 6.6, заменив "FirFilterROCm" → "IirFilterROCm".

Структура `ProcessFromCPU()` в `iir_filter_rocm.cpp`:
```
hipMalloc(input_ptr) → hipMemcpyHtoDAsync → hipStreamSynchronize → Process(input_ptr,...) → hipFree(input_ptr)
```

---

### TASK 9: Создать `tests/filters_benchmark.hpp` — OpenCL benchmark-классы

**Файл**: `modules/filters/tests/filters_benchmark.hpp` (НОВЫЙ)
**Namespace**: `test_filters`

Два класса-наследника `drv_gpu_lib::GpuBenchmarkBase`:

#### `FirFilterBenchmark`
```cpp
class FirFilterBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
    FirFilterBenchmark(
        drv_gpu_lib::IBackend* backend,
        filters::FirFilter& filter,
        cl_mem input_buf,    // уже на GPU
        uint32_t channels,
        uint32_t points,
        GpuBenchmarkBase::Config cfg = {
            .n_warmup   = 5,
            .n_runs     = 20,
            .output_dir = "Results/Profiler/GPU_00_FirFilter"})
      : GpuBenchmarkBase(backend, "FirFilter", cfg),
        filter_(filter), input_buf_(input_buf),
        channels_(channels), points_(points) {}

protected:
    /// Warmup — без timing
    void ExecuteKernel() override {
        auto result = filter_.Process(input_buf_, channels_, points_);
        clReleaseMemObject(result.data);   // освобождаем выходной буфер
    }

    /// Замер — с ProfEvents → RecordEvent → GPUProfiler
    void ExecuteKernelTimed() override {
        filters::FirFilter::ProfEvents pe;
        auto result = filter_.Process(input_buf_, channels_, points_, &pe);
        for (auto& [name, ev] : pe)
            RecordEvent(name, ev);       // clWaitForEvents + профдата + clReleaseEvent
        clReleaseMemObject(result.data);
    }

private:
    filters::FirFilter& filter_;
    cl_mem              input_buf_;
    uint32_t            channels_;
    uint32_t            points_;
};
```

#### `IirFilterBenchmark`
Идентично `FirFilterBenchmark` — заменить `FirFilter` → `IirFilter`, имя `"IirFilter"`,
`output_dir = "Results/Profiler/GPU_00_IirFilter"`.

---

### TASK 10: Создать `tests/test_filters_benchmark.hpp` — OpenCL test runner

**Файл**: `modules/filters/tests/test_filters_benchmark.hpp` (НОВЫЙ)
**Namespace**: `test_filters_benchmark`

```cpp
namespace test_filters_benchmark {

inline void run() {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "Filters[Benchmark]", "=== OpenCL Benchmark: FirFilter + IirFilter ===");

    // OpenCL init с CL_QUEUE_PROFILING_ENABLE (обязательно для cl_event timing!)
    drv_gpu_lib::ExternalOpenCLContext ctx;
    if (!ctx.Initialize(/* CL_QUEUE_PROFILING_ENABLE */)) {
        con.Print(0, "Filters[Benchmark]", "[SKIP] OpenCL init failed");
        return;
    }

    auto* backend = ctx.GetBackend();
    if (!backend->GetConfigGPU().is_prof) {
        con.Print(0, "Filters[Benchmark]", "[SKIP] is_prof=false in configGPU.json");
        return;
    }

    // Параметры теста
    constexpr uint32_t CHANNELS = 8;
    constexpr uint32_t POINTS   = 4096;

    // Подготовка входного cl_mem буфера (заполнить тестовыми данными)
    // ... (аналогично test_fft_maxima_benchmark.hpp)

    // FirFilter Benchmark
    {
        filters::FirFilter fir(backend);
        // Установить коэффициенты (64-тапный ФНЧ, идентично test_fir_basic.hpp)
        fir.SetCoefficients(/* coeffs */);

        test_filters::FirFilterBenchmark bench(backend, fir, input_buf, CHANNELS, POINTS);
        bench.Run();
        bench.Report();
    }

    // IirFilter Benchmark
    {
        filters::IirFilter iir(backend);
        // Установить секции (Butterworth 2nd order LP)
        iir.SetBiquadSections(/* sections */);

        test_filters::IirFilterBenchmark bench(backend, iir, input_buf, CHANNELS, POINTS,
            {.n_warmup=5, .n_runs=20, .output_dir="Results/Profiler/GPU_00_IirFilter"});
        bench.Run();
        bench.Report();
    }

    // Освободить input_buf
    clReleaseMemObject(input_buf);
}

}  // namespace test_filters_benchmark
```

> **Ключевое**: OpenCL queue должен быть создан с `CL_QUEUE_PROFILING_ENABLE`.
> Смотреть как это делается в `test_fft_maxima_benchmark.hpp`.

---

### TASK 11: Создать `tests/filters_benchmark_rocm.hpp` — ROCm benchmark-классы

**Файл**: `modules/filters/tests/filters_benchmark_rocm.hpp` (НОВЫЙ)
**Условие**: `#if ENABLE_ROCM`
**Namespace**: `test_filters_rocm`

#### `FirFilterROCmBenchmark`
```cpp
#if ENABLE_ROCM

class FirFilterROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
    FirFilterROCmBenchmark(
        drv_gpu_lib::IBackend* backend,
        filters::FirFilterROCm& filter,
        const std::vector<std::complex<float>>& input_data,
        uint32_t channels,
        uint32_t points,
        GpuBenchmarkBase::Config cfg = {
            .n_warmup   = 5,
            .n_runs     = 20,
            .output_dir = "Results/Profiler/GPU_00_FirFilter_ROCm"})
      : GpuBenchmarkBase(backend, "FirFilter_ROCm", cfg),
        filter_(filter), input_data_(input_data),
        channels_(channels), points_(points) {}

protected:
    void ExecuteKernel() override {
        auto result = filter_.ProcessFromCPU(input_data_, channels_, points_);
        hipFree(result.data);
    }

    void ExecuteKernelTimed() override {
        filters::FirFilterROCm::ROCmProfEvents events;
        auto result = filter_.ProcessFromCPU(input_data_, channels_, points_, &events);
        for (auto& [name, data] : events)
            RecordROCmEvent(name, data);
        hipFree(result.data);
    }

private:
    filters::FirFilterROCm&              filter_;
    std::vector<std::complex<float>>     input_data_;
    uint32_t                             channels_;
    uint32_t                             points_;
};
```

#### `IirFilterROCmBenchmark`
Идентично — заменить `FirFilterROCm` → `IirFilterROCm`, имя `"IirFilter_ROCm"`,
`output_dir = "Results/Profiler/GPU_00_IirFilter_ROCm"`.

---

### TASK 12: Создать `tests/test_filters_benchmark_rocm.hpp` — ROCm test runner

**Файл**: `modules/filters/tests/test_filters_benchmark_rocm.hpp` (НОВЫЙ)
**Условие**: `#if ENABLE_ROCM`
**Namespace**: `test_filters_benchmark_rocm`

```cpp
#if ENABLE_ROCM

namespace test_filters_benchmark_rocm {

inline void run() {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "Filters[Benchmark][ROCm]", "=== ROCm Benchmark: FirFilter + IirFilter ===");

    // Проверить AMD GPU
    if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
        con.Print(0, "Filters[Benchmark][ROCm]", "[SKIP] No AMD GPU available");
        return;
    }

    try {
        // ROCm backend init
        drv_gpu_lib::ROCmBackend backend;
        backend.Initialize();

        constexpr uint32_t CHANNELS = 8;
        constexpr uint32_t POINTS   = 4096;
        size_t total = CHANNELS * POINTS;

        // Тестовые данные: CW сигнал
        std::vector<std::complex<float>> input_data(total);
        // ... заполнить (аналогично test_filters_rocm.hpp)

        // FirFilterROCm Benchmark
        {
            filters::FirFilterROCm fir(&backend);
            fir.SetCoefficients(/* coeffs */);

            test_filters_rocm::FirFilterROCmBenchmark bench(&backend, fir, input_data, CHANNELS, POINTS);
            bench.Run();
            bench.Report();
        }

        // IirFilterROCm Benchmark
        {
            filters::IirFilterROCm iir(&backend);
            iir.SetBiquadSections(/* sections */);

            test_filters_rocm::IirFilterROCmBenchmark bench(&backend, iir, input_data, CHANNELS, POINTS,
                {.n_warmup=5, .n_runs=20, .output_dir="Results/Profiler/GPU_00_IirFilter_ROCm"});
            bench.Run();
            bench.Report();
        }

    } catch (const std::exception& e) {
        con.Print(0, "Filters[Benchmark][ROCm]", std::string("[SKIP] ") + e.what());
    }
}

}  // namespace test_filters_benchmark_rocm

#endif  // ENABLE_ROCM
```

---

### TASK 13: Обновить `tests/all_test.hpp`

**Файл**: `modules/filters/tests/all_test.hpp`

#### Добавить #include новых файлов:
```cpp
// Было:
#include "test_fir_basic.hpp"
#include "test_iir_basic.hpp"
#include "test_filters_rocm.hpp"

// Стало:
#include "test_fir_basic.hpp"
#include "test_iir_basic.hpp"
#include "test_filters_rocm.hpp"
#include "test_filters_benchmark.hpp"
#if ENABLE_ROCM
#include "test_filters_benchmark_rocm.hpp"
#endif
```

#### Обновить `run()`:
```cpp
namespace filters_all_test {

inline void run() {
    filters::tests::run_fir_basic();
    filters::tests::run_iir_basic();
    test_filters_rocm::run();  // ROCm — Linux only, uncomment on AMD GPU

    // BENCHMARK: FirFilter + IirFilter (OpenCL, GpuBenchmarkBase)
    // test_filters_benchmark::run();

    // BENCHMARK: FirFilter + IirFilter (ROCm, GpuBenchmarkBase)
#if ENABLE_ROCM
//  test_filters_benchmark_rocm::run();
#endif
}

}  // namespace filters_all_test
```

---

### TASK 14: Обновить `tests/README.md`

**Файл**: `modules/filters/tests/README.md`

Добавить секции для новых файлов:

```markdown
## filters_benchmark.hpp
**Namespace**: `test_filters`
**Дата**: 2026-03-01

Benchmark-классы наследники `GpuBenchmarkBase` (OpenCL):

| Класс | Метод | Стадия | Результаты |
|-------|-------|--------|------------|
| `FirFilterBenchmark` | `Process(cl_mem)` | Kernel | `Results/Profiler/GPU_00_FirFilter/` |
| `IirFilterBenchmark` | `Process(cl_mem)` | Kernel | `Results/Profiler/GPU_00_IirFilter/` |

- Входные данные (cl_mem) фиксированы между прогонами
- 5 warmup + 20 замерных → PrintReport + ExportJSON + ExportMarkdown

## test_filters_benchmark.hpp
**Namespace**: `test_filters_benchmark`
**Дата**: 2026-03-01

Test runner для OpenCL бенчмарков:
- 8 каналов × 4096 точек
- OpenCL init с `CL_QUEUE_PROFILING_ENABLE`
- Если `is_prof=false` — выводит `[SKIP]`

## filters_benchmark_rocm.hpp
**Namespace**: `test_filters_rocm`
**Дата**: 2026-03-01
**Условие**: `#if ENABLE_ROCM`

| Класс | Метод | Стадии | Результаты |
|-------|-------|--------|------------|
| `FirFilterROCmBenchmark` | `ProcessFromCPU` | Upload(H2D) + Kernel | `Results/Profiler/GPU_00_FirFilter_ROCm/` |
| `IirFilterROCmBenchmark` | `ProcessFromCPU` | Upload(H2D) + Kernel | `Results/Profiler/GPU_00_IirFilter_ROCm/` |

## test_filters_benchmark_rocm.hpp
**Namespace**: `test_filters_benchmark_rocm`
**Дата**: 2026-03-01
**Условие**: `#if ENABLE_ROCM`

ROCm test runner:
- Если нет AMD GPU — `[SKIP]`, не падает
- 5 warmup + 20 замерных → PrintReport + ExportJSON + ExportMarkdown
```

---

## Сводная таблица изменений

| # | Файл | Тип | Изменение |
|---|------|-----|-----------|
| 1 | `include/filters/fir_filter.hpp` | Изменить | ProfEvents + CollectOrRelease + `prof_events*` к Process() |
| 2 | `src/fir_filter.cpp` | Изменить | Убрать RecordProfilingEvent, убрать старые includes, CollectOrRelease |
| 3 | `include/filters/iir_filter.hpp` | Изменить | То же что TASK 1 |
| 4 | `src/iir_filter.cpp` | Изменить | То же что TASK 2 |
| 5 | `include/filters/fir_filter_rocm.hpp` | Изменить | ROCmProfEvents + `prof_events*` к Process/ProcessFromCPU |
| 6 | `src/fir_filter_rocm.cpp` | Изменить | Helpers + hipEvent в Process() + Upload timing в ProcessFromCPU() |
| 7 | `include/filters/iir_filter_rocm.hpp` | Изменить | То же что TASK 5 |
| 8 | `src/iir_filter_rocm.cpp` | Изменить | То же что TASK 6 |
| 9 | `tests/filters_benchmark.hpp` | НОВЫЙ | FirFilterBenchmark + IirFilterBenchmark (OpenCL) |
| 10 | `tests/test_filters_benchmark.hpp` | НОВЫЙ | OpenCL test runner |
| 11 | `tests/filters_benchmark_rocm.hpp` | НОВЫЙ | ROCm benchmark-классы |
| 12 | `tests/test_filters_benchmark_rocm.hpp` | НОВЫЙ | ROCm test runner |
| 13 | `tests/all_test.hpp` | Изменить | Добавить includes + закомментированные вызовы |
| 14 | `tests/README.md` | Изменить | Добавить описание новых файлов |

**Итого**: 8 изменённых файлов + 4 новых файла = **12 файлов**

---

## Важные правила (из GPU_Profiling_Mechanism.md)

1. **Порядок CollectOrRelease**: event сначала используется как wait для следующей
   операции (если есть), ПОТОМ `CollectOrRelease`. Для одиночного ядра — просто
   `clFinish(queue_)` + `CollectOrRelease`.

2. **CL_QUEUE_PROFILING_ENABLE**: Обязателен для `clGetEventProfilingInfo`.
   Создавать OpenCL queue с этим флагом в test runner.

3. **hipEvent порядок**:
   ```
   hipEventRecord(ev_s, stream_) → kernel → hipEventRecord(ev_e, stream_)
   → hipStreamSynchronize → MakeROCmDataFromEvents(ev_s, ev_e)
   ```
   `MakeROCmDataFromEvents` уничтожает events внутри — не трогать после вызова.

4. **Освобождение выходного буфера в benchmark**:
   - OpenCL: `clReleaseMemObject(result.data)` в каждом Execute*()
   - ROCm: `hipFree(result.data)` в каждом Execute*()

5. **Вывод**: ТОЛЬКО через `GPUProfiler.PrintReport()` / `ExportJSON()` / `ExportMarkdown()`.
   ЗАПРЕЩЕНО: `GetStats()` + цикл + `con.Print`.

---

*Создан: 2026-03-01*
*Автор: Кодо (AI Assistant)*
