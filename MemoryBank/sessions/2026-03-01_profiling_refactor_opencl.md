# Сессия: Рефакторинг профилирования fft_maxima (OpenCL)
**Дата**: 2026-03-01
**Задача**: `TASK_fft_maxima_profiling_opencl.md`
**Ветка**: Profiller01

---

## ✅ Что сделано

### TASK 1 — `modules/fft_maxima/include/spectrum_maxima_finder.h` — ЗАВЕРШЁН
- Добавлен `#include <utility>` (для `std::pair`)
- Добавлен `using ProfEvents = std::vector<std::pair<const char*, cl_event>>;` в `public:` секцию
- Добавлен `static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* prof_events)` в `private:` секцию
- Обновлены **публичные** шаблонные методы (добавлен `ProfEvents* prof_events = nullptr`):
  - `Process<T>()` — 4-й параметр
  - `FindAllMaxima<T>()` — 6-й параметр
  - `AllMaxima<T>()` — 6-й параметр
  - `FindAllMaxima(cl_mem,...)` — 11-й параметр
- Обновлены **приватные** объявления (`ProfEvents* pe = nullptr`):
  - `ProcessFromCPU`, `ProcessBatch`, `ProcessFromGPU`, `ProcessBatchFromGPU`
  - `ReadResults`, `FindAllMaximaFromCPU`, `FindAllMaximaFromGPUPipeline`, `AllMaximaFromCPU`
- Обновлены **реализации шаблонов** в заголовке (передают `prof_events` в приватные методы)

### TASK 2 — `modules/fft_maxima/src/spectrum_maxima_finder.cpp` — ЗАВЕРШЁН
- Удалены `#include "backends/opencl/opencl_profiling.hpp"` и `#include "services/gpu_profiler.hpp"`
- **ProcessBatch**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, заменены 3 `RecordProfilingEvent` → `CollectOrRelease` (Upload/FFT/PostKernel), передаёт `prof_events` в `ReadResults`
- **ReadResults**: добавлен `ProfEvents* pe`, удалён `gpu_id`, заменены `RecordProfilingEvent + clReleaseEvent` → `clWaitForEvents + CollectOrRelease` (Download)

---

## ⏳ Что осталось

### TASK 3 — `spectrum_maxima_finder_process.cpp` — ЗАВЕРШЁН
Файл: `modules/fft_maxima/src/spectrum_maxima_finder_process.cpp`

- Удалены `#include "backends/opencl/opencl_profiling.hpp"` и `#include "services/gpu_profiler.hpp"`
- **ProcessFromCPU**: добавлен `ProfEvents* prof_events`, оба вызова `ProcessBatch()` передают `prof_events`
- **ProcessFromGPU**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, 3 `RecordProfilingEvent` → `CollectOrRelease` (GPU→GPU Copy / FFT / PostKernel), `ReadResults(post, prof_events)`, передаёт `prof_events` в `ProcessBatchFromGPU()`
- **ProcessBatchFromGPU**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, те же 3 замены

### TASK 4 — `spectrum_maxima_finder_all_maxima.cpp` — ЗАВЕРШЁН
Файл: `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`

- Удалены `#include "backends/opencl/opencl_profiling.hpp"` и `#include "services/gpu_profiler.hpp"`
- **FindAllMaximaFromCPU**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, 2 замены (single-batch: Upload/FFT+PostCallback), 2 замены (batch: Upload/FFT), передаёт `prof_events` в `FindAllMaxima(...)`
- **FindAllMaximaFromGPUPipeline**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, 2 замены (single-batch: GPU→GPU Copy/FFT+PostCallback), 2 замены (batch: GPU→GPU Copy/FFT), передаёт `prof_events` в `FindAllMaxima(...)`
- **AllMaximaFromCPU**: добавлен `ProfEvents* prof_events`, удалены `gpu_id/profiler/do_prof`, передаёт `prof_events` в `FindAllMaxima(...)`
- **FindAllMaxima(cl_mem)**: добавлен `ProfEvents* prof_events`, удалён `gpu_id`, 4 замены:
  - `mag_event`: старый `RecordProfilingEvent` удалён, `clReleaseEvent(mag_event)` → `CollectOrRelease(mag_event, "ComputeMagnitudes", pe)` (после detect enqueue)
  - `detect_event`: `clReleaseEvent(detect_event)` → `CollectOrRelease(detect_event, "Detect", pe)` (после compact enqueue)
  - `scan_all_event`: `clReleaseEvent(scan_all_event)` → `CollectOrRelease(scan_all_event, "Scan", pe)` (после compact enqueue)
  - `compact_event`: `clReleaseEvent(compact_event)` → `CollectOrRelease(compact_event, "Compact", pe)` (после clWaitForEvents)

---

### TASK 5 — `tests/fft_maxima_benchmark.hpp` — ЗАВЕРШЁН

- **НОВЫЙ ФАЙЛ**: `modules/fft_maxima/tests/fft_maxima_benchmark.hpp`
- `SpectrumMaximaFinderBenchmark : GpuBenchmarkBase` — Process(ONE_PEAK): Upload+FFT+PostKernel
- `SpectrumMaximaAllMaximaBenchmark : GpuBenchmarkBase` — FindAllMaxima: Upload+FFT+Detect+Scan+Compact
- `ExecuteKernel()` — warmup без timing (prof_events = nullptr)
- `ExecuteKernelTimed()` — с ProfEvents → `RecordEvent()` для каждого cl_event

### TASK 6 — `tests/test_fft_maxima_benchmark.hpp` — ЗАВЕРШЁН

- **НОВЫЙ ФАЙЛ**: `modules/fft_maxima/tests/test_fft_maxima_benchmark.hpp`
- OpenCL init с `CL_QUEUE_PROFILING_ENABLE`
- GenerateSignal: 10 лучей × 8192 точек, Fs=100 кГц, комплексные синусоиды
- Benchmark 1: Process → `Results/Profiler/GPU_00_SpectrumMaxima_Process/`
- Benchmark 2: FindAllMaxima → `Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/`
- `IsProfEnabled()=false` → `[SKIP]`, не падает

### TASK 7 — `tests/all_test.hpp` — ЗАВЕРШЁН

- Добавлен `#include "test_fft_maxima_benchmark.hpp"`
- `test_benchmark_all_maxima::run()` помечен как DEPRECATED
- `test_fft_maxima_benchmark::run()` добавлен закомментированным

### TASK 8 — `tests/README.md` — ЗАВЕРШЁН

- `test_benchmark_all_maxima.hpp` помечен как DEPRECATED
- Добавлены описания `fft_maxima_benchmark.hpp` и `test_fft_maxima_benchmark.hpp`

---

### Как продолжить (оставшееся)

Нужно:
1. Удалить `#include "backends/opencl/opencl_profiling.hpp"` и `#include "services/gpu_profiler.hpp"`
2. **ProcessFromCPU** — добавить `ProfEvents* prof_events` к сигнатуре, передать в `ProcessBatch(data, batch.start, batch.count, prof_events)`
3. **ProcessFromGPU** — добавить `ProfEvents* prof_events`, в single-batch пути заменить 3 `RecordProfilingEvent` → `CollectOrRelease`:
   ```cpp
   // БЫЛО:
   drv_gpu_lib::RecordProfilingEvent(copy_event, gpu_id, "SpectrumMaxima", "GPU→GPU Copy");
   cl_event fft_event = ExecuteFFT(copy_event);
   clReleaseEvent(copy_event);
   drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");
   cl_event post_event = ExecutePostKernel(fft_event);
   clReleaseEvent(fft_event);
   drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");
   auto results = ReadResults(post_event);
   clReleaseEvent(post_event);

   // СТАЛО:
   cl_event fft_event = ExecuteFFT(copy_event);
   CollectOrRelease(copy_event, "GPU→GPU Copy", prof_events);
   cl_event post_event = ExecutePostKernel(fft_event);
   CollectOrRelease(fft_event, "FFT", prof_events);
   auto results = ReadResults(post_event, prof_events);
   CollectOrRelease(post_event, "PostKernel", prof_events);
   ```
   В batch-пути передать `prof_events` в `ProcessBatchFromGPU`
4. **ProcessBatchFromGPU** — аналогично: добавить `ProfEvents* prof_events`, те же 3 замены (Copy/FFT/PostKernel), `ReadResults(post_event, prof_events)`

### TASK 5 — Создать `tests/fft_maxima_benchmark.hpp` (НОВЫЙ файл)

12 вызовов `RecordProfilingEvent` в 4 методах → `CollectOrRelease`:

| Метод | Старое событие | Новое имя |
|-------|---------------|-----------|
| FindAllMaximaFromCPU (single) | Upload | `"Upload"` |
| FindAllMaximaFromCPU (single) | FFT+PostCallback | `"FFT+PostCallback"` |
| FindAllMaximaFromCPU (batch) | Upload_Batch | `"Upload"` |
| FindAllMaximaFromCPU (batch) | FFT_Batch | `"FFT"` |
| FindAllMaximaFromGPUPipeline (single) | GPU→GPU Copy | `"GPU→GPU Copy"` |
| FindAllMaximaFromGPUPipeline (single) | FFT+PostCallback | `"FFT+PostCallback"` |
| FindAllMaximaFromGPUPipeline (batch) | GPU→GPU Copy_Batch | `"GPU→GPU Copy"` |
| FindAllMaximaFromGPUPipeline (batch) | FFT_Batch | `"FFT"` |
| FindAllMaxima(cl_mem) | ComputeMagnitudes | `"ComputeMagnitudes"` |
| FindAllMaxima(cl_mem) | Detect | `"Detect"` |
| FindAllMaxima(cl_mem) | Scan | `"Scan"` |
| FindAllMaxima(cl_mem) | Compact | `"Compact"` |

**Важно про порядок** (wait_event должен быть передан ПЕРЕД CollectOrRelease):
```cpp
// FindAllMaximaFromCPU:
cl_event upload_event = UploadData(data);
cl_event fft_event = ExecuteAllMaximaFFT(upload_event);
CollectOrRelease(upload_event, "Upload", pe);       // после ExecuteAllMaximaFFT!
AllMaximaResult result = FindAllMaxima(..., pe);
CollectOrRelease(fft_event, "FFT+PostCallback", pe);// после FindAllMaxima (fft_event не нужен ей)

// FindAllMaxima(cl_mem):
// mag_event — используется как wait для detect kernel, потом CollectOrRelease вместо clReleaseEvent
// detect_event — передать в ExecutePrefixSum, потом CollectOrRelease
// scan_all_event — передать как wait в compact kernel, потом CollectOrRelease
// compact_event — передать как wait в ReadBuffer, потом CollectOrRelease
```

Изменить **AllMaximaFromCPU** (строка ~503): убрать `GPUProfiler` доступ, передать `pe` в `FindAllMaxima(gpu_fft, ..., pe)`.

### TASK 5 — Создать `tests/fft_maxima_benchmark.hpp` (НОВЫЙ файл)
Два класса-наследника `GpuBenchmarkBase`:
- `SpectrumMaximaFinderBenchmark` — `Process<T>()` с `ProfEvents`
- `SpectrumMaximaAllMaximaBenchmark` — `FindAllMaxima<T>()` с `ProfEvents`

Готовый код — в `MemoryBank/tasks/TASK_fft_maxima_profiling_opencl.md` → раздел TASK 5

### TASK 6 — Создать `tests/test_fft_maxima_benchmark.hpp` (НОВЫЙ файл)
Test runner с OpenCL init (`CL_QUEUE_PROFILING_ENABLE`!), 2 бенчмарка.
Готовый код — в `MemoryBank/tasks/TASK_fft_maxima_profiling_opencl.md` → раздел TASK 6

### TASK 7 — Обновить `tests/all_test.hpp`
```cpp
// Добавить include:
#include "test_fft_maxima_benchmark.hpp"
// Добавить (закомментировать):
// test_fft_maxima_benchmark::run();
// Закомментировать старый:
// test_benchmark_all_maxima::run();
```

### TASK 8 — Обновить `tests/README.md`
Добавить таблицу новых benchmark-файлов + пометить `test_benchmark_all_maxima.hpp` как DEPRECATED.

### После OpenCL — ROCm (TASK 9-14)
Детали в `MemoryBank/tasks/TASK_fft_maxima_profiling_rocm.md`

---

## Ключевые паттерны

```cpp
// CollectOrRelease (в private секции SpectrumMaximaFinder):
static void CollectOrRelease(cl_event ev, const char* name, ProfEvents* pe) {
    if (!ev) return;
    if (pe) { pe->push_back({name, ev}); }
    else    { clReleaseEvent(ev); }
}

// Правило: сначала передать event как wait, ПОТОМ CollectOrRelease:
cl_event a = Operation1();
cl_event b = Operation2(a);       // a — wait для b
CollectOrRelease(a, "Stage1", pe); // теперь можно освободить/сохранить a
cl_event c = Operation3(b);
CollectOrRelease(b, "Stage2", pe);

// ReadResults: CL_FALSE → нужен clWaitForEvents перед CollectOrRelease:
clWaitForEvents(1, &read_event);
CollectOrRelease(read_event, "Download", pe);
```

---

## Как продолжить в следующем чате

Сказать: **"Продолжи рефакторинг профилирования fft_maxima. TASK 4 готов, начни TASK 5."**

Кодо прочитает этот файл + `MemoryBank/tasks/TASK_fft_maxima_profiling_opencl.md` и продолжит с TASK 5.
