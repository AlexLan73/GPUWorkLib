# TASK: Профилирование модуля heterodyne (OpenCL + ROCm)

> **Статус**: ✅ DONE (14/14)
> **Ветка**: Profiller01
> **Дата**: 2026-03-01
> **Референс**: `modules/fft_processor/` + `Doc_Addition/GPU_Profiling_Mechanism.md`
> **Итого тасков**: 14 (A1-A4 · B1-B4 · C1 · D1 · E1 · F1 · G1-G2)

---

## Контекст модуля

**Структура heterodyne**:
```
modules/heterodyne/
├── include/
│   ├── i_heterodyne_processor.hpp          ← Интерфейс (4 virtual методы)
│   ├── heterodyne_params.hpp               ← HeterodyneParams, HeterodyneResult
│   └── processors/
│       ├── heterodyne_processor_opencl.hpp ← OpenCL prod-класс
│       └── heterodyne_processor_rocm.hpp   ← ROCm prod-класс (#if ENABLE_ROCM)
├── src/
│   ├── heterodyne_processor_opencl.cpp
│   └── heterodyne_processor_rocm.cpp
└── tests/
    ├── all_test.hpp
    ├── test_heterodyne_basic.hpp
    ├── test_heterodyne_pipeline.hpp
    ├── test_heterodyne_rocm.hpp
    └── README.md
```

**Методы-кандидаты для профилирования**:

| Метод | Стадии | Примечание |
|-------|--------|------------|
| `Dechirp(rx, ref, params)` | Upload_Rx + Upload_Ref + Kernel_Multiply + Download | Основной метод |
| `Correct(dc, f_beat, params)` | Upload_DC + Upload_PhaseStep + Kernel_Correct + Download | Частотная коррекция |
| `DechirpFromGPU(rx_cl_mem, ref, params)` | Upload_Ref + Kernel_Multiply + Download | rx уже на GPU |
| `DechirpWithGPURef(rx_cl_mem, ref_cl_mem, params)` | Kernel_Multiply + Download | оба буфера на GPU |

**Ключевой архитектурный нюанс**: `HeterodyneProcessorOpenCL` и `HeterodyneProcessorROCm`
наследуют от `IHeterodyneProcessor` через `override`. Добавлять `prof_events` параметр
**нельзя к override-методам** (нарушение сигнатуры = больше не override).

**Решение**: Паттерн "делегирование":
```cpp
// В .hpp — оставляем override (делегирующий):
std::vector<std::complex<float>> Dechirp(rx, ref, params) override {
  return Dechirp(rx, ref, params, nullptr);  // ← делегирует к impl
}

// Добавляем новый публичный метод с prof_events:
std::vector<std::complex<float>> Dechirp(rx, ref, params,
    std::vector<std::pair<const char*, cl_event>>* prof_events);

// В .cpp — реализация только с prof_events версии (не override)
```

**Тип prof_events**:
- OpenCL: `std::vector<std::pair<const char*, cl_event>>*`
- ROCm: `std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>*`

**Alias для удобства** (в .hpp рядом с классом):
```cpp
// OpenCL
using HeterodyneOCLProfEvents = std::vector<std::pair<const char*, cl_event>>;

// ROCm (в heterodyne_processor_rocm.hpp)
using HeterodyneROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

**Параметры бенчмарков**:
```
num_antennas = 5
num_samples  = 4000
sample_rate  = 12 MHz
n_warmup = 5, n_runs = 20
output_dir = "Results/Profiler/GPU_00_Heterodyne"        (OpenCL)
output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"   (ROCm)
```

---

## A — OpenCL Production: добавить prof_events

### A1 — `HeterodyneProcessorOpenCL::Dechirp()` — ProfEvents

**Файлы**: `include/processors/heterodyne_processor_opencl.hpp` + `src/heterodyne_processor_opencl.cpp`

#### `.hpp` — изменения:

1. Добавить include `<utility>` (для `std::pair`) — если нет
2. Добавить alias рядом с классом (перед `class HeterodyneProcessorOpenCL`):
   ```cpp
   using HeterodyneOCLProfEvents = std::vector<std::pair<const char*, cl_event>>;
   ```
3. Изменить `Dechirp()` override — сделать делегирующим (inline в .hpp):
   ```cpp
   std::vector<std::complex<float>> Dechirp(
       const std::vector<std::complex<float>>& rx_data,
       const std::vector<std::complex<float>>& ref_data,
       const HeterodyneParams& params) override {
     return Dechirp(rx_data, ref_data, params, nullptr);
   }
   ```
4. Добавить новую публичную перегрузку (декларация в .hpp):
   ```cpp
   std::vector<std::complex<float>> Dechirp(
       const std::vector<std::complex<float>>& rx_data,
       const std::vector<std::complex<float>>& ref_data,
       const HeterodyneParams& params,
       HeterodyneOCLProfEvents* prof_events);
   ```

#### `.cpp` — изменения:

1. В начало файла добавить helper `CollectOrRelease` (static, один раз на файл):
   ```cpp
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
   ```

2. Старый `Dechirp()` из .cpp УДАЛИТЬ (теперь он inline делегирует).

3. Добавить новую реализацию `Dechirp(..., prof_events)`:
   - Upload rx: `cl_event ev_rx = nullptr;` → `clEnqueueWriteBuffer(..., &ev_rx)` → `CollectOrRelease(ev_rx, "Upload_Rx", prof_events)`
   - Upload ref: `cl_event ev_ref = nullptr;` → `clEnqueueWriteBuffer(..., &ev_ref)` → `CollectOrRelease(ev_ref, "Upload_Ref", prof_events)`
   - Kernel: `cl_event ev_k = nullptr;` → `clEnqueueNDRangeKernel(..., &ev_k)` → `CollectOrRelease(ev_k, "Kernel_Multiply", prof_events)`
   - Download: `cl_event ev_dl = nullptr;` → `clEnqueueReadBuffer(CL_TRUE, ..., &ev_dl)` → `CollectOrRelease(ev_dl, "Download", prof_events)`

   ⚠️ **Паттерн CollectOrRelease**: event передаётся в CollectOrRelease ПОСЛЕ операции
   (не используется как wait_list после передачи).

---

### A2 — `HeterodyneProcessorOpenCL::Correct()` — ProfEvents

Аналогично A1.

**Стадии**:
- `Upload_DC` — `clEnqueueWriteBuffer(buf_dc_, dc_data, ...)`
- `Upload_PhaseStep` — `clEnqueueWriteBuffer(buf_freq_, phase_step, ...)`
- `Kernel_Correct` — `clEnqueueNDRangeKernel(kernel_correct_, ...)`
- `Download` — `clEnqueueReadBuffer(buf_corr_, result, ...)`

#### `.hpp` — изменения:
- Изменить `Correct()` override → inline делегирующий к `Correct(..., nullptr)`
- Добавить перегрузку `Correct(..., HeterodyneOCLProfEvents* prof_events)`

#### `.cpp` — изменения:
- Удалить старую реализацию из .cpp (теперь в .hpp inline)
- Добавить `Correct(..., prof_events)` с четырьмя CollectOrRelease вызовами

---

### A3 — `HeterodyneProcessorOpenCL::DechirpFromGPU()` — ProfEvents

**Стадии** (rx уже на GPU — нет Upload_Rx):
- `Upload_Ref` — `clEnqueueWriteBuffer(buf_ref_, ref_data, ...)`
- `Kernel_Multiply` — `clEnqueueNDRangeKernel(kernel_multiply_, ...)`
- `Download` — `clEnqueueReadBuffer(buf_dc_, result, ...)`

#### `.hpp` — изменения:
- Изменить `DechirpFromGPU()` override → inline делегирующий к `DechirpFromGPU(..., nullptr)`
- Добавить перегрузку `DechirpFromGPU(..., HeterodyneOCLProfEvents* prof_events)`

---

### A4 — `HeterodyneProcessorOpenCL::DechirpWithGPURef()` — ProfEvents

**Стадии** (оба буфера на GPU — нет uploads):
- `Kernel_Multiply` — `clEnqueueNDRangeKernel(kernel_multiply_, ...)`
- `Download` — `clEnqueueReadBuffer(buf_dc_, result, ...)`

#### `.hpp` — изменения:
- Изменить `DechirpWithGPURef()` → inline делегирующий к `DechirpWithGPURef(..., nullptr)`
- Добавить перегрузку `DechirpWithGPURef(..., HeterodyneOCLProfEvents* prof_events)`

---

## B — ROCm Production: добавить ROCmProfEvents

### B1 — `HeterodyneProcessorROCm::Dechirp()` — ROCmProfEvents

**Файлы**: `include/processors/heterodyne_processor_rocm.hpp` + `src/heterodyne_processor_rocm.cpp`

#### `.hpp` — изменения (внутри `#if ENABLE_ROCM`):

1. Добавить include `"DrvGPU/services/profiling_types.hpp"` — для `ROCmProfilingData`
2. Добавить alias:
   ```cpp
   using HeterodyneROCmProfEvents =
       std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
   ```
3. Изменить `Dechirp()` override → inline делегирующий:
   ```cpp
   std::vector<std::complex<float>> Dechirp(rx, ref, params) override {
     return Dechirp(rx, ref, params, nullptr);
   }
   ```
4. Добавить перегрузку `Dechirp(..., HeterodyneROCmProfEvents* prof_events)`

#### `.cpp` — изменения (внутри `#if ENABLE_ROCM`):

1. Добавить helper'ы `MakeROCmDataFromEvents` и `MakeROCmDataFromClock` (static):
   ```cpp
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
     uint64_t ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
     d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
     d.kind = kind; d.op_string = op_string;
     return d;
   }
   ```

2. Реализация `Dechirp(..., prof_events)`:
   - Upload_Rx (H2D async → hipEvent):
     ```cpp
     hipEvent_t ev_rx_s = nullptr, ev_rx_e = nullptr;
     if (prof_events) { hipEventCreate(&ev_rx_s); hipEventCreate(&ev_rx_e);
                        hipEventRecord(ev_rx_s, stream_); }
     hipMemcpyHtoDAsync(buf_rx_, ..., stream_);
     if (prof_events) hipEventRecord(ev_rx_e, stream_);
     ```
   - Upload_Ref (H2D async → hipEvent): аналогично
   - Kernel_Multiply:
     ```cpp
     hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
     if (prof_events) { hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
                        hipEventRecord(ev_k_s, stream_); }
     hipModuleLaunchKernel(...);
     if (prof_events) hipEventRecord(ev_k_e, stream_);
     ```
   - Download (D2H async → hipEvent):
     ```cpp
     hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
     if (prof_events) { hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
                        hipEventRecord(ev_dl_s, stream_); }
     hipMemcpyDtoHAsync(result.data(), buf_dc_, ..., stream_);
     if (prof_events) hipEventRecord(ev_dl_e, stream_);
     ```
   - После `hipStreamSynchronize(stream_)` → собрать в prof_events:
     ```cpp
     if (prof_events) {
       prof_events->push_back({"Upload_Rx",   MakeROCmDataFromEvents(ev_rx_s,  ev_rx_e,  1, "H2D")});
       prof_events->push_back({"Upload_Ref",  MakeROCmDataFromEvents(ev_ref_s, ev_ref_e, 1, "H2D")});
       prof_events->push_back({"Kernel_Multiply", MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "dechirp_multiply")});
       prof_events->push_back({"Download",    MakeROCmDataFromEvents(ev_dl_s,  ev_dl_e,  1, "D2H")});
     }
     ```

---

### B2 — `HeterodyneProcessorROCm::Correct()` — ROCmProfEvents

**Стадии**:
- `Upload_DC` (H2D async) → `MakeROCmDataFromEvents`
- `Upload_PhaseStep` (H2D async) → `MakeROCmDataFromEvents`
- `Kernel_Correct` → `MakeROCmDataFromEvents`
- `Download` (D2H async+sync) → `MakeROCmDataFromEvents`

Аналогично B1.

---

### B3 — `HeterodyneProcessorROCm::DechirpFromGPU()` — ROCmProfEvents

**Стадии** (rx уже на GPU):
- `Upload_Ref` → `MakeROCmDataFromEvents`
- `Kernel_Multiply` → `MakeROCmDataFromEvents`
- `Download` → `MakeROCmDataFromEvents`

---

### B4 — `HeterodyneProcessorROCm::DechirpWithGPURef()` — ROCmProfEvents

**Стадии** (оба буфера на GPU):
- `Kernel_Multiply` → `MakeROCmDataFromEvents`
- `Download` → `MakeROCmDataFromEvents`

---

## C1 — `tests/heterodyne_benchmark.hpp` — OpenCL benchmark классы

**Новый файл**: `modules/heterodyne/tests/heterodyne_benchmark.hpp`

**namespace**: `test_heterodyne_opencl`

**Два класса**:

### Класс `HeterodyneDechirpBenchmark : GpuBenchmarkBase`

```cpp
class HeterodyneDechirpBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneDechirpBenchmark(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorOpenCL& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_Heterodyne"})
    : GpuBenchmarkBase(backend, "Heterodyne_Dechirp", cfg),
      proc_(proc), params_(params), rx_data_(rx_data), ref_data_(ref_data) {}

protected:
  void ExecuteKernel() override {
    proc_.Dechirp(rx_data_, ref_data_, params_);  // без prof_events → warmup
  }
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneOCLProfEvents events;
    proc_.Dechirp(rx_data_, ref_data_, params_, &events);
    for (auto& [name, ev] : events) RecordEvent(name, ev);
  }

private:
  drv_gpu_lib::HeterodyneProcessorOpenCL& proc_;
  drv_gpu_lib::HeterodyneParams           params_;
  std::vector<std::complex<float>>        rx_data_;
  std::vector<std::complex<float>>        ref_data_;
};
```

### Класс `HeterodyneCorrectBenchmark : GpuBenchmarkBase`

```cpp
class HeterodyneCorrectBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneCorrectBenchmark(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorOpenCL& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_Heterodyne"})
    : GpuBenchmarkBase(backend, "Heterodyne_Correct", cfg),
      proc_(proc), params_(params), dc_data_(dc_data), f_beat_hz_(f_beat_hz) {}

protected:
  void ExecuteKernel() override {
    proc_.Correct(dc_data_, f_beat_hz_, params_);
  }
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneOCLProfEvents events;
    proc_.Correct(dc_data_, f_beat_hz_, params_, &events);
    for (auto& [name, ev] : events) RecordEvent(name, ev);
  }

private:
  drv_gpu_lib::HeterodyneProcessorOpenCL& proc_;
  drv_gpu_lib::HeterodyneParams           params_;
  std::vector<std::complex<float>>        dc_data_;
  std::vector<float>                      f_beat_hz_;
};
```

**Includes** необходимые:
```cpp
#include "processors/heterodyne_processor_opencl.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"
#include <complex>
#include <vector>
```

---

## D1 — `tests/test_heterodyne_benchmark.hpp` — OpenCL test runner

**Новый файл**: `modules/heterodyne/tests/test_heterodyne_benchmark.hpp`

**namespace**: `test_heterodyne_benchmark`

```cpp
inline int run() {
  // OpenCL init с CL_QUEUE_PROFILING_ENABLE
  // Параметры:
  //   params.num_antennas = 5
  //   params.num_samples  = 4000
  //   params.sample_rate  = 12e6f
  //   params.f_start      = 0.0f
  //   params.f_end        = 1e6f
  //
  // Генерация тестовых данных:
  //   rx_data: num_antennas * num_samples complex (заполнить нулями или chirp-сигналом)
  //   ref_data: num_samples complex
  //   dc_data: num_antennas * num_samples complex
  //   f_beat_hz: {300e3f, 600e3f, 900e3f, 1200e3f, 1500e3f} (5 антенн)
  //
  // Создать proc(backend)
  // Запустить HeterodyneDechirpBenchmark → Run() → Report()
  // Запустить HeterodyneCorrectBenchmark → Run() → Report()
  // IsProfEnabled() проверка перед каждым Run()
  // Cleanup: backend.reset() + clReleaseCommandQueue + clReleaseContext
}
```

**Генерация тестовых данных для Dechirp**:
- rx_data: LFM сигнал с задержкой: `exp(j * 2*pi * (f_start*t + mu/2 * t^2))` для каждой антенны
- ref_data: опорный LFM без задержки
- Или просто: заполнить единицами (для бенчмарка данные не критичны)

**Для простоты**: заполнить `{1.0f, 0.0f}` (pure real = 1+0i).

---

## E1 — `tests/heterodyne_benchmark_rocm.hpp` — ROCm benchmark классы

**Новый файл**: `modules/heterodyne/tests/heterodyne_benchmark_rocm.hpp`

**namespace**: `test_heterodyne_rocm`

```cpp
#if ENABLE_ROCM

class HeterodyneDechirpBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
  // аналогично C1, но:
  // - proc_ типа HeterodyneProcessorROCm&
  // - ExecuteKernelTimed: HeterodyneROCmProfEvents events; proc_.Dechirp(..., &events);
  // - for (auto& [name, data] : events) RecordROCmEvent(name, data);
  // - output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"
};

class HeterodyneCorrectBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
  // аналогично для Correct()
};

#endif  // ENABLE_ROCM
```

---

## F1 — `tests/test_heterodyne_benchmark_rocm.hpp` — ROCm test runner

**Новый файл**: `modules/heterodyne/tests/test_heterodyne_benchmark_rocm.hpp`

```cpp
#if ENABLE_ROCM
namespace test_heterodyne_benchmark_rocm {
inline int run() {
  // Проверка наличия ROCm устройств (ROCmCore::GetAvailableDeviceCount())
  // ROCmBackend backend; backend.Initialize(0);
  // HeterodyneProcessorROCm proc(&backend);
  // HeterodyneDechirpBenchmarkROCm dechirp_bench(...) → Run() → Report()
  // HeterodyneCorrectBenchmarkROCm correct_bench(...) → Run() → Report()
}
}
#endif
```

---

## G1 — Обновить `tests/all_test.hpp`

**Файл**: `modules/heterodyne/tests/all_test.hpp`

Добавить includes:
```cpp
#include "test_heterodyne_benchmark.hpp"
#if ENABLE_ROCM
#include "test_heterodyne_benchmark_rocm.hpp"
#endif
```

Добавить в `heterodyne_all_test::run()`:
```cpp
// Heterodyne OpenCL Benchmark (GpuBenchmarkBase)
//  test_heterodyne_benchmark::run();    ← раскомментировать для запуска

// Heterodyne ROCm Benchmark
#if ENABLE_ROCM
//  test_heterodyne_benchmark_rocm::run();
#endif
```

---

## G2 — Обновить `tests/README.md`

Добавить таблицу с описанием бенчмарков:

```markdown
## Benchmark Tests (GpuBenchmarkBase)

| # | File | Class | Method | Стадии профилирования |
|---|------|-------|--------|----------------------|
| B1 | heterodyne_benchmark.hpp | HeterodyneDechirpBenchmark | Dechirp() | Upload_Rx, Upload_Ref, Kernel_Multiply, Download |
| B2 | heterodyne_benchmark.hpp | HeterodyneCorrectBenchmark | Correct() | Upload_DC, Upload_PhaseStep, Kernel_Correct, Download |
| B3 | heterodyne_benchmark_rocm.hpp | HeterodyneDechirpBenchmarkROCm | Dechirp() | Upload_Rx, Upload_Ref, Kernel_Multiply, Download |
| B4 | heterodyne_benchmark_rocm.hpp | HeterodyneCorrectBenchmarkROCm | Correct() | Upload_DC, Upload_PhaseStep, Kernel_Correct, Download |
```

---

## Итоговый чеклист

```
OpenCL Production
[x] A1 — heterodyne_processor_opencl: CollectOrRelease + Dechirp prof_events
[x] A2 — heterodyne_processor_opencl: Correct prof_events
[x] A3 — heterodyne_processor_opencl: DechirpFromGPU prof_events
[x] A4 — heterodyne_processor_opencl: DechirpWithGPURef prof_events

ROCm Production
[x] B1 — heterodyne_processor_rocm: MakeROCmDataFromEvents + Dechirp prof_events
[x] B2 — heterodyne_processor_rocm: Correct prof_events
[x] B3 — heterodyne_processor_rocm: DechirpFromGPU prof_events
[x] B4 — heterodyne_processor_rocm: DechirpWithGPURef prof_events

Benchmark Files
[x] C1 — tests/heterodyne_benchmark.hpp (OpenCL: Dechirp + Correct классы)
[x] D1 — tests/test_heterodyne_benchmark.hpp (OpenCL test runner)
[x] E1 — tests/heterodyne_benchmark_rocm.hpp (ROCm: Dechirp + Correct классы)
[x] F1 — tests/test_heterodyne_benchmark_rocm.hpp (ROCm test runner)

Integration
[x] G1 — tests/all_test.hpp (includes + закомментированные вызовы)
[x] G2 — tests/README.md (описание бенчмарков)
```

---

## Порядок реализации (рекомендуется)

1. **A1** (CollectOrRelease + Dechirp) — самый важный метод
2. **A2, A3, A4** — по той же схеме
3. **B1** (MakeROCmDataFromEvents + Dechirp ROCm)
4. **B2, B3, B4** — по той же схеме
5. **C1** — benchmark класс OpenCL (использует A1-A4)
6. **D1** — test runner OpenCL
7. **E1** — benchmark класс ROCm (использует B1-B4)
8. **F1** — test runner ROCm
9. **G1, G2** — интеграция и документация

---

## Важные замечания

### ⚠️ CL_QUEUE_PROFILING_ENABLE
В test runner (D1) команда очереди ОБЯЗАТЕЛЬНО создаётся с флагом:
```cpp
cl_command_queue queue = clCreateCommandQueue(
    context, device, CL_QUEUE_PROFILING_ENABLE, &err);
```
Иначе `clGetEventProfilingInfo` вернёт `CL_PROFILING_INFO_NOT_AVAILABLE`.

### ⚠️ Не трогать интерфейс IHeterodyneProcessor
`i_heterodyne_processor.hpp` — не изменяем. Используем паттерн делегирования
в конкретных классах.

### ⚠️ ROCm D2H — async, не sync
В `heterodyne_processor_rocm.cpp` используется `hipMemcpyDtoHAsync` (не sync).
Поэтому для Download тоже используем `MakeROCmDataFromEvents`, а не `MakeROCmDataFromClock`.
После всех hipEvent::Record идёт `hipStreamSynchronize(stream_)`, затем collect events.

### ⚠️ DechirpWithGPURef stub
В интерфейсе `IHeterodyneProcessor` метод `DechirpWithGPURef` не pure virtual
(бросает `runtime_error` по умолчанию). В `HeterodyneProcessorROCm` он реализован.
Паттерн делегирования тот же.

### Результаты профилирования
После реализации результаты будут в:
```
Results/Profiler/GPU_00_Heterodyne/
├── report.md
└── report.json
Results/Profiler/GPU_00_Heterodyne_ROCm/
├── report.md
└── report.json
```
