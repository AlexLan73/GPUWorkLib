# TASK: Профилирование FFTProcessor — первая реализация единого механизма

> **ID**: PROF-001
> **Приоритет**: Средний
> **Статус**: ✅ РЕАЛИЗОВАНО
> **Backend**: OpenCL (NVIDIA)
> **Модуль**: `modules/fft_processor`
> **Создан**: 2026-03-01
> **Спека**: `MemoryBank/specs/Profil_GPU.md`

---

## Цель

Реализовать единый механизм профилирования (Template Method Pattern) на примере
`FFTProcessor`. Это **пилотная реализация** — после проверки перенести паттерн
на все остальные модули.

**Результат:** вместо ручного накопления → стандартный `FFTProcessorBenchmark`
который прогревает GPU, собирает >= 20 замеров и отдаёт в `GPUProfiler`.

---

## Контекст

### Что реализовано

- `GpuBenchmarkBase` — базовый класс Template Method в `DrvGPU/services/`
- `FFTProcessorBenchmark` — наследник для FFTProcessor в `tests/`
- `FFTProcessor` — ЧИСТЫЙ production-код (ноль RecordProfilingEvent внутри)
- Опциональный `prof_events` параметр для сбора cl_event

### Ключевой принцип: Production = Чистый

```
FFTProcessor::ProcessComplex(data, params)              ← production (ноль overhead)
FFTProcessor::ProcessComplex(data, params, &prof_events) ← benchmark (собирает cl_event)
```

---

## Шаги реализации

### Шаг 1 — GpuBenchmarkBase ✅

**Файл:** `DrvGPU/services/gpu_benchmark_base.hpp`

Базовый класс с двумя виртуальными методами:

```cpp
class GpuBenchmarkBase {
public:
  struct Config {
    int         n_warmup   = 5;
    int         n_runs     = 20;
    std::string output_dir = "Results/Profiler";
  };

  void Run();      // InitProfiler → Warmup → Reset → Measure
  void Report();   // profiler.PrintReport + ExportJSON + ExportMarkdown + Stop
  bool IsProfEnabled() const;

protected:
  virtual void ExecuteKernel() = 0;       // warmup — без timing
  virtual void ExecuteKernelTimed() = 0;  // measure — с timing

  // Helper: wait + extract + Record + release
  void RecordEvent(const char* event_name, cl_event ev);

#ifdef ENABLE_ROCM
  void RecordROCmEvent(const char* event_name, const ROCmProfilingData& data);
#endif

  IBackend* backend_ = nullptr;
  int       gpu_id_  = 0;

private:
  void InitProfiler();  // SetGPUInfo + profiler.Start()
  std::string module_name_;
  Config      cfg_;
  bool        is_prof_ = false;
};
```

**Поток Run():**
1. `InitProfiler()` — `SetGPUInfo` (GPU name, driver) + `profiler.Start()`
2. Warmup — `n_warmup` раз `ExecuteKernel()` (без timing)
3. `profiler.Reset()` — очистить warmup данные
4. Measure — `n_runs` раз `ExecuteKernelTimed()` (с timing → GPUProfiler)

---

### Шаг 2 — Опциональный prof_events в FFTProcessor ✅

**Файл:** `modules/fft_processor/include/fft_processor.hpp`

Добавлен опциональный параметр (обратно совместимо — `nullptr` по умолчанию):

```cpp
// Production — ноль overhead:
proc.ProcessComplex(data, params);

// Benchmark — собирает cl_event:
std::vector<std::pair<const char*, cl_event>> events;
proc.ProcessComplex(data, params, &events);
// events = {{"Upload", ev1}, {"FFT", ev2}, {"Download", ev3}}
```

**В реализации (.cpp):** `CollectOrRelease(ev, name, prof_events)` —
если `prof_events != nullptr` → сохраняет событие, иначе → `clReleaseEvent`.

**Удалены:** все вызовы `RecordProfilingEvent()` из `fft_processor.cpp`.
Production-код полностью чистый.

---

### Шаг 3 — FFTProcessorBenchmark ✅

**Файл:** `modules/fft_processor/tests/fft_processor_benchmark.hpp`

```cpp
class FFTProcessorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
protected:
  // Warmup — без timing
  void ExecuteKernel() override {
    proc_.ProcessComplex(input_data_, params_);  // prof_events = nullptr
  }

  // Measure — с timing → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    std::vector<std::pair<const char*, cl_event>> events;
    proc_.ProcessComplex(input_data_, params_, &events);
    for (auto& [name, ev] : events) {
      RecordEvent(name, ev);  // → GPUProfiler
    }
  }
};
```

---

### Шаг 4 — Тест-файл бенчмарка ✅

**Файл:** `modules/fft_processor/tests/test_fft_benchmark.hpp`

Создаёт OpenCL context с `CL_QUEUE_PROFILING_ENABLE`, инициализирует backend,
генерирует тестовые данные (64 луча × 1024 точки), запускает `bench.Run()` + `bench.Report()`.

---

### Шаг 5 — Подключение в all_test.hpp ✅

**Файл:** `modules/fft_processor/tests/all_test.hpp`

```cpp
#include "test_fft_benchmark.hpp"
// test_fft_benchmark::run();  ← раскомментировать для запуска
```

---

### Шаг 6 — Проверка результата

**Поток данных при вызове:**

```
bench.Run()
  ├─ InitProfiler()     SetGPUInfo + profiler.Start()
  ├─ Warmup: 5×         ExecuteKernel() — без timing, GPU прогрет
  ├─ profiler.Reset()   Очистить warmup данные
  └─ Measure: 20×       ExecuteKernelTimed()
       └─ ProcessComplex(data, params, &events)
          └─ events: Upload, FFT, Download
             └─ RecordEvent → profiler.Record()

bench.Report()
  ├─ profiler.PrintReport()              ← стандартный формат GPUProfiler
  ├─ profiler.ExportJSON(output_dir)     ← JSON файл
  ├─ profiler.ExportMarkdown(output_dir) ← Markdown файл
  └─ profiler.Stop()
```

> ⚠️ `bench.Report()` **ничего своего не печатает**.
> Весь вывод — только через GPUProfiler.

**Ожидаемые файлы в `Results/Profiler/GPU_00_FFT/`:**
```
FFTProcessor_benchmark_YYYY-MM-DD_HH-MM-SS.json
FFTProcessor_benchmark_YYYY-MM-DD_HH-MM-SS.md
```

---

## Критерии завершения (Definition of Done)

- [x] `DrvGPU/services/gpu_benchmark_base.hpp` создан и компилируется
- [x] `FFTProcessor` — чистый (нет RecordProfilingEvent внутри)
- [x] `FFTProcessor::ProcessComplex()` принимает `prof_events = nullptr` (обратно совместимо)
- [x] `FFTProcessorBenchmark` создан, наследует `GpuBenchmarkBase`
- [x] `test_fft_benchmark.hpp` создан
- [x] `bench.Run()` — ровно 5 warmup + 20 measure запусков
- [x] `bench.Report()` вызывает только `profiler.PrintReport()` + `ExportJSON()` + `ExportMarkdown()`
- [ ] JSON и MD файлы появляются в `Results/Profiler/GPU_00_FFT/` *(требует запуска)*
- [ ] GPU name и driver info — в шапке отчёта GPUProfiler (не «Unknown») *(требует запуска)*
- [x] Нет `std::cout`, `con.Print()`, ручного цикла накопления в бенчмарке
- [x] Существующие тесты (`test_fft_processor.hpp`) компилируются без изменений

---

## После завершения → следующие модули

| Модуль | Файл бенчмарка |
|--------|---------------|
| Statistics | `modules/statistics/tests/statistics_benchmark.hpp` |
| Filters | `modules/filters/tests/filters_benchmark.hpp` |
| Heterodyne | `modules/heterodyne/tests/heterodyne_benchmark.hpp` |
| SignalGenerators | `modules/signal_generators/tests/generators_benchmark.hpp` |

Создать `Doc/Profiling_Migration_Guide.md` — гайд по переходу для каждого модуля.

---

*Создан: 2026-03-01 | Кодо*
*Спека: `MemoryBank/specs/Profil_GPU.md`*
