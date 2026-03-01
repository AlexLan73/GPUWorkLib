# Профилирование GPU — Единый механизм

> **Статус**: ✅ Реализовано (PROF-001 FFTProcessor)
> **Дата**: 2026-03-01
> **Автор**: Кодо + Alex
> **Первый модуль**: `modules/fft_processor` (OpenCL) — задача: `tasks/TASK_profiling_fftprocessor.md`

---

## 1. Проблема

Каждый модуль делает профилирование по-своему. Пример из `vector_algebra/tests/test_stage_profiling.hpp`:

```cpp
// ❌ Как сейчас — каждый раз вручную
constexpr int kRuns   = 10;   // мало! нужно >= 20
constexpr int kWarmup = 3;    // мало!

for (int w = 0; w < kWarmup; ++w)
    RunKernel(...);           // warmup вручную

StageTiming avg = {};
for (int r = 0; r < kRuns; ++r) {
    auto t = RunKernel(...);
    avg.total_ms += t.total_ms;
    avg.potrf_ms += t.potrf_ms;
    // ... 7 полей вручную ...
}
avg.total_ms /= kRuns;

PrintStageAvg("...", avg);    // ❌ con.Print вместо GPUProfiler
                              // ❌ Record() не вызывается вообще
```

**Последствия:** нет единого формата отчёта, нет JSON/MD экспорта, нет GPU info в шапке,
сравнивать отчёты разных модулей нельзя.

---

## 2. Требования

| # | Требование | Значение по умолчанию |
|---|-----------|----------------------|
| R1 | Минимум **20 измерений** для статистики | `n_runs = 20` |
| R2 | **Прогрев GPU** до старта замеров | `n_warmup = 5` |
| R3 | **Автовыбор backend** в базовом классе | OpenCL или ROCm — один раз |
| R4 | Базовый класс в `DrvGPU/services/` | Переиспользуется всеми модулями |
| R5 | Работает **только если `is_prof=true`** | Из `configGPU.json` per-GPU |
| R6 | Вывод **только через GPUProfiler** | `PrintReport()` / `Export*()` |
| R7 | Production-класс модуля — **чистый** | Ноль кода профилирования внутри |

---

## 3. Архитектура — Template Method Pattern

### 3.1 Общая схема

```
DrvGPU/services/gpu_benchmark_base.hpp
═══════════════════════════════════════════════════════════════

  GpuBenchmarkBase  (абстрактный)
  ┌──────────────────────────────────────────────────────┐
  │  Поля:                                               │
  │  · backend_     — IBackend* (все операции GPU)       │
  │  · gpu_id_      — ID текущего GPU (из 10)            │
  │  · module_name_ — имя модуля для GPUProfiler         │
  │  · cfg_         — {n_warmup=5, n_runs=20, output_dir}│
  │  · is_prof_     — из configGPU.json                  │
  │                                                      │
  │  Публичный API (реализован здесь — не трогать):      │
  │  + Run()    → Init → Warmup → Reset → Measure       │
  │  + Report() → profiler.PrintReport + Export*         │
  │  + IsProfEnabled() → is_prof_ flag                   │
  │                                                      │
  │  Protected (переопределяет каждый модуль):           │
  │  # ExecuteKernel()      — warmup БЕЗ timing         │
  │  # ExecuteKernelTimed() — measure С timing           │
  │                                                      │
  │  Helper (для наследников):                           │
  │  # RecordEvent(name, cl_event)                       │
  │    → clWaitForEvents + FillOpenCLProfilingData       │
  │    → profiler.Record() + clReleaseEvent              │
  │                                                      │
  │  Приватные (реализованы здесь):                      │
  │  - InitProfiler() → SetGPUInfo + profiler.Start      │
  │  - Warmup: n_warmup × ExecuteKernel → profiler.Reset │
  │  - Measure: n_runs × ExecuteKernelTimed              │
  └──────────────────────────────────────────────────────┘
            ▲ наследует
            │
  modules/fft_processor/tests/fft_processor_benchmark.hpp
  ═══════════════════════════════════════════════════════

  FFTProcessorBenchmark : public GpuBenchmarkBase
  ┌──────────────────────────────────────────────────────┐
  │  · proc_       — ссылка на FFTProcessor              │
  │  · params_     — параметры FFT (фиксированы)         │
  │  · input_data_ — входные данные (фиксированы)        │
  │                                                      │
  │  # ExecuteKernel() override                          │
  │      → proc_.ProcessComplex(data, params)            │
  │        (без prof_events — warmup, ноль overhead)     │
  │                                                      │
  │  # ExecuteKernelTimed() override                     │
  │      → proc_.ProcessComplex(data, params, &events)   │
  │      → for each (name, ev) : RecordEvent(name, ev)   │
  │        (Upload, FFT, Download → GPUProfiler)         │
  └──────────────────────────────────────────────────────┘
            │ composition (ссылка)
            ▼
  modules/fft_processor/include/fft_processor.hpp
  ════════════════════════════════════════════════

  FFTProcessor  ← ЧИСТЫЙ, не знает о GpuBenchmarkBase
  ┌──────────────────────────────────────────────────────┐
  │  ProcessComplex(data, params)                        │
  │  ProcessComplex(data, params, &prof_events)          │
  │  ProcessMagPhase(data, params)                       │
  │  ProcessMagPhase(data, params, &prof_events)         │
  │                                                      │
  │  prof_events = nullptr → ноль overhead (production)  │
  │  prof_events != nullptr → собирает cl_event для      │
  │                           внешнего профилирования    │
  └──────────────────────────────────────────────────────┘
```

### 3.2 Поток данных — как работает

```
bench.Run()
  ├─ InitProfiler()  → SetGPUInfo + profiler.Start()
  ├─ Warmup: 5 × ExecuteKernel()
  │    └─ ProcessComplex(data, params)  — без events, ноль overhead
  ├─ profiler.Reset()  → очистить warmup данные
  └─ Measure: 20 × ExecuteKernelTimed()
       └─ ProcessComplex(data, params, &events)
          └─ events: {"Upload", ev1}, {"FFT", ev2}, {"Download", ev3}
             └─ RecordEvent(name, ev) для каждого:
                → clWaitForEvents + FillOpenCLProfilingData
                → profiler.Record(gpu_id, "FFTProcessor", name, pdata)
                → clReleaseEvent
          GPUProfiler копит все вызовы → min/max/avg автоматически

bench.Report()
  ├─ profiler.PrintReport()     ← вся таблица стандартная
  ├─ profiler.ExportJSON()      ← JSON
  ├─ profiler.ExportMarkdown()  ← Markdown
  └─ profiler.Stop()
```

### 3.3 Production vs Benchmark — разделение ответственности

```
Production (чистый код):          Benchmark (профилирование):
─────────────────────────         ──────────────────────────────
FFTProcessor                      FFTProcessorBenchmark
  · ProcessComplex(data, params)    · ExecuteKernel()  → warmup
  · ProcessMagPhase(data, params)   · ExecuteKernelTimed() → measure
  · Никакого профилирования         · RecordEvent → GPUProfiler
  · prof_events = nullptr (default) · prof_events = &vector
  · cl_events освобождаются внутри  · cl_events собираются → Record
```

### 3.4 Использование в тест-файле

```cpp
// Весь бенчмарк — ~10 строк

FFTProcessor          proc(backend);               // чистый рабочий класс
FFTProcessorBenchmark bench(backend, proc, params,  // профилирующий наследник
    input_data,
    {.n_warmup = 5, .n_runs = 20,
     .output_dir = "Results/Profiler/GPU_00_FFT"});

bench.Run();     // warmup(5) + measure(20) → GPUProfiler → всё автоматически
bench.Report();  // PrintReport + ExportJSON + ExportMarkdown
```

### 3.5 Статистика (из 20 замеров × количество событий)

| Поле | Описание |
|------|----------|
| `min_ms` | Минимальное время |
| `max_ms` | Максимальное время |
| `avg_ms` | Среднее время |
| `queue_delay_avg` | Ожидание в очереди хоста |
| `submit_delay_avg` | Задержка до старта на GPU |
| `exec_avg` | Чистое время выполнения кернела |

> Дисперсию (std dev) можно добавить позже — одно поле `sum_sq_ms` в `DetailedTimingStats`.

---

## 4. Размещение файлов

```
DrvGPU/
└── services/
    ├── gpu_profiler.hpp           ← существующий (не трогаем)
    └── gpu_benchmark_base.hpp     ← базовый класс (Template Method)

modules/fft_processor/
├── include/
│   └── fft_processor.hpp          ← ЧИСТЫЙ + опциональный prof_events
├── src/
│   └── fft_processor.cpp          ← ЧИСТЫЙ (без RecordProfilingEvent)
└── tests/
    ├── fft_processor_benchmark.hpp  ← наследник GpuBenchmarkBase
    ├── test_fft_benchmark.hpp       ← точка входа бенчмарка
    ├── test_fft_processor.hpp       ← существующий (не трогаем)
    └── all_test.hpp                 ← вызов бенчмарка
```

---

## 5. configGPU.json — `is_prof` уже есть

```json
{
  "gpus": [
    { "id": 0, "is_prof": true, ... },
    { "id": 1, "is_prof": true, ... }
  ]
}
```

`GpuBenchmarkBase` читает флаг в конструкторе через `GPUConfig::IsProfilingEnabled()`.
При `is_prof = false` → `Run()` и `Report()` — no-op, нулевой overhead в production.

---

## 6. Правила

```
✅ ПРАВИЛЬНО                          ❌ ЗАПРЕЩЕНО
──────────────────────────────────    ──────────────────────────────────────
bench.Run()   + bench.Report()        ручной for-loop warmup/накопление
profiler.PrintReport()                con.Print() для вывода статистики
profiler.ExportJSON / ExportMarkdown  GetStats() + цикл + cout / std::cout
ExecuteKernelTimed() override         std::chrono для GPU timing
n_runs >= 20                          n_runs < 20 (мало для статистики)
код профилирования только в tests/    профилирование в prod-классе модуля
prof_events для сбора cl_event        RecordProfilingEvent внутри модуля
```

---

## 7. Дорожная карта

| Задача | Модуль | Backend | Статус |
|--------|--------|---------|--------|
| PROF-001 | `fft_processor` | OpenCL | ✅ Реализовано |
| PROF-002 | `statistics` | ROCm | ⏳ После PROF-001 |
| PROF-003 | `filters` | OpenCL | ⏳ После PROF-001 |
| PROF-004 | `heterodyne` | OpenCL | ⏳ После PROF-001 |
| PROF-005 | `Doc/Profiling_Migration_Guide.md` | — | ⏳ После PROF-002 |

---

*Обновлено: 2026-03-01 | Кодо + Alex*
*Задача PROF-001: `tasks/TASK_profiling_fftprocessor.md`*
