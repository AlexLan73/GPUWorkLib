---
name: check-profiling-implementation
description: Verifies that a GPU module implements profiling correctly according to GPU_Profiling_Mechanism.md. Use when auditing a module for profiling compliance, checking profiling setup, or validating GpuBenchmarkBase usage.
---

# Check Profiling Implementation

Проверяет соответствие реализации профилирования модуля документу [GPU_Profiling_Mechanism.md](../../Doc_Addition/GPU_Profiling_Mechanism.md).

## Trigger

Применять когда пользователь говорит: проверить профилирование модуля, аудит профилирования, check profiling, validate profiling, соответствует ли модуль GPU_Profiling_Mechanism.

## Быстрый старт

1. Запустить проверку:
   ```bash
   python scripts/check_profiling.py [module_name]   # один модуль
   python scripts/check_profiling.py --all           # все модули
   ./scripts/check_profiling.sh [module_name]        # или через shell
   ```
2. Прочитать вывод и исправить отмеченные проблемы (❌ FAIL, ⚠ warning)
3. При необходимости — ручной аудит по чеклисту ниже

## Чеклист (GPU_Profiling_Mechanism.md §6)

### Production-класс

- [ ] Добавлен параметр `prof_events*` к публичным методам (default = `nullptr`)
- [ ] **OpenCL**: используется `CollectOrRelease(ev, "Name", prof_events)` вместо прямого `clReleaseEvent`
- [ ] **ROCm**: обёрнуты `hipEventCreate/Record` в `if (prof_events)` блоки
- [ ] При `prof_events = nullptr` — **ноль дополнительного overhead**

### Benchmark-класс (: GpuBenchmarkBase)

- [ ] Наследует от `drv_gpu_lib::GpuBenchmarkBase`
- [ ] Реализован `ExecuteKernel()` — вызов БЕЗ prof_events (warmup)
- [ ] Реализован `ExecuteKernelTimed()` — вызов С prof_events + цикл `RecordEvent()` / `RecordROCmEvent()`
- [ ] Задан `output_dir = "Results/Profiler/GPU_NN_ModuleName"`

### Test runner

- [ ] **OpenCL**: очередь создана с `CL_QUEUE_PROFILING_ENABLE`
- [ ] Перед `Run()` проверяется `bench.IsProfEnabled()`
- [ ] ROCm-код обёрнут в `#if ENABLE_ROCM`

### Запрещено

- [ ] **ЗАПРЕЩЕНО**: `GetStats()` + цикл + `con.Print` / `std::cout` для вывода профилирования
- [ ] Вывод **ТОЛЬКО** через `bench.Report()` → `GPUProfiler::PrintReport()`

## Эталонная реализация

Референс — `modules/fft_processor/` (OpenCL и ROCm). Для сравнения с другими модулями:

```
modules/<module>/
  include/   ← prof_events* / ROCmProfEvents* в сигнатурах
  src/       ← CollectOrRelease (OpenCL) / MakeROCmDataFrom* (ROCm)
  tests/     ← *benchmark*.hpp (GpuBenchmarkBase) + test_*benchmark*.hpp (runner)
```

Кастомные реализации без GpuBenchmarkBase (`fm_correlator`, `vector_algebra`) —
проверяются по ключевым принципам: hipEvent + profiler.Record + PrintReport.

## Отчёт о несоответствии

Скрипт сохраняет отчёт при наличии ошибок:
```
Results/profiling_audit/YYYY-MM-DD_HH-MM-SS.txt
```
Запуск: `python scripts/check_profiling.py --all`

## Частые ошибки (§7 документа)

| Ошибка | Решение |
|--------|---------|
| `cl_event` возвращает 0 | Очередь без `CL_QUEUE_PROFILING_ENABLE` |
| `hipEventElapsedTime` для sync D2H | Использовать `MakeROCmDataFromClock()` для sync операций |
| is_prof=false → no-op | Проверить configGPU.json, backend device index |
| N ≠ n_runs в отчёте | `Report()` должен вызывать `WaitEmpty()` перед PrintReport (уже в GpuBenchmarkBase) |

## Workflow аудита

1. Вызвать `python scripts/check_profiling.py <module>` — получить отчёт
2. Открыть модуль: `modules/<module>/`
3. Пройти по чеклисту вручную для edge-cases
4. Исправить найденные нарушения
5. Перезапустить скрипт для проверки
