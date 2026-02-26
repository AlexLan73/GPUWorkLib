# Task 10 / PLAN Vector Algebra Cholesky — исправления и вопросы

> **Дата**: 2026-02-25
> **Назначение**: Файл для внесения исправлений и вопросов по плану и реализации

---

## Мои исправления

<!-- Внеси сюда свои правки -->



---

## Мои вопросы

<!-- Задай здесь свои вопросы -->

1. **Создать дополнительный таск** — строго: всё делается внутри kernel, по возможности в одном kernel. Референс: `/home/alex/C++/LCH-Farrow01/Matrix`
2. **LCH-Farrow01 сделано правильно?** — проверено
3. **Где значения профилирования?** — должны быть

---

## Ответы (Кодо)

### LCH-Farrow01 — анализ

В [matrix_invert.cpp](/home/alex/C++/LCH-Farrow01/Matrix/matrix_invert.cpp):

- **POTRF + POTRI** — вызовы rocSOLVER (библиотека), выполняются на GPU. Это не custom kernel.
- **Symmetrize** — делается на CPU после `hipMemcpy` (download). Референс всегда возвращает результат на хост.
- **Профилирование** — `GPUTimer` с `hipEventRecord`/`hipEventElapsedTime` (корректно, GPU time).

**Вывод**: Референс не использует один общий kernel. POTRF/POTRI — это библиотечные алгоритмы LAPACK, их нельзя «собрать в один kernel». Реалистичная цель: **всё на GPU** (POTRF + POTRI + Symmetrize kernel) без лишнего download/upload.

### Профилирование в vector_algebra

В [test_cholesky_inverter_rocm.hpp](modules/vector_algebra/tests/test_cholesky_inverter_rocm.hpp) (Test 6):

- Используется `std::chrono::high_resolution_clock` — это CPU wall-clock, не GPU time.
- `profiler.Record("Cholesky", "POTRF_POTRI_341x341", elapsed_ms)` — неверная сигнатура: нет `gpu_id`, `elapsed_ms` не является `OpenCLProfilingData`/`ROCmProfilingData`.

**Правильно** (по образцу LCH-Farrow01 и DrvGPU):

- Использовать `hipEventRecord` / `hipEventElapsedTime` для измерения GPU времени.
- Вызывать `profiler.Record(gpu_id, "Cholesky", "POTRF_POTRI_341x341", ROCmProfilingData)` с заполненными полями `start_ns`, `end_ns` и т.п.
- Или использовать `MakeOpenCLFromDurationMs(elapsed_ms)` для ROCm (если ROCmProfilingData требует hipEvent — см. [profiling_types.hpp](../../DrvGPU/services/profiling_types.hpp)).

---

## Задание (от Alex)

1. **Повторить как в LCH-Farrow01** — использовать обвязки из DrvGPU, не городить отсебятину.
2. **Профилировать** — как в [CLAUDE.md](../../CLAUDE.md) и [Examples/GPUProfiler_SetGPUInfo.md](../../Examples/GPUProfiler_SetGPUInfo.md): GPUProfiler, `SetGPUInfo` перед `Start()`, вывод **только** через `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`.
3. **Вывести значения в файл** — `ExportMarkdown("Results/Profiler/cholesky_*.md")`, `ExportJSON("Results/Profiler/cholesky_*.json")`.
4. **Повторить как в примере** — для одной матрицы и для batched матриц (как в [matrix_invert.cpp](/home/alex/C++/LCH-Farrow01/Matrix/matrix_invert.cpp) и matrix_invert_advanced.cpp).

---

## Контекст (для справки)

- **Plan**: [PLAN_Vector_Algebra_Cholesky.md](PLAN_Vector_Algebra_Cholesky.md)
- **Task**: [Task_10_VectorAlgebraCholesky.md](../tasks/Task_10_VectorAlgebraCholesky.md)
- **Task_11**: [Task_11_Cholesky_GPU_Only_Profiling.md](../tasks/Task_11_Cholesky_GPU_Only_Profiling.md)
- **Реализация**: `modules/vector_algebra/`
- **DrvGPU**: `DrvGPU/` — backend, GPUProfiler, ConsoleOutput
