# 🚀 ИНСТРУКЦИЯ ДЛЯ СЛЕДУЮЩЕЙ СЕССИИ

> **Дата**: продолжение 2026-03-14
> **Ветка**: `main`
> **Задача**: Ref03 — написать код единой архитектуры GPU-операций

---

## ЧТО ЧИТАТЬ ПЕРВЫМ

1. `Doc_Addition/PLAN/Ref03_Unified_Architecture.md` — **ГЛАВНЫЙ ДОКУМЕНТ** (6-слойная модель, все классы, диаграммы)
2. `MemoryBank/tasks/TASK_Ref03_unified_architecture.md` — **ВСЕ ТАСКИ** с тонкими местами
3. `MemoryBank/sessions/2026-03-14.md` — что уже сделано

---

## ЧТО УЖЕ НАПИСАНО (код)

### ✅ Готово:
- `DrvGPU/services/buffer_set.hpp` — BufferSet<N> template (Layer 4)

### ❌ Нужно написать (порядок):

**Ref03-A: Foundation (DrvGPU)**
1. `DrvGPU/interface/i_gpu_operation.hpp` — IGpuOperation interface (Layer 2)
2. `DrvGPU/services/gpu_kernel_op.hpp` — GpuKernelOp base class (Layer 3)
3. `DrvGPU/interface/gpu_context.hpp` — GpuContext per-module (Layer 1)

**Ref03-B: Statistics (6 Op-классов + thin Facade)**
4. `modules/statistics/include/operations/mean_reduction_op.hpp`
5. `modules/statistics/include/operations/welford_fused_op.hpp`
6. `modules/statistics/include/operations/welford_float_op.hpp`
7. `modules/statistics/include/operations/median_radix_sort_op.hpp`
8. `modules/statistics/include/operations/median_histogram_op.hpp`
9. `modules/statistics/include/operations/median_histogram_complex_op.hpp`
10. ПЕРЕПИСАТЬ `modules/statistics/include/statistics_processor.hpp` → thin Facade
11. ПЕРЕПИСАТЬ `modules/statistics/src/statistics_processor.cpp` → delegates to Ops

**Ref03-C: Strategies Pipeline**
12. `modules/strategies/include/i_pipeline_step.hpp` — IPipelineStep interface
13. `modules/strategies/include/pipeline_context.hpp` — PipelineContext struct
14. `modules/strategies/include/pipeline.hpp` — Pipeline class (execute + profiling)
15. `modules/strategies/include/pipeline_builder.hpp` — PipelineBuilder fluent API
16. `modules/strategies/include/steps/gemm_step.hpp`
17. `modules/strategies/include/steps/window_fft_step.hpp`
18. `modules/strategies/include/steps/debug_stats_step.hpp`
19. `modules/strategies/include/steps/one_max_step.hpp`
20. `modules/strategies/include/steps/all_maxima_step.hpp`
21. `modules/strategies/include/steps/minmax_step.hpp`
22. ПЕРЕПИСАТЬ `modules/strategies/src/antenna_processor_v1.cpp` → uses Pipeline

---

## ТОНКИЕ МЕСТА (запомнить!)

### BufferSet<N>:
- Specialization для N=0 уже есть (welford_fused не нужны приватные буферы)
- hipMalloc обёрнут в `#if ENABLE_ROCM` — на Windows (nvidia ветка) не компилируется

### GpuContext:
- CompileModule() — один hiprtc вызов на ВСЕ kernels модуля (не по одному!)
- Kernel sources: `kernels::GetStatisticsKernelSource()` — уже есть, не менять
- WARP_SIZE: `arch_name.find("gfx9") == 0` → 64, иначе → 32
- KernelCacheService: disk HSACO cache — как в текущем CompileKernels()

### Statistics Op-классы:
- Каждая Op читает/пишет shared buffers через GpuContext (kInput, kMagnitudes, kResult, kMediansCompact)
- MedianRadixSortOp: вызывает `gpu_sort::ExecuteSort()` из `statistics_sort_gpu.hip` — extern C, не менять .hip файл
- MedianRadixSortOp: `QuerySortTempSize()` при первом Require() → кешировать sort_temp_size_
- Offsets buffer: upload segment offsets при аллокации (hipMemcpyAsync)

### Strategies Pipeline:
- 7 streams остаются (stream_main_ + stream_debug1-3 + stream_bench3a-c)
- hipBLAS handle → в PipelineContext
- hipFFT plan → в WindowFftStep (с LRU-2 cache как сейчас)
- Parallel steps: OneMax + AllMaxima + MinMax на разных streams
- DebugStatsStep: использует StatisticsProcessor (другой модуль!) → передаётся через PipelineContext
- Event synchronization: event_gemm_done_, event_fft_done_ → в PipelineContext

### Python bindings:
- НЕ ТРОГАТЬ py_statistics.hpp — Facade API не меняется
- НЕ ТРОГАТЬ Python тесты — отдельная задача

---

## КАК ПРОВЕРИТЬ

1. Файлы должны компилироваться (но без GPU — только синтаксис)
2. На GPU (понедельник): `test_statistics_rocm::run()` → 11/11 passed
3. На GPU: `strategies_all_test::run()` → все тесты passed
4. Benchmark regression: timing не должно деградировать

---

## ТАКЖЕ В ОЧЕРЕДИ (не Ref03)

- Histogram median ждёт тестирования на ROCm (commit dc11bc6)
- В nvidia ветке stash с .gitignore (git stash pop при возврате)
