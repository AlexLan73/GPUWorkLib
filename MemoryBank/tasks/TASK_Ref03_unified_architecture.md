# TASK: Ref03 — Единая архитектура GPU-операций

> **Статус**: BACKLOG → IN_PROGRESS (понедельник 2026-03-17)
> **План**: `Doc_Addition/PLAN/Ref03_Unified_Architecture.md`
> **Зависит от**: histogram median протестирован на ROCm (dc11bc6)

---

## Ref03-A: Foundation (DrvGPU base classes)

### A1: BufferSet<N> template
**Файл**: `DrvGPU/services/buffer_set.hpp`
**Что**: Compile-time fixed array GPU буферов с lazy alloc + reuse
**Тонкие места**:
- `hipMalloc` может вернуть `hipErrorOutOfMemory` — нужен throw с понятным сообщением
- Move semantics: `memcpy` + `memset(0)` — НЕ `std::swap` (entries_ — stack array)
- Деструктор: `ReleaseAll()` — RAII, вызывает `hipFree` для каждого ненулевого ptr
- OpenCL вариант: `clCreateBuffer`/`clReleaseMemObject` — нужен `#if ENABLE_ROCM` или backend-agnostic через IBackend::Allocate/Free

### A2: IGpuOperation interface
**Файл**: `DrvGPU/interface/i_gpu_operation.hpp`
**Что**: Минимальный контракт: Name, Initialize, IsReady, Release
**Тонкие места**:
- Чисто виртуальный деструктор `= default` (не `= 0`)
- Не шаблонизировать — конкретные типы result/params в Concrete Ops

### A3: GpuKernelOp base
**Файл**: `DrvGPU/services/gpu_kernel_op.hpp`
**Что**: Base class с доступом к compiled kernels через GpuContext
**Тонкие места**:
- НЕ компилирует kernels сам — берёт из GpuContext (Facade компилирует один раз)
- `kernel("name")` — простой lookup, throw если не найден
- `stream()` — проксирует из ctx_

### A4: GpuContext
**Файл**: `DrvGPU/interface/gpu_context.hpp`
**Что**: Per-module shared state (backend, stream, compiled module, shared buffers)
**Тонкие места**:
- `CompileModule(source, names)` — одна hiprtc компиляция на ВСЕ kernels модуля (как сейчас)
- Disk cache через KernelCacheService (как сейчас)
- SharedBuf enum — фиксированный набор (kInput, kMagnitudes, kResult, kMediansCompact)
- НЕ template — конкретный enum. Каждый модуль может расширить enum в своём header
- Thread safety: per-module instance, операции последовательны ВНУТРИ модуля
- **WARP_SIZE determination**: arch_name.find("gfx9") → 64, else → 32 (как сейчас в statistics)

### A5: Unit tests for BufferSet
**Файл**: `DrvGPU/tests/test_buffer_set.hpp`
**Что**: alloc/reuse/release/move semantics tests

---

## Ref03-B: Statistics refactoring

### B1: MeanReductionOp
**Файл**: `modules/statistics/include/operations/mean_reduction_op.hpp` + `.cpp`
**Из**: `ExecuteMeanReduction()` из statistics_processor.cpp (строки 757-818)
**Kernels**: `mean_reduce_phase1`, `mean_reduce_final`
**Буферы**: `BufferSet<1>` — reduce_buf
**Тонкие места**:
- `blocks_per_beam` = `(n_point + kDoubleLoadElements - 1) / kDoubleLoadElements`
- `final_block` = power of 2, capped at kBlockSize
- Shared memory = 0 (static LDS в kernel)
- Результат пишется в ctx_->RequireShared(kResult)

### B2: WelfordFusedOp
**Файл**: `modules/statistics/include/operations/welford_fused_op.hpp` + `.cpp`
**Из**: `ExecuteWelfordFusedKernel()` (строки 851-876)
**Kernels**: `welford_fused`
**Буферы**: `BufferSet<0>` — нет приватных буферов!
**Тонкие места**:
- Shared memory = `4 * (kBlockSize + 1) * sizeof(float)` — передаётся при launch
- Один блок на beam → grid = (beam_count, 1, 1)
- Читает ctx_->RequireShared(kInput), пишет в ctx_->RequireShared(kResult)

### B3: WelfordFloatOp
**Файл**: `modules/statistics/include/operations/welford_float_op.hpp` + `.cpp`
**Из**: `ExecuteWelfordFloatKernel()` (строки 938-963)
**Kernels**: `welford_float`
**Буферы**: `BufferSet<0>`
**Тонкие места**:
- Shared memory = `2 * (kBlockSize + 1) * sizeof(float)` (только 2 массива: sum_mag, sum_sq)
- Читает ctx_->RequireShared(kMagnitudes)

### B4: MedianRadixSortOp
**Файл**: `modules/statistics/include/operations/median_radix_sort_op.hpp` + `.cpp`
**Из**: `ExecuteMagnitudesKernel() + ExecuteMedianSort() + ExecuteExtractMediansKernel()`
**Kernels**: `compute_magnitudes`, `extract_medians`
**Буферы**: `BufferSet<3>` — sort_buf, sort_temp_buf, offsets_buf
**Тонкие места**:
- `statistics_sort_gpu.hip` остаётся как есть (extern C function, compiled hipcc)
- `QuerySortTempSize()` вызывается при AllocateBuffers — результат кешируется
- offsets_buf: `hipMemcpyAsync` upload segment offsets при аллокации
- rocPRIM: `segmented_radix_sort_keys` — все beams параллельно

### B5: MedianHistogramOp
**Файл**: `modules/statistics/include/operations/median_histogram_op.hpp` + `.cpp`
**Из**: `ExecuteHistogramMedian()` с `is_complex=false`
**Kernels**: `histogram_median_pass`, `find_median_bucket`
**Буферы**: `BufferSet<3>` — hist_buf, target_prefix, target_value
**Тонкие места**:
- 4-pass loop: memset hist → launch histogram → launch find_bucket
- blocks_per_beam capped at 1024
- Результат: uint32 → float conversion на host (beam_count мал, <1ms)
- Пишет финальные medians в ctx_->RequireShared(kMediansCompact)

### B6: MedianHistogramComplexOp
**Файл**: `modules/statistics/include/operations/median_histogram_complex_op.hpp` + `.cpp`
**Из**: `ExecuteHistogramMedian()` с `is_complex=true`
**Kernels**: `histogram_median_pass_complex`, `find_median_bucket`
**Буферы**: `BufferSet<3>` — hist_buf, target_prefix, target_value
**Тонкие места**:
- Идентичен B5, но kernel вычисляет |z| на лету (как welford_fused)
- Читает ctx_->RequireShared(kInput) вместо kMagnitudes

### B7: StatisticsProcessor → thin Facade
**Файл**: переписать `statistics_processor.hpp` + `.cpp`
**Что**: ComputeMean → mean_op_.Execute(), ComputeMedian → strategy select, etc.
**Тонкие места**:
- GpuContext создаётся в конструкторе (backend → stream → cache)
- CompileModule() вызывается lazy при первом Execute
- Ops создаются как member variables (не unique_ptr — проще move)
- API не меняется! Python bindings не трогаем

### B8: Тесты — убедиться что ВСЕ 11/11 тестов проходят
**Файл**: `modules/statistics/tests/test_statistics_rocm.hpp` — без изменений
**Тонкие места**:
- Запустить НА GPU перед и после рефакторинга
- Сравнить результаты бенчмарков (не должно быть regression)

---

## Ref03-C: Strategies pipeline refactoring

### C1: IPipelineStep interface
**Файл**: `modules/strategies/include/i_pipeline_step.hpp`
**Что**: Execute(PipelineContext&), IsEnabled(config), Name()
**Тонкие места**:
- `PipelineContext` — расширение GpuContext для strategies: d_S, d_W, d_X, d_spectrum, d_magnitudes, config, result
- IsEnabled() проверяет config (например stats == NONE → skip debug step)

### C2: PipelineContext struct
**Файл**: `modules/strategies/include/pipeline_context.hpp`
**Что**: Все буферы pipeline + конфиг + текущий результат
**Тонкие места**:
- Не наследует GpuContext — СОДЕРЖИТ GpuContext& (composition over inheritance)
- Буферы: d_S, d_W, d_X, d_fft_input, d_spectrum, d_magnitudes, d_hamming_window
- Result accumulator: AntennaResult (заполняется step'ами)

### C3: Concrete Steps
**Файлы**: `modules/strategies/include/steps/gemm_step.hpp`, `window_fft_step.hpp`, `debug_stats_step.hpp`, `one_max_step.hpp`, `all_maxima_step.hpp`, `minmax_step.hpp`
**Тонкие места**:
- **GemmStep**: использует hipBLAS handle из PipelineContext. Event synchronization (event_gemm_done)
- **WindowFftStep**: fused hamming_pad + hipFFT + magnitudes. Зависит от event_gemm_done
- **DebugStatsStep**: параметризуется DebugPoint (PRE_INPUT, POST_GEMM, POST_FFT) + StatisticsSet bitmask. Использует StatisticsProcessor (другой модуль!). Запускается на отдельном stream (stream_debug1/2/3)
- **OneMaxStep, AllMaximaStep, MinMaxStep**: могут запускаться параллельно на разных streams

### C4: Pipeline class
**Файл**: `modules/strategies/include/pipeline.hpp`
**Что**: Хранит vector<unique_ptr<IPipelineStep>>, Execute() с optional profiling
**Тонкие места**:
- Parallel steps: group of steps launched on different streams, sync before next group
- Per-step profiling: hipEvent_t start/stop → GPUProfiler::Record()
- Суммарное время: accumulate step times

### C5: PipelineBuilder
**Файл**: `modules/strategies/include/pipeline_builder.hpp`
**Что**: Fluent API: add(), add_if(), add_parallel(), build()
**Тонкие места**:
- `add_parallel()` создаёт ParallelGroup (внутренний класс)
- `build()` валидирует: нет дублей, зависимости соблюдены

### C6: AntennaProcessor_v1 → thin wrapper over Pipeline
**Файл**: переписать `antenna_processor_v1.cpp`
**Что**: Конструктор строит Pipeline через PipelineBuilder. process() = pipeline.Execute()
**Тонкие места**:
- 7 streams остаются (main + debug1-3 + bench3a-c)
- hipBLAS handle остаётся в PipelineContext
- hipFFT plan остаётся в WindowFftStep
- **API не меняется!** AntennaProcessorTest.step_*() → вызывают конкретные Steps напрямую

### C7: Per-step profiling integration
**Что**: Pipeline::Execute() с GPUProfiler → per-step timing + total
**Тонкие места**:
- hipEvent timing: start перед step, stop после step, elapsed после sync
- Parallel steps: timing = max(step_a, step_b, step_c)
- PrintReport показывает каждый step + total

### C8: Тесты
**Что**: Все существующие тесты должны проходить (test_full_pipeline, test_external_weights, benchmark_streams)
**Тонкие места**:
- AntennaProcessorTest.step_*() должны работать как раньше
- Benchmark: 1 stream vs 3 streams — результат не должен деградировать

---

## Ref03-D: Filters refactoring (после B + C)

### D1: FirFilterOp
### D2: IirBiquadOp
### D3: KalmanFilterOp, KaufmanFilterOp, MovingAverageOp
### D4: FilterProcessor → thin Facade
### D5: Тесты

---

## Ref03-E: fft_func refactoring (после B + C)

### E1: FFTForwardOp (wraps hipFFT plan)
### E2: MagnitudeOp
### E3: PeakFinderOp (wraps AllMaximaPipeline)
### E4: FFTProcessor → thin Facade
### E5: Тесты

---

## Checklist перед началом (понедельник)

- [ ] Собрать main на ROCm (Debian-Radeon9070)
- [ ] Прогнать ВСЕ тесты statistics (11/11)
- [ ] Прогнать тесты strategies (pipeline + benchmark)
- [ ] Зафиксировать baseline timing (для regression check)
- [ ] Начать с Ref03-A (foundation) → A1 BufferSet<N>
