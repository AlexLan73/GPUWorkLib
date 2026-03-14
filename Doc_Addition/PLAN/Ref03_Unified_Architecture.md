# Ref03 — Единая архитектура GPU-операций GPUWorkLib

> **Статус**: ОБСУЖДЕНИЕ → APPROVED → IN_PROGRESS
> **Автор**: Alex + Кодо
> **Дата**: 2026-03-14
> **Применяется к**: ВСЕ модули (statistics, filters, fft_func, strategies, heterodyne, lch_farrow)

---

## 1. Проблема

Каждый модуль — свой стиль. `StatisticsProcessor`: 10 kernel handles, 13 GPU буферов, 30+ полей в move semantics. Добавление histogram median потребовало правок в 6+ местах одного файла. Нет единой концепции — каждый новый модуль/алгоритм пишется «с нуля».

## 2. Цель

**Одна архитектура на весь проект.** Любой модуль строится из маленьких Op-классов, объединённых Facade. Публичный API НЕ меняется.

---

## 3. Шестислойная модель

```
┌─────────────────────────────────────────────────────────┐
│  СЛОЙ 6: Facades + Strategies                           │
│  StatisticsProcessor, FilterProcessor, FFTProcessor     │
│  MedianStrategy, PipelineBuilder                        │
├─────────────────────────────────────────────────────────┤
│  СЛОЙ 5: Concrete Operations (маленькие классы)         │
│  MeanReductionOp, WelfordFusedOp, MedianHistogramOp     │
│  FirFilterOp, IirBiquadOp, FFTForwardOp, GemmStepOp     │
├─────────────────────────────────────────────────────────┤
│  СЛОЙ 4: BufferedGpuOp (base: buffer management)        │
│  BufferSet<N> — compile-time array, zero overhead        │
├─────────────────────────────────────────────────────────┤
│  СЛОЙ 3: GpuKernelOp (base: kernel compilation)         │
│  hiprtc/OpenCL compile, KernelCacheService               │
├─────────────────────────────────────────────────────────┤
│  СЛОЙ 2: IGpuOperation (abstract interface)             │
│  Name(), Initialize(), IsReady(), Release()              │
├─────────────────────────────────────────────────────────┤
│  СЛОЙ 1: GpuContext (shared state per-module)           │
│  backend, stream, module, console, shared buffers        │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Ключевые классы

### 4.1 GpuContext (СЛОЙ 1) — per-module

```cpp
// DrvGPU/interface/gpu_context.hpp
class GpuContext {
public:
  GpuContext(IBackend* backend);

  // Immutable (thread-safe read)
  IBackend* backend() const;
  hipStream_t stream() const;
  ConsoleOutput& console() const;

  // Kernel compilation (lazy, one-time, thread-safe: per-module instance)
  void CompileModule(const char* source, const std::vector<std::string>& kernel_names);
  hipFunction_t GetKernel(const char* name) const;

  // Shared buffers (managed by Facade, used by multiple Ops within module)
  enum SharedBuf { kInput, kMagnitudes, kResult, kMediansCompact, kSharedCount };
  void* RequireShared(SharedBuf id, size_t bytes);
  void ReleaseShared();

private:
  IBackend* backend_;
  hipStream_t stream_;
  hipModule_t module_ = nullptr;
  std::unordered_map<std::string, hipFunction_t> kernels_;
  BufferSet<kSharedCount> shared_;
  KernelCacheService cache_;
};
```

**Почему per-module:**
- Каждый модуль — свой stream → параллельное выполнение на одном GPU
- Thread-safe: модули не шарят mutable state
- Свой compiled module → независимая kernel compilation

### 4.2 IGpuOperation (СЛОЙ 2)

```cpp
// DrvGPU/interface/i_gpu_operation.hpp
class IGpuOperation {
public:
  virtual ~IGpuOperation() = default;
  virtual const char* Name() const = 0;
  virtual void Initialize(GpuContext& ctx) = 0;
  virtual bool IsReady() const = 0;
  virtual void Release() = 0;
};
```

### 4.3 GpuKernelOp (СЛОЙ 3)

```cpp
// DrvGPU/services/gpu_kernel_op.hpp
class GpuKernelOp : public IGpuOperation {
protected:
  GpuContext* ctx_ = nullptr;

  // Get compiled kernel from GpuContext (Facade compiled module once)
  hipFunction_t kernel(const char* name) const {
    return ctx_->GetKernel(name);
  }

  hipStream_t stream() const { return ctx_->stream(); }

public:
  void Initialize(GpuContext& ctx) override { ctx_ = &ctx; OnInitialize(); }
  bool IsReady() const override { return ctx_ != nullptr; }
  void Release() override { OnRelease(); ctx_ = nullptr; }

protected:
  virtual void OnInitialize() {}  // Override for custom init
  virtual void OnRelease() {}     // Override for custom cleanup
};
```

### 4.4 BufferSet<N> (СЛОЙ 4)

```cpp
// DrvGPU/services/buffer_set.hpp
template<size_t N>
class BufferSet {
  struct Entry { void* ptr = nullptr; size_t size = 0; };
  Entry entries_[N];

public:
  void* Require(size_t idx, size_t bytes) {
    auto& e = entries_[idx];
    if (e.size >= bytes) return e.ptr;  // reuse
    if (e.ptr) hipFree(e.ptr);
    hipMalloc(&e.ptr, bytes);
    e.size = bytes;
    return e.ptr;
  }

  void ReleaseAll() {
    for (auto& e : entries_) {
      if (e.ptr) { hipFree(e.ptr); e.ptr = nullptr; e.size = 0; }
    }
  }

  // Move: trivial (memcpy + zero source)
  BufferSet(BufferSet&& o) noexcept { std::memcpy(entries_, o.entries_, sizeof(entries_)); std::memset(o.entries_, 0, sizeof(o.entries_)); }
  BufferSet& operator=(BufferSet&& o) noexcept { ReleaseAll(); std::memcpy(entries_, o.entries_, sizeof(entries_)); std::memset(o.entries_, 0, sizeof(o.entries_)); return *this; }

  ~BufferSet() { ReleaseAll(); }
  BufferSet() = default;
};
```

**Свойства:**
- Zero overhead (stack array, no heap, no hash)
- Compile-time size → ошибки при компиляции
- Move = memcpy → тривиально
- Per-instance → thread-safe

### 4.5 Concrete Op (СЛОЙ 5, пример)

```cpp
// modules/statistics/include/operations/median_histogram_op.hpp
class MedianHistogramOp : public GpuKernelOp {
  enum Buf { kHist, kPrefix, kValue, kBufCount };
  BufferSet<kBufCount> bufs_;

public:
  const char* Name() const override { return "MedianHistogram"; }

  void Execute(size_t beam_count, size_t n_point, bool is_complex) {
    bufs_.Require(kHist,   beam_count * 256 * sizeof(uint32_t));
    bufs_.Require(kPrefix, beam_count * sizeof(uint32_t));
    bufs_.Require(kValue,  beam_count * sizeof(uint32_t));
    // ... 4-pass histogram logic ...
  }

protected:
  void OnRelease() override { bufs_.ReleaseAll(); }
};
```

### 4.6 Facade + Strategy (СЛОЙ 6)

```cpp
// modules/statistics/include/statistics_processor.hpp
class StatisticsProcessor {
  GpuContext ctx_;                    // per-module context
  MeanReductionOp mean_op_;          // lazy-init
  WelfordFusedOp welford_op_;
  MedianRadixSortOp median_sort_op_;
  MedianHistogramOp median_hist_op_;
  MedianHistogramComplexOp median_hist_complex_op_;

  static constexpr size_t kHistogramThreshold = 100'000;

public:
  // ПУБЛИЧНЫЙ API НЕ МЕНЯЕТСЯ!
  std::vector<MedianResult> ComputeMedian(const std::vector<std::complex<float>>& data,
                                           const StatisticsParams& params) {
    ctx_.RequireShared(GpuContext::kInput, ...);
    UploadData(...);

    if (params.n_point > kHistogramThreshold) {
      median_hist_complex_op_.Execute(params.beam_count, params.n_point);
    } else {
      median_sort_op_.Execute(params.beam_count, params.n_point);
    }
    return ReadMedianResults();
  }
};
```

---

## 5. Strategies Pipeline — гибкая архитектура

### Проблема
Текущий pipeline жёстко захардкожен: Step0→Step1→Step2→Step3→Step4→Step5→Step6.
Нужно: добавлять/удалять шаги, профилировать каждый, ветвить pipeline.

### Решение: IPipelineStep + PipelineBuilder

```cpp
// modules/strategies/include/i_pipeline_step.hpp
class IPipelineStep : public IGpuOperation {
public:
  virtual void Execute(PipelineContext& ctx) = 0;
  virtual bool IsEnabled(const AntennaProcessorConfig& cfg) const = 0;
};

// Concrete steps:
class GemmStep      : public IPipelineStep { ... };  // Step 2
class WindowFftStep : public IPipelineStep { ... };  // Step 4
class OneMaxStep    : public IPipelineStep { ... };  // Step 6.1
class AllMaximaStep : public IPipelineStep { ... };  // Step 6.2
class MinMaxStep    : public IPipelineStep { ... };  // Step 6.3

// Debug/Stats steps (optional):
class DebugStatsStep : public IPipelineStep {
  enum DebugPoint { PRE_INPUT, POST_GEMM, POST_FFT };
  DebugPoint point_;
  StatisticsSet stat_fields_;
  // Пропускается если stat_fields_ == NONE
};
```

### PipelineBuilder

```cpp
class PipelineBuilder {
public:
  PipelineBuilder& add(std::unique_ptr<IPipelineStep> step);
  PipelineBuilder& add_if(bool condition, std::unique_ptr<IPipelineStep> step);
  PipelineBuilder& add_parallel(std::vector<std::unique_ptr<IPipelineStep>> steps);
  std::unique_ptr<Pipeline> build();
};

// Usage:
auto pipeline = PipelineBuilder()
  .add(make_unique<DebugStatsStep>(PRE_INPUT, cfg.pre_input_stats))
  .add(make_unique<GemmStep>())
  .add(make_unique<DebugStatsStep>(POST_GEMM, cfg.post_gemm_stats))
  .add(make_unique<WindowFftStep>())
  .add(make_unique<DebugStatsStep>(POST_FFT, cfg.post_fft_stats))
  .add_parallel({
    make_unique<OneMaxStep>(),
    make_unique<AllMaximaStep>(),
    make_unique<MinMaxStep>()
  })
  .build();
```

### Per-Step профилирование

```cpp
class Pipeline {
  std::vector<std::unique_ptr<IPipelineStep>> steps_;
  GPUProfiler* profiler_ = nullptr;

  AntennaResult Execute(PipelineContext& ctx) {
    for (auto& step : steps_) {
      if (!step->IsEnabled(ctx.config())) continue;

      hipEvent_t start, stop;
      if (profiler_) { hipEventRecord(start, stream); }

      step->Execute(ctx);

      if (profiler_) {
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms; hipEventElapsedTime(&ms, start, stop);
        profiler_->Record(gpu_id, step->Name(), ms);
      }
    }
  }
};
```

### Ветвление (branch)

```cpp
// Добавить ветку с дополнительными расчётами:
auto pipeline_extended = PipelineBuilder()
  .add(make_unique<GemmStep>())
  .add(make_unique<WindowFftStep>())
  // Основная ветка:
  .add(make_unique<OneMaxStep>())
  // Дополнительная ветка (параллельно):
  .add_parallel({
    make_unique<AllMaximaStep>(),
    make_unique<CustomAnalysisStep>(),  // ← НОВОЕ
    make_unique<MinMaxStep>()
  })
  .build();
```

---

## 6. Многопоточность

### Правило: per-module GpuContext

```
GPU_00
├── StatisticsProcessor → GpuContext(stream_A)  // Thread A
├── FilterProcessor     → GpuContext(stream_B)  // Thread B (параллельно!)
└── FFTProcessor        → GpuContext(stream_C)  // Thread C (параллельно!)
```

### Правило: операции внутри модуля — последовательны

```
StatisticsProcessor.ComputeMedian()
  → MeanReductionOp.Execute()    // stream_A
  → MedianHistogramOp.Execute()  // stream_A (после mean)
  // НЕ параллельно — один stream на модуль
```

### Исключение: strategies с parallel steps

```
AntennaProcessor.process()
  → GemmStep          // stream_main
  → WindowFftStep     // stream_main
  → [parallel]:
      OneMaxStep       // stream_bench3a
      AllMaximaStep    // stream_bench3b
      MinMaxStep       // stream_bench3c
```

---

## 7. Файловая структура (target)

```
DrvGPU/
  interface/
    i_gpu_operation.hpp       ← НОВЫЙ
    gpu_context.hpp           ← НОВЫЙ
  services/
    gpu_kernel_op.hpp         ← НОВЫЙ
    buffer_set.hpp            ← НОВЫЙ

modules/statistics/
  include/
    statistics_processor.hpp  ← ПЕРЕПИСАТЬ (thin Facade)
    operations/
      mean_reduction_op.hpp   ← НОВЫЙ
      welford_fused_op.hpp    ← НОВЫЙ
      welford_float_op.hpp    ← НОВЫЙ
      median_radix_sort_op.hpp ← НОВЫЙ
      median_histogram_op.hpp  ← НОВЫЙ
      median_histogram_complex_op.hpp ← НОВЫЙ

modules/strategies/
  include/
    i_pipeline_step.hpp       ← НОВЫЙ
    pipeline_builder.hpp      ← НОВЫЙ
    pipeline.hpp              ← НОВЫЙ
    steps/
      gemm_step.hpp           ← НОВЫЙ
      window_fft_step.hpp     ← НОВЫЙ
      debug_stats_step.hpp    ← НОВЫЙ
      one_max_step.hpp        ← НОВЫЙ
      all_maxima_step.hpp     ← НОВЫЙ
      minmax_step.hpp         ← НОВЫЙ
```

---

## 8. Порядок реализации

| Ref | Модуль | Файлов | Приоритет | Зависимости |
|-----|--------|--------|-----------|-------------|
| **Ref03-A** | DrvGPU Foundation | ~5 | 🔴 FIRST | — |
| **Ref03-B** | statistics | ~8 | 🟡 SECOND | Ref03-A |
| **Ref03-C** | strategies (pipeline) | ~10 | 🟡 SECOND | Ref03-A |
| **Ref03-D** | filters | ~8 | 🟢 THIRD | Ref03-A |
| **Ref03-E** | fft_func | ~6 | 🟢 THIRD | Ref03-A |
