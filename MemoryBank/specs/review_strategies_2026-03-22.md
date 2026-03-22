# 🔍 Code Review: modules/strategies

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~40 файлов
> **Методы анализа**: sequential-thinking, grep analysis
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | WARP_SIZE=32 hardcoded → `ROCmCore::GetWarpSize()` (2 места + 3 в filters) | `antenna_processor_v1.cpp`, `strategies_float_api.hpp`, 3 файла filters |
| 2 | 🟡→✅ | v1 compile_kernels() → GpuContext::CompileModule(). Удалено ~100 строк hiprtc+KernelCacheService | `antenna_processor_v1.hpp/cpp` |

---

## 📊 Общая оценка

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| **Архитектура** | ⭐⭐⭐⭐⭐ | **Лучшая в проекте** — Pipeline + Builder + multi-stream + parallel groups |
| Pipeline pattern | ⭐⭐⭐⭐⭐ | IPipelineStep → PipelineStepBase → Steps (GEMM, FFT, OneMax...) |
| PipelineBuilder | ⭐⭐⭐⭐⭐ | Fluent API: add() / add_if() / add_parallel() / build() |
| Multi-stream | ⭐⭐⭐⭐⭐ | 7 streams + 4 events для inter-stream sync |
| Steps quality | ⭐⭐⭐⭐⭐ | Маленькие, focused, все проверяют return codes |
| v1 legacy | ⭐⭐⭐ | ~500 строк manual hiprtc + raw void* |
| Тесты | ⭐⭐⭐⭐⭐ | 12 тестов + signal strategies + benchmarks |

**Вердикт**: 🏆 Архитектурно — самый продвинутый модуль проекта. Pipeline + Builder + multi-stream parallel groups. Steps чистые и маленькие.

---

## 📐 Архитектура (Pipeline)

```
AntennaProcessor_v1 (Facade)
  ├── PipelineBuilder
  │     .add(GemmStep)                    → stream_main
  │     .add(WindowFftStep)               → stream_main
  │     .add_parallel([OneMax, AllMaxima, MinMax],
  │                    [stream_a, stream_b, stream_c])
  │     .build()
  ├── Pipeline::Execute(ctx)
  │     for each entry:
  │       if SEQUENTIAL → step.Execute(ctx)
  │       if PARALLEL   → all steps.Execute() + sync all streams
  └── PipelineContext
        ├── backend, gpu_ctx, cfg
        ├── hipblas_handle, fft_plan
        ├── 7 streams + 4 events
        ├── BufferSet<7> buffers
        └── stats_processor, all_maxima_pipeline, complex_to_mag
```

---

## 🔴 Критические проблемы: 0

---

## 🟡 Важные замечания (2)

### 1. WARP_SIZE=32 hardcoded (**ИСПРАВЛЕНО**)

**Файлы**:
- `src/antenna_processor_v1.cpp:267`
- `include/strategies_float_api.hpp:287`

### 2. AntennaProcessor_v1 — legacy dual implementation

Проект содержит **две параллельные реализации**:

| Реализация | Архитектура | Состояние |
|-----------|-------------|-----------|
| AntennaProcessor_v1 | Legacy (manual hiprtc, raw void*, KernelCacheService) | Production, ~500 строк |
| Pipeline + Steps | Ref03-C (PipelineContext, BufferSet, Builder) | Modern, чистый |

v1 используется в `process()` и `do_*()` методах. Pipeline Steps (GemmStep, WindowFftStep, OneMaxStep...) — современная замена.

**Рекомендация**: Постепенно мигрировать `process()` на Pipeline::Execute(), вызывая Steps вместо do_*() методов. v1 сохранить как fallback.

---

## 🟢 Что отлично

### Multi-stream pipeline ⭐⭐⭐⭐⭐
```
stream_main:   GEMM → Window+FFT → event_fft_done
stream_debug1: stats(d_S)  ← параллельно с GEMM
stream_debug2: stats(d_X)  ← параллельно с Window+FFT
stream_post_a: OneMax       ← параллельно после FFT
stream_post_b: AllMaxima    ← параллельно после FFT
stream_post_c: MinMax       ← параллельно после FFT
```

### PipelineBuilder ✅
```cpp
auto pipe = PipelineBuilder()
    .add(make_unique<GemmStep>())
    .add(make_unique<WindowFftStep>())
    .add_parallel({OneMax, AllMaxima, MinMax}, {s1, s2, s3})
    .build();
```

### Steps — маленькие и чистые ✅
- GemmStep: 15 строк, hipblasCgemm + error check
- WindowFftStep: 40 строк, memset + kernel + FFT + magnitudes
- OneMaxStep, MinMaxStep, AllMaximaStep: каждый < 50 строк

### PipelineContext — non-owning composition ✅
Единый struct с указателями на всё что нужно Steps. Owned by Facade.

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | Статус |
|----------|--------|
| DrvGPU | ✅ IBackend + GpuContext |
| ConsoleOutput | ✅ |
| hipFFT checks | ✅ (WindowFftStep) |
| hipBLAS checks | ✅ (GemmStep) |
| Ref03-C | ✅ Pipeline/Steps; ❌ v1 legacy |
| Multi-GPU | ✅ Per-instance, no globals |
| Windows stub | ❌ (нет — `#if ENABLE_ROCM` без else) |

---

## 📋 Сводка задач

| # | Приоритет | Описание | Сложность |
|---|-----------|----------|-----------|
| 1 | 🟡 ✅ | WARP_SIZE → GetWarpSize() | Низкая (**DONE**) |
| 2 | 🟢 | Мигрировать v1 process() → Pipeline::Execute() | Высокая |
| 3 | 🟢 | Windows stubs для strategies | Низкая |

---

*Ревью подготовлено с: sequential-thinking (2 шага), grep analysis по ~40 файлам*
