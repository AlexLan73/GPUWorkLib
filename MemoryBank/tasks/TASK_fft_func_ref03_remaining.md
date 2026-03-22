# TASK: fft_func — оставшийся рефакторинг (Ref03 + namespace)

> **Создано**: 2026-03-22
> **Источник**: Code Review `MemoryBank/specs/review_fft_func_2026-03-22.md`
> **Приоритет**: Medium
> **Статус**: Задача A ✅ DONE, Задача B — BACKLOG

---

## Контекст

По результатам ревью fft_func (2026-03-22) исправлены 7 из 9 пунктов.
Два оставшихся — крупные архитектурные задачи, требующие отдельного подхода.

### Что уже сделано (2026-03-22):
- [x] hipfftExecC2C — проверка result во всех 4 вызовах
- [x] sqrtf → __fsqrt_rn во всех runtime kernel sources
- [x] warp_size → ROCmCore::GetWarpSize() в 4 файлах
- [x] C2MP мигрирован на Ref03 (GpuContext + BufferSet + MagPhaseOp + MagnitudeOp)
- [x] MakeROCmDataFromEvents → shared utils/rocm_profiling_helpers.hpp
- [x] In-place FFT (экономия batch × nFFT × 8 bytes)

---

## Задача A: SpectrumProcessorROCm → Ref03

**Сложность**: Высокая (~600 строк, 15+ точек вызова CompileKernels)
**Файлы**: `src/spectrum_processor_rocm.cpp`, `include/processors/spectrum_processor_rocm.hpp`

### Текущее состояние
- Ручной hiprtc: CompileKernels() ~100 строк (hiprtcCreateProgram → hiprtcCompileProgram → hipModuleLoadData → hipModuleGetFunction)
- Ручные буферы: input_buffer_, fft_input_, fft_output_, maxima_output_, magnitudes_buffer_ (5 raw void*)
- Ручные kernel launches: ExecutePadKernel, ExecuteFFT, ExecutePostKernel, ExecuteComputeMagnitudes
- Ручной KernelCacheService (нет — загрузка/сохранение кеша не реализована для Spectrum)
- hipFFT plan management (2 плана: plan_ + allmax_plan_)
- AllMaximaPipelineROCm — отдельный pipeline-класс

### План миграции (поэтапно)

**Этап 1**: GpuContext для компиляции
- Добавить `GpuContext ctx_` в класс
- Заменить CompileKernels() → `ctx_.CompileModule(source, kernel_names)`
- Заменить `module_`, `pad_kernel_`, `compute_mag_kernel_`, `kernels_compiled_` → `ctx_.GetKernel(name)`
- Оставить post_kernel_ отдельно (CompilePostKernel компилирует другой source)
- **Бонус**: автоматически получаем disk cache (KernelCacheService в GpuContext)

**Этап 2**: Ops для kernel launches
- Создать `ComputeMagnitudesOp : GpuKernelOp`
- Переиспользовать `PadDataOp` (уже есть в FFTProcessorROCm)
- Создать `SpectrumPostOp : GpuKernelOp` (для ONE_PEAK/TWO_PEAKS)

**Этап 3**: BufferSet для буферов
- Заменить 5 raw void* на `BufferSet<5>` с enum индексами
- Адаптировать AllocateBuffers(), ReallocateBuffersForBatch()
- Адаптировать ReleaseResources()

### Риски
- 15+ мест вызова CompileKernels — нужно аккуратно подставить EnsureCompiled()
- CompilePostKernel() компилирует ДРУГОЙ source (post_kernel) из того же hiprtc module — нужно объединить sources или использовать второй CompileModule вызов
- AllMaximaPipelineROCm тоже использует свои буферы и kernel'ы — может потребовать отдельную GpuContext
- hipFFT plan management остаётся в facade (не в Op) — корректно по Ref03

---

## Задача B: Namespace unification (fft_processor + antenna_fft)

**Сложность**: Средняя, но высокий blast radius (30 файлов)
**Текущее состояние**: Два namespace в одном модуле `fft_func`

| Namespace | Классы | Файлы |
|-----------|--------|-------|
| `fft_processor` | FFTProcessorROCm, ComplexToMagPhaseROCm, PadDataOp, MagPhaseOp, MagnitudeOp | ~15 файлов |
| `antenna_fft` | SpectrumProcessorROCm, AllMaximaPipelineROCm, ISpectrumProcessor, SpectrumParams, SpectrumResult | ~20 файлов |

### Зависимости antenna_fft (30 файлов!):
- `modules/strategies/` — PipelineContext, AntennaProcessorV1, AllMaximaStep
- `modules/heterodyne/` — HeterodyneDechirp
- `Doc/Modules/fft_func/` — Full.md, API.md, Quick.md
- `Doc/Modules/strategies/` — Full.md, API.md
- Python bindings (если есть)

### Варианты
1. **Rename antenna_fft → fft_processor**: чистое решение, но ~30 файлов + возможная поломка Python bindings
2. **using namespace alias**: `namespace fft_processor { using namespace antenna_fft; }` — грязный хак
3. **Оставить как есть**: два namespace = исторически сложившаяся конвенция, не ломает ничего

### Рекомендация
Вариант 1 (полное переименование) — но делать ОТДЕЛЬНЫМ коммитом с grep/sed, чтобы легко откатить. Проверить Python bindings первым делом.

---

## Зависимости
- Задача A не зависит от B (можно делать независимо)
- Задача B лучше делать ПОСЛЕ задачи A (когда Spectrum уже на Ref03)
