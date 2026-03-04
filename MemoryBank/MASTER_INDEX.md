# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-02-23

---

## Модули проекта

| Модуль | Статус | Описание |
|--------|--------|----------|
| **DrvGPU** | Active | Базовый драйвер GPU (OpenCL backend). См. `specs/drvgpu.md` |
| **SpectrumMaximaFinder** (fft_maxima) | Active | Поиск максимума спектра FFT. См. `Doc/Modules/fft_maxima/Full.md` |
| **SignalGenerators** | Active | CW, LFM, Noise, Script, **FormSignal** DONE. См. `Doc/Modules/signal_generators/Full.md` |
| **FFTProcessor** | Active | FFT с режимами Complex/MagPhase. OpenCL (clFFT) + **ROCm (hipFFT)** + **ComplexToMagPhaseROCm**. ROCm: 11/11 тестов. `Doc/Modules/fft_processor/Full.md` |
| **ScriptGenerator** | Active | Text DSL -> OpenCL kernel compiler |
| **LchFarrow** | Active | Lagrange fractional delay (48x5 matrix) |
| **Filters** | **Active** | **FIR + IIR GPU фильтры (Stage 1 MVP + Stage 3 AI Pipeline DONE)** |
| **Python Bindings** | Active | pybind11 модуль gpuworklib |
| **vector_algebra** | **COMPLETED** | Cholesky инверсия матриц (rocBLAS+rocSOLVER). **v2**: SymmetrizeMode (Roundtrip/GpuKernel), hiprtc, RAII CholeskyResult. C++ 23 PASSED, Python 6 PASSED. API: `Doc/Python/vector_algebra_api.md` |
| **Statistics** | 🟢 **Active** | ROCm hiprtc: welford_fused, extract_medians, KernelCacheService. C++ 7/7 PASSED, Python 9/9 PASSED. `Doc/Modules/statistics/Full.md` |
| **Heterodyne** | **Active** | LFM Dechirp (stretch-processing). См. `Doc/Modules/heterodyne/Full.md` |
| **FM Correlator** | **Planned** | ФМ-корреляция M-последовательностями в частотной области (ROCm). См. `Doc/Modules/fm_correlator/Full.md` |
| **Python_test** | **Active** | Тестовая инфраструктура Python (10 модулей, 24 файла). См. `Doc/Python_test/Full.md` |


## Текущий статус

**В работе**:
- ROCm Backend — см. [tasks/PLAN_AMD_Radeon_9070_ROCm.md](tasks/PLAN_AMD_Radeon_9070_ROCm.md)
- **heterodyne профилирование (OpenCL + ROCm)** — 14 тасков 📋 PLAN → [tasks/TASK_heterodyne_profiling.md](tasks/TASK_heterodyne_profiling.md)
- **signal_generators профилирование (OpenCL + ROCm)** — 17 тасков 📋 PLAN → [tasks/TASK_signal_generators_profiling.md](tasks/TASK_signal_generators_profiling.md)
- **lch_farrow профилирование (OpenCL + ROCm)** — 12 тасков 📋 PLAN → [tasks/TASK_lch_farrow_profiling.md](tasks/TASK_lch_farrow_profiling.md)
**Завершено**:
- **fft_maxima профилирование (OpenCL + ROCm)** — TASK 1–14 DONE → [tasks/TASK_fft_maxima_profiling_opencl.md](tasks/TASK_fft_maxima_profiling_opencl.md)
- **filters профилирование (OpenCL + ROCm)** — TASK 1–14 DONE → [tasks/TASK_filters_profiling.md](tasks/TASK_filters_profiling.md)
- SpectrumMaximaFinder — OpenCL kernel optimizations → [tasks/fft_maxima_optimization_plan.md](tasks/fft_maxima_optimization_plan.md)

---

## Python модуль (gpuworklib)

```
build/python/Release/gpuworklib.cp312-win_amd64.pyd
```

### Классы:
- `GPUContext(device_index)` — OpenCL контекст
- `SignalGenerator(ctx)` — CW/LFM/Noise генерация
- `ScriptGenerator(ctx)` — Text DSL -> GPU kernel
- `FormSignalGenerator(ctx)` — Мультиканальный генератор (getX формула)
- `FormScriptGenerator(ctx)` — DSL + on-disk kernel cache
- `FFTProcessor(ctx)` — GPU FFT (clFFT)
- `LchFarrow(ctx)` — Fractional delay processor
- `HeterodyneDechirp(ctx)` — LFM dechirp (stretch-processing)
- `FirFilter(ctx)` — FIR фильтр (GPU, scipy-коэффициенты) **NEW**
- `IirFilter(ctx)` — IIR фильтр (biquad cascade, GPU)
- `CholeskyInverterROCm(ctx, mode)` — Инверсия матриц (Cholesky, ROCm). SymmetrizeMode: Roundtrip/GpuKernel **v2**
- `StatisticsProcessor(ctx)` — mean/median/variance/std per beam (ROCm). `compute_mean()`, `compute_median()`, `compute_statistics()`

### Тесты: `Python_test/test_*.py`

---

## Перспективные задачи

| Задача | Приоритет | Описание |
|--------|-----------|----------|
| Filters Stage 2 | Средний | Text->kernel pipeline (FormScriptGenerator-like) |
| ~~Filters Stage 3~~ | **DONE** | ~~Groq AI micro-agent~~ -> **AI Filter Pipeline DONE** |
| ROCm backend | Средний | Миграция на AMD Radeon 9070. **План**: `tasks/PLAN_AMD_Radeon_9070_ROCm.md` |
| Оконные функции | Низкий | Hann, Hamming, Blackman |
| Streaming filters | Низкий | State persistence для непрерывного потока |
| Overlap-Save/Add | Низкий | Длинные FIR через FFT |

---

*Последнее обновление: 2026-03-01*
