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
| **FFTProcessor** | Active | FFT с режимами Complex/MagPhase |
| **ScriptGenerator** | Active | Text DSL -> OpenCL kernel compiler |
| **LchFarrow** | Active | Lagrange fractional delay (48x5 matrix) |
| **Filters** | **Active** | **FIR + IIR GPU фильтры (Stage 1 MVP + Stage 3 AI Pipeline DONE)** |
| **Python Bindings** | Active | pybind11 модуль gpuworklib |
| **Statistics** | Planned | Статистика. См. `specs/statistics_module.md` |
| **Heterodyne** | **Active** | LFM Dechirp (stretch-processing). См. `Doc/Modules/heterodyne/Full.md` |


## Текущий статус

**В работе**: ROCm Backend — см. [tasks/PLAN_AMD_Radeon_9070_ROCm.md](tasks/PLAN_AMD_Radeon_9070_ROCm.md)

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
- `IirFilter(ctx)` — IIR фильтр (biquad cascade, GPU) **NEW**

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

*Последнее обновление: 2026-02-23*
