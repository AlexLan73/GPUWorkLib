# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-02-18

---

## Модули проекта

| Модуль | Статус | Описание |
|--------|--------|----------|
| **DrvGPU** | Active | Базовый драйвер GPU (OpenCL backend). См. `specs/drvgpu.md` |
| **SpectrumMaximaFinder** (fft_maxima) | Active | Поиск максимума спектра FFT. См. `specs/fft_maxima.md` |
| **SignalGenerators** | Active | CW, LFM, Noise, Script, **FormSignal** DONE. См. `specs/Form_signals.md` |
| **FFTProcessor** | Active | FFT с режимами Complex/MagPhase |
| **ScriptGenerator** | Active | Text DSL -> OpenCL kernel compiler |
| **LchFarrow** | Active | Lagrange fractional delay (48x5 matrix) |
| **Filters** | **Active** | **FIR + IIR GPU фильтры (Stage 1 MVP DONE)** |
| **Python Bindings** | Active | pybind11 модуль gpuworklib |
| **Statistics** | Planned | Статистика. См. `specs/statistics_module.md` |
| **Heterodyne** | Planned | Гетеродин (перенос частоты) |


## Текущий статус

### Filters Module — Stage 1 MVP DONE (2026-02-18)
- FIR: direct-form, 2D NDRange, __constant/__global авто-выбор
- IIR: biquad cascade DFII-T, все секции в одном kernel
- C++ тесты: GPU vs CPU err = 1e-6 (PASSED)
- Python тесты: vs scipy err = 4.77e-7 (PASSED)
- 4-panel plot: `Results/JSON/test_filters_stage1.png`

### FormSignalGenerator — ЗАВЕРШЕНО (все 6 этапов)
- **Этап 1** — FormSignalGenerator: C++ (6/6), Python (7/7 + 6 графиков)
- **Этап 2** — FormScriptGenerator: DSL + kernel cache, C++ (7/7)
- **Этап 3** — SignalService + Factory: CreateForm/CreateFormROCm
- **Этап 4** — Python bindings: PyFormSignalGenerator + PyFormScriptGenerator
- **Этап 5** — Документация: `Doc/Python/signal_generators_api.md` + example
- **Этап 6** — ROCm stubs: FormSignalGeneratorROCm + form_signal.hip

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
- `FirFilter(ctx)` — FIR фильтр (GPU, scipy-коэффициенты) **NEW**
- `IirFilter(ctx)` — IIR фильтр (biquad cascade, GPU) **NEW**

### Тесты: `Python_test/test_*.py`

---

## Перспективные задачи

| Задача | Приоритет | Описание |
|--------|-----------|----------|
| Filters Stage 2 | Средний | Text->kernel pipeline (FormScriptGenerator-like) |
| Filters Stage 3 | Средний | Groq AI micro-agent (NL -> filter design -> GPU) |
| ROCm backend | Средний | Добавить hipFFT для AMD GPU |
| Оконные функции | Низкий | Hann, Hamming, Blackman |
| Streaming filters | Низкий | State persistence для непрерывного потока |
| Overlap-Save/Add | Низкий | Длинные FIR через FFT |

---

*Последнее обновление: 2026-02-18*
