# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-02-17

---

## Модули проекта

| Модуль | Статус | Описание |
|--------|--------|----------|
| **DrvGPU** | Active | Базовый драйвер GPU (OpenCL backend). См. `specs/drvgpu.md` |
| **SpectrumMaximaFinder** (fft_maxima) | Active | Поиск максимума спектра FFT. См. `specs/fft_maxima.md` |
| **SignalGenerators** | Active | CW, LFM, Noise, Script, **FormSignal** ✅ DONE. См. `specs/Form_signals.md` |
| **FFTProcessor** | Active | FFT с режимами Complex/MagPhase |
| **ScriptGenerator** | Active | Text DSL -> OpenCL kernel compiler |
| **Python Bindings** | Active | pybind11 модуль gpuworklib |
| **Filters** | Planned | ЦОС фильтры (FIR, IIR) |
| **Statistics** | Planned | Статистика. См. `specs/statistics_module.md` |
| **Heterodyne** | Planned | Гетеродин (перенос частоты) |


## Текущий статус

### FormSignalGenerator — ✅ ЗАВЕРШЕНО (все 6 этапов)
- **Этап 1** ✅ — FormSignalGenerator: C++ (6/6), Python (7/7 + 6 графиков)
- **Этап 2** ✅ — FormScriptGenerator: DSL + kernel cache, C++ (7/7)
- **Этап 3** ✅ — SignalService + Factory: CreateForm/CreateFormROCm
- **Этап 4** ✅ — Python bindings: PyFormSignalGenerator + PyFormScriptGenerator
- **Этап 5** ✅ — Документация: `Doc/Python/signal_generators_api.md` + example
- **Этап 6** ✅ — ROCm stubs: FormSignalGeneratorROCm + form_signal.hip

---

## Python модуль (gpuworklib)

```
build/python/Release/gpuworklib.cp312-win_amd64.pyd
```

### Классы:
- `GPUContext(device_index)` — OpenCL контекст
- `SignalGenerator(ctx)` — CW/LFM/Noise генерация
- `ScriptGenerator(ctx)` — Text DSL -> GPU kernel
- `FormSignalGenerator(ctx)` — Мультиканальный генератор (getX формула) ✅
- `FormScriptGenerator(ctx)` — DSL + on-disk kernel cache ✅ NEW
- `FFTProcessor(ctx)` — GPU FFT (clFFT)

### Тесты: `D:\Python\С++ to Python\test_gpuworklib.py` (9 тестов)

---

## Перспективные задачи

| Задача | Приоритет | Описание |
|--------|-----------|----------|
| Fractional Delay | Низкий | Оценка дробной задержки (Tretter) |
| ROCm backend | Средний | Добавить hipFFT для AMD GPU |
| Оконные функции | Низкий | Hann, Hamming, Blackman |
| Real-to-Complex FFT | Низкий | R2C оптимизация |

---

*Последнее обновление: 2026-02-17*
