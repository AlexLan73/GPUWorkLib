# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-02-13

---

## Модули проекта

| Модуль | Статус | Описание |
|--------|--------|----------|
| **DrvGPU** | Active | Базовый драйвер GPU (OpenCL backend) |
| **SpectrumMaximaFinder** | Active | Поиск максимума спектра FFT |
| **SignalGenerators** | Active | CW, LFM, Noise, Script генераторы |
| **FFTProcessor** | Active | FFT с режимами Complex/MagPhase |
| **ScriptGenerator** | Active | Text DSL -> OpenCL kernel compiler |
| **Python Bindings** | Active | pybind11 модуль gpuworklib |
| **Filters** | Planned | ЦОС фильтры (FIR, IIR) |
| **Statistics** | Planned | Статистическая обработка |
| **Heterodyne** | Planned | Гетеродин (перенос частоты) |


## Текущий статус


## Задачи
---

## Python модуль (gpuworklib)

```
build/python/Release/gpuworklib.cp312-win_amd64.pyd
```

### Классы:
- `GPUContext(device_index)` — OpenCL контекст
- `SignalGenerator(ctx)` — CW/LFM/Noise генерация
- `ScriptGenerator(ctx)` — Text DSL -> GPU kernel
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

*Последнее обновление: 2026-02-13*
