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

---

## Завершённые темы (2026-02)

| Тема | Дата | Описание |
|------|------|----------|
| **ТЕМА 4: DrvGPU Optimization** | 02-10 | Анализ, объединение OpenCLBackend |
| **ТЕМА 3: Kernel Refactoring** | 02-10 | OnePeak/TwoPeaks кернелы, PeakSearchMode |
| **ТЕМА 2: Batch Processing** | 02-11 | BatchManager, 256x1.3M точек |
| **ТЕМА 1: API Refactoring** | 02-12 | Новый API: Process(InputData, Mode, Driver) |
| **Signal Generators** | 02-13 | CW/LFM/Noise на GPU, Strategy+Factory+DI |
| **FFT Processor** | 02-13 | clFFT wrapper, Complex/MagPhase/MagPhaseFreq |
| **Python Bindings** | 02-13 | pybind11 модуль, 7 тестов + графики |
| **ScriptGenerator** | 02-13 | Text DSL -> OpenCL, [Params]/[Defs]/[Signal] |

---

## Текущий статус

### API v2.0 (2026-02-12)
```cpp
// Новый API
InputData<T> input{.antenna_count, .n_point, .data, .repeat_count, .sample_rate};
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

### Документация
- Полная: `Doc/spectrum_maxima_api_guide.md`
- Краткая: `Doc/spectrum_maxima_api_quick.md`

### Пути
- Логи: `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log`
- Профайлер: `Results/Profiler/GPU_XX_Profiler/*.md|json`

---

## Задачи

| Файл | Описание |
|------|----------|
| [tasks/BACKLOG.md](tasks/BACKLOG.md) | Очередь задач |
| [tasks/COMPLETED.md](tasks/COMPLETED.md) | Завершённые задачи |

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
