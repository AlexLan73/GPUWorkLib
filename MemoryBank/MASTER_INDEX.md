# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-02-12

---

## Модули проекта

| Модуль | Статус | Описание |
|--------|--------|----------|
| **DrvGPU** | Active | Базовый драйвер GPU (OpenCL backend) |
| **SpectrumMaximaFinder** | Active | Поиск максимума спектра FFT |
| **Filters** | Planned | ЦОС фильтры (FIR, IIR) |
| **Statistics** | Planned | Статистическая обработка |
| **Heterodyne** | Planned | Гетеродин (перенос частоты) |
| **SignalSynth** | Planned | Синтезатор сигналов |

---

## Завершённые темы (2026-02)

| Тема | Дата | Описание |
|------|------|----------|
| **ТЕМА 4: DrvGPU Optimization** | 02-10 | Анализ, объединение OpenCLBackend |
| **ТЕМА 3: Kernel Refactoring** | 02-10 | OnePeak/TwoPeaks кернелы, PeakSearchMode |
| **ТЕМА 2: Batch Processing** | 02-11 | BatchManager, 256×1.3M точек |
| **ТЕМА 1: API Refactoring** | 02-12 | Новый API: Process(InputData, Mode, Driver) |

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

## Перспективные задачи

| Задача | Приоритет | Описание |
|--------|-----------|----------|
| ROCm backend | Средний | Добавить hipFFT для AMD GPU |
| Оконные функции | Низкий | Hann, Hamming, Blackman |
| Real-to-Complex FFT | Низкий | R2C оптимизация |

---

*Последнее обновление: 2026-02-12*
