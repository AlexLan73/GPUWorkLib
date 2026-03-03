# GPUWorkLib — Documentation Index

> Главный индекс документации проекта.
> Здесь: навигация по всем разделам, статус модулей, структура файлов.
>
> **Date**: 2026-03-03 | **Maintained by**: Кодо (AI Assistant)

---

## Быстрый старт

| Цель | Документ |
|------|----------|
| Посмотреть API одной функцией | [**Quick_Reference.md**](Quick_Reference.md) |
| Понять классы и структуры подробно | [**Full_Reference.md**](Full_Reference.md) |
| Изучить архитектуру системы | [Architecture/Architecture_INDEX.md](Architecture/Architecture_INDEX.md) |
| Python API и биндинги | [→ раздел ниже](#python) |
| Примеры и паттерны | [Examples/](../Examples/) |

---

## Модули

Порядок по приоритету использования (от обработки сигналов к инфраструктуре):

### Вычислительные модули

| # | Модуль | Backend | Статус | Quick | Full | Python API | Python тесты |
|---|--------|---------|--------|-------|------|------------|-------------|
| 1 | **FFT Processor** | OpenCL / ROCm | 🟢 Active | [QR §1](Quick_Reference.md#1-fft-processor) | [FR §1](Full_Reference.md#1-fft-processor) | — | [Python_test/fft_maxima/](../Python_test/fft_maxima/) |
| 2 | **Statistics** | ROCm only | 🟢 Active | [QR §2](Quick_Reference.md#2-statistics-rocm) | [FR §2](Full_Reference.md#2-statistics-rocm) | [rocm_modules_api.md](Python/rocm_modules_api.md) | [Python_test/statistics/](../Python_test/statistics/) |
| 3 | **Vector Algebra** | ROCm only | 🟢 Active | [QR §3](Quick_Reference.md#3-vector-algebra-rocm) | [FR §3](Full_Reference.md#3-vector-algebra-rocm) | [vector_algebra_api.md](Python/vector_algebra_api.md) | [Python_test/vector_algebra/](../Python_test/vector_algebra/) |
| 4 | **FFT Maxima** | OpenCL / ROCm | 🟢 Active | [QR §4](Quick_Reference.md#4-fft-maxima) | [FR §4](Full_Reference.md#4-fft-maxima) | [spectrum_maxima_api.md](Python/spectrum_maxima_api.md) | [Python_test/fft_maxima/](../Python_test/fft_maxima/) |
| 5 | **Filters** | OpenCL / ROCm | 🟢 Active | [QR §5](Quick_Reference.md#5-filters) | [FR §5](Full_Reference.md#5-filters) | [rocm_modules_api.md](Python/rocm_modules_api.md) | [Python_test/filters/](../Python_test/filters/) |
| 6 | **Signal Generators** | OpenCL / ROCm | 🟢 Active | [QR §6](Quick_Reference.md#6-signal-generators) | [FR §6](Full_Reference.md#6-signal-generators) | [signal_generators_api.md](Python/signal_generators_api.md) | [Python_test/signal_generators/](../Python_test/signal_generators/) |
| 7 | **LCH Farrow** | OpenCL / ROCm | 🟢 Active | [QR §7](Quick_Reference.md#7-lch-farrow) | [FR §7](Full_Reference.md#7-lch-farrow) | [lch_farrow_api.md](Python/lch_farrow_api.md) | [Python_test/lch_farrow/](../Python_test/lch_farrow/) |
| 8 | **Heterodyne** | OpenCL / ROCm | 🟢 Active | [QR §8](Quick_Reference.md#8-heterodyne) | [FR §8](Full_Reference.md#8-heterodyne) | [rocm_modules_api.md](Python/rocm_modules_api.md) | [Python_test/heterodyne/](../Python_test/heterodyne/) |

### Инфраструктура

| # | Модуль | Статус | Quick | Full | Подробная документация |
|---|--------|--------|-------|------|------------------------|
| 9 | **DrvGPU** | 🟢 Active | [QR §9](Quick_Reference.md#9-drvgpu) | [FR §9](Full_Reference.md#9-drvgpu--core-driver) | [DrvGPU/Architecture.md](DrvGPU/Architecture.md) |

---

## Детальная документация модулей

### `Doc/Modules/` — полные описания C++ модулей

| Модуль | Full | Quick | Дополнительно |
|--------|------|-------|---------------|
| FFT Processor | [fft_processor/Full.md](Modules/fft_processor/Full.md) | [fft_processor/Quick.md](Modules/fft_processor/Quick.md) | |
| Statistics | [statistics/](Modules/statistics/) | | |
| Vector Algebra | [vector_algebra/](Modules/vector_algebra/) | | |
| FFT Maxima | [fft_maxima/Full.md](Modules/fft_maxima/Full.md) | [fft_maxima/Quick.md](Modules/fft_maxima/Quick.md) | [FindAllMaxima_Guide.md](Modules/fft_maxima/FindAllMaxima_MaxValue_Guide.md) |
| Filters | [filters/Full.md](Modules/filters/Full.md) | [filters/Quick.md](Modules/filters/Quick.md) | [gpu_filters_research.md](Modules/filters/gpu_filters_research.md) |
| Signal Generators | [signal_generators/Full.md](Modules/signal_generators/Full.md) | [signal_generators/Quick.md](Modules/signal_generators/Quick.md) | [ScriptGenerator.md](Modules/signal_generators/ScriptGenerator.md) |
| LCH Farrow | [lch_farrow/Full.md](Modules/lch_farrow/Full.md) | [lch_farrow/Quick.md](Modules/lch_farrow/Quick.md) | |
| Heterodyne | [heterodyne/Full.md](Modules/heterodyne/Full.md) | [heterodyne/Quick.md](Modules/heterodyne/Quick.md) | |

### `Doc/DrvGPU/` — документация ядра

| Документ | Описание |
|----------|----------|
| [Architecture.md](DrvGPU/Architecture.md) | Полная архитектура DrvGPU: слои, паттерны, зависимости |
| [Classes.md](DrvGPU/Classes.md) | Справочник всех 50+ классов с методами |
| [Quick.md](DrvGPU/Quick.md) | Краткий старт: основные классы и паттерны |
| [Memory.md](DrvGPU/Memory.md) | Система памяти GPU: GPUBuffer, SVMBuffer, HIPBuffer |
| [OpenCL.md](DrvGPU/OpenCL.md) | OpenCL backend: командные очереди, профилирование |
| [Services/Full.md](DrvGPU/Services/Full.md) | Все сервисы: GPUProfiler, ConsoleOutput, BatchManager |
| [Services/Quick.md](DrvGPU/Services/Quick.md) | Краткий справочник сервисов |

---

## Python

> Python биндинги через **pybind11**. Конвертация типов C++ ↔ Python — автоматическая.
> Подробнее о механике: [Full_Reference.md §10](Full_Reference.md#10-python-api)

### `Doc/Python/` — Python API документация

| Файл | Модули | Описание |
|------|--------|----------|
| [signal_generators_api.md](Python/signal_generators_api.md) | SignalGenerator, FormSignal, LfmAnalyticalDelay, DelayedFormSignal | Все генераторы сигналов: CW, LFM, Form, DSL-скрипт, задержка Farrow |
| [spectrum_maxima_api.md](Python/spectrum_maxima_api.md) | SpectrumMaximaFinder | Поиск пиков: OnePeak, AllMaxima, GPU/CPU режимы |
| [lch_farrow_api.md](Python/lch_farrow_api.md) | LchFarrow | Дробная задержка: set_delays, process |
| [vector_algebra_api.md](Python/vector_algebra_api.md) | CholeskyInverterROCm, SymmetrizeMode | Cholesky инверсия матриц на ROCm |
| [rocm_modules_api.md](Python/rocm_modules_api.md) | ROCmGPUContext, FirFilterROCm, IirFilterROCm, LchFarrowROCm, HeterodyneROCm, StatisticsProcessor | Все ROCm модули |

### `Doc/Python_test/` — описание Python тестов

| Файл | Описание |
|------|----------|
| [Python_test/Full.md](Python_test/Full.md) | Полный список тестов с описанием каждого |
| [Python_test/Quick.md](Python_test/Quick.md) | Как запускать тесты, структура |

### Структура Python тестов (`Python_test/`)

```
Python_test/
├── signal_generators/
│   ├── test_form_signal.py              # FormSignalGenerator: 7 тестов + 6 графиков
│   ├── test_delayed_form_signal.py      # DelayedFormSignalGenerator + Farrow
│   └── test_lfm_analytical_delay.py    # LfmAnalyticalDelay: 5 тестов
├── fft_maxima/
│   ├── test_spectrum_find_all_maxima.py # AllMaxima pipeline
│   ├── test_find_all_maxima_maxvalue.py # MaxValue структура
│   └── test_spectrum_find_all_maxima_rocm.py
├── filters/
│   ├── test_filters_stage1.py           # Базовые FIR/IIR тесты
│   ├── test_fir_filter_rocm.py
│   ├── test_iir_filter_rocm.py
│   ├── test_iir_plot.py                 # Визуализация АЧХ
│   ├── test_ai_filter_pipeline.py       # Полный pipeline
│   └── test_ai_fir_demo.py
├── heterodyne/
│   ├── test_heterodyne.py               # OpenCL dechirp
│   ├── test_heterodyne_rocm.py          # ROCm dechirp
│   ├── test_heterodyne_comparison.py    # OpenCL vs ROCm
│   └── test_heterodyne_step_by_step.py # Пошаговая диагностика
├── lch_farrow/
│   ├── test_lch_farrow.py               # OpenCL Farrow
│   └── test_lch_farrow_rocm.py         # ROCm Farrow
├── statistics/
│   └── test_statistics_rocm.py         # Welford + radix sort
├── vector_algebra/
│   ├── test_cholesky_inverter_rocm.py   # Cholesky POTRF/POTRI
│   └── test_matrix_csv_comparison.py   # Сравнение с NumPy
├── integration/
│   └── test_gpuworklib.py              # Интеграционные тесты
├── hybrid/
│   └── test_hybrid_backend.py          # OpenCL + ROCm одновременно
└── zero_copy/
    └── test_zero_copy.py               # Zero-copy буферы
```

### Биндинги `python/`

```
python/
├── gpu_worklib_bindings.cpp    # Главный файл: регистрация всех классов
├── py_filters.hpp              # FirFilter, IirFilter (OpenCL)
├── py_filters_rocm.hpp         # FirFilterROCm, IirFilterROCm
├── py_heterodyne.hpp           # HeterodyneDechirp (OpenCL)
├── py_heterodyne_rocm.hpp      # HeterodyneROCm
├── py_lch_farrow.hpp           # LchFarrow (OpenCL)
├── py_lch_farrow_rocm.hpp      # LchFarrowROCm
├── py_lfm_analytical_delay.hpp # LfmAnalyticalDelay
├── py_statistics.hpp           # StatisticsProcessor (ROCm)
├── py_vector_algebra_rocm.hpp  # CholeskyInverterROCm
└── CMakeLists.txt              # Сборка pybind11 модуля
```

---

## Архитектура

| Документ | Уровень | Описание |
|----------|---------|----------|
| [Architecture/Architecture_INDEX.md](Architecture/Architecture_INDEX.md) | — | Индекс всех архитектурных диаграмм |
| [Architecture/Architecture_C1_SystemContext.md](Architecture/Architecture_C1_SystemContext.md) | C1 | Акторы, внешние системы, границы |
| [Architecture/Architecture_C2_Container.md](Architecture/Architecture_C2_Container.md) | C2 | Контейнеры: модули, зависимости |
| [Architecture/Architecture_C3_Component.md](Architecture/Architecture_C3_Component.md) | C3 | Компоненты внутри модулей |
| [Architecture/Architecture_C4_Code.md](Architecture/Architecture_C4_Code.md) | C4 | Код: интерфейсы, UML |
| [Architecture/Architecture_DFD.md](Architecture/Architecture_DFD.md) | DFD | Потоки данных Level 0/1/2 |
| [Architecture/Architecture_Seq.md](Architecture/Architecture_Seq.md) | Seq | Диаграммы последовательностей |

---

## Дополнительная документация

| Документ | Описание |
|----------|----------|
| [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) | Оптимизация HIP/ROCm ядер: теория + паттерны + чеклист |
| [Examples/GPUProfiler_SetGPUInfo.md](../Examples/GPUProfiler_SetGPUInfo.md) | Как передать GPU info в профайлер |
| [Debian_Radeon9070_Setup.md](Debian_Radeon9070_Setup.md) | Настройка ROCm на Debian |
| [../ROCm_Setup_Instructions.md](../ROCm_Setup_Instructions.md) | Полная инструкция установки ROCm |
| [../configGPU.json](../configGPU.json) | Конфигурация GPU устройств |

---

## Структура проекта

```
GPUWorkLib/
├── Doc/                         # ← Эта папка (документация)
│   ├── INDEX.md                 #   Главный индекс (этот файл)
│   ├── Quick_Reference.md       #   Краткий справочник API
│   ├── Full_Reference.md        #   Полный справочник API
│   ├── Architecture/            #   C4 диаграммы
│   ├── DrvGPU/                  #   Документация ядра
│   ├── Modules/                 #   Документация C++ модулей
│   ├── Python/                  #   Python API документация
│   └── Python_test/             #   Описание Python тестов
├── DrvGPU/                      # Ядро: OpenCL/ROCm абстракция
├── modules/                     # Вычислительные модули
│   ├── fft_processor/
│   ├── statistics/
│   ├── vector_algebra/
│   ├── fft_maxima/
│   ├── filters/
│   ├── signal_generators/
│   ├── lch_farrow/
│   └── heterodyne/
├── python/                      # Python биндинги (pybind11)
├── Python_test/                 # Python тесты по модулям
├── Results/                     # Результаты тестов и профилирования
│   ├── Plots/                   #   Графики из Python тестов
│   ├── JSON/                    #   JSON результаты
│   └── Profiler/                #   Данные GPUProfiler
├── MemoryBank/                  # Управление проектом (задачи, спеки)
└── Examples/                    # Паттерны и примеры для разработки
```

---

*Last updated: 2026-03-03 | Maintained by: Кодо (AI Assistant)*
