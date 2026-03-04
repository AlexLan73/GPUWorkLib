# GPUWorkLib — GPU-библиотека для цифровой обработки сигналов

**GPUWorkLib** — модульная библиотека GPU-вычислений для задач ЦОС: FFT, фильтрация, статистика, гетеродин, генераторы сигналов, корреляция. Поддерживает бэкенды OpenCL и ROCm/HIP.

---

## Модули

| Модуль | Каталог | Классы | Backend | Python API | Статус |
|--------|---------|--------|---------|------------|--------|
| **DrvGPU** | `DrvGPU/` | `GPUContext`, `ROCmGPUContext`, `IBackend` | OpenCL, ROCm | `GPUContext`, `ROCmGPUContext` | ✅ Активен |
| **SignalGenerators** | `modules/signal_generators/` | `CwGenerator`, `LfmGenerator`, `NoiseGenerator`, `FormSignalGenerator` | OpenCL, ROCm | `SignalGenerator` | ✅ Активен |
| **FFTProcessor** | `modules/fft_processor/` | `FFTProcessor` | OpenCL (clFFT), ROCm (hipFFT) | `FFTProcessor` | ✅ Активен |
| **FFTMaxima** | `modules/fft_maxima/` | `SpectrumMaximaFinder`, `SpectrumProcessorOpenCL`, `SpectrumProcessorROCm` | OpenCL, ROCm | частично | ✅ Активен |
| **Filters** | `modules/filters/` | `FirFilter`, `IirFilter`, `FirFilterROCm`, `IirFilterROCm`, `MovingAverageFilterROCm`, `KalmanFilterROCm`, `KaufmanFilterROCm` | OpenCL, ROCm | `FirFilter`, `IirFilter` | ✅ Активен |
| **Heterodyne** | `modules/heterodyne/` | `HeterodyneDechirp`, `HeterodyneProcessorOpenCL`, `HeterodyneProcessorROCm` | OpenCL, ROCm | `HeterodyneDechirp` | ✅ Активен |
| **Statistics** | `modules/statistics/` | `StatisticsProcessor` | ROCm only | `StatisticsProcessor` | ✅ Активен |
| **VectorAlgebra** | `modules/vector_algebra/` | `VectorAlgebra` (Cholesky POTRF/POTRI) | ROCm only | `VectorAlgebra` | ✅ Активен |
| **LchFarrow** | `modules/lch_farrow/` | `LchFarrow`, `LchFarrowROCm` | OpenCL, ROCm | `LchFarrow` | ✅ Активен |
| **FMCorrelator** | `modules/fm_correlator/` | `FMCorrelator` | ROCm only | `FMCorrelatorROCm` | 🟡 Planned |

---

## Структура проекта

```
GPUWorkLib/
├── CMakeLists.txt             # Корневой CMake — add_subdirectory() на все модули
├── CMakePresets.json          # Конфигурации сборки (Radeon9070, AI100, ...)
├── DrvGPU/config/configGPU.json  # Конфигурация GPU (is_prof, is_logger, ...)
├── run_test                   # Скрипт запуска тестов (C++ + Python)
│
├── DrvGPU/                    # Базовый GPU-драйвер
│   ├── backends/              # OpenCL и ROCm реализации IBackend
│   ├── services/              # GPUProfiler, GpuBenchmarkBase, BatchManager, ConsoleOutput
│   ├── config/                # GPUConfig (configGPU.json)
│   └── tests/
│
├── modules/                   # Вычислительные модули
│   ├── fft_maxima/            # SpectrumMaximaFinder (OnePeak, AllMaxima)
│   ├── fft_processor/         # FFTProcessor (Complex / MagPhase)
│   ├── signal_generators/     # CW, LFM, Noise, Form, Script
│   ├── filters/               # FIR, IIR, SMA/EMA/DEMA/TEMA, Kalman, KAMA
│   ├── heterodyne/            # LFM Dechirp pipeline
│   ├── statistics/            # Welford mean/var, медиана, radix sort (ROCm)
│   ├── vector_algebra/        # Cholesky POTRF/POTRI (ROCm + rocBLAS/rocSOLVER)
│   ├── lch_farrow/            # Lagrange fractional delay 48×5
│   └── fm_correlator/         # ФМ-корреляция M-последовательностями (ROCm)
│
├── src/
│   └── main.cpp               # Точка входа C++ тестов (вызывает all_test.hpp каждого модуля)
│
├── python/                    # Python bindings (pybind11)
│   ├── gpu_worklib_bindings.cpp
│   ├── py_heterodyne.hpp
│   ├── py_filters.hpp
│   ├── py_lch_farrow.hpp
│   └── py_fm_correlator_rocm.hpp
│
├── Python_test/               # Python тесты по модулям
│   ├── filters/
│   ├── fft_maxima/
│   ├── signal_generators/
│   ├── statistics/
│   ├── heterodyne/
│   ├── lch_farrow/
│   ├── vector_algebra/
│   └── fm_correlator/
│
├── scripts/
│   ├── run_agent_tests.sh     # Основной test runner (C++ + Python)
│   └── run_agent_tests.py     # Python-часть test runner
│
├── Results/                   # Артефакты тестов
│   ├── Plots/                 # Графики Python тестов (по модулям)
│   └── Profiler/              # Отчёты GPUProfiler (.md, .json)
│
├── Logs/                      # Per-GPU логи (plog)
│   └── DRVGPU_00/
│
├── Doc/                       # Документация
│   ├── Architecture/          # C1-C4, DFD, Seq диаграммы
│   ├── Modules/               # Full.md + Quick.md по каждому модулю
│   └── Python/                # Python API документация
│
└── Doc_Addition/              # Доп. документация (ROCm guide, планы)
```

---

## Сборка

### Требования

- Debian Linux
- CMake 3.20+
- GCC / g++, C++17
- ROCm 7.2+ (`/opt/rocm`) + OpenCL (входит в ROCm)

### Конфигурации (CMakePresets.json)

| Preset | Платформа | Build dir |
|--------|-----------|-----------|
| `Radeon9070` | AMD Radeon 9070 + ROCm 7.2 | `build/Radeon9070/` |
| `AI100` | AMD AI100 + ROCm 7.2/7.5 | `build/AI100/` |

### Полная сборка

```bash
cmake --preset Radeon9070
cmake --build build/Radeon9070 -j$(nproc)
```

### Debug-сборка

```bash
cmake --preset Radeon9070-Debug
cmake --build build/Radeon9070-debug -j$(nproc)
```

---

## Выборочная сборка — отключение модулей

По умолчанию **все модули собираются вместе**. Для рабочего режима, когда часть модулей не нужна (экономия времени компиляции, уменьшение бинаря), нужно закомментировать соответствующие строки в корневом `CMakeLists.txt`:

```cmake
# CMakeLists.txt — закомментируй ненужные модули:

add_subdirectory(DrvGPU)            # ⛔ Нельзя отключать — базовый драйвер

add_subdirectory(modules/fft_maxima)
add_subdirectory(modules/fft_processor)
add_subdirectory(modules/lch_farrow)
add_subdirectory(modules/filters)
add_subdirectory(modules/statistics)
add_subdirectory(modules/vector_algebra)
add_subdirectory(modules/fm_correlator)

# Эти два можно отключить, если не нужны:
# add_subdirectory(modules/heterodyne)       # ← закомментировать
# add_subdirectory(modules/signal_generators) # ← закомментировать
```

> **Как работает**: `src/CMakeLists.txt` использует `if(TARGET GPUWorkLib::heterodyne)` перед линковкой — если модуль не собран, он просто не линкуется. Ошибок компиляции не будет.

> **⚠️ Важно**: если в `src/main.cpp` (или подключённых `all_test.hpp`) есть `#include` заголовков отключённого модуля — их нужно закомментировать вручную или обернуть в `#ifdef`.

**Пример**: только FFT + Filters, без генераторов и гетеродина:

```cmake
add_subdirectory(DrvGPU)
add_subdirectory(modules/fft_maxima)
add_subdirectory(modules/fft_processor)
add_subdirectory(modules/filters)
add_subdirectory(modules/statistics)
# add_subdirectory(modules/signal_generators)  # отключено
# add_subdirectory(modules/heterodyne)          # отключено
# add_subdirectory(modules/lch_farrow)          # отключено
# add_subdirectory(modules/vector_algebra)      # отключено
# add_subdirectory(modules/fm_correlator)       # отключено (ROCm-only, planned)
```

---

## Сборка Python-биндингов

Python-биндинги **отключены по умолчанию** (`BUILD_PYTHON=OFF`). Включить:

```bash
# Сборка с Python-биндингами
cmake --preset Radeon9070 -DBUILD_PYTHON=ON
cmake --build build/Radeon9070 -j$(nproc)
```

Собранный модуль:
```
build/Radeon9070/python/gpuworklib.cpython-312-x86_64-linux-gnu.so
```

Подключение в Python:
```python
import sys
sys.path.insert(0, "build/Radeon9070/python")
import gpuworklib as gw

# OpenCL-контекст
ctx = gw.GPUContext(0)

# ROCm-контекст
ctx_rocm = gw.ROCmGPUContext(0)

# FIR фильтр (OpenCL)
fir = gw.FirFilter(ctx, coeffs)
result = fir.process(signal)

# FFT (OpenCL)
fft = gw.FFTProcessor(ctx, nfft=4096)
spectrum = fft.process(signal)

# Heterodyne Dechirp (OpenCL)
het = gw.HeterodyneDechirp(ctx)
het.set_params(f_start=0, f_end=1e6, sample_rate=12e6, num_samples=4000, num_antennas=5)
result = het.process(rx_signal)
```

> Полная документация Python API: [`Doc/Python/`](Doc/Python/)

---

## Запуск тестов

```bash
./run_test                                # все модули
./run_test filters                        # один модуль
./run_test fft_processor
./run_test heterodyne
./run_test --file config/tests_order.txt  # из файла (порядок соблюдается)
```

### Порядок модулей при `all`

1.  `drvgpu`
2.  `fft_processor`
3.  `statistics`
4.  `vector_algebra`
5.  `fft_maxima`
6.  `filters`
7.  `signal_generators`
8.  `lch_farrow`
9.  `heterodyne`
10. `fm_correlator`

### Что происходит при запуске

```
./run_test filters
  └─ scripts/run_agent_tests.sh filters
       ├─ cmake --build (incremental)
       ├─ build/GPUWorkLib filters        # C++ тесты
       └─ python3 scripts/run_agent_tests.py filters
            └─ pytest Python_test/filters/ -v -s
```

### Поведение по типу GPU

| GPU | Пропускается |
|-----|-------------|
| AMD | clFFT-тесты (не поддерживает RDNA4+, gfx1201) |
| NVIDIA | ROCm-тесты |

---

## Конфигурация GPU (`configGPU.json`)

```json
{
  "description": "GPU Configuration for DrvGPU",
  "version": "1.0",
  "gpus": [
    {
      "id": 0,
      "name": "TEST",
      "is_active": true,
      "is_prof": true,
      "is_logger": true,
      "is_console": true,
      "is_db": false,
      "max_memory_percent": 70,
      "log_level": "INFO"
    }
  ]
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | int | Индекс GPU (с 0) |
| `name` | string | Имя GPU — выводится в логах и консоли |
| `is_active` | bool | Инициализировать при старте |
| `is_prof` | bool | Включить GPUProfiler |
| `is_logger` | bool | Включить файловые логи (`Logs/DRVGPU_XX/`) |
| `is_console` | bool | Вывод в консоль (мультиGPU-безопасный) |
| `is_db` | bool | Вывод в БД (зарезервировано) |
| `max_memory_percent` | int | Макс. использование памяти GPU, % |
| `log_level` | string | Уровень логирования: `DEBUG` (всё) → `INFO` (нормальная работа) → `WARNING` (только предупреждения+ошибки) → `ERROR` (только фатальные). Каждый уровень включает все уровни выше. |

---

## Артефакты тестов

| Тип | Путь |
|-----|------|
| Графики Python | `Results/Plots/<module>/` |
| Профайлер GPU (JSON/MD) | `Results/Profiler/<GPU_ID_ModuleName>/` |
| Логи | `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log` |

---

## Документация

| Раздел | Путь |
|--------|------|
| Индекс архитектуры | [`Doc/Architecture/Architecture_INDEX.md`](Doc/Architecture/Architecture_INDEX.md) |
| Архитектура DrvGPU | [`Doc/DrvGPU/Architecture.md`](Doc/DrvGPU/Architecture.md) |
| Модуль FFT Maxima | [`Doc/Modules/fft_maxima/Full.md`](Doc/Modules/fft_maxima/Full.md) |
| Модуль FFT Processor | [`Doc/Modules/fft_processor/Full.md`](Doc/Modules/fft_processor/Full.md) |
| Модуль Signal Generators | [`Doc/Modules/signal_generators/Full.md`](Doc/Modules/signal_generators/Full.md) |
| Модуль Filters | [`Doc/Modules/filters/Full.md`](Doc/Modules/filters/Full.md) |
| Модуль Heterodyne | [`Doc/Modules/heterodyne/Full.md`](Doc/Modules/heterodyne/Full.md) |
| Модуль Statistics | [`Doc/Modules/statistics/Full.md`](Doc/Modules/statistics/Full.md) |
| Модуль Vector Algebra | [`Doc/Modules/vector_algebra/Full.md`](Doc/Modules/vector_algebra/Full.md) |
| Модуль LCH Farrow | [`Doc/Modules/lch_farrow/Full.md`](Doc/Modules/lch_farrow/Full.md) |
| Модуль FM Correlator | [`Doc/Modules/fm_correlator/Full.md`](Doc/Modules/fm_correlator/Full.md) |
| ROCm/HIP оптимизация | [`Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`](Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) |

---

**Статус**: активная разработка
**Платформа**: Debian Linux, AMD GPU (ROCm 7.2+)
