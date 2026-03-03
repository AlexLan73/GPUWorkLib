# GPUWorkLib — GPU-библиотека для цифровой обработки сигналов

**GPUWorkLib** — модульная библиотека GPU-вычислений для задач ЦОС: FFT, фильтрация, статистика, гетеродин, генераторы сигналов. Поддерживает бэкенды OpenCL и ROCm/HIP.

---

## Модули

| Модуль | Класс | Backend | Python API | Статус |
|--------|-------|---------|------------|--------|
| **DrvGPU** | `GPUManager`, `IBackend` | OpenCL, ROCm | `GPUContext` | ✅ Активен |
| **SignalGenerators** | `CwGenerator`, `LfmGenerator`, `NoiseGenerator`, `FormSignalGenerator` | OpenCL, ROCm | `SignalGenerator` | ✅ Активен |
| **FFTProcessor** | `FFTProcessor` | OpenCL, ROCm (hipFFT) | `FFTProcessor` | ✅ Активен |
| **SpectrumMaximaFinder** | `SpectrumMaximaFinder` | OpenCL, ROCm | частично | ✅ Активен |
| **Statistics** | `StatisticsProcessor` | ROCm | `StatisticsProcessor` | ✅ Активен |
| **Heterodyne** | `HeterodyneProcessorOpenCL`, `HeterodyneProcessorROCm` | OpenCL, ROCm | `HeterodyneDechirp` | ✅ Активен |
| **Filters** | `FirFilter`, `IirFilter`, `FirFilterROCm`, `IirFilterROCm` | OpenCL, ROCm | `FirFilter`, `IirFilter` | ✅ Активен |
| **VectorAlgebra** | `VectorOpsModule`, матрица, Холецкий | ROCm | `VectorAlgebra` | ✅ Активен |
| **LchFarrow** | `LchFarrowProcessor` | OpenCL, ROCm | `LchFarrow` | ✅ Активен |

---

## Структура проекта

```
GPUWorkLib/
├── CMakePresets.json          # Конфигурации сборки
├── CMakeLists.txt
├── configGPU.json             # Конфигурация GPU (is_prof, is_logger, ...)
├── run_test                   # Запуск тестов (точка входа)
│
├── DrvGPU/                    # Базовый GPU-драйвер (OpenCL + ROCm бэкенды)
│   ├── backends/              # OpenCL и ROCm реализации
│   ├── services/              # GPUProfiler, GpuBenchmarkBase, Logger
│   ├── config/                # GPUConfig (configGPU.json)
│   └── tests/
│
├── modules/                   # Вычислительные модули
│   ├── filters/
│   ├── fft_processor/
│   ├── fft_maxima/
│   ├── signal_generators/
│   ├── statistics/
│   ├── heterodyne/
│   ├── vector_algebra/
│   └── lch_farrow/
│
├── src/
│   └── main.cpp               # Точка входа C++ тестов
│
├── Python_test/               # Python тесты по модулям
│   ├── filters/
│   ├── fft_maxima/
│   ├── signal_generators/
│   ├── statistics/
│   ├── heterodyne/
│   ├── lch_farrow/
│   └── vector_algebra/
│
├── scripts/
│   ├── run_agent_tests.sh     # Основной test runner (C++ + Python)
│   └── run_agent_tests.py     # Python-часть test runner
│
├── Results/                   # Артефакты тестов
│   ├── Plots/                 # Графики Python тестов
│   │   ├── filters/
│   │   ├── fft_maxima/
│   │   ├── signal_generators/
│   │   ├── heterodyne/
│   │   └── ...
│   └── Profiler/              # Отчёты GPUProfiler (.md, .json)
│       ├── GPU_00_FirFilter/
│       ├── GPU_00_FFT/
│       └── ...
│
├── Logs/                      # Per-GPU логи (plog)
│   └── DRVGPU_00/
│
├── Doc/                       # Документация модулей
├── Doc_Addition/              # Доп. документация (ROCm guide, планы)
└── MemoryBank/                # Задачи, спецификации, история сессий
```

---

## Сборка

### Требования

- Debian Linux
- CMake 3.20+
- GCC / g++, C++17
- ROCm 7.2+ (`/opt/rocm`)
- OpenCL (входит в ROCm)

### Конфигурации

| Preset | Платформа | build dir |
|--------|-----------|-----------|
| `Radeon9070` | Debian + AMD Radeon 9070 + ROCm 7.2 + OpenCL | `build/Radeon9070/` |
| `AI100` | Debian + AMD AI100 + ROCm 7.2/7.5 + OpenCL | `build/AI100/` |

### Команды сборки

```bash
# AMD Radeon 9070
cmake --preset Radeon9070
cmake --build build/Radeon9070 -j$(nproc)

# AMD AI100
cmake --preset AI100
cmake --build build/AI100 -j$(nproc)

# Debug (флаг к любому пресету)
cmake --preset Radeon9070-Debug
cmake --build build/Radeon9070-debug -j$(nproc)
```

> По умолчанию `./run_test` ищет бинарник в `build/`. Для другого каталога:
> ```bash
> BUILD_DIR=build/Radeon9070 ./run_test filters
> ```

---

## Запуск тестов

```bash
./run_test                                # все модули
./run_test all                            # все модули (то же самое)
./run_test filters                        # один модуль
./run_test fft_processor
./run_test --file config/tests_order.txt  # из файла (порядок соблюдается)
```

### Порядок модулей при `all`

1. `drvgpu`
2. `fft_processor`
3. `statistics`
4. `vector_algebra`
5. `fft_maxima`
6. `filters`
7. `signal_generators`
8. `lch_farrow`
9. `heterodyne`

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
| AMD | clFFT-тесты (устаревшая библиотека, не поддерживает RDNA4+) |
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
| `log_level` | string | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Артефакты тестов

| Тип | Путь |
|-----|------|
| Графики Python | `Results/Plots/<module>/` |
| Профайлер GPU (JSON/MD) | `Results/Profiler/<GPU_ID_ModuleName>/` |
| Логи | `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log` |

---

## Python API

Python-биндинги собираются при `cmake --preset ...` (pybind11). Путь к `.so`:

```
build/<preset>/python/gpuworklib.*.so
```

Пример использования:

```python
import sys
sys.path.insert(0, "build/Radeon9070/python")
import gpuworklib as gw

ctx = gw.GPUContext(0)

# FIR фильтр
fir = gw.FirFilter(ctx, coeffs)
result = fir.process(signal)

# FFT
fft = gw.FFTProcessor(ctx, nfft=4096)
spectrum = fft.process(signal)
```

Полная документация Python API: [`Doc/Python/`](Doc/Python/)

---

## Документация

| Раздел | Путь |
|--------|------|
| Архитектура DrvGPU | [`Doc/DrvGPU/Architecture.md`](Doc/DrvGPU/Architecture.md) |
| Модуль Filters | [`Doc/Modules/filters/Full.md`](Doc/Modules/filters/Full.md) |
| Модуль FFTProcessor | [`Doc/Modules/fft_processor/Full.md`](Doc/Modules/fft_processor/Full.md) |
| Модуль SignalGenerators | [`Doc/Modules/signal_generators/Full.md`](Doc/Modules/signal_generators/Full.md) |
| ROCm/HIP оптимизация | [`Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`](Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) |
| Python Bindings | [`Doc/Modules/python_bindings/README.md`](Doc/Modules/python_bindings/README.md) |

---

**Статус**: активная разработка
**Платформа**: Debian Linux, AMD GPU (ROCm 7.2+)
