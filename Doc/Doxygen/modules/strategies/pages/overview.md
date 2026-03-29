@page strategies_overview strategies — Обзор модуля

@tableofcontents

@section strategies_overview_purpose Назначение

Полный pipeline **цифрового формирования диаграммы направленности** (Digital Beamforming):
сигнальная матрица [N_ant x M_samples] x весовая матрица → FFT спектры → post-FFT анализ.

7 шагов обработки, 4 сценария post-FFT, 3 checkpoint-а статистики, 4 параллельных HIP stream-а.

> **Namespace**: `strategies` | **Backend**: ROCm (hipBLAS, hipFFT) | **Статус**: Active

@section strategies_overview_classes Ключевые классы

| Класс | Описание |
|-------|----------|
| `AntennaProcessor_v1` | Главный фасад: 4 HIP streams, hipBLAS CGEMM, 7-step pipeline |
| `AntennaProcessorTest` | Debug-версия: step-by-step API для пошаговой отладки |
| `WeightGenerator` | Генерация delay-and-sum весовых матриц |
| `Pipeline` | Универсальный GPU pipeline orchestrator |
| `PipelineBuilder` | Builder pattern для конструирования pipeline |

@section strategies_overview_architecture Архитектура (7-step pipeline)

```
Signal [N_ant × M_samples] + Weights [N_ant × N_ant]
  │
  ▼
┌──────────────────────────────────────────────────┐
│ Step 1: Statistics(S)         → pre_input_stats  │
│         Входная статистика (mean, std, power)    │
├──────────────────────────────────────────────────┤
│ Step 2: hipBLAS CGEMM         X = W · S          │
│         Формирование лучей                       │
├──────────────────────────────────────────────────┤
│ Step 3: Statistics(X)         → post_gemm_stats  │
│         Статистика после формирования            │
├──────────────────────────────────────────────────┤
│ Step 4: Hamming window + zero-pad + hipFFT       │
│         Спектральный анализ                      │
├──────────────────────────────────────────────────┤
│ Step 5: Statistics(|spectrum|) → post_fft_stats  │
│         Статистика спектра                       │
├──────────────────────────────────────────────────┤
│ Step 6: Post-FFT scenario                        │
│         OneMax / AllMaxima / MinMax               │
├──────────────────────────────────────────────────┤
│ Step 7: Sync and return                          │
│         Синхронизация 4 HIP streams              │
└──────────────────────────────────────────────────┘
```

@section strategies_overview_scenarios Post-FFT сценарии

| Сценарий | Описание |
|----------|----------|
| `OneMax` | Один максимум спектра с параболической интерполяцией |
| `AllMaxima` | Все максимумы выше порога |
| `GlobalMinMax` | Глобальные min/max → динамический диапазон (дБ) |
| `ALL_REQUIRED` | Выполнить все три сценария |

@section strategies_overview_streams 4 HIP Streams

Модуль использует 4 параллельных HIP stream-а для максимальной загрузки GPU:
- **Stream 0**: основной pipeline (CGEMM + FFT)
- **Stream 1**: статистика pre_input
- **Stream 2**: статистика post_gemm
- **Stream 3**: статистика post_fft

@section strategies_overview_quickstart Быстрый старт

@subsection strategies_overview_quickstart_cpp C++

@code{.cpp}
#include "modules/strategies/include/antenna_processor_v1.hpp"

strategies::AntennaProcessorConfig cfg;
cfg.n_ant = 256;            // количество антенн
cfg.n_samples = 1200000;    // отсчётов
cfg.sample_rate = 12e6;     // частота дискретизации (Гц)
cfg.scenario_mode = strategies::PostFftScenarioMode::ALL_REQUIRED;

strategies::AntennaProcessor_v1 proc(backend);
proc.Initialize(cfg);

auto result = proc.Process(signal, weights);
// result.pre_input_stats  — статистика входа
// result.post_gemm_stats  — статистика после CGEMM
// result.post_fft_stats   — статистика спектра
// result.one_max_results  — OneMax (частота, амплитуда, интерполяция)
// result.all_maxima       — все максимумы выше порога
// result.minmax           — GlobalMinMax (min, max, dynamic range dB)
@endcode

@subsection strategies_overview_quickstart_python Python

@code{.py}
from Python_test.strategies.pipeline_runner import PipelineRunner, PipelineConfig

config = PipelineConfig(n_ant=5, n_samples=8000, fs=12e6)
runner = PipelineRunner(config)
result = runner.run(signal, weights)
# result.pre_input_stats, result.post_fft_stats, ...
@endcode

@section strategies_overview_patterns GoF паттерны

Модуль активно использует паттерны проектирования:

| Паттерн | Реализация |
|---------|------------|
| **Strategy** (GoF) | `ISignalStrategy` — подключаемые алгоритмы post-FFT |
| **Template Method** (GoF) | `StrategyTestBase` — шаблон для тестов |
| **Builder** (GoF) | `PipelineBuilder` — конструирование pipeline |
| **Factory Method** (GoF) | Создание сценариев по enum |
| **Controller** (GRASP) | `AntennaProcessor_v1` — координация шагов |

@section strategies_overview_dependencies Зависимости

- **DrvGPU** — ROCm backend, GpuContext, GPUProfiler
- **hipBLAS** — CGEMM для матричного умножения
- **hipFFT** — спектральный анализ
- **statistics** — checkpoint статистика (mean, std, power)

@see strategies_formulas
@see strategies_tests
