# strategies module tests

## Test Architecture — OOP/SOLID/GRASP/GoF

Все тесты строятся по единой архитектуре:

```
ISignalStrategy (Strategy GoF)
  ├── SinSignalStrategy
  ├── LfmNoDelayStrategy
  ├── LfmWithDelayStrategy
  └── LfmFarrowStrategy

StrategyTestBase (Template Method GoF)
  Run() → Setup → GenerateSignals → PrepareMatrix → Execute → Validate → Save → Teardown
  ├── BaseStrategyTest      (T1: полный pipeline)
  ├── DebugStepTest         (T2: step-by-step)
  ├── StrategiesProfilingBenchmark (T3: GPUProfiler per step)
  └── TimingPerStepTest     (T4: hipEvent timing table)
```

---

## Test files

### Существующие

| Файл | Описание | Namespace |
|------|---------|-----------|
| `test_strategies_pipeline.hpp` | Full pipeline (5 ant, 8k pts, 12MHz) | `test_strategies` |
| `test_strategies_step_profiling.hpp` | GPU profiling per step (hipEvent + GPUProfiler) | `test_strategies_profiling` |
| `test_strategies_benchmark_streams.hpp` | Parallel stream benchmarking | `test_strategies_benchmark_streams` |

### Новые (OOP/SOLID/GRASP/GoF framework)

| Файл | Паттерн | Описание | Namespace |
|------|---------|---------|-----------|
| `antenna_test_params.hpp` | Information Expert (GRASP) | `AntennaTestParams`, `SignalVariant` | `test_strategies` |
| `i_signal_strategy.hpp` | Strategy (GoF) | `ISignalStrategy` interface | `test_strategies` |
| `signal_strategies.hpp` | Strategy (GoF) | Sin / LfmNoDelay / LfmWithDelay / LfmFarrow | `test_strategies` |
| `signal_strategy_factory.hpp` | Factory Method (GoF) | `SignalStrategyFactory::Create()` | `test_strategies` |
| `strategy_test_base.hpp` | Template Method (GoF) | `StrategyTestBase::Run()` skeleton | `test_strategies` |
| `base_strategy_test.hpp` | Controller (GRASP) | `BaseStrategyTest` — полный pipeline | `test_strategies` |
| `debug_step_test.hpp` | — | `DebugStepTest` — step-by-step | `test_strategies` |
| `strategies_profiling_benchmark.hpp` | Template Method | `StrategiesProfilingBenchmark` | `test_strategies` |
| `timing_per_step_test.hpp` | — | `TimingPerStepTest` — таблица timing | `test_strategies` |
| `test_base_strategy.hpp` | Composite (GoF) | runner: 4 сигнала × BaseStrategyTest | `test_base_strategy` |
| `test_debug_steps.hpp` | — | runner: DebugStepTest × 4 сигнала | `test_debug_steps` |

---

## Базовые параметры (TestStrategia.md)

| Параметр | Значение |
|----------|---------|
| n_ant (full) | 2500 |
| n_ant (small) | 100 (квадратная матрица, быстро) |
| n_samples | 5000 |
| fs | 0.5 МГц |
| fdev | 90 кГц |
| f0 | 100 кГц |
| tau_step | 2 мкс/антенну (для LFM_WITH_DELAY, LFM_FARROW) |
| W matrix | identity-like n_ant × n_ant |

> **TODO**: non-square W (2500×100) требует добавить `n_beams` в `AntennaProcessorConfig`.
> До этого — тесты используют квадратную матрицу `AntennaTestParams::Small()`.

---

## Варианты сигнала

| SignalVariant | Описание | fdev | tau_step |
|---------------|---------|------|---------|
| `SIN` | Синус (fdev=0) | 0 | 0 |
| `LFM_NO_DELAY` | ЛЧМ без задержек | 90 кГц | 0 |
| `LFM_WITH_DELAY` | ЛЧМ + линейные задержки | 90 кГц | 2 мкс |
| `LFM_FARROW` | ЛЧМ + дробные задержки (LchFarrowROCm) | 90 кГц | 2 мкс |

---

## 4 типа тестов

| Тест | Класс | Цель |
|------|-------|------|
| T1 | `BaseStrategyTest` | Полный pipeline, валидация freq + DR |
| T2 | `DebugStepTest` | Каждый Step отдельно + опц. запись файлов |
| T3 | `StrategiesProfilingBenchmark` | GPUProfiler per Step (PrintReport/JSON/MD) |
| T4 | `TimingPerStepTest` | hipEvent timing table → JSON для Python |

---

## Порядок тестирования (от простого к сложному)

| # | Тест | GPU? |
|---|------|------|
| P1 | Компиляция с новыми headers | Нет |
| P2 | `run_sin_only()` — smoke тест | Да |
| P3 | `run_all_variants()` — все 4 сигнала T1 | Да |
| P4 | `run_all()` (debug steps) — T2 | Да |
| P5 | `StrategiesProfilingBenchmark` — T3 | Да |
| P6 | `TimingPerStepTest` — T4, JSON export | Да |
| P7 | `python Python_test/strategies/test_timing_analysis.py` — анализ JSON | Нет |

---

## Вызов из all_test.hpp

```cpp
test_base_strategy::run_sin_only(backend);        // T1 smoke
test_base_strategy::run_all_variants(backend);    // T1 full
test_debug_steps::run_all(backend);               // T2

// С записью в файлы:
test_debug_steps::run_with_save(backend, SignalVariant::LFM_FARROW);
```

---

## Результаты

| Артефакт | Путь |
|----------|------|
| GPUProfiler отчёт (MD) | `Results/Profiler/strategies/YYYY-MM-DD.md` |
| GPUProfiler отчёт (JSON) | `Results/Profiler/strategies/YYYY-MM-DD.json` |
| Timing JSON (T4) | `Results/strategies/timing_{SIGNAL}.json` |
| Debug данные (T2) | `Results/strategies/debug/{test}_{signal}/` |
| Python графики | `Results/Plots/strategies/timing.png` |
