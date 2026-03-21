# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-03-15
> **Статус**: Ref03-B/E/C — написаны, ждут GPU-тестирования. Тест-архитектура strategies создана.

---

## 🔥 БЛИЖАЙШИЙ ПОНЕДЕЛЬНИК — Порядок тестирования

> Полный план: `MemoryBank/tasks/TEST_ORDER_PLAN.md`

```
УРОВЕНЬ 1 → DrvGPU         ./gpu_work_lib drvgpu          (18 тестов)
УРОВЕНЬ 2 → statistics      ./gpu_work_lib statistics       (T1-T11)
УРОВЕНЬ 3 → fft_func        ./gpu_work_lib fft_func         (L3-1..L3-3)
УРОВЕНЬ 4 → signal_generators (ПИСАТЬ ТЕСТЫ + запускать)    (L4-1..L4-4)
УРОВЕНЬ 5 → lch_farrow      ./gpu_work_lib lch_farrow       (регрессия)
УРОВЕНЬ 6 → fft_maxima      (external d_spectrum consumers)
УРОВЕНЬ 7 → strategies      ./gpu_work_lib strategies       (L7-1..L7-9)
```

---

## ⏳ Написано, НЕ тестировано на GPU

### Ref03-B — Statistics (6 Op-классов)
- `MeanReductionOp`, `WelfordFusedOp`, `WelfordFloatOp`
- `MedianRadixSortOp`, `MedianHistogramOp`, `MedianHistogramComplexOp`
- `StatisticsProcessor` — thin Facade (-75% кода)
- **Тест**: `./gpu_work_lib statistics` → T1-T11 PASSED?

### Ref03-E — fft_func / fft_processor
- `PadDataOp` + `MagPhaseOp`, `FftProcessorROCm` thin Facade (-46% кода)
- **Тест**: `./gpu_work_lib fft_func` → все PASSED?

### Ref03-C — strategies pipeline infrastructure
- 6 Step-классов: `PrepareInputStep`, `GemmStep`, `WindowFftStep`, `OneMaxStep`, `AllMaximaStep`, `MinMaxStep`
- `AntennaProcessorPipeline` с `IPipelineStep` + `PipelineBuilder`
- **Тест**: Part of L7-2..L7-9

### Signal Generators ROCm port (commit `0e6e395`)
- `CwGeneratorROCm`, `LfmGeneratorROCm`, `NoiseGeneratorROCm`
- HIP kernels: `__sincosf`, Philox PRNG, Box-Muller
- **Нужно НАПИСАТЬ тесты**: `test_cw_rocm.hpp`, `test_lfm_rocm.hpp`, `test_noise_rocm.hpp`
- **Тест**: `./gpu_work_lib signal_generators` → GPU vs CPU max_err < 0.01

---

## ✅ Написана в сессии 2026-03-15 — Тест-архитектура strategies

**C++ `modules/strategies/tests/` (11 новых файлов)**:
```
antenna_test_params.hpp       ← AntennaTestParams + SignalVariant enum
i_signal_strategy.hpp         ← ISignalStrategy interface (Strategy GoF)
signal_strategies.hpp         ← Sin/LfmNoDelay/LfmDelay/LfmFarrow
signal_strategy_factory.hpp   ← Factory Method GoF
strategy_test_base.hpp        ← Template Method GoF (Run skeleton)
base_strategy_test.hpp        ← T1: полный pipeline test
debug_step_test.hpp           ← T2: step-by-step AntennaProcessorTest
strategies_profiling_benchmark.hpp  ← T3: GPUProfiler per step
timing_per_step_test.hpp      ← T4: hipEvent timing table → JSON
test_base_strategy.hpp        ← runner: 4 сигнала × BaseStrategyTest
test_debug_steps.hpp          ← runner: DebugStepTest × 4 сигнала
```

**Python `Python_test/strategies/` (5 новых файлов)**:
```
test_params.py                ← AntennaTestParams dataclass + SignalVariant
signal_generators_strategy.py ← ISignalStrategy + 4 реализации + Factory
strategy_test_base.py         ← StrategyTestBase (extends TestBase)
test_base_pipeline.py         ← TestRunner T1: 4 варианта, без GPU ✅
test_debug_steps.py           ← TestRunner T2: per-step validation, без GPU ✅
test_timing_analysis.py       ← парсинг JSON + bar chart (после L7-9)
```

---

## TODO после тестирования

- [ ] Non-square матрица 2500×100: добавить `n_beams` в `AntennaProcessorConfig`
- [ ] Ref03-C Facade rewrite: `antenna_processor_v1.hpp+.cpp` → Pipeline delegation
- [ ] Python TestRunner: P1-P7 можно запустить на Windows уже сейчас (без GPU)

*Последнее обновление: 2026-03-15*
