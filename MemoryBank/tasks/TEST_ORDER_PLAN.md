# Порядок тестирования — от простого к сложному

> **Дата**: 2026-03-15
> **Ветка**: `main`
> **GPU**: AMD Radeon 9070 (gfx1201), ROCm 7.2+

Тестируем в строгом порядке: каждый следующий модуль ЗАВИСИТ от предыдущих.
Если тест на уровне N падает — решаем ДО перехода к N+1.

---

## Слои зависимостей

```
DrvGPU (фундамент)
  └─ statistics (Ref03-B)
  └─ fft_func / fft_processor (Ref03-E)
  └─ signal_generators ROCm
        └─ lch_farrow ROCm
        └─ fft_maxima (AllMaxima/OneMax/MinMax)
              └─ strategies (использует ВСЁ выше)
```

---

## УРОВЕНЬ 1 — DrvGPU (фундамент)

> Цель: убедиться что GpuContext, BufferSet, GpuKernelOp работают.
> Без этого ничего не работает.

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L1-1 | `cmake .. -DENABLE_ROCM=ON && make -j$(nproc)` | — | Компиляция без ошибок |
| L1-2 | external context ROCm | `DrvGPU/tests/test_rocm_external_context.hpp` | InitializeFromExternalStream |
| L1-3 | external context Hybrid | `DrvGPU/tests/test_hybrid_external_context.hpp` | OpenCL+ROCm |
| L1-4 | DrvGPU external factories | `DrvGPU/tests/test_drv_gpu_external.hpp` | Static factories |

```bash
./gpu_work_lib drvgpu
```

**Критерий**: 3×6 = 18 тестов PASSED ✅

---

## УРОВЕНЬ 2 — statistics (Ref03-B — написано, НЕ тестировано)

> 6 новых Op-классов: MeanReductionOp, WelfordFusedOp, WelfordFloatOp,
> MedianRadixSortOp, MedianHistogramOp, MedianHistogramComplexOp
> + StatisticsProcessor thin Facade

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L2-1 | T1-T4: Welford fused | `tests/test_statistics_rocm.hpp` | Mean/Std через kernel |
| L2-2 | T5-T7: RadixSort median | `tests/test_statistics_rocm.hpp` | Median через sort |
| L2-3 | T8-T11: Histogram median | `tests/test_statistics_rocm.hpp` | Median histogram |

```bash
./gpu_work_lib statistics
```

**Критерий**: T1-T11 PASSED ✅

> ⚠️ Histogram median (T8-T11) — ещё не тестировался на GPU!

---

## УРОВЕНЬ 3 — fft_func / fft_processor (Ref03-E — написано, НЕ тестировано)

> PadDataOp + MagPhaseOp, FftProcessorROCm thin Facade (-46% кода)

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L3-1 | FFT processor ROCm | `fft_func/tests/test_fft_processor_rocm.hpp` | hipFFT + magnitudes |
| L3-2 | AllMaxima pipeline | `fft_func/tests/test_all_maxima_rocm.hpp` | Spectrum maxima finding |
| L3-3 | Timing benchmark | `fft_func/tests/test_fft_benchmark_rocm.hpp` | Нет regression vs baseline |

```bash
./gpu_work_lib fft_func
```

**Критерий**: все тесты PASSED, timing ≈ pre-Ref03-E baseline ✅

---

## УРОВЕНЬ 4 — signal_generators ROCm (написано, НЕ тестировано)

> CwGeneratorROCm, LfmGeneratorROCm, NoiseGeneratorROCm
> HIP kernels: sincos→__sincosf, Philox PRNG, Box-Muller

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L4-1 | CW generator ROCm | `tests/test_cw_rocm.hpp` | GPU vs CPU reference |
| L4-2 | LFM generator ROCm | `tests/test_lfm_rocm.hpp` | Chirp + FFT peak |
| L4-3 | Noise generator ROCm | `tests/test_noise_rocm.hpp` | PRNG distribution |
| L4-4 | FormSignalGeneratorROCm | `tests/test_form_signal_rocm.hpp` | FormParams → GPU signal |

```bash
./gpu_work_lib signal_generators
```

**Критерий**: CW/LFM/Noise PASSED, GPU vs CPU max_err < 0.01 ✅

> ⚠️ Нужно написать L4-1..L4-4 (файлы не существуют ещё!)
> Порядок: написать прямо на GPU → compile → run

---

## УРОВЕНЬ 5 — lch_farrow ROCm (уже тестировался, верификация)

> LchFarrowROCm — дробные задержки Лагранжем 48×5

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L5-1 | lch_farrow baseline | `tests/test_lch_farrow_rocm.hpp` | GPU vs CPU |
| L5-2 | Benchmark regression | `tests/test_lch_farrow_benchmark_rocm.hpp` | Нет regression |

```bash
./gpu_work_lib lch_farrow
```

**Критерий**: все PASSED ✅ (должны работать — уже тестировались ранее)

---

## УРОВЕНЬ 6 — fft_maxima (AllMaxima / OneMax / MinMax для strategies)

> `modules/fft_func` содержит AllMaximaPipelineROCm.
> Эти тесты проверяют именно consumers которые strategies использует.

| # | Тест | Файл | Что проверяет |
|---|------|------|--------------|
| L6-1 | AllMaxima от external d_spectrum | `fft_func/tests/test_all_maxima_ext.hpp` | Spectrum → peaks |
| L6-2 | OneMax parabola | `fft_func/tests/` | Peak + parabolic interp |
| L6-3 | GlobalMinMax | `fft_func/tests/` | Min/max + dynamic range |

**Критерий**: Step2.1/2.2/2.3 работают от external d_spectrum ✅

---

## УРОВЕНЬ 7 — strategies (новая тестовая архитектура)

> Все зависимости проверены. Теперь полный pipeline strategies.

### Шаг 7.0: Компиляция с новыми test headers

```bash
cmake .. -DENABLE_ROCM=ON && make -j$(nproc) 2>&1 | head -50
```

Новые файлы в `modules/strategies/tests/`:
- `antenna_test_params.hpp`, `i_signal_strategy.hpp`
- `signal_strategies.hpp`, `signal_strategy_factory.hpp`
- `strategy_test_base.hpp`, `base_strategy_test.hpp`
- `debug_step_test.hpp`, `test_debug_steps.hpp`
- `strategies_profiling_benchmark.hpp`, `timing_per_step_test.hpp`
- `test_base_strategy.hpp`

### Порядок тестов strategies (от простого к сложному)

| # | Тест | C++ вызов | GPU? | Что проверяет |
|---|------|----------|------|--------------|
| L7-1 | Компиляция | `make` | Нет | Include paths, types |
| L7-2 | Smoke SIN | `run_sin_only(backend)` | Да | Полный pipeline SIN |
| L7-3 | Существующий full_pipeline | `test_full_pipeline(backend)` | Да | Regression |
| L7-4 | External weights | `test_external_weights(backend)` | Да | Managed weights |
| L7-5 | Base×4 variants | `run_all_variants(backend)` | Да | SIN/LFM/delay/Farrow |
| L7-6 | Debug steps | `run_all(backend)` | Да | Per-step validation |
| L7-7 | Benchmark streams | `run_benchmark_streams(backend)` | Да | Parallel streams |
| L7-8 | Profiling | `StrategiesProfilingBenchmark` | Да | GPUProfiler report |
| L7-9 | Timing table | `TimingPerStepTest` | Да | hipEvent JSON export |

### Python тесты strategies (не требуют GPU!)

| # | Тест | Команда | GPU? |
|---|------|---------|------|
| P1 | AntennaTestParams | `pytest test_params.py -v` | Нет |
| P2 | NumPy pipeline SIN | `pytest test_base_pipeline.py::test_sin_full_pipeline` | Нет |
| P3 | Все 4 варианта | `pytest test_base_pipeline.py -v` | Нет |
| P4 | GEMM shape+gain | `pytest test_debug_steps.py::test_gemm_shape_and_gain` | Нет |
| P5 | FFT peak | `pytest test_debug_steps.py::test_fft_peak_location` | Нет |
| P6 | OneMax accuracy | `pytest test_debug_steps.py::test_one_max_accuracy` | Нет |
| P7 | MinMax DR | `pytest test_debug_steps.py::test_minmax_dynamic_range` | Нет |
| P8 | Timing analysis | `pytest test_timing_analysis.py -v` | После L7-9 |

```bash
# Запуск из корня GPUWorkLib:
pytest Python_test/strategies/test_base_pipeline.py -v
pytest Python_test/strategies/test_debug_steps.py -v
```

---

## Порядок выполнения в понедельник

```
Утро (компиляция):
  L1-1  →  cmake + make (убеждаемся что собирается)

Тестирование от фундамента:
  L1-2..L1-4  →  DrvGPU external context tests
  L2-1..L2-3  →  statistics ROCm (T1-T11)
  L3-1..L3-3  →  fft_func ROCm
  L4-1..L4-4  →  signal_generators ROCm (писать тесты прямо здесь!)
  L5-1..L5-2  →  lch_farrow regression

Стратегии:
  L6-1..L6-3  →  fft_maxima consumers
  L7-1        →  compiles with new test files
  L7-2..L7-3  →  smoke + regression
  L7-4..L7-6  →  full T1+T2
  L7-7..L7-9  →  benchmark + profiling + timing

Python (параллельно, без GPU):
  P1..P7 → pytest (можно на Windows пока С++ тесты на Linux)
  P8    → после L7-9 (нужен JSON от TimingPerStepTest)
```

---

## Файлы тест-архитектуры strategies (новые, 2026-03-15)

Созданы в этой сессии:

**C++ `modules/strategies/tests/`**:
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

**Python `Python_test/strategies/`**:
```
test_params.py                ← AntennaTestParams dataclass + SignalVariant
signal_generators_strategy.py ← ISignalStrategy + 4 реализации + Factory
strategy_test_base.py         ← StrategyTestBase (extends TestBase)
test_base_pipeline.py         ← pytest T1: 4 варианта, без GPU
test_debug_steps.py           ← pytest T2: per-step validation, без GPU
test_timing_analysis.py       ← парсинг JSON + bar chart
```

**Паттерны**: Template Method + Strategy + Factory + Composite + Information Expert (GRASP) + Controller (GRASP) + SRP/OCP/DIP (SOLID)

---

## TODO: non-square матрица 2500×100

Текущие тесты используют квадратную матрицу (n_ant=n_beams=100).
Для полного TestStrategia.md (2500×100):
1. Добавить `n_beams` в `AntennaProcessorConfig`
2. Обновить `GemmStep` — использовать n_beams вместо n_ant для output
3. Обновить `PrepareMatrix()` в `StrategyTestBase` — generate_identity_rectangular(n_ant, n_beams)
4. Тест: `AntennaTestParams::FullSpec()` → 2500×5000, W=2500×100

---

*Создано: 2026-03-15 | Обновить после каждого пройденного уровня*
