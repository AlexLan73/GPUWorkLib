@page strategies_tests strategies — Тесты и бенчмарки

@tableofcontents

@section strategies_tests_overview Обзор тестирования

Модуль `strategies` покрыт многоуровневыми тестами: полный pipeline, пошаговая отладка,
4 варианта сигналов (GoF Strategy), параллельные stream-бенчмарки и per-step профилирование.

@section strategies_tests_cpp C++ тесты

Расположение: `modules/strategies/tests/`

@subsection strategies_tests_cpp_pipeline test_strategies_pipeline.hpp — Полный pipeline

| Тест | Описание |
|------|----------|
| `FullPipeline` | 5 антенн, 8000 отсчётов, 12 МГц — полный 7-step pipeline |
| `AllScenarios` | Все 3 post-FFT сценария: OneMax, AllMaxima, MinMax |
| `CheckpointStats` | Проверка 3 checkpoint-ов статистики |

@subsection strategies_tests_cpp_strategy test_base_strategy.hpp — GoF Strategy

4 варианта сигналов через паттерн **Strategy** (`ISignalStrategy`):

| Сигнал | Описание |
|--------|----------|
| `PureTone` | Чистая синусоида → OneMax = точная частота |
| `TwoTones` | Два тона → AllMaxima = две частоты |
| `Broadband` | Широкополосный шум → MinMax = малый DR |
| `Narrowband` | Узкополосный + шум → OneMax + высокий DR |

Паттерны: **Strategy** (ISignalStrategy), **Template Method** (StrategyTestBase), **Factory Method**.

@subsection strategies_tests_cpp_debug test_debug_steps.hpp — Пошаговая отладка

| Тест | Описание |
|------|----------|
| `DebugStep_Signal1` | Step-by-step через AntennaProcessorTest (PureTone) |
| `DebugStep_Signal2` | TwoTones: проверка каждого шага |
| `DebugStep_Signal3` | Broadband: статистика на каждом шаге |
| `DebugStep_Signal4` | Narrowband: промежуточные результаты |

@subsection strategies_tests_cpp_profiling test_strategies_step_profiling.hpp — Per-step профилирование

Замер каждого шага pipeline через hipEvent + GPUProfiler:

| Шаг | Метрика |
|-----|---------|
| Step 1 | Statistics(S) time |
| Step 2 | CGEMM time |
| Step 3 | Statistics(X) time |
| Step 4 | Window + FFT time |
| Step 5 | Statistics(spectrum) time |
| Step 6 | Post-FFT scenario time |
| Step 7 | Sync time |

@subsection strategies_tests_cpp_streams test_strategies_benchmark_streams.hpp — Stream benchmark

Тестирование параллельной работы 4 HIP stream-ов:

| Тест | Описание |
|------|----------|
| `StreamParallel` | 4 stream-а параллельно: проверка корректности |
| `StreamOverlap` | Перекрытие compute/transfer между streams |
| `StreamScaling` | 1 vs 2 vs 4 stream-а: speedup |

@subsection strategies_tests_cpp_timing timing_per_step_test.hpp — Timing table

hipEvent timing для каждого шага → экспорт в JSON:

| Формат | Расположение |
|--------|--------------|
| JSON | `Results/JSON/strategies_timing.json` |
| Markdown | `Results/Profiler/strategies_steps.md` |

@section strategies_tests_python Python тесты

Расположение: `Python_test/strategies/`

@subsection strategies_tests_python_pipeline test_strategies_pipeline.py — E2E pipeline

| Тест | Описание |
|------|----------|
| `FullE2E` | Полный end-to-end pipeline через Python bindings |
| `ResultValidation` | Проверка stats, one_max, minmax |

@subsection strategies_tests_python_base test_base_pipeline.py — Базовый pipeline

| Тест | Описание |
|------|----------|
| `Construction` | Создание PipelineRunner с конфигурацией |
| `Defaults` | Проверка значений по умолчанию |

@subsection strategies_tests_python_debug test_debug_steps.py — Per-step валидация

| Тест | Описание |
|------|----------|
| `StepByStep` | Вызов каждого шага отдельно через Python |
| `IntermediateBuffers` | Проверка промежуточных буферов |

@subsection strategies_tests_python_farrow test_farrow_pipeline.py — Farrow delay + alignment

| Тест | Описание |
|------|----------|
| `DelayAlignment` | Farrow дробная задержка для выравнивания сигналов |
| `RawVsAligned` | Сравнение raw vs aligned |

@subsection strategies_tests_python_scenario test_scenario_builder.py — Scenario builder

| Тест | Описание |
|------|----------|
| `ScenarioConstruction` | Создание сценариев через builder |
| `GeometryBuilder` | Геометрия антенной решётки |

@subsection strategies_tests_python_steps test_strategies_step_by_step.py — Detailed step tracing

| Тест | Описание |
|------|----------|
| `DetailedTrace` | Подробная трассировка каждого шага |

@subsection strategies_tests_python_timing test_timing_analysis.py — JSON timing analysis

| Тест | Описание |
|------|----------|
| `JSONParse` | Парсинг JSON timing данных |
| `BottleneckAnalysis` | Определение узких мест pipeline |

@section strategies_tests_benchmarks Бенчмарки

| Класс | Метрики |
|-------|---------|
| `StrategiesProfilingBenchmark` | Per-step GPUProfiler (все 7 шагов) |
| Stream benchmark | Multi-stream parallel timing |
| `TimingPerStepTest` | hipEvent table → JSON export |

@section strategies_tests_plots Графики

@subsection strategies_tests_plots_checkpoints Post-FFT checkpoints (Step 2.1 / 2.2 / 2.3)

@image html strategies/checkpoints_2_1_2_2_2_3.png "Spectrum peaks: OneMax vs AllMaxima vs MinMax" width=800px

Три checkpoint-а статистики спектра: OneMax (параболическая интерполяция),
AllMaxima (все пики выше порога), MinMax (динамический диапазон).

@subsection strategies_tests_plots_farrow Farrow delay alignment

@image html strategies/farrow_raw_vs_aligned.png "Raw vs Farrow-aligned signal" width=700px

Сравнение необработанного сигнала с сигналом после Farrow дробной задержки.

@subsection strategies_tests_plots_peak Pipeline A vs B — peak comparison

@image html strategies/peak_comparison_a_vs_b.png "Pipeline A vs B: peak frequency comparison" width=700px

Сравнение частот максимумов при разных конфигурациях pipeline.

@subsection strategies_tests_plots_spectra FFT spectra side-by-side

@image html strategies/spectra_pipeline_a_vs_b.png "FFT spectra: Pipeline A vs Pipeline B" width=800px

FFT спектры двух конфигураций pipeline: сравнение уровней боковых лепестков и динамического диапазона.

@see strategies_overview
@see strategies_formulas
