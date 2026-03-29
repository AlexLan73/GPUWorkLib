@page range_angle_tests range_angle — Тесты и бенчмарки

@tableofcontents

@section range_angle_tests_overview Обзор тестирования

Модуль `range_angle` находится в стадии **Beta** — операции содержат пустые заглушки (stubs).
Текущие тесты проверяют корректность конструирования, инициализации параметров
и вызова pipeline без фактических GPU-вычислений.

@warning Op Execute() methods are **empty stubs** — тесты проверяют только интерфейсы,
не числовые результаты.

@section range_angle_tests_cpp C++ тесты

Расположение: `modules/range_angle/tests/`

@subsection range_angle_tests_cpp_basic test_range_angle_basic.hpp — Базовые тесты

| Тест | Описание |
|------|----------|
| `Construct` | Создание `RangeAngleProcessor` с ROCm backend |
| `SetParams` | Установка параметров: n_ant_az, n_ant_el, n_samples, f_start, f_end |
| `Process` | Вызов полного pipeline (stubs) — проверка отсутствия crash |

@subsection range_angle_tests_cpp_benchmark test_range_angle_benchmark.hpp — Бенчмарк (stubs)

| Тест | Описание |
|------|----------|
| `LatencyStubs` | GPUProfiler замер латентности pipeline из stubs |

@note Бенчмарк стадий будет информативен только после реализации Op Execute() методов.

@section range_angle_tests_python Python тесты

Расположение: `Python_test/range_angle/`

@subsection range_angle_tests_python_main test_range_angle.py — Базовый тест

| Тест | Описание |
|------|----------|
| `BasicConstruct` | Создание процессора через Python bindings |
| `BasicProcess` | Вызов process() — проверка отсутствия exception |

Пример запуска:

@code{.py}
python Python_test/range_angle/test_range_angle.py
@endcode

@section range_angle_tests_planned Планируемые тесты

После реализации Op Execute() методов будут добавлены:

| Тест | Описание |
|------|----------|
| `SingleTarget` | Одна цель: проверка дальности и углов |
| `MultiTarget` | TOP_N: несколько целей разной дальности |
| `RangeResolution` | Проверка \f$ \Delta R = c / 2B \f$ |
| `AngleResolution` | Проверка углового разрешения 2D FFT |
| `NoiseFloor` | SNR на фоне шума |
| `NumPyReference` | Сравнение с NumPy/SciPy реализацией |
| `FullBenchmark` | Per-stage GPUProfiler (dechirp, FFT, beam, peak) |

@section range_angle_tests_pipeline_status Статус стадий pipeline

| Стадия | Op | Статус |
|--------|----|--------|
| 1 | `DechirpWindowOp` | Stub |
| 2 | `RangeFftOp` | Stub |
| 3 | `TransposeOp` | Stub |
| 4 | `BeamFftOp` | Stub |
| 5 | `PeakSearchOp` | Stub |

@see range_angle_overview
@see range_angle_formulas
