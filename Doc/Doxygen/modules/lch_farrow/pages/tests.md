@page lch_farrow_tests lch_farrow — Тесты и бенчмарки

@tableofcontents

@section lch_farrow_tests_overview Обзор тестирования

Модуль `lch_farrow` покрыт тестами для обоих бэкендов (OpenCL и ROCm):
от нулевой задержки до мультиантенных конфигураций. Бенчмарки замеряют
каждую стадию pipeline (upload, kernel) для обоих бэкендов.

@section lch_farrow_tests_cpp C++ тесты

Расположение: `modules/lch_farrow/tests/`

@subsection lch_farrow_tests_cpp_rocm test_lch_farrow_rocm.hpp — ROCm тесты

4 теста с возрастающей сложностью:

| Тест | Задержка | Описание |
|------|----------|----------|
| `ZeroDelay` | \f$ \tau = 0 \f$ | Нулевая задержка — выход = вход (identity) |
| `IntegerDelay` | \f$ D = 5,\; \mu = 0 \f$ | Целочисленная задержка 5 отсчётов (только сдвиг) |
| `FractionalDelay` | \f$ D = 2,\; \mu = 0.7 \f$ | Дробная задержка 2.7 отсчёта (интерполяция) |
| `MultiAntenna` | Разные \f$ \tau_a \f$ | 8 антенн с разными задержками одновременно |

@note Тест `ZeroDelay` проверяет, что при \f$ \mu = 0 \f$ интерполяция не вносит искажений
(выход точно равен входу с точностью до float).

@section lch_farrow_tests_python Python тесты

Расположение: `Python_test/lch_farrow/`

@subsection lch_farrow_tests_python_main test_lch_farrow.py — NumPy эталон

| Тест | Описание |
|------|----------|
| `NumpyReference` | NumPy реализация Lagrange интерполяции — эталон для сравнения |
| `GPUVerification` | GPU результат vs NumPy эталон: max absolute error |
| `LagrangeMatrixLoading` | Загрузка и проверка матрицы 48x5 |

Пример запуска:

@code{.py}
python Python_test/lch_farrow/test_lch_farrow.py
@endcode

@subsection lch_farrow_tests_python_rocm test_lch_farrow_rocm.py — ROCm-specific

| Тест | Описание |
|------|----------|
| `ROCmPipeline` | Полный pipeline на ROCm: Initialize → LoadMatrix → Process |
| `HSACOCache` | Проверка disk cache для скомпилированных ядер |
| `MultiAntROCm` | 8 антенн на ROCm |

@note Требуется AMD GPU + Linux для запуска ROCm тестов.

Пример запуска:

@code{.py}
python Python_test/lch_farrow/test_lch_farrow_rocm.py
@endcode

@section lch_farrow_tests_benchmarks Бенчмарки

@subsection lch_farrow_tests_bench_opencl LchFarrowBenchmark — OpenCL

| Стадия | Описание |
|--------|----------|
| `Upload_delay` | Загрузка массива задержек на GPU |
| `Kernel` | Выполнение ядра `lch_farrow_delay` |

@subsection lch_farrow_tests_bench_rocm LchFarrowBenchmarkROCm — ROCm

| Стадия | Описание |
|--------|----------|
| `Upload_input` | Загрузка входного сигнала Host → Device |
| `Upload_delay` | Загрузка массива задержек Host → Device |
| `Kernel` | Выполнение HIP ядра `lch_farrow_delay` |

@subsection lch_farrow_tests_bench_params Параметры бенчмарков

Стандартная конфигурация:

| Параметр | Значение |
|----------|----------|
| Антенн | 8 |
| Отсчётов | 4096 |
| Задержки (мкс) | [0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9] |

@section lch_farrow_tests_kernel Тестирование ядра

Ядро `lch_farrow_delay` проверяется по следующим критериям:

| Критерий | Метод проверки |
|----------|---------------|
| Корректность интерполяции | Сравнение с NumPy Lagrange |
| Граничные условия | \f$ \mu = 0 \f$ (identity), \f$ \mu \to 1 \f$ |
| Coalesced access | GPU Profiler: bandwidth utilization |
| Отсутствие div/mod | Анализ PTX / ISA |
| Multi-antenna | Параллельная обработка 8 антенн |

@section lch_farrow_tests_plots Графики

@subsection lch_farrow_tests_plots_delay Задержка сигнала

@image html lch_farrow/delay_comparison.png "Исходный vs задержанный сигнал (дробная задержка)" width=700px

@subsection lch_farrow_tests_plots_error Ошибка интерполяции

@image html lch_farrow/interpolation_error.png "Ошибка интерполяции Lagrange vs истинная задержка" width=700px

@subsection lch_farrow_tests_plots_multi Multi-antenna задержки

@image html lch_farrow/multi_antenna_delays.png "8 антенн с различными задержками" width=800px

@section lch_farrow_tests_accuracy Точность интерполяции

| Метрика | Значение |
|---------|----------|
| Max absolute error (vs NumPy) | < \f$ 10^{-5} \f$ |
| Квантование \f$ \mu \f$ | 48 уровней, \f$ \Delta\mu < 0.0104 \f$ |
| Identity test (\f$ \mu = 0 \f$) | Точное совпадение (bitwise) |

@see lch_farrow_overview
@see lch_farrow_formulas
