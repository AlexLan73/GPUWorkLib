@page statistics_tests Statistics -- Тесты и бенчмарки

@tableofcontents

@section stat_tests_intro Введение

Тестирование модуля **statistics** включает C++ unit-тесты на GPU,
Python-тесты с NumPy-эталоном и бенчмарки производительности.
Все тесты проверяют корректность вычислений mean, Welford variance/std,
и медианы (radix sort / histogram) для различных размеров данных и количества лучей.

---

@section stat_tests_cpp C++ тесты

@subsection stat_tests_cpp_main Основные тесты (test_statistics_rocm.hpp)

Файл: `modules/statistics/tests/test_statistics_rocm.hpp`

| # | Тест | Описание |
|---|------|----------|
| 1 | SingleBeamMean | Комплексное среднее, 1 луч, 4096 отсчётов |
| 2 | MultiBeamMean | Комплексное среднее, 4 луча, 8192 отсчётов |
| 3 | WelfordSingleBeam | Welford one-pass: mean_mag + var + std, 1 луч |
| 4 | WelfordMultiBeam | Welford one-pass, 4 луча, проверка всех полей |
| 5 | MedianRadixSmall | Медиана radix sort, N=256 (bitonic fallback) |
| 6 | MedianRadixMedium | Медиана radix sort, N=8192 |
| 7 | MedianHistogramLarge | Медиана histogram, N=1M |
| 8 | MedianHistogramComplexLarge | Медиана histogram (complex input), N=1M |
| 9 | ComputeAllSingleBeam | ComputeAll: mean + Welford + median, 1 луч |
| 10 | ComputeAllMultiBeam | ComputeAll, 4 луча, полная проверка |
| 11 | GPUInputDirect | Вход -- device pointer (без upload) |
| 12 | EdgeCaseZeros | Все нули: mean=0, var=0, std=0 |
| 13 | EdgeCaseConstant | Константный сигнал: var=0, median=const |
| 14 | EdgeCaseSingleElement | N=1: mean=z, median=|z| |
| 15 | LargeBeamCount | 64 луча, 4096 отсчётов -- проверка масштабируемости |

@subsection stat_tests_cpp_float Float-тесты (test_statistics_float_rocm.hpp)

Файл: `modules/statistics/tests/test_statistics_float_rocm.hpp`

| # | Тест | Описание |
|---|------|----------|
| 1 | FloatWelford | WelfordFloatOp на pre-computed magnitudes |
| 2 | FloatMedianRadix | Медиана radix sort на float-данных |
| 3 | FloatMedianHistogram | Медиана histogram на float-данных, N=500K |
| 4 | FloatComputeAll | ComputeAllFloat pipeline |
| 5 | FloatGPUInput | Float-данные уже на GPU (device pointer) |

@subsection stat_tests_cpp_example Пример C++ теста

@code{.cpp}
// test_statistics_rocm.hpp -- ComputeAllMultiBeam
void TestComputeAllMultiBeam(DrvGPU& drv) {
  auto& con = ConsoleOutput::GetInstance();
  statistics::StatisticsProcessor proc(drv.GetBackend());
  statistics::StatisticsParams p{4, 8192, 0};
  proc.Initialize(p);

  // Генерация тестовых данных (4 луча x 8192 complex)
  auto signal = GenerateTestSignal(4, 8192);
  auto result = proc.ComputeAll(signal);

  // Проверка: NumPy reference
  for (int b = 0; b < 4; ++b) {
    auto ref = ComputeReferenceStats(signal, b, 8192);
    ASSERT_NEAR(result.stats[b].mean_magnitude, ref.mean_mag, 1e-3f);
    ASSERT_NEAR(result.stats[b].variance, ref.variance, 1e-2f);
    ASSERT_NEAR(result.stats[b].std_dev, ref.std_dev, 1e-2f);
    ASSERT_NEAR(result.medians[b].median_magnitude, ref.median, 1e-2f);
  }
  con.Print("TestComputeAllMultiBeam: PASSED");
}
@endcode

---

@section stat_tests_python Python тесты

@subsection stat_tests_python_main Основные тесты (test_statistics_rocm.py)

Файл: `Python_test/statistics/test_statistics_rocm.py`

**NumPy reference тесты (5):**
| # | Тест | Описание |
|---|------|----------|
| 1 | test_complex_mean_reference | `np.mean(signal, axis=1)` vs GPU |
| 2 | test_magnitude_mean_reference | `np.mean(np.abs(signal), axis=1)` vs GPU |
| 3 | test_variance_reference | `np.var(np.abs(signal), ddof=0)` vs GPU |
| 4 | test_std_reference | `np.std(np.abs(signal), ddof=0)` vs GPU |
| 5 | test_median_reference | `np.median(np.abs(signal), axis=1)` vs GPU |

**GPU тесты (7):**
| # | Тест | Описание |
|---|------|----------|
| 6 | test_compute_all_4beams | ComputeAll, 4 луча, все метрики |
| 7 | test_compute_all_64beams | ComputeAll, 64 луча (stress-test) |
| 8 | test_median_small_array | N=128 (bitonic sort путь) |
| 9 | test_median_large_array | N=1M (histogram путь) |
| 10 | test_zeros_input | Вход -- нули: все метрики = 0 |
| 11 | test_constant_input | Константа: var=0, std=0, median=const |
| 12 | test_gpu_input_direct | Данные уже на GPU (без NumPy upload) |

@subsection stat_tests_python_float Float-тесты (test_statistics_float_rocm.py)

Файл: `Python_test/statistics/test_statistics_float_rocm.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_float_welford | WelfordFloat: pre-computed magnitudes |
| 2 | test_float_median_radix | Float median, radix sort |
| 3 | test_float_median_histogram | Float median, histogram, N=500K |

@subsection stat_tests_python_computeall ComputeAll тесты (test_compute_all.py)

Файл: `Python_test/statistics/test_compute_all.py`

| # | Тест | Описание |
|---|------|----------|
| 1 | test_compute_all_vs_separate | ComputeAll == ComputeMean + ComputeStatistics + ComputeMedian |
| 2 | test_compute_all_float | ComputeAllFloat pipeline |

@subsection stat_tests_python_example Пример Python теста

@code{.py}
# test_statistics_rocm.py -- test_complex_mean_reference
def test_complex_mean_reference(self):
    """Проверка комплексного среднего: GPU vs NumPy."""
    signal = np.random.randn(4, 8192) + 1j * np.random.randn(4, 8192)
    signal = signal.astype(np.complex64)

    # NumPy эталон
    np_mean = np.mean(signal, axis=1)

    # GPU вычисление
    result = self.proc.compute_mean(signal)

    for b in range(4):
        np.testing.assert_allclose(
            [result[b].real, result[b].imag],
            [np_mean[b].real, np_mean[b].imag],
            atol=1e-3,
            err_msg=f"Beam {b}: complex mean mismatch"
        )
@endcode

---

@section stat_tests_benchmarks Бенчмарки

@subsection stat_tests_bench_computeall ComputeAllBenchmarkROCm

Полный бенчмарк `ComputeAll` через `GpuBenchmarkBase`:

| Стадия | Описание |
|--------|----------|
| Upload | Загрузка complex float2 данных на GPU |
| Welford_Fused | One-pass: magnitude + S1 + S2 + финальный расчёт |
| Median_RadixSort | rocPRIM segmented radix sort (для N < 256K) |
| Median_Histogram | Histogram-based median (для N >= 256K) |
| MeanReduction | Hierarchical complex mean reduction |
| Download | Скачивание результатов с GPU |

@subsection stat_tests_bench_example Запуск бенчмарка

@code{.cpp}
// Из all_test.hpp:
#include "modules/statistics/tests/test_statistics_benchmark_rocm.hpp"

// В RunAllTests():
test_statistics_benchmark_rocm::RunBenchmarks(drv);
@endcode

Результат сохраняется в `Results/Profiler/` через `GPUProfiler::ExportMarkdown()`.

---

@section stat_tests_plots Графики

@subsection stat_tests_plot_reference Welford reference -- NumPy vs GPU

@image html statistics/test_statistics_rocm_reference.png "Statistics: NumPy reference vs GPU (mean, std, median)" width=700px

Сравнение GPU-результатов с NumPy-эталоном по всем лучам:
- Верхний ряд: mean_magnitude (GPU) vs `np.mean(np.abs(signal), axis=1)`
- Средний ряд: std_dev (GPU) vs `np.std(np.abs(signal), ddof=0, axis=1)`
- Нижний ряд: median (GPU) vs `np.median(np.abs(signal), axis=1)`

---

@section stat_tests_tolerances Допустимые погрешности

| Метрика | Абсолютная | Относительная | Примечание |
|---------|------------|---------------|------------|
| complex mean | \f$ 10^{-3} \f$ | \f$ 10^{-4} \f$ | float32 precision |
| mean_magnitude | \f$ 10^{-3} \f$ | \f$ 10^{-4} \f$ | -- |
| variance | \f$ 10^{-2} \f$ | \f$ 10^{-3} \f$ | Welford accumulation error |
| std_dev | \f$ 10^{-2} \f$ | \f$ 10^{-3} \f$ | sqrt amplifies error |
| median | \f$ 10^{-2} \f$ | \f$ 10^{-3} \f$ | Histogram binning precision |

@note Histogram-медиана может иметь бОльшую погрешность для очень неравномерных распределений
из-за конечной ширины бинов на последней итерации.

---

@section stat_tests_see_also Смотрите также

- @ref statistics_overview -- Обзор модуля, классы и быстрый старт
- @ref statistics_formulas -- Математические формулы
