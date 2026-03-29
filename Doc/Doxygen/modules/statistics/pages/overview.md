@page statistics_overview Statistics -- Обзор модуля

@tableofcontents

@section stat_overview_purpose Назначение

Модуль **statistics** выполняет per-beam GPU-статистику на комплексных данных:
mean (комплексное среднее), median (медиана магнитуды), variance, std_dev.
Численно стабильный Welford one-pass алгоритм для дисперсии,
radix sort и histogram подходы для медианы различных размеров данных.

> **Namespace**: `statistics` | **Backend**: ROCm | **Статус**: Active

@section stat_overview_classes Ключевые классы

| Класс | Описание | Заголовок |
|-------|----------|-----------|
| @ref statistics::StatisticsProcessor | Facade: ComputeMean, ComputeMedian, ComputeStatistics, ComputeAll | `statistics_processor.hpp` |
| @ref statistics::MeanReductionOp | Иерархическая редукция комплексного среднего (multi-pass reduce) | `operations/mean_reduction_op.hpp` |
| @ref statistics::WelfordFusedOp | One-pass Welford: mean_magnitude + variance + std (complex input) | `operations/welford_fused_op.hpp` |
| @ref statistics::WelfordFloatOp | Welford на float magnitudes (для pre-computed \f$ |z| \f$) | `operations/welford_float_op.hpp` |
| @ref statistics::MedianRadixSortOp | Медиана через rocPRIM radix sort (малые/средние массивы) | `operations/median_radix_sort_op.hpp` |
| @ref statistics::MedianHistogramOp | Медиана через гистограмму (большие float данные) | `operations/median_histogram_op.hpp` |
| @ref statistics::MedianHistogramComplexOp | Медиана гистограммой (большие complex данные, magnitude pre-step) | `operations/median_histogram_complex_op.hpp` |

@section stat_overview_arch Архитектура

Модуль строится по единой 6-слойной архитектуре Ref03:

- **Слой 1** -- `GpuContext`: per-module контекст (backend, stream, compiled module, shared buffers)
- **Слой 2** -- `IGpuOperation`: интерфейс (Name, Initialize, IsReady, Release)
- **Слой 3** -- `GpuKernelOp`: базовый доступ к compiled kernels через GpuContext
- **Слой 5** -- Конкретные операции: `MeanReductionOp`, `WelfordFusedOp`, `MedianRadixSortOp`, `MedianHistogramOp`
- **Слой 6** -- Facade: `StatisticsProcessor` объединяет все операции, автоматически выбирает стратегию медианы

@section stat_overview_pipeline Конвейер обработки

```
complex float2* (GPU) --> StatisticsProcessor
                               |
            +------------------+------------------+
            |                  |                  |
      MeanReductionOp   WelfordFusedOp     MedianStrategy
      (complex mean)   (mag mean+var+std)        |
                                          +------+------+
                                          |             |
                                    RadixSort     Histogram
                                    (N < 256K)    (N >= 256K)
```

@section stat_overview_strategy Стратегия выбора медианы

Модуль автоматически выбирает алгоритм медианы в зависимости от размера данных:

| Размер данных | Алгоритм | Операция |
|---------------|----------|----------|
| \f$ N < 256 \f$ | Bitonic sort (in-register) | `MedianRadixSortOp` |
| \f$ 256 \le N < 256\text{K} \f$ | rocPRIM segmented radix sort | `MedianRadixSortOp` |
| \f$ N \ge 256\text{K} \f$ | Histogram-based (O(passes)) | `MedianHistogramOp` / `MedianHistogramComplexOp` |

@note Для complex-данных медиана вычисляется по магнитуде \f$ |z| \f$, а не по Re/Im отдельно.

@section stat_overview_quickstart Быстрый старт

@subsection stat_overview_cpp C++ пример

@code{.cpp}
#include "modules/statistics/include/statistics_processor.hpp"

// Создание процессора статистики
statistics::StatisticsProcessor proc(backend);

// Параметры: 4 луча, 8192 отсчета на луч, auto memory_limit
statistics::StatisticsParams p{4, 8192, 0};
proc.Initialize(p);

// Полная статистика: mean + variance + std + median
auto result = proc.ComputeAll(signal);

// Результаты по каждому лучу
for (int b = 0; b < 4; ++b) {
  auto& s = result.stats[b];
  // s.mean          -- complex mean (float2)
  // s.mean_magnitude -- средняя магнитуда (float)
  // s.variance       -- дисперсия (float)
  // s.std_dev        -- СКО (float)
  auto& m = result.medians[b];
  // m.median_magnitude -- медиана магнитуды (float)
}
@endcode

@code{.cpp}
// Раздельные вызовы
auto mean_result = proc.ComputeMean(signal);
auto stat_result = proc.ComputeStatistics(signal);  // Welford: mean_mag + var + std
auto med_result  = proc.ComputeMedian(signal);       // auto-select radix/histogram
@endcode

@subsection stat_overview_python Python пример

@code{.py}
import gpuworklib as gw
import numpy as np

# Инициализация GPU контекста
ctx = gw.ROCmGPUContext(0)

# Создание процессора: 4 луча, 8192 отсчетов
proc = gw.StatisticsProcessor(ctx, beam_count=4, n_point=8192)

# Подготовка данных
signal = np.random.randn(4, 8192) + 1j * np.random.randn(4, 8192)
signal = signal.astype(np.complex64)

# Полная статистика
result = proc.compute_all(signal)
# result.stats[beam].mean, .variance, .std_dev, .mean_magnitude
# result.medians[beam].median_magnitude
@endcode

@code{.py}
# Проверка через NumPy-эталон
for b in range(4):
    mag = np.abs(signal[b])
    np_mean = np.mean(mag)
    np_std  = np.std(mag)
    np_med  = np.median(mag)
    print(f"Beam {b}: GPU mean={result.stats[b].mean_magnitude:.6f}, "
          f"NumPy mean={np_mean:.6f}")
@endcode

@warning Population variance (ddof=0) используется по умолчанию -- как в NumPy `np.var(x, ddof=0)`.

@section stat_overview_deps Зависимости

- **DrvGPU** -- управление GPU-контекстом, потоками, памятью
- **ROCm / HIP** -- запуск HIP kernels
- **rocPRIM** -- segmented radix sort для медианы

@section stat_overview_see_also Смотрите также

- @ref statistics_formulas -- Математические формулы (mean, Welford, median)
- @ref statistics_tests -- Тесты, бенчмарки и графики
- @ref heterodyne_overview -- Гетеродин (использует статистику для SNR)
- @ref fft_func_overview -- FFT-обработка (статистика спектра)
