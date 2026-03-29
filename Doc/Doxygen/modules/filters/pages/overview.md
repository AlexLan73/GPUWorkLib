@page filters_overview Filters — Обзор

@tableofcontents

@section filters_overview_purpose Назначение

GPU-фильтрация комплексных многоканальных сигналов (float2).
5 типов фильтров: FIR, IIR, Moving Average, Kalman, KAMA.

> **Namespace**: `filters` | **Backend**: ROCm (HIP) | **Статус**: Active

@section filters_overview_classes Ключевые классы

| Класс | Описание |
|-------|----------|
| `FirFilterROCm` | Direct-form FIR: 2D NDRange (channels × samples), до 16000 тапов |
| `IirFilterROCm` | Biquad cascade DFII-T: 1 thread per channel, sequential |
| `MovingAverageFilterROCm` | SMA/EMA/MMA/DEMA/TEMA: ring buffer, N≤128 |
| `KalmanFilterROCm` | 1D scalar Kalman: Re/Im независимо |
| `KaufmanFilterROCm` | KAMA: Efficiency Ratio adaptive smoothing |

@section filters_overview_architecture Архитектура

```
Input Signal → [FirFilterROCm]   → Filtered (direct convolution)
             → [IirFilterROCm]   → Filtered (biquad cascade DFII-T)
             → [MovingAverageFilterROCm] → Smoothed (SMA/EMA/MMA/DEMA/TEMA)
             → [KalmanFilterROCm] → Denoised (predict-update)
             → [KaufmanFilterROCm] → Adaptive (KAMA)
```

@subsection filters_arch_fir FIR

2D NDRange: каждый thread обрабатывает один sample одного канала.
Поддержка до 16000 тапов через глобальную память коэффициентов.

@subsection filters_arch_iir IIR Biquad

1 thread per channel, sequential processing.
Cascade biquad DFII-Transposed: каждая секция имеет 5 коэффициентов (b0, b1, b2, a1, a2).

@section filters_overview_quickstart Быстрый старт

@subsection filters_qs_cpp C++

@code{.cpp}
#include "modules/filters/include/filters/fir_filter_rocm.hpp"

filters::FirFilterROCm fir(backend);
fir.Initialize(coefficients, n_channels);
auto output = fir.Process(input_signal);
@endcode

@subsection filters_qs_python Python

@code{.py}
import gpuworklib as gw
ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilterROCm(ctx, coefficients=coeffs, channels=8)
output = fir.process(signal_np)
@endcode

@section filters_overview_seealso См. также

- @ref filters_formulas — Математика фильтров
- @ref filters_tests — Тесты и бенчмарки
- @ref drvgpu_main — DrvGPU (базовый драйвер)
