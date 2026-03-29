@page lch_farrow_overview lch_farrow — Обзор модуля

@tableofcontents

@section lch_farrow_overview_purpose Назначение

Дробная задержка (sub-sample) комплексных сигналов через **5-точечную интерполяцию Лагранжа**
с предвычисленной матрицей коэффициентов 48x5.

Применяется для **когерентного формирования диаграммы направленности**, когда задержка
не кратна периоду дискретизации. Обеспечивает точность интерполяции до 48 уровней
квантования дробной части задержки.

> **Namespace**: `lch_farrow` | **Backend**: OpenCL + ROCm | **Статус**: Active

@section lch_farrow_overview_classes Ключевые классы

| Класс | Backend | Описание |
|-------|---------|----------|
| `LchFarrow` | OpenCL | OpenCL реализация дробной задержки |
| `LchFarrowROCm` | ROCm/HIP | ROCm реализация (hiprtc + HSACO disk cache) |

@note Оба класса имеют идентичный API. `LchFarrowROCm` использует HSACO disk cache
для ускорения повторной компиляции ядер.

@section lch_farrow_overview_architecture Архитектура

```
Input signal [n_antennas × n_points] complex
  + Delays [n_antennas] float (мкс)
  + Lagrange matrix [48 × 5] float
  │
  ▼
┌───────────────────────────────────────┐
│ 1. Разделение задержки               │
│    τ = D + μ                         │
│    D — целая часть (integer delay)   │
│    μ — дробная часть ∈ [0, 1)        │
├───────────────────────────────────────┤
│ 2. Квантование μ                     │
│    row = floor(μ × 48)              │
│    Выбор строки из матрицы 48×5      │
├───────────────────────────────────────┤
│ 3. GPU Kernel: lch_farrow_delay      │
│    2D grid (x=sample, y=antenna)     │
│    5-точечная интерполяция            │
│    Без div/mod в kernel              │
└───────────────────────────────────────┘
  │
  ▼
Output signal [n_antennas × n_points] complex
```

@section lch_farrow_overview_quickstart Быстрый старт

@subsection lch_farrow_overview_quickstart_cpp C++

@code{.cpp}
#include "modules/lch_farrow/include/lch_farrow_rocm.hpp"

lch_farrow::LchFarrowROCm farrow(backend);
farrow.Initialize(n_antennas, n_points, sample_rate);

// Загрузка предвычисленной матрицы Лагранжа [48 x 5]
farrow.LoadMatrix(lagrange_48x5);

// Установка задержек для каждой антенны (мкс)
farrow.SetDelays(delays_us);

// Обработка: CPU -> GPU -> CPU
auto output = farrow.ProcessFromCPU(input_signal);

// Или: данные уже на GPU
auto output_gpu = farrow.Process(input_gpu_buffer);
@endcode

@subsection lch_farrow_overview_quickstart_python Python

@code{.py}
import gpuworklib as gw

ctx = gw.ROCmGPUContext(0)
farrow = gw.LchFarrowROCm(ctx,
                          n_ant=8,
                          n_points=4096,
                          fs=1e6)

# Установка задержек (мкс)
delays = [0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9]
output = farrow.process(signal_np, delays)
@endcode

@section lch_farrow_overview_kernel GPU Kernel

Ядро `lch_farrow_delay`:

| Характеристика | Значение |
|----------------|----------|
| Grid | 2D: (x = sample index, y = antenna index) |
| Без div/mod | Все индексы предвычислены на хосте |
| Доступ к памяти | Coalesced по x-оси (samples) |
| Коэффициенты | В constant memory (48x5 = 240 float) |
| Тип данных | float2 (комплексные отсчёты) |

@section lch_farrow_overview_matrix Матрица коэффициентов 48x5

Предвычисленная lookup-таблица 5-точечных базисных полиномов Лагранжа:

- **48 строк**: 48 уровней квантования дробной части \f$ \mu \in [0, 1) \f$
- **5 столбцов**: коэффициенты \f$ L_0(\mu), L_1(\mu), L_2(\mu), L_3(\mu), L_4(\mu) \f$
- Индекс строки: \f$ \text{row} = \lfloor \mu \times 48 \rfloor \f$

@note Матрица вычисляется один раз и загружается на GPU. Это устраняет необходимость
вычисления полиномов в runtime, обеспечивая максимальную производительность ядра.

@section lch_farrow_overview_backends Сравнение бэкендов

| Характеристика | OpenCL (`LchFarrow`) | ROCm (`LchFarrowROCm`) |
|----------------|----------------------|------------------------|
| Компиляция kernel | Online (clBuildProgram) | hiprtc + HSACO disk cache |
| Поддержка AMD | Да (все GPU) | Да (gfx900+) |
| Поддержка NVIDIA | Да | Нет |
| Профилирование | OpenCL events | hipEvent |

@section lch_farrow_overview_dependencies Зависимости

- **DrvGPU** — OpenCL/ROCm backend, GpuContext, GPUProfiler

@see lch_farrow_formulas
@see lch_farrow_tests
