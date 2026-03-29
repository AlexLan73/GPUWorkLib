@page range_angle_overview range_angle — Обзор модуля

@tableofcontents

@section range_angle_overview_purpose Назначение

3D FFT обработка для 2D антенной решётки (например, 16x16 элементов).
Детекция **дальности** и **2D угла** (азимут + угол места) радарных целей.

Pipeline из 5 стадий: dechirp → range FFT → transpose → 2D beam FFT → peak search.

> **Namespace**: `range_angle` | **Backend**: ROCm | **Статус**: Beta (ops = stubs)

@warning Текущий статус — **Beta**: методы `Execute()` операций являются пустыми заглушками (stubs).
Модуль находится на ранней стадии разработки. Интерфейсы стабильны, реализация будет добавлена.

@section range_angle_overview_classes Ключевые классы

| Класс | Слой Ref03 | Описание |
|-------|------------|----------|
| `RangeAngleProcessor` | Layer 6 (Facade) | Главный фасад — 5-стадийный pipeline |
| `DechirpWindowOp` | Layer 5 (Concrete Op) | Умножение на \f$ \overline{\text{ref\_lfm}} \f$ + окно Хэмминга |
| `RangeFftOp` | Layer 5 (Concrete Op) | Batched hipFFT по антеннам → спектр дальности |
| `TransposeOp` | Layer 5 (Concrete Op) | Перестановка данных для пространственной FFT |
| `BeamFftOp` | Layer 5 (Concrete Op) | 2D FFT (азимут × угол места) + fftshift |
| `PeakSearchOp` | Layer 5 (Concrete Op) | 3D GPU max reduction → `TargetInfo` |

@section range_angle_overview_pipeline Pipeline (5 стадий)

Полный pipeline обработки радарного сигнала:

```
Signal [n_ant_az × n_ant_el × n_samples]
  │
  ▼
┌─────────────────────────────────────┐
│ 1. DechirpWindowOp                  │
│    rx × conj(ref_lfm) + Hamming     │
│    → baseband beat signal           │
├─────────────────────────────────────┤
│ 2. RangeFftOp                       │
│    Batched FFT per antenna          │
│    → range spectrum                 │
├─────────────────────────────────────┤
│ 3. TransposeOp                      │
│    Rearrange [n_range × n_az × n_el]│
├─────────────────────────────────────┤
│ 4. BeamFftOp                        │
│    2D spatial FFT + fftshift        │
│    → 3D power cube                  │
├─────────────────────────────────────┤
│ 5. PeakSearchOp                     │
│    3D max reduction                 │
│    → TargetInfo{R, θ_az, θ_el, P}  │
└─────────────────────────────────────┘
```

@section range_angle_overview_quickstart Быстрый старт

@subsection range_angle_overview_quickstart_cpp C++

@code{.cpp}
#include "modules/range_angle/include/range_angle_processor.hpp"

range_angle::RangeAngleProcessor proc(backend);
range_angle::RangeAngleParams p;
p.n_ant_az = 16;        // 16 антенн по азимуту
p.n_ant_el = 16;        // 16 антенн по углу места
p.n_samples = 4096;     // отсчётов на антенну
p.f_start = 1e9;        // начальная частота ЛЧМ
p.f_end = 1.1e9;        // конечная частота ЛЧМ
p.sample_rate = 200e6;  // частота дискретизации
p.peak_mode = range_angle::PeakSearchMode::TOP_1;
proc.Initialize(p);

auto result = proc.Process(signal);
// result.targets[0].range_m      — дальность (метры)
// result.targets[0].angle_az_deg — азимут (градусы)
// result.targets[0].angle_el_deg — угол места (градусы)
@endcode

@subsection range_angle_overview_quickstart_python Python

@code{.py}
import gpuworklib as gw

ctx = gw.ROCmGPUContext(0)
proc = gw.RangeAngleProcessor(ctx,
                              n_az=16, n_el=16,
                              n_samples=4096,
                              f_start=1e9, f_end=1.1e9,
                              fs=200e6)
result = proc.process(signal_np)
# result.targets[0] — {range_m, angle_az_deg, angle_el_deg, power_dB, snr_dB}
@endcode

@section range_angle_overview_target_info Структура TargetInfo

Результат обнаружения цели:

| Поле | Тип | Описание |
|------|-----|----------|
| `range_m` | float | Дальность до цели (метры) |
| `angle_az_deg` | float | Азимут (градусы) |
| `angle_el_deg` | float | Угол места (градусы) |
| `power_dB` | float | Мощность сигнала (дБ) |
| `snr_dB` | float | Отношение сигнал/шум (дБ) |

Режимы поиска: `TOP_1` (одна самая сильная цель), `TOP_N` (N целей).

@section range_angle_overview_guides Учебные материалы

Два подробных описания метода 2FFT/3FFT для обработки ЛЧМ-сигналов:

- **Для начинающих**: [2FFT для ЛЧМ — Объяснение для чайника](md__doc_2_modules_2range__angle_23fft__lfm__processing__simple.html) — аналогии, картинки, сравнение с Capon, реальные числа проекта (16x16 URA, 1.3M точек)
- **Для инженеров**: [2FFT для ЛЧМ — Инженерно-технический анализ](md__doc_2_modules_2range__angle_23fft__lfm__processing__technical.html) — математика, алгоритмы (Matched Filter, Stretch Processing, Range-Doppler 2D, FrFT), GPU-реализация (hipFFT), Python-код, таблицы сложности

@note Эти документы описывают 4 метода обработки ЛЧМ: Matched Filter FFT, Stretch Processing (dechirp+FFT), Range-Doppler 2D FFT, Fractional Fourier Transform. Наш модуль реализует 3D вариант (Range + 2D Beam FFT) для URA 16x16.

@section range_angle_overview_dependencies Зависимости

- **DrvGPU** — ROCm backend, GpuContext, GPUProfiler
- **hipFFT** — batched FFT и 2D FFT
- **HeterodyneDechirp** — Stretch Processing (dechirp шаг 1)

@see range_angle_formulas
@see range_angle_tests
