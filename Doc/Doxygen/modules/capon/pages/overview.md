@page capon_overview capon — Обзор модуля

@tableofcontents

@section capon_overview_purpose Назначение

Adaptive beamformer **Capon** (MVDR — Minimum Variance Distortionless Response).
Вычисляет адаптивные весовые коэффициенты, минимизируя выходную мощность
при сохранении сигнала от целевого направления. Эффективное подавление помех
с использованием ROCm (rocBLAS, rocsolver) на GPU.

> **Namespace**: `capon` | **Backend**: ROCm (rocBLAS, rocsolver) | **Статус**: Framework Ready

@section capon_overview_modes Режимы работы

Модуль поддерживает два основных режима:

- **ComputeRelief** — вычисление пространственного спектра \f$ z[m] \f$ для \f$ M \f$ направлений.
  Позволяет визуализировать угловую зависимость мощности и определить направления приходящих сигналов.
- **AdaptiveBeamform** — формирование выходных сигналов \f$ Y_{\text{out}} \f$ для каждого луча
  с адаптивным подавлением помех.

@section capon_overview_classes Ключевые классы

| Класс | Слой Ref03 | Описание |
|-------|------------|----------|
| `CaponProcessor` | Layer 6 (Facade) | Главный фасад — единственная точка входа |
| `CovarianceMatrixOp` | Layer 5 (Concrete Op) | Ковариационная матрица \f$ R = YY^H/N + \mu I \f$ через rocBLAS CGEMM |
| `CaponInvertOp` | Layer 5 (Concrete Op) | Обращение \f$ R^{-1} \f$ через `CholeskyInverterROCm` (vector_algebra) |
| `ComputeWeightsOp` | Layer 5 (Concrete Op) | Весовые коэффициенты \f$ W = R^{-1} U \f$ через rocBLAS CGEMM |
| `CaponReliefOp` | Layer 5 (Concrete Op) | Пространственный спектр \f$ z[m] = 1/\text{Re}(u^H R^{-1} u) \f$ |
| `AdaptBeamformOp` | Layer 5 (Concrete Op) | Адаптивное формирование \f$ Y_{\text{out}} = W^H Y \f$ |

@note Модуль зависит от `vector_algebra` (`CholeskyInverterROCm`) для Cholesky-разложения и обращения матриц.

@section capon_overview_architecture Архитектура

Модуль построен по единой 6-слойной архитектуре Ref03:

```
CaponProcessor (Facade, Layer 6)
  ├── CovarianceMatrixOp    — rocBLAS CGEMM: R = Y·Yᴴ/N + μI
  ├── CaponInvertOp          — CholeskyInverterROCm: R⁻¹
  ├── ComputeWeightsOp       — rocBLAS CGEMM: W = R⁻¹·U
  ├── CaponReliefOp          — HIP kernel: z[m] = 1/Re(uᴴR⁻¹u)
  └── AdaptBeamformOp        — rocBLAS CGEMM: Yout = WᴴY
```

@section capon_overview_quickstart Быстрый старт

@subsection capon_overview_quickstart_cpp C++

@code{.cpp}
#include "modules/capon/include/capon_processor.hpp"

capon::CaponProcessor proc(backend);
capon::CaponParams p;
p.n_channels = 85; p.n_samples = 1000;
p.n_directions = 181; p.mu = 0.01f;
proc.Initialize(p);

// Режим 1: пространственный спектр (relief)
auto relief = proc.ComputeRelief(Y, steering);
// relief.relief[m] — мощность для направления m

// Режим 2: адаптивное формирование
auto beam = proc.AdaptiveBeamform(Y, steering);
// beam.output[m * N + n] — выходной сигнал луча m, отсчёт n
@endcode

@subsection capon_overview_quickstart_python Python

@code{.py}
import gpuworklib as gw

ctx = gw.ROCmGPUContext(0)
proc = gw.CaponProcessor(ctx, P=85, N=1000, M=181, mu=0.01)

# Пространственный спектр
relief = proc.compute_relief(Y_np, steering_np)

# Адаптивное формирование
beam = proc.adaptive_beamform(Y_np, steering_np)
@endcode

@section capon_overview_dependencies Зависимости

- **vector_algebra** — `CholeskyInverterROCm` для POTRF/POTRI
- **DrvGPU** — ROCm backend, GpuContext, GPUProfiler
- **rocBLAS** — CGEMM для матричных операций
- **rocsolver** — POTRF/POTRI для Cholesky

@section capon_overview_testdata Тестовые данные

Директория `modules/capon/tests/data/` содержит MATLAB-данные от заказчика:
- `signal_matlab.txt` — эталонный сигнал
- `x_data.txt`, `y_data.txt` — координаты антенн (P=85, N=1000, M=1369)

@section capon_overview_todo TODO

- rocBLAS CGEMM в CovarianceMatrixOp (текущая задача)

@see capon_formulas
@see capon_tests
@see @ref vector_algebra_overview
