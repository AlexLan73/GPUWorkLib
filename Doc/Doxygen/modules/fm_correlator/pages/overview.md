@page fm_correlator_overview fm_correlator — Обзор модуля

@tableofcontents

@section fm_correlator_overview_purpose Назначение

Частотная корреляция принятых сигналов с M-последовательностью (LFSR) и циклическими сдвигами.
Применяется для **пассивной бистатической радиолокации**, FM-детекции,
многогипотезного поиска задержек.

Модуль выполняет R2C/C2R FFT через hipFFT и реализует 4 специализированных HIP ядра
для эффективной обработки на GPU.

> **Namespace**: `drv_gpu_lib` | **Backend**: ROCm (hipFFT) | **Статус**: Active

@section fm_correlator_overview_classes Ключевые классы

| Класс | Описание |
|-------|----------|
| `FMCorrelator` | Facade: `SetParams()`, `Process()` — единая точка входа |
| `FMCorrelatorProcessorROCm` | ROCm реализация: hipFFT R2C/C2R + HIP kernels |
| `FMCorrelatorParams` | Параметры: N (fft_size), K (shifts), S (signals), n_kg |
| `FMCorrelatorResult` | Результат: `peaks[S x K x n_kg]` float |

@section fm_correlator_overview_architecture Архитектура

Pipeline обработки:

```
Reference M-seq [N] float
  │
  ▼
┌────────────────────────────────────────┐
│ 1. apply_cyclic_shifts                 │
│    ref[N] → ref_complex[K × N] float2  │
│    K циклических сдвигов               │
├────────────────────────────────────────┤
│ 2. hipFFT R2C                          │
│    ref_complex → ref_fft [K × (N/2+1)] │
│    inp_signals → inp_fft [S × (N/2+1)] │
│    Hermitian symmetry: экономия 2x     │
├────────────────────────────────────────┤
│ 3. multiply_conj_fused                 │
│    conj(ref_fft) × inp_fft            │
│    3D grid: (N/2+1) × K × S           │
├────────────────────────────────────────┤
│ 4. hipFFT C2R (IFFT)                   │
│    corr_fft → corr_time [S × K × N]   │
├────────────────────────────────────────┤
│ 5. extract_magnitudes_real             │
│    |corr_time| / N → peaks [S×K×n_kg] │
│    bitwise abs (1 instruction)         │
└────────────────────────────────────────┘
```

@section fm_correlator_overview_params Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `fft_size` (N) | int | Размер FFT (степень двойки) |
| `num_shifts` (K) | int | Количество циклических сдвигов |
| `num_signals` (S) | int | Количество входных сигналов |
| `n_kg` | int | Количество выходных точек (пиков) на сдвиг |

@section fm_correlator_overview_quickstart Быстрый старт

@subsection fm_correlator_overview_quickstart_cpp C++

@code{.cpp}
#include "modules/fm_correlator/include/fm_correlator.hpp"

drv_gpu_lib::FMCorrelator corr(backend);
drv_gpu_lib::FMCorrelatorParams p;
p.fft_size = 4096;     // N — размер FFT
p.num_shifts = 8;      // K — циклических сдвигов
p.num_signals = 4;     // S — входных сигналов
p.n_kg = 200;          // точек вывода
corr.SetParams(p);

auto result = corr.Process(reference, signals);
// result.peaks[signal][shift][j] — корреляционные пики
@endcode

@subsection fm_correlator_overview_quickstart_python Python

@code{.py}
import gpuworklib as gw

ctx = gw.ROCmGPUContext(0)
corr = gw.FMCorrelator(ctx,
                       fft_size=4096,
                       num_shifts=8,
                       num_signals=4,
                       n_kg=200)

result = corr.process(ref_np, signals_np)
# result.peaks — массив [S × K × n_kg]
@endcode

@section fm_correlator_overview_kernels HIP Kernels

4 специализированных ядра:

| Kernel | Описание |
|--------|----------|
| `apply_cyclic_shifts` | `ref[N]` float → `ref_complex[K × N]` float2, K циклических сдвигов |
| `multiply_conj_fused` | 3D grid (N/2+1, K, S): комплексное сопряжённое умножение |
| `extract_magnitudes_real` | Bitwise abs (1 инструкция, без ветвлений) |
| `generate_test_inputs` | GPU-генерация тестовых входов с circshift-паттерном |

@note Hermitian symmetry (R2C FFT): обрабатываются только \f$ N/2+1 \f$ бинов —
экономия памяти и вычислений в 2 раза.

@section fm_correlator_overview_dependencies Зависимости

- **DrvGPU** — ROCm backend, GpuContext, GPUProfiler
- **hipFFT** — R2C и C2R FFT

@see fm_correlator_formulas
@see fm_correlator_tests
