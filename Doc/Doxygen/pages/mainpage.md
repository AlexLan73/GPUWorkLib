# GPUWorkLib {#mainpage}

> **Библиотека GPU-вычислений для цифровой обработки сигналов**
> **Платформы**: OpenCL 3.0 (все GPU), ROCm 7.2+ (AMD), Hybrid
> **Архитектура**: Linux/AMD (ветка `main`), Windows/NVIDIA (ветка `nvidia`)

---

## Модули

| # | Модуль | Namespace | Назначение | Docs |
|---|--------|-----------|-----------|------|
| 0 | **DrvGPU** | `drv_gpu_lib` | Unified GPU management (OpenCL / ROCm / Hybrid) | @ref drvgpu_main |
| 1 | **fft_func** | `fft_processor`, `antenna_fft` | Пакетный FFT (hipFFT) + поиск максимумов спектра | @ref fft_func_overview |
| 2 | **statistics** | `statistics` | Mean, Std, Median (Welford, Radix Sort) | @ref statistics_overview |
| 3 | **vector_algebra** | `vector_algebra` | Cholesky inversion (rocsolver) | @ref vector_algebra_overview |
| 4 | **filters** | `filters` | FIR, IIR, MovingAverage, Kalman, KAMA | @ref filters_overview |
| 5 | **signal_generators** | `signal_gen` | CW, LFM, Noise, FormSignal, DelayedFormSignal | @ref signal_generators_overview |
| 6 | **lch_farrow** | `lch_farrow` | Lagrange 48x5 дробная задержка | @ref lch_farrow_overview |
| 7 | **heterodyne** | `drv_gpu_lib` | LFM Dechirp -> beat frequency -> range | @ref heterodyne_overview |
| 8 | **fm_correlator** | `drv_gpu_lib` | FM-корреляция (M-sequence, cyclic shifts) | @ref fm_correlator_overview |
| 9 | **strategies** | `strategies` | Цифровое ДН: CGEMM + FFT + post-FFT scenarios | @ref strategies_overview |
| 10 | **capon** | `capon` | MVDR Capon beamformer | @ref capon_overview |
| 11 | **range_angle** | `range_angle` | 3D: dechirp -> range FFT -> 2D beam FFT | @ref range_angle_overview |

---

## Архитектура

```
Приложение / Модули
       | принимают IBackend*
       v
   DrvGPU (Facade)
       |
       +-- OpenCLBackend  -> cl_mem / cl_command_queue
       +-- ROCmBackend    -> hipStream_t  (ENABLE_ROCM=1)
       +-- HybridBackend  -> OpenCL + ROCm + ZeroCopyBridge
```

**Принцип**: модуль пишется один раз, работает на любом GPU.

**Подробнее**: @ref architecture_page "Архитектура (Ref03)"

---

## Навигация

### Модули (подробно)

Каждый модуль имеет собственную HTML-документацию, связанную через TAGFILES:

- @ref drvgpu_main "DrvGPU" — backends, memory, profiling, services
- @ref fft_func_overview "fft_func" — DFT, zero-padding, mag/phase, peak search
- @ref statistics_overview "statistics" — Welford, radix sort median
- @ref vector_algebra_overview "vector_algebra" — Cholesky, symmetrize
- @ref filters_overview "filters" — FIR, IIR, MA, Kalman, KAMA
- @ref signal_generators_overview "signal_generators" — CW, LFM, Noise, FormSignal
- @ref lch_farrow_overview "lch_farrow" — Lagrange interpolation
- @ref heterodyne_overview "heterodyne" — LFM dechirp, range estimation
- @ref fm_correlator_overview "fm_correlator" — M-sequence correlation
- @ref strategies_overview "strategies" — digital beamforming pipeline
- @ref capon_overview "capon" — MVDR adaptive beamformer
- @ref range_angle_overview "range_angle" — 3D range-angle processing

### Формулы каждого модуля

- @ref fft_func_formulas "FFT — DFT, zero-padding, Blelloch scan"
- @ref signal_generators_formulas "Генераторы — CW, LFM, Box-Muller, Lagrange"
- @ref filters_formulas "Фильтры — FIR, IIR, MA, Kalman, KAMA"
- @ref statistics_formulas "Статистика — Welford, radix sort, histogram"
- @ref heterodyne_formulas "Гетеродин — dechirp, beat freq, range"
- @ref vector_algebra_formulas "Матричная алгебра — Cholesky, POTRF/POTRI"
- @ref capon_formulas "Capon — MVDR, covariance, relief"
- @ref fm_correlator_formulas "FM-корреляция — LFSR, cyclic shift"
- @ref strategies_formulas "Strategies — CGEMM, parabolic interpolation"
- @ref lch_farrow_formulas "Farrow — Lagrange 48x5"

### Тесты

- @ref tests_overview_page "Сводка тестов всех модулей"
- Для каждого модуля: `{module}_tests` — подробные тесты и бенчмарки

### Справка

- @ref modules_overview_page "Обзор модулей" — краткая сводка
- @ref build_guide_page "Сборка проекта" — CMake, ROCm, OpenCL
- **API Reference** — меню "Классы" и "Файлы"

---

## Быстрый старт

**C++ — FFT одного луча:**
```cpp
#include "DrvGPU/include/drv_gpu.hpp"
#include "modules/fft_func/include/fft_processor_rocm.hpp"

drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::ROCM, 0);
gpu.Initialize();

fft_processor::FFTProcessorROCm fft(&gpu.GetBackend());
fft_processor::FFTProcessorParams params;
params.n_point   = 1024;
params.nFFT      = 1024;
params.beam_count = 1;
fft.Initialize(params);

std::vector<std::complex<float>> signal(1024);
// ... заполнить signal ...
auto result = fft.ProcessMagPhase(signal);
// result[0].magnitudes — амплитудный спектр
// result[0].frequencies — частотная ось (Гц)
```

**Python:**
```python
import gpuworklib
ctx = gpuworklib.ROCmGPUContext(0)
fft = gpuworklib.FFTProcessorROCm(ctx, n_point=1024, nFFT=1024, beam_count=1)
result = fft.process_mag_phase(signal_np)  # np.ndarray complex64
```
