@page fft_func_overview FFT Functions — Обзор

@tableofcontents

@section fft_overview_purpose Назначение

Пакетный 1D FFT через hipFFT + поиск максимумов спектра.
Объединяет бывшие модули `fft_processor` и `fft_maxima`.

> **Namespace**: `fft_processor`, `antenna_fft` | **Backend**: ROCm (hipFFT) | **Статус**: Active

@section fft_overview_classes Ключевые классы

| Класс | Namespace | Описание |
|-------|-----------|----------|
| `FFTProcessorROCm` | `fft_processor` | hipFFT facade: forward/inverse, multi-beam batch |
| `ComplexToMagPhaseROCm` | `fft_processor` | IQ → magnitude + phase (HIP kernel) |
| `SpectrumMaximaFinder` | `antenna_fft` | Поиск пиков: ONE_PEAK, TWO_PEAKS modes |
| `SpectrumProcessorROCm` | `antenna_fft` | ROCm реализация ISpectrumProcessor |
| `AllMaximaPipelineROCm` | `antenna_fft` | ALL_MAXIMA: Detect → Scan → Compact (Blelloch) |
| `PadDataOp` | `fft_processor` | Ref03 Layer 5: zero-padding kernel |
| `MagPhaseOp` | `fft_processor` | Ref03 Layer 5: complex → mag+phase kernel |
| `SpectrumProcessorFactory` | `antenna_fft` | Factory: `Create(BackendType, IBackend*)` |

@section fft_overview_architecture Архитектура

@subsection fft_arch_pipeline Pipeline обработки

```
Signal → [PadData] → [hipFFT C2C] → [MagPhase] → [SpectrumMaxima] → Result
          zero-pad     batch FFT     |X|+∠X        ONE_PEAK / ALL_MAXIMA
```

@subsection fft_arch_all_maxima ALL_MAXIMA Pipeline (Blelloch scan)

4-kernel stream compaction:

| Шаг | Kernel | Описание |
|-----|--------|----------|
| 1 | `detect_all_maxima` | Флаги: `is_max[k] = (\|X[k]\| > \|X[k-1]\|) && (\|X[k]\| > \|X[k+1]\|)` |
| 2 | `block_scan` | Prefix scan (Blelloch) — подсчёт позиций |
| 3 | `block_add` | Добавление сумм блоков (второй проход) |
| 4 | `compact_maxima` | Запись результатов в компактный массив |

@note BLOCK_SIZE = 512, LDS: (BLOCK_SIZE+1) × sizeof(uint32_t) — +1 против bank conflicts.

@section fft_overview_quickstart Быстрый старт

@subsection fft_qs_cpp C++

@code{.cpp}
#include "modules/fft_func/include/fft_processor_rocm.hpp"

fft_processor::FFTProcessorROCm fft(backend);
fft_processor::FFTProcessorParams p{1024, 1024, 4}; // n_point, nFFT, beams
fft.Initialize(p);
auto result = fft.ProcessMagPhase(signal);
// result[beam].magnitudes — амплитуды
// result[beam].frequencies — частоты (Гц)
@endcode

@subsection fft_qs_python Python

@code{.py}
import gpuworklib as gw
ctx = gw.ROCmGPUContext(0)
fft = gw.FFTProcessorROCm(ctx, n_point=1024, nFFT=1024, beam_count=4)
result = fft.process_mag_phase(signal_np)  # np.ndarray complex64
@endcode

@section fft_overview_seealso См. также

- @ref fft_func_formulas — Математика FFT
- @ref fft_func_tests — Тесты и бенчмарки
- @ref drvgpu_main — DrvGPU (базовый драйвер)
