# FFT Maxima Module (SpectrumMaximaFinder)

> Поиск максимумов спектра FFT на GPU с параболической интерполяцией

**Namespace**: `antenna_fft`
**Каталог**: `modules/fft_maxima/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Содержание

| Файл | Описание |
|------|----------|
| [API.md](API.md) | Полный API Reference |

---

## Обзор

Модуль ищет **максимумы** (пики) в спектре FFT для массива антенн/лучей.

Основной pipeline:
```
Input Data → Zero-Pad → clFFT → Post-Processing → Peak Search → Parabolic Interpolation → Results
```

### Ключевые возможности

- **Batch processing**: обработка 256+ антенн с автоматическим разбиением по GPU памяти
- **Two peak modes**: ONE_PEAK (лучший из левого/правого) и TWO_PEAKS (оба)
- **Parabolic interpolation**: уточнение частоты до долей бина
- **clFFT pre-callback**: zero-padding в kernel'е для максимальной производительности
- **FFTPlanCache**: кеширование clFFT планов для разных batch_size
- **Universal API**: CPU vectors, cl_mem, SVM pointers

---

## Быстрый старт

### Новый API (v2.0)

```cpp
#include "interface/spectrum_input_data.hpp"

// Подготовка данных
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = my_vector,
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

// Обработка
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

// Результат
for (auto& r : results) {
    printf("Antenna %d: freq=%.4f Hz, mag=%.2f\n",
           r.antenna_id, r.interpolated.refined_frequency, r.interpolated.magnitude);
}
```

### GPU данные

```cpp
antenna_fft::InputData<cl_mem> gpu_input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = my_cl_mem_buffer,
    .gpu_memory_bytes = actual_size,
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

auto results = finder.Process(gpu_input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                  SpectrumMaximaFinder                        │
│  Process(InputData<T>, PeakSearchMode, DriverType)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           AntennaFFTProcMax                          │   │
│  │  ProcessBatch(input, start, count, profiling)        │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │             AntennaFFTCore                           │   │
│  │  Initialize() → AllocateBuffers() → CreateFFTPlan()  │   │
│  │  ExecuteFFT() → ExecutePostKernel() → FindPeaks()    │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │          Support Classes                             │   │
│  │  FFTPlanCache — кеш clFFT планов                     │   │
│  │  FFTBatchAdapter — расчёт batch через BatchManager   │   │
│  │  FFTResultWriter — вывод результатов (MD/JSON)       │   │
│  │  FFTKernelSources — OpenCL kernel source strings     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Файлы

```
modules/fft_maxima/
├── include/
│   ├── interface/
│   │   ├── spectrum_input_data.hpp    # InputData<T>, DriverType, ProcessingParams
│   │   ├── spectrum_maxima_types.h    # PeakSearchMode, MaxValue, SpectrumResult
│   │   └── antenna_fft_params.h       # AntennaFFTParams, FFTResult, FFTMaxResult
│   ├── fft_plan_cache.hpp             # FFTPlanCache (RAII, cache hits tracking)
│   ├── fft_batch_adapter.hpp          # FFTBatchAdapter (DrvGPU::BatchManager)
│   ├── fft_result_writer.hpp          # FFTResultWriter (MD + JSON output)
│   └── kernels/
│       └── fft_kernel_sources.hpp     # OpenCL kernel sources
├── src/
│   ├── antenna_fft_core.cpp
│   ├── antenna_fft_proc_max.cpp
│   └── spectrum_maxima_finder.cpp
├── tests/
│   ├── test_spectrum_maxima.hpp
│   ├── test_fft_maxima.hpp
│   ├── test_large_batch.hpp
│   ├── test_fft_svm.hpp
│   ├── test_external_context_fft.hpp
│   ├── test_signal_generator.hpp
│   ├── test_gpu_generator_integration.hpp
│   └── cpu_fft_reference.hpp
└── CMakeLists.txt
```

---

*Обновлено: 2026-02-13*
