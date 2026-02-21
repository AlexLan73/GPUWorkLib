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
SpectrumMaximaFinder
  → ISpectrumProcessor (Strategy)
  → SpectrumProcessorOpenCL
      → clFFT + pre/post callbacks
      → AllMaximaPipelineOpenCL (Detect → Scan → Compact)
```

## Файлы

```
modules/fft_maxima/
├── include/
│   ├── spectrum_maxima_finder.h
│   ├── interface/
│   │   ├── spectrum_input_data.hpp
│   │   ├── spectrum_maxima_types.h
│   │   ├── i_spectrum_processor.hpp
│   │   └── i_all_maxima_pipeline.hpp
│   ├── processors/
│   │   ├── spectrum_processor_opencl.hpp
│   │   └── spectrum_processor_rocm.hpp
│   ├── pipelines/
│   │   └── all_maxima_pipeline_opencl.hpp
│   └── kernels/
│       ├── fft_kernel_sources.hpp
│       └── all_maxima_kernel_sources.hpp
├── src/
│   ├── spectrum_maxima_finder.cpp
│   ├── spectrum_maxima_finder_process.cpp
│   ├── spectrum_maxima_finder_all_maxima.cpp
│   ├── spectrum_processor_opencl.cpp
│   └── all_maxima_pipeline_opencl.cpp
├── kernels/
│   └── fft_kernels.cl
└── tests/
    ├── test_spectrum_maxima.hpp
    ├── test_find_all_maxima.hpp
    ├── test_batch_all_maxima.hpp
    ├── test_large_batch.hpp
    ├── test_gpu_generator_integration.hpp
    ├── test_benchmark_all_maxima.hpp
    └── cpu_fft_reference.hpp
```

---

*Обновлено: 2026-02-13*
