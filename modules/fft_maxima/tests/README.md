# FFT Maxima — Тесты

> **Модуль**: `modules/fft_maxima/`
> **Пространство имён**: `antenna_fft`
> **Обновлено**: 2026-02-14

## Обзор

Тесты для FFT-обработки и поиска максимумов спектра. Все тесты — header-only (`*.hpp`), вызываются из `src/main.cpp`.

Каждый тест создаёт собственный `OpenCLBackend` через `InitializeFromExternalContext()`.

---

## Файлы тестов

### test_spectrum_maxima.hpp
**Пространство имён**: `test_spectrum_maxima`

Тесты `SpectrumMaximaFinder` в режимах ONE_PEAK и TWO_PEAKS:
- Генерация синусоид для 5 антенн (2.5, 5.0, 7.5, 10.0, 12.5 Гц)
- FFT-обработка с параболической интерполяцией
- Сравнение результата GPU с аналитической частотой
- Профилирование через GPUProfiler

### test_large_batch.hpp
**Пространство имён**: `test_large_batch`

Стресс-тест пакетной обработки:
- **256 антенн x 1 300 000 точек** (не помещается в GPU-память)
- Автоматическое разбиение на пакеты через BatchManager
- Проверка корректности CPU vs GPU (первые 10 лучей)
- Интеграция GPUProfiler

### test_gpu_generator_integration.hpp
**Пространство имён**: `test_gpu_generator_integration`

Полный интеграционный тест конвейера:
1. `ExternalOpenCLContext` создаёт контекст
2. `CwGenerator` (signal_gen) генерирует данные на GPU (`cl_mem`)
3. `SpectrumMaximaFinder.Process(InputData<cl_mem>)` обрабатывает без загрузки
4. Валидация результатов
5. Бенчмарк + отчёт GPUProfiler

### test_find_all_maxima.hpp
**Пространство имён**: `test_find_all_maxima`
**Дата**: 2026-02-14

Тесты методов `FindAllMaxima` и `AllMaxima` (поиск ВСЕХ локальных максимумов):

| # | Тест | API | Описание |
|---|------|-----|----------|
| 1 | `TestThreePeaks` | `FindAllMaxima(cl_mem, ...)` | 3 синусоиды (50, 120, 200 Гц), эталон CPU DFT, проверка всех максимумов |
| 2 | `TestMultiBeam` | `FindAllMaxima(cl_mem, ...)` | 5 лучей с разными частотами, параллельный beam-aware scan |
| 3 | `TestGpuOutput` | `FindAllMaxima(..., GPU)` | OutputDestination::GPU, проверка возврата cl_mem буферов |
| 4 | `TestFullPipelineCPU` | `FindAllMaxima(InputData<vector>)` | Полный конвейер: CPU данные -> FFT -> обнаружение -> scan -> compaction |
| 5 | `TestFullPipelineGPU` | `FindAllMaxima(InputData<cl_mem>)` | Полный конвейер: GPU данные -> FFT -> обнаружение -> scan -> compaction |
| 6 | `TestAllMaximaCPU` | `AllMaxima(InputData<vector>)` | FFT данные на CPU -> обнаружение -> scan -> compaction (без FFT) |
| 7 | `TestAllMaximaGPU` | `AllMaxima(InputData<cl_mem>)` | FFT данные на GPU -> обнаружение -> scan -> compaction (без FFT) |

**GPU-конвейер**: Обнаружение -> Префиксная сумма (Blelloch Scan) -> Stream Compaction

**Вспомогательные функции**:
- `CpuFindAllMaxima()` — CPU-эталон (аналог SciPy `find_peaks`)
- `CpuDFT()` — наивный DFT для тестовых массивов

### test_batch_all_maxima.hpp
**Пространство имён**: `test_batch_all_maxima`
**Дата**: 2026-02-15

Тесты batch-обработки FindAllMaxima с **CPU и GPU данными**:

| # | Тест | Вход | Dest | Описание |
|---|------|------|------|----------|
| 1 | `TestBatchVectorInput_DestCPU` | CPU (vector) | CPU | Данные с хоста → результат на CPU |
| 2 | `TestBatchVectorInput_DestGPU` | CPU (vector) | GPU | Данные с хоста → результат на GPU |
| 3 | `TestBatchGPUInput_DestCPU` | GPU (cl_mem) | CPU | Данные на GPU → результат на CPU |
| 4 | `TestBatchGPUInput_DestGPU` | GPU (cl_mem) | GPU | Данные на GPU → результат на GPU |
| 5 | `TestBatchWithProfiling` | CPU | CPU | Профилирование (Upload, FFT, Detect, Scan, Compact) |

**Формат результата**: `beams[].maxima` — `vector<MaxValue>` (index, real, imag, magnitude, phase, refined_frequency).

### test_benchmark_all_maxima.hpp
**Пространство имён**: `test_benchmark_all_maxima`
**Дата**: 2026-02-14
**⚠️ DEPRECATED** — заменён `test_fft_maxima_benchmark.hpp` (GpuBenchmarkBase)

Бенчмарк производительности FindAllMaxima (старый стиль):
- **10 лучей x 500 000 точек**, Fs=100 кГц
- Интеграция GPUProfiler с полным отчётом
- ConsoleOutput для мультиGPU-безопасного вывода
- Экспорт профилирования: `Results/Profiler/GPU_00_Profiler/benchmark_all_maxima_*.{md,json}`

### fft_maxima_benchmark.hpp
**Пространство имён**: `test_fft_maxima`
**Дата**: 2026-03-01

Benchmark-классы наследники `GpuBenchmarkBase` (новый стиль):

| Класс | Метод | Стадии | Результаты |
|-------|-------|--------|------------|
| `SpectrumMaximaFinderBenchmark` | `Process(ONE_PEAK)` | Upload, FFT, PostKernel | `Results/Profiler/GPU_00_SpectrumMaxima_Process/` |
| `SpectrumMaximaAllMaximaBenchmark` | `FindAllMaxima` | Upload, FFT, Detect, Scan, Compact | `Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/` |

- `ExecuteKernel()` — warmup без timing (prof_events = nullptr)
- `ExecuteKernelTimed()` — с ProfEvents → RecordEvent → GPUProfiler (min/max/avg автоматически)

### test_fft_maxima_benchmark.hpp
**Пространство имён**: `test_fft_maxima_benchmark`
**Дата**: 2026-03-01

Test runner для двух бенчмарков:
- **10 лучей x 8192 точек**, Fs=100 кГц, комплексные синусоиды
- OpenCL init с `CL_QUEUE_PROFILING_ENABLE` (обязательно для cl_event timing)
- 5 прогревочных прогонов + 20 замерных → PrintReport + ExportJSON + ExportMarkdown
- Если `is_prof=false` в configGPU.json — выводит `[SKIP]`, не падает

### fft_maxima_benchmark_rocm.hpp
**Пространство имён**: `test_fft_maxima_rocm`
**Дата**: 2026-03-01
**Условие компиляции**: `#if ENABLE_ROCM`

ROCm benchmark-классы наследники `GpuBenchmarkBase`:

| Класс | Метод | Стадии | Результаты |
|-------|-------|--------|------------|
| `SpectrumProcessorROCmBenchmark` | `ProcessFromCPU(ONE_PEAK)` | Upload(H2D), PadKernel, FFT, PostKernel, Download(D2H) | `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/` |
| `SpectrumProcessorROCmAllMaximaBenchmark` | `FindAllMaximaFromCPU` | Upload(H2D), PadKernel, FFT, ComputeMagnitudes, Pipeline | `Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/` |

- Timing через `hipEvent_t` (GPU) и `std::chrono` (D2H sync)
- `ExecuteKernel()` — warmup без overhead (prof_events = nullptr)
- `ExecuteKernelTimed()` — с ROCmProfEvents → RecordROCmEvent → GPUProfiler

### test_fft_maxima_benchmark_rocm.hpp
**Пространство имён**: `test_fft_maxima_benchmark_rocm`
**Дата**: 2026-03-01
**Условие компиляции**: `#if ENABLE_ROCM`

Test runner для двух ROCm бенчмарков:
- **10 лучей x 8192 точек**, Fs=100 кГц
- ROCmBackend init → если нет AMD GPU — выводит `[SKIP]`, не падает
- 5 прогревочных + 20 замерных прогонов → PrintReport + ExportJSON + ExportMarkdown

### cpu_fft_reference.hpp

CPU-эталон FFT (обёртка pocketfft) для проверки корректности.

---

## Как запускать

### Все тесты из main:
```bash
cmake --build build --config Release
./build/Release/GPUWorkLib.exe
```

### Выборочный запуск:
Тесты включаются/отключаются комментированием в `src/main.cpp`:
```cpp
test_find_all_maxima::run();         // FindAllMaxima: 7 тестов
test_benchmark_all_maxima::run();    // Бенчмарк: 10 лучей x 500k
test_gpu_generator_integration::run(); // Полный конвейер
test_signal_generators::run();       // Генераторы сигналов
test_fft_processor::run();           // FFT Processor
test_fft_vs_cpu::run();             // FFT vs CPU-эталон
```

---

## Обзор API

### FindAllMaxima (старый API — FFT данные уже на GPU)
```cpp
SpectrumMaximaFinder finder(backend);
AllMaximaResult result = finder.FindAllMaxima(
    fft_data_cl_mem,   // cl_mem с комплексным результатом FFT
    beam_count,        // количество лучей
    nFFT,              // размер FFT
    sample_rate,       // Гц
    OutputDestination::CPU  // CPU, GPU или ALL
);

for (auto& beam : result.beams) {
    // beam.positions[], beam.magnitudes[], beam.frequencies[]
    // beam.num_maxima
}
```

### FindAllMaxima<T> (новый API — полный конвейер с FFT)
```cpp
InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 256;
input.n_point = 1300000;
input.data = signal;
input.sample_rate = 1000.0f;

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
```

### AllMaxima<T> (новый API — обнаружение+scan+compaction, без FFT)
```cpp
InputData<cl_mem> input;
input.antenna_count = 3;
input.n_point = nFFT;       // данные уже после FFT!
input.data = fft_gpu_buffer;
input.sample_rate = 1000.0f;

auto result = finder.AllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);
```

---

## Структура результата

### AllMaximaResult
```cpp
struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams;
    OutputDestination destination;
    void* gpu_positions;    // cl_mem (если GPU/ALL)
    void* gpu_magnitudes;   // cl_mem (если GPU/ALL)
    void* gpu_counts;       // cl_mem (если GPU/ALL)
    size_t total_maxima;
};

struct AllMaximaBeamResult {
    uint32_t antenna_id;
    uint32_t num_maxima;
    std::vector<MaxValue> maxima;  // index, real, imag, magnitude, phase, refined_frequency
};

// GPU: gpu_maxima (MaxValue[]), gpu_counts (uint32_t[])
```

---

## Производительность (RTX 2080 Ti)

| Конфигурация | Время | На луч |
|-------------|-------|--------|
| 1 луч x 1024 FFT | ~0.03 мс | 0.03 мс |
| 5 лучей x 512 FFT | ~0.09 мс | 0.02 мс |
| 256 лучей x 4096 FFT | ~56 мс | 0.22 мс |
| 10 лучей x 500k точек (с FFT) | TBD | TBD |
