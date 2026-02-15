# 📝 FFT Maxima (SpectrumMaximaFinder) — Спецификация

> **Модуль**: `fft_maxima` / `SpectrumMaximaFinder`
> **Статус**: 🟢 Active
> **Платформы**: OpenCL (clFFT), ROCm (hipFFT) — planned
> **Namespace**: `antenna_fft`
> **Автор**: Alex
> **Создано**: 2026-02-15
> **Обновлено**: 2026-02-15

---

## 🎯 Назначение

Модуль поиска максимумов в FFT-спектре с параболической интерполяцией. Поддерживает:
- **Process** — поиск одного/двух пиков (ONE_PEAK, TWO_PEAKS)
- **FindAllMaxima** — полный pipeline: сырой сигнал → FFT → поиск ВСЕХ локальных максимумов
- **AllMaxima** — поиск всех максимумов в готовом FFT-спектре (без FFT)

---

## 📐 API и логика работы

### Главный класс: SpectrumMaximaFinder

```cpp
class SpectrumMaximaFinder {
public:
    explicit SpectrumMaximaFinder(drv_gpu_lib::IBackend* backend);

    void Initialize();

    // Универсальный Process — шаблон по типу данных
    template<typename T>
    std::vector<SpectrumResult> Process(
        const InputData<T>& input,
        PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
        DriverType driver = DriverType::ROCM);

    // FindAllMaxima — полный pipeline (сигнал → FFT → detect → scan → compact)
    template<typename T>
    AllMaximaResult FindAllMaxima(
        const InputData<T>& input,
        OutputDestination dest = OutputDestination::CPU,
        DriverType driver = DriverType::OPENCL,
        uint32_t search_start = 0,
        uint32_t search_end = 0);

    // AllMaxima — только detect+scan+compact (данные уже FFT!)
    template<typename T>
    AllMaximaResult AllMaxima(
        const InputData<T>& input,
        OutputDestination dest = OutputDestination::CPU,
        DriverType driver = DriverType::OPENCL,
        uint32_t search_start = 0,
        uint32_t search_end = 0);

    // Low-level: FFT данные уже на GPU
    AllMaximaResult FindAllMaxima(cl_mem fft_data, uint32_t beam_count,
        uint32_t nFFT, float sample_rate, ...);

    const ProfilingData& GetProfilingData() const;
};
```

### Внутренняя логика Process (ONE_PEAK / TWO_PEAKS)

```
1. PrepareParams() — вычисление nFFT, base_fft из antenna_count, n_point, repeat_count
2. Диспетчеризация по типу T:
   - vector<complex<float>> → ProcessFromCPU() (upload → FFT → post-kernel → read)
   - cl_mem → ProcessFromGPU() (copy → FFT → post-kernel → read, batch при необходимости)
   - void* (SVM) → TODO
3. Pre-callback (GPU): zero-padding n_point → nFFT, repeat_count копий
4. clFFT batched: Complex-to-Complex FFT
5. Post-kernel: поиск max(|FFT[i]|²) + параболическая интерполяция
6. ReadResults() → vector<SpectrumResult>
```

### Внутренняя логика FindAllMaxima (полный pipeline)

```
1. PrepareParams() с PeakSearchMode::ALL_MAXIMA
   - max_maxima_per_beam (дефолт 1000) — лимит максимумов на луч
2. Диспетчеризация:
   - CPU → FindAllMaximaFromCPU: Upload → FFT(pre+post callback) → Detect → Scan → Compact
   - GPU → FindAllMaximaFromGPUPipeline: Copy → FFT → Detect → Scan → Compact
3. Batch-обработка (автоматическая):
   - BatchManager проверяет вместимость (CalculateOptimalBatchSize)
   - Если не влезает → разбивка на batch с beam_offset
   - При Dest=GPU: создаёт общие буферы positions/magnitudes/counts
   - Каждый batch записывает с правильным beam_offset → корректные antenna_id
4. FFT с post-callback: |FFT[i]| → magnitudes_buffer_
5. Detect: локальные максимумы (mag[i] > mag[i-1] && mag[i] > mag[i+1])
6. Prefix Sum (Blelloch Scan): beam-aware параллельный scan
7. Compact: stream compaction → позиции и амплитуды (с beam_offset для batch)
8. Output: CPU (clEnqueueReadBuffer) / GPU (cl_mem) / ALL
```

**Особенности batch-обработки FindAllMaxima:**
- При Dest=CPU: каждый batch возвращает результаты, которые мерджатся с коррекцией antenna_id
- При Dest=GPU: все batch записывают в общие GPU-буферы с beam_offset — нет лишних копирований
- Профилирование: GPUProfiler собирает метрики по всем batch, вывод через PrintReport()

### Внутренняя логика AllMaxima (без FFT)

```
1. input.n_point = nFFT (данные уже FFT!)
2. CPU: AllMaximaFromCPU — Upload → ComputeMagnitudes → Detect → Scan → Compact
3. GPU: FindAllMaxima(cl_mem) — ComputeMagnitudes → Detect → Scan → Compact
```

---

## 📥 Интерфейс входных данных InputData\<T\>

### Структура

```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;     // Количество антенн (лучей)
    uint32_t n_point = 0;           // Точек на антенну (или nFFT для AllMaxima!)
    T data{};                       // Данные
    size_t gpu_memory_bytes = 0;   // Реальный размер GPU буфера (для cl_mem)

    uint32_t repeat_count = 2;      // nFFT = nextPow2(n_point) × repeat_count
    float sample_rate = 1000.0f;    // Гц
    uint32_t search_range = 0;      // 0 = auto = nFFT/4
    float memory_limit = 0.80f;     // Доля GPU памяти для batch (0.0-1.0)
    size_t max_maxima_per_beam = 1000;  // Макс. кол-во максимумов на луч (FindAllMaxima)

    size_t TotalPoints() const;
    size_t SizeBytes() const;
    size_t ActualGpuMemory() const;
};
```

### Варианты типов T и использования

| T | Источник данных | Когда использовать |
|---|-----------------|-------------------|
| `std::vector<std::complex<float>>` | CPU (host) | Данные сгенерированы на CPU, нужен upload |
| `cl_mem` | GPU (OpenCL) | Данные от CwGenerator, FFTProcessor или заказчика |
| `void*` | SVM | Zero-copy (TODO, не реализовано) |

### Варианты использования по сценариям

#### 1. CPU данные — Process (один пик)

```cpp
std::vector<std::complex<float>> signal = GenerateSignal();  // 256 * 1024 точек

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1024,
    .data = signal,
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

#### 2. GPU данные от генератора — Process (zero-copy)

```cpp
// CwGenerator уже создал cl_mem
cl_mem gpu_signal = generator.GetOutputBuffer();

InputData<cl_mem> input{
    .antenna_count = 256,
    .n_point = 1024,
    .data = gpu_signal,
    .gpu_memory_bytes = 256 * 1024 * sizeof(std::complex<float>),
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

#### 3. Большие данные — batch автоматически

```cpp
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,   // 1.3M точек на антенну
    .data = huge_signal,
    .memory_limit = 0.80f // 80% свободной GPU памяти на batch
};

// BatchManager разобьёт на пакеты автоматически
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK);
```

#### 4. FindAllMaxima — полный pipeline из CPU

```cpp
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,
    .n_point = 1024,
    .data = raw_signal,   // Сырой сигнал (НЕ FFT!)
    .sample_rate = 1000.0f
};

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
// result.beams[i].positions, .magnitudes, .frequencies
```

#### 5. AllMaxima — FFT уже посчитан

```cpp
// FFT результат уже на GPU
InputData<cl_mem> fft_input{
    .antenna_count = 5,
    .n_point = 1024,       // = nFFT (размер FFT!)
    .data = gpu_fft_result, // Комплексный спектр
    .sample_rate = 1000.0f
};

auto result = finder.AllMaxima(fft_input, OutputDestination::CPU);
```

---

## 📂 Организация кода

### Структура модуля

```
modules/fft_maxima/
├── include/
│   ├── spectrum_maxima_finder.h      # Главный фасад
│   ├── antenna_fft_core.h           # Legacy: AntennaFFTCore (Release/Debug)
│   ├── antenna_fft_release.h
│   ├── interface/
│   │   ├── spectrum_input_data.hpp  # InputData<T>, ProcessingParams, DriverType
│   │   ├── spectrum_maxima_types.h  # SpectrumResult, AllMaximaResult, MaxValue
│   │   ├── antenna_fft_params.h    # AntennaFFTParams, FFTResult (legacy)
│   │   ├── i_spectrum_processor.hpp # Strategy interface
│   │   ├── i_all_maxima_pipeline.hpp
│   │   └── i_backend.hpp
│   ├── processors/
│   │   ├── spectrum_processor_opencl.hpp  # OpenCL реализация
│   │   └── spectrum_processor_rocm.hpp    # ROCm stub
│   ├── pipelines/
│   │   └── all_maxima_pipeline_opencl.hpp # Detect→Scan→Compact
│   ├── factory/
│   │   └── spectrum_processor_factory.hpp
│   ├── kernels/
│   │   ├── fft_kernel_sources.hpp
│   │   └── all_maxima_kernel_sources.hpp
│   ├── fft_plan_cache.hpp
│   ├── fft_batch_adapter.hpp
│   └── fft_result_writer.hpp
├── src/
│   ├── spectrum_maxima_finder.cpp
│   ├── spectrum_maxima_finder_process.cpp
│   ├── spectrum_maxima_finder_all_maxima.cpp
│   ├── spectrum_processor_opencl.cpp
│   ├── spectrum_processor_rocm.cpp
│   ├── spectrum_processor_factory.cpp
│   ├── all_maxima_pipeline_opencl.cpp
│   ├── antenna_fft_core.cpp
│   └── antenna_fft_release.cpp
├── kernels/
│   └── fft_kernels.cl
├── tests/
│   ├── README.md
│   ├── test_spectrum_maxima.hpp
│   ├── test_find_all_maxima.hpp
│   ├── test_large_batch.hpp
│   ├── test_gpu_generator_integration.hpp
│   ├── test_benchmark_all_maxima.hpp
│   ├── cpu_fft_reference.hpp
│   └── ...
└── CMakeLists.txt
```

### Паттерны проектирования

| Паттерн | Где используется | Описание |
|---------|------------------|----------|
| **Facade** | `SpectrumMaximaFinder` | Единая точка входа, скрывает ProcessFromCPU/GPU, batch |
| **Strategy** | `ISpectrumProcessor` | Выбор OpenCL vs ROCm реализации |
| **Factory** | `SpectrumProcessorFactory` | Создание процессора по `DriverType` |
| **Pipeline** | `AllMaximaPipelineOpenCL` | Detect → Scan → Compact (цепочка шагов) |
| **Adapter** | `FFTBatchAdapter` | Адаптация к BatchManager для расчёта batch'ей |
| **Template Method** | `Process<T>` | Единый интерфейс для разных типов данных |
| **RAII** | `FFTPlanCache` | Кеш clFFT планов, освобождение в деструкторе |

---

## 🔌 Подключение ROCm

### Текущее состояние

- **SpectrumProcessorROCm** — stub, все методы `throw std::runtime_error("ROCm not implemented")`
- **SpectrumProcessorFactory::Create(DriverType::ROCM, backend)** — создаёт stub, при вызове методов — throw

### Место для реализации ROCm

**Файлы:**
- `include/processors/spectrum_processor_rocm.hpp`
- `src/spectrum_processor_rocm.cpp`

**Зависимости:**
- `hipFFT` (rocFFT) вместо clFFT
- `hipDeviceptr_t` вместо `cl_mem`
- HIP kernels вместо OpenCL kernels (или hipify)

**План интеграции:**

1. **IBackend** — расширить для ROCm: `GetHipContext()`, `GetHipStream()` (или аналог)
2. **SpectrumProcessorROCm** — реализовать методы `ISpectrumProcessor`:
   - `Initialize()` — создание hipFFT плана
   - `ProcessFromCPU()` — hipMalloc + hipMemcpy + hipFFT + kernel
   - `ProcessFromGPU()` — hipFFT + kernel (данные уже на GPU)
   - `FindAllMaxima()`, `AllMaximaFromCPU()` — аналогично OpenCL pipeline
3. **AllMaximaPipelineROCm** — создать по аналогии с `AllMaximaPipelineOpenCL`
4. **Kernels** — портировать `.cl` в HIP (hipify или ручной перевод)

**Документация ROCm:**
- Сохранить план в `MemoryBank/research/` или `Doc_Addition/Info_rocm_integration.md`
- После реализации — обновить эту спецификацию

### Условия компиляции

```cpp
#ifdef USE_ROCM
    // SpectrumProcessorROCm с hipFFT
#else
    // SpectrumProcessorOpenCL с clFFT
#endif
```

Или выбор в runtime через `DriverType` (текущий подход).

---

## 📋 Требования

### Функциональные
- [x] REQ-001: Process — один пик (ONE_PEAK) с параболической интерполяцией
- [x] REQ-002: Process — два пика (TWO_PEAKS)
- [x] REQ-003: FindAllMaxima — полный pipeline (сигнал → FFT → все максимумы)
- [x] REQ-004: AllMaxima — поиск в готовом FFT (без FFT)
- [x] REQ-005: InputData\<T\> — CPU (vector), GPU (cl_mem)
- [ ] REQ-006: InputData\<void*\> — SVM (zero-copy)
- [ ] REQ-007: ROCm backend (hipFFT)

### Нефункциональные
- [x] NFR-001: Batch processing через BatchManager (memory_limit)
- [x] NFR-002: Профилирование через GPUProfiler
- [x] NFR-003: Multi-GPU через IBackend

---

## 📊 Примеры (логика вызова)

### Пример 1: Простой Process из CPU

```cpp
SpectrumMaximaFinder finder(backend);
finder.Initialize();  // Вызывается автоматически при первом Process

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,
    .n_point = 1000,
    .data = my_signal,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input);
// results.size() == 5
// results[i].interpolated.refined_frequency — частота пика
// results[i].interpolated.magnitude — амплитуда
```

### Пример 2: Интеграция с CwGenerator (GPU→GPU)

```cpp
CwGenerator gen(backend);
gen.Generate(256, 1024, 50.0f, 1000.0f);  // 256 лучей, 1024 точки, 50 Гц
cl_mem gpu_signal = gen.GetOutputBuffer();

InputData<cl_mem> input{
    .antenna_count = 256,
    .n_point = 1024,
    .data = gpu_signal,
    .gpu_memory_bytes = 256 * 1024 * 8,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input);
// Нет upload! Только GPU copy → FFT → post-kernel
```

### Пример 3: FindAllMaxima — все пики

```cpp
InputData<cl_mem> input{
    .antenna_count = 3,
    .n_point = 1024,
    .data = raw_signal_gpu,  // Сырой сигнал!
    .sample_rate = 1000.0f
};

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);

for (const auto& beam : result.beams) {
    for (uint32_t j = 0; j < beam.num_maxima; ++j) {
        printf("Beam %u: freq=%.2f Hz, mag=%.4f\n",
               beam.antenna_id, beam.frequencies[j], beam.magnitudes[j]);
    }
}
```

---

## 📚 Ссылки

- **Спецификация FFT**: `MemoryBank/specs/fft_module.md`
- **API Reference**: `Doc/Modules/fft_maxima/API.md`
- **Python API**: `Doc/Python/spectrum_maxima_api.md`
- **Тесты**: `modules/fft_maxima/tests/README.md`

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-02-15 | Кодо | Создание полной спецификации fft_maxima |
