# API Reference — SpectrumMaximaFinder (fft_maxima)

> Поиск максимумов FFT-спектра на GPU с параболической интерполяцией.
> Два режима: **один пик** (быстро, sub-bin уточнение) и **все локальные максимумы** (Detect → Scan → Compact).

**Namespace**: `antenna_fft`

---

## Подключение

```cpp
// C++
#include "modules/fft_maxima/include/spectrum_maxima_finder.h"
```

```python
# Python
import gpuworklib
finder = gpuworklib.SpectrumMaximaFinder(ctx)  # ctx = gpuworklib.GPUContext(0)
```

---

## Backend — откуда берётся

`SpectrumMaximaFinder` принимает `IBackend*` — указатель на GPU-бэкенд из DrvGPU.
**Не владеет указателем** — backend должен жить дольше объекта.

```cpp
// Single GPU (OpenCL — рекомендуется в production)
drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
gpu.Initialize();
antenna_fft::SpectrumMaximaFinder finder(&gpu.GetBackend());

// Через OpenCLBackend напрямую (в тестах/бенчмарках)
auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
backend->InitializeFromExternalContext(context, device, queue);
antenna_fft::SpectrumMaximaFinder finder(backend.get());
```

---

## Типы данных

### PeakSearchMode

```cpp
enum class PeakSearchMode {
    ONE_PEAK,    // Один пик (наибольший) + параболическая интерполяция → 4 MaxValue на луч
    TWO_PEAKS,   // Два пика (левый и правый диапазон) → 8 MaxValue на луч
    ALL_MAXIMA   // Все локальные максимумы (pipeline Detect → Scan → Compact)
};
```

### DriverType (= BackendType)

```cpp
// DriverType — алиас drv_gpu_lib::BackendType
using DriverType = drv_gpu_lib::BackendType;
// Значения: AUTO, OPENCL, ROCm
```

### OutputDestination

```cpp
enum class OutputDestination {
    CPU,   // Результат скачивается на CPU (beams заполнены, gpu_maxima = nullptr)
    GPU,   // Результат остаётся на GPU (beams пусты, caller освобождает gpu_maxima/gpu_counts!)
    ALL    // И CPU и GPU (caller освобождает gpu_maxima/gpu_counts!)
};
```

### InputData\<T\>

Универсальная структура входных данных. Тип `T`:
- `std::vector<std::complex<float>>` — CPU данные
- `cl_mem` — OpenCL буфер на GPU
- `void*` — SVM pointer

```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;  // Число антенн/лучей
    uint32_t n_point = 0;        // Точек на антенну (до zero-padding)
                                 // ⚠️ Для AllMaxima(): n_point = nFFT (данные уже FFT!)
    T data{};                    // Данные (CPU или GPU)
    size_t gpu_memory_bytes = 0; // Размер GPU буфера (0 = auto, только для cl_mem)

    uint32_t repeat_count = 2;   // nFFT = nextPow2(n_point) * repeat_count
    float sample_rate = 1000.0f; // Частота дискретизации, Гц
    uint32_t search_range = 0;   // Диапазон поиска бинов (0 = auto = nFFT/4)
    float memory_limit = 0.80f;  // Доля GPU памяти для батча (0.0–1.0)

    size_t TotalPoints() const;      // antenna_count * n_point
    size_t SizeBytes() const;        // TotalPoints() * sizeof(complex<float>)
    size_t ActualGpuMemory() const;  // gpu_memory_bytes или SizeBytes()
};
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `antenna_count` | `uint32_t` | Число параллельных лучей/антенн |
| `n_point` | `uint32_t` | Точек на луч (для Process: входные сэмплы; для AllMaxima: nFFT) |
| `data` | `T` | CPU-вектор, `cl_mem` или SVM-указатель |
| `gpu_memory_bytes` | `size_t` | Для `cl_mem`: реальный размер буфера; 0 = auto |
| `repeat_count` | `uint32_t` | Zero-padding коэффициент: nFFT = nextPow2(n_point) × repeat_count |
| `sample_rate` | `float` | Гц; нужна для freq[k] = k × fs/nFFT |
| `search_range` | `uint32_t` | Число бинов для поиска пика (0 = nFFT/4) |
| `memory_limit` | `float` | BatchManager не превысит эту долю свободной GPU памяти |

### MaxValue

GPU-совместимая структура (32 байта, aligned).

```cpp
struct MaxValue {
    uint32_t index;           // Бин FFT [0, nFFT)
    float real, imag;         // Re/Im FFT[index]
    float magnitude;          // |FFT[index]| = sqrt(real²+imag²)
    float phase;              // arg(FFT[index]) в ГРАДУСАХ = atan2(imag,real)×180/π
    float freq_offset;        // Дробная поправка δ ∈ [-0.5, +0.5] (параболическая интерполяция)
    float refined_frequency;  // Уточнённая частота Гц = (index + δ) × sample_rate / nFFT
                              // В AllMaxima/FindAllMaxima: δ=0 (без интерполяции)
    uint32_t pad;             // Выравнивание до 32 байт
};
```

### SpectrumResult

Результат для одной антенны (режим ONE_PEAK или TWO_PEAKS).

```cpp
struct SpectrumResult {
    uint32_t antenna_id;   // Индекс антенны в батче
    MaxValue interpolated; // Параболически уточнённый пик
    MaxValue left_point;   // FFT[peak_bin - 1] — левый сосед
    MaxValue center_point; // FFT[peak_bin]     — сам пик
    MaxValue right_point;  // FFT[peak_bin + 1] — правый сосед
};
```

> При ONE_PEAK: `vector<SpectrumResult>` содержит `antenna_count` элементов.
> При TWO_PEAKS: `2 × antenna_count` элементов — `[left0, right0, left1, right1, ...]`.

### AllMaximaBeamResult

Все максимумы одного луча (режим ALL_MAXIMA).

```cpp
struct AllMaximaBeamResult {
    uint32_t antenna_id;          // Индекс луча
    uint32_t num_maxima;          // Реальное число найденных максимумов
    std::vector<MaxValue> maxima; // CPU-копия (пусто при dest=GPU)
};
```

### AllMaximaResult

Выходной контейнер `FindAllMaxima` / `AllMaxima`.

```cpp
struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams; // По лучам (заполнено при dest=CPU/ALL)
    OutputDestination destination;
    void* gpu_maxima = nullptr;  // GPU буфер MaxValue[beams×max_per_beam] — ⚠️ caller owns при GPU/ALL!
    void* gpu_counts = nullptr;  // GPU буфер uint32[beams] — ⚠️ caller owns при GPU/ALL!
    size_t total_maxima = 0;     // Суммарное число найденных максимумов
    size_t gpu_bytes = 0;        // Размер gpu_maxima в байтах
    size_t TotalMaxima() const;
};
```

> ⚠️ **Ownership при dest=GPU или dest=ALL** — caller обязан освободить:
> - OpenCL: `clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima))`
> - ROCm: `hipFree(result.gpu_maxima)`
>
> При `dest=CPU` — `gpu_maxima/gpu_counts == nullptr` (освобождены внутри).

### SpectrumParams

```cpp
struct SpectrumParams {
    uint32_t antenna_count = 5;
    uint32_t n_point = 1000;
    uint32_t repeat_count = 2;
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;   // 0 = auto = nFFT/4
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK;
    float memory_limit = 0.80f;
    // Вычисляемые (заполняются после Initialize/Process)
    uint32_t nFFT = 0;    // nextPow2(n_point) × repeat_count
    uint32_t base_fft = 0; // nextPow2(n_point)
};
```

### ProfilingData

```cpp
struct ProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_kernel_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};
```

---

## SpectrumMaximaFinder

**Файл**: `modules/fft_maxima/include/spectrum_maxima_finder.h`
**Платформа**: OpenCL (clFFT). ROCm backend — в разработке.

### Конструкторы / Деструктор

```cpp
// ✅ Новый API (рекомендуется)
explicit SpectrumMaximaFinder(drv_gpu_lib::IBackend* backend);

// ⚠️ Устаревший API (deprecated)
[[deprecated("Use SpectrumMaximaFinder(IBackend*) + Process(InputData<T>, ...)")]]
explicit SpectrumMaximaFinder(const SpectrumParams& params, drv_gpu_lib::IBackend* backend);

~SpectrumMaximaFinder();  // Освобождает GPU ресурсы (RAII)

SpectrumMaximaFinder(const SpectrumMaximaFinder&) = delete;
SpectrumMaximaFinder(SpectrumMaximaFinder&&) noexcept;
```

### Initialize()

```cpp
void Initialize();
```

Создаёт GPU буферы, FFT план с pre-callback, компилирует post-kernel.
При новом API вызывается **автоматически** внутри `Process()` при первом вызове с CPU данными.

**Исключения**: `std::runtime_error` при ошибке GPU инициализации.

---

### Process\<T\>() — один пик на антенну

```cpp
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::ROCm,
    ProfEvents* prof_events = nullptr);
```

Полный pipeline: **сырой сигнал → Zero-Pad → FFT → post-kernel (поиск пика + парабола)**.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input` | `InputData<T>` | Входной сигнал (не FFT!) — CPU/GPU/SVM |
| `mode` | `PeakSearchMode` | ONE_PEAK или TWO_PEAKS |
| `driver` | `DriverType` | OPENCL / ROCm (ROCm пока не реализован) |
| `prof_events` | `ProfEvents*` | `nullptr` = production; не-`nullptr` = сбор cl_event |

**Возвращает**: `vector<SpectrumResult>` — один результат на антенну (ONE_PEAK) или два (TWO_PEAKS).

---

### FindAllMaxima\<T\>() — все пики из сырого сигнала

```cpp
template<typename T>
AllMaximaResult FindAllMaxima(
    const InputData<T>& input,
    OutputDestination dest = OutputDestination::CPU,
    DriverType driver = DriverType::OPENCL,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    ProfEvents* prof_events = nullptr);
```

Полный pipeline: **сырой сигнал → Zero-Pad → FFT → Detect → Scan → Compact**.
`input.n_point` = размер входных сэмплов (до FFT).

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input` | `InputData<T>` | Сырой сигнал (не FFT!) |
| `dest` | `OutputDestination` | CPU / GPU / ALL |
| `driver` | `DriverType` | OPENCL / ROCm |
| `search_start` | `uint32_t` | Первый бин поиска (0 = auto = 1, пропуск DC) |
| `search_end` | `uint32_t` | Последний бин поиска (0 = auto = nFFT/2) |
| `prof_events` | `ProfEvents*` | Опциональный сбор событий |

**Возвращает**: `AllMaximaResult`

---

### AllMaxima\<T\>() — все пики из готового FFT

```cpp
template<typename T>
AllMaximaResult AllMaxima(
    const InputData<T>& input,
    OutputDestination dest = OutputDestination::CPU,
    DriverType driver = DriverType::OPENCL,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    ProfEvents* prof_events = nullptr);
```

Pipeline без FFT: **ComputeMag → Detect → Scan → Compact**.
`input.data` содержит **уже посчитанный FFT-спектр**, `input.n_point` = **nFFT** (не n_point сигнала!).

> ⚠️ **Критично**: `input.n_point = nFFT` — размер FFT-спектра, не исходного сигнала!
> Если передать сырой сигнал, compute_magnitudes посчитает sqrt(Re²+Im²) от отсчётов — результат неверный.

---

### FindAllMaxima() — низкоуровневая перегрузка (cl_mem)

```cpp
AllMaximaResult FindAllMaxima(
    cl_mem fft_data,           // GPU буфер с FFT (beam_count * nFFT float2)
    uint32_t beam_count,
    uint32_t nFFT,
    float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 0,
    uint32_t search_end = 0,
    uint32_t beam_offset = 0,              // Смещение начального луча (для батч-pipeline)
    cl_mem external_out_maxima = nullptr,  // Внешний GPU буфер для MaxValue (или nullptr)
    cl_mem external_out_counts = nullptr,  // Внешний GPU буфер для counts (или nullptr)
    ProfEvents* prof_events = nullptr);
```

Низкоуровневая версия: принимает FFT уже загруженный на GPU.
Используется внутри `AllMaxima<cl_mem>` и при ручном батч-pipeline.

---

### Вспомогательные методы

```cpp
ProfilingData GetProfilingData() const; // Тайминги последнего вызова
const SpectrumParams& GetParams() const;
bool IsInitialized() const;
void PrintInfo() const;
```

---

## Python API

### Конструктор

```python
finder = gpuworklib.SpectrumMaximaFinder(ctx)
# ctx: gpuworklib.GPUContext(device_index=0)
```

### find_all_maxima()

```python
result = finder.find_all_maxima(
    fft_data,           # numpy complex64 — FFT-спектр (1D или 2D)
    sample_rate,        # float, Гц
    beam_count=0,       # int — 0 = авто из shape
    nFFT=0,             # int — 0 = авто из shape
    search_start=0,     # int — первый бин (0 = auto = 1)
    search_end=0        # int — последний бин (0 = auto = nFFT/2)
)
```

**Формат `fft_data`**:
- 1D `(nFFT,)` → один луч → возвращает `dict`
- 2D `(beams, nFFT)` → несколько лучей → возвращает `list[dict]`

> ⚠️ Принимает **FFT-спектр**, не сырой сигнал! Сначала: `fft.process_complex(signal, sample_rate=fs)`

**Возвращает** (один луч — `dict`, несколько — `list[dict]`):

```python
# Один луч (1D input)
{
    "positions":   np.ndarray(uint32),   # Бины найденных пиков
    "magnitudes":  np.ndarray(float32),  # |FFT[bin]| для каждого пика
    "frequencies": np.ndarray(float32),  # bin × fs/nFFT, Гц
    "num_maxima":  int                   # Число найденных максимумов
}

# Несколько лучей (2D input) — list[dict], дополнительно:
{ "antenna_id": int, ...те же поля... }
```

---

## Цепочки вызовов

### ONE_PEAK — CPU данные

```
SpectrumMaximaFinder finder(backend)
  └─→ Process(input)
      ├─→ Initialize() (при первом вызове)    // GPU буферы + FFT план
      ├─→ UploadData()                         // H2D: vector → pre_callback_userdata_
      ├─→ ExecuteFFT()                         // clFFT с pre-callback (zero-padding inline)
      ├─→ ExecutePostKernel()                  // GPU reduction: max(|FFT|) + парабола
      └─→ ReadResults()                        // D2H: 4×MaxValue на антенну
```

### FindAllMaxima — GPU данные (cl_mem → AllMaxima)

```
SignalGenerator → cl_mem (сырой сигнал на GPU)
  └─→ FindAllMaxima(InputData<cl_mem>{...}, OutputDestination::CPU)
      ├─→ CopyGpuData()                        // D2D: cl_mem → pre_callback_userdata_
      ├─→ ExecuteAllMaximaFFT()                // clFFT с post-callback (→ magnitudes_buffer_)
      ├─→ ExecuteDetectKernel()                // GPU: detect_flags[i] = 1 if local_max
      ├─→ ExecutePrefixSum()                   // GPU: beam-aware parallel prefix sum
      ├─→ ExecuteCompactKernel()               // GPU: compact → MaxValue[]
      └─→ D2H download                          // result.beams заполнены
```

### AllMaxima — готовый FFT спектр

```
FFTProcessor::ProcessComplex() → vector<FFTComplexResult>
  └─→ AllMaxima(InputData<vector>{
          .n_point = nFFT,      // ⚠️ n_point = nFFT спектра!
          .data = spectrum
      })
      ├─→ Upload FFT на GPU                    // H2D
      ├─→ compute_magnitudes kernel            // GPU: |FFT[i]| → magnitudes_buffer_
      ├─→ Detect → PrefixSum → Compact
      └─→ D2H download
```

### Python pipeline

```
GPUContext(0)
  └─→ FFTProcessor(ctx)
      └─→ process_complex(signal, sample_rate=fs)   // → numpy complex64 (FFT)
          └─→ SpectrumMaximaFinder(ctx)
              └─→ find_all_maxima(spectrum, sample_rate=fs)
                  └─→ dict / list[dict] с positions/magnitudes/frequencies
```

---

## Примеры

### C++ — ONE_PEAK, CPU данные

```cpp
#include "modules/fft_maxima/include/spectrum_maxima_finder.h"
#include "DrvGPU/include/drv_gpu.hpp"

drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
gpu.Initialize();

antenna_fft::SpectrumMaximaFinder finder(&gpu.GetBackend());

antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,
    .n_point = 1000,
    .data = signal_data,    // 5 * 1000 complex<float>
    .repeat_count = 2,      // nFFT = 2048
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, antenna_fft::PeakSearchMode::ONE_PEAK,
                              antenna_fft::DriverType::OPENCL);

for (const auto& r : results) {
    printf("Antenna %u: freq=%.4f Hz, mag=%.2f, phase=%.2f deg\n",
           r.antenna_id,
           r.interpolated.refined_frequency,
           r.interpolated.magnitude,
           r.interpolated.phase);
}
```

### C++ — FindAllMaxima из GPU буфера (cl_mem)

```cpp
// Сигнал уже на GPU (например от SignalGenerator)
cl_mem gpu_signal = ...;
size_t buf_bytes = beam_count * n_point * sizeof(std::complex<float>);

antenna_fft::InputData<cl_mem> input{
    .antenna_count = 256,
    .n_point = 1024,
    .data = gpu_signal,
    .gpu_memory_bytes = buf_bytes,
    .repeat_count = 1,      // nFFT = 1024
    .sample_rate = 1000.0f
};

auto result = finder.FindAllMaxima(input, antenna_fft::OutputDestination::CPU,
                                   antenna_fft::DriverType::OPENCL);

printf("Total maxima: %zu\n", result.total_maxima);
for (const auto& beam : result.beams) {
    printf("Beam %u: %u maxima\n", beam.antenna_id, beam.num_maxima);
    for (const auto& m : beam.maxima)
        printf("  bin=%u, freq=%.2f Hz, mag=%.4f\n", m.index, m.refined_frequency, m.magnitude);
}
```

### C++ — AllMaxima с Dest=GPU (zero-copy в следующий модуль)

```cpp
auto result = finder.AllMaxima(fft_input, antenna_fft::OutputDestination::GPU);
// result.gpu_maxima — cl_mem с MaxValue[], передать в следующий модуль

// ⚠️ ОБЯЗАТЕЛЬНО освободить после использования!
clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima));
clReleaseMemObject(static_cast<cl_mem>(result.gpu_counts));
```

### C++ — Batch processing (256 антенн × 1.3M точек)

```cpp
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = huge_data,        // 256 * 1.3M * 8 = ~2.66 GB
    .repeat_count = 2,
    .sample_rate = 10000.0f,
    .memory_limit = 0.80f     // BatchManager делит на батчи автоматически
};

auto results = finder.Process(input, antenna_fft::PeakSearchMode::ONE_PEAK,
                              antenna_fft::DriverType::OPENCL);
// results.size() == 256
```

### Python — найти все пики (один луч)

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

fs = 1000.0
nFFT = 1024
freqs = [100.0, 200.0, 350.0]

# Сигнал: три тона
t = np.arange(nFFT, dtype=np.float32)
signal = sum(np.sin(2 * np.pi * f * t / fs) for f in freqs).astype(np.complex64)

# FFT на GPU, затем поиск максимумов
spectrum = fft.process_complex(signal, sample_rate=fs)  # numpy complex64
result = finder.find_all_maxima(spectrum, sample_rate=fs)

print(f"Found {result['num_maxima']} maxima:")
for i in range(result['num_maxima']):
    print(f"  bin={result['positions'][i]}, "
          f"freq={result['frequencies'][i]:.2f} Hz, "
          f"mag={result['magnitudes'][i]:.2f}")
```

### Python — несколько лучей (2D array)

```python
# shape: (3, nFFT) — три луча
spectra = np.vstack([
    fft.process_complex(
        np.sin(2 * np.pi * f * t / fs).astype(np.complex64), sample_rate=fs
    )
    for f in [100.0, 200.0, 300.0]
])  # shape: (3, 1024)

results = finder.find_all_maxima(spectra, sample_rate=fs)
# results — list[dict], каждый с antenna_id
for r in results:
    print(f"Beam {r['antenna_id']}: {r['num_maxima']} maxima, "
          f"top freq = {r['frequencies'][0]:.1f} Hz")
```

---

## Константы

| Константа | Значение | Описание |
|-----------|---------|----------|
| `LOCAL_SIZE` | 256 | Work-group для post-kernel (ONE_PEAK/TWO_PEAKS) |
| `SCAN_LOCAL_SIZE` | 256 | Work-group для block_scan (AllMaxima prefix sum) |
| `SCAN_BLOCK_SIZE` | 512 | = 2 × SCAN_LOCAL_SIZE — элементов на блок |
| `PRE_CALLBACK_HEADER_SIZE` | 32 байта | Заголовок `pre_callback_userdata_` |

---

## Ограничения и нюансы

1. **`AllMaxima` vs `FindAllMaxima`** — разные входы:
   - `FindAllMaxima(InputData<T>)` принимает **сырой сигнал** и делает FFT
   - `AllMaxima(InputData<T>)` принимает **FFT-спектр**, `n_point` = nFFT

2. **Python принимает FFT-спектр** — `find_all_maxima(fft_data, ...)` ждёт `complex64`.
   Сначала: `spectrum = fft.process_complex(signal, sample_rate=fs)`

3. **Dest=GPU — caller освобождает** — `result.gpu_maxima` и `result.gpu_counts` при `dest=GPU/ALL` принадлежат caller'у. Обязателен `clReleaseMemObject()`.

4. **clFFT не работает на AMD RDNA4+** (gfx1201, Radeon 9070). OpenCL тесты закомментированы. На AMD — только ROCm.

5. **Параболическая интерполяция** — только ONE_PEAK/TWO_PEAKS. В AllMaxima/FindAllMaxima: `refined_frequency = index × fs/nFFT` (без δ).

6. **search_start=0 → auto=1** — DC-бин пропускается. `search_end=0` → nFFT/2 (только положительные частоты).

7. **Deprecated конструктор** — `SpectrumMaximaFinder(SpectrumParams, IBackend*)` → `[[deprecated]]`. Используй `SpectrumMaximaFinder(IBackend*)`.

8. **Batch processing автоматический** — `memory_limit` управляет долей GPU памяти. BatchManager делит `antenna_count` на батчи без участия пользователя.

---

## Параболическая интерполяция (теория)

```
δ = 0.5 × (L - R) / (L - 2×C + R)
refined_frequency = (index + δ) × sample_rate / nFFT
```

Где `L, C, R` — магнитуды бинов `(index-1, index, index+1)`.
Точность уточнения: до `0.5 × bin_width` Гц.

---

## См. также

- [Full.md](Full.md) — полная документация с математикой и описанием ядер
- [Quick.md](Quick.md) — шпаргалка
- [FindAllMaxima_MaxValue_Guide.md](FindAllMaxima_MaxValue_Guide.md) — подробный гайд по AllMaxima
- Заголовочный файл: `modules/fft_maxima/include/spectrum_maxima_finder.h`
- Типы данных: `modules/fft_maxima/include/types/`

---

*Обновлено: 2026-03-09*
