# API Reference — FFTProcessor

> FFT 1..N лучей с тремя режимами вывода (комплексный спектр, магнитуда+фаза, магнитуда+фаза+частота).
> Поддерживает два бэкенда: **OpenCL** (`FFTProcessor`) и **ROCm/HIP** (`FFTProcessorROCm`).

**Namespace**: `fft_processor`

---

## Подключение

```cpp
// C++ — OpenCL backend
#include "modules/fft_processor/include/fft_processor.hpp"

// C++ — ROCm backend (только при ENABLE_ROCM=1, только AMD GPU)
#include "modules/fft_processor/include/fft_processor_rocm.hpp"
```

```python
# Python — OpenCL backend
proc = gpuworklib.FFTProcessor(ctx)          # ctx = GPUContext

# Python — ROCm backend (только AMD GPU, требует ENABLE_ROCM=1)
proc = gpuworklib.FFTProcessorROCm(ctx)      # ctx = ROCmGPUContext
```

---

## Backend — откуда берётся и как получить

`FFTProcessor` и `FFTProcessorROCm` принимают `IBackend*` — указатель на GPU-бэкенд из DrvGPU.
**Не владеет указателем** — backend должен жить дольше FFTProcessor.

### OpenCL — через DrvGPU (рекомендуется в production)

```cpp
#include "DrvGPU/include/drv_gpu.hpp"
#include "modules/fft_processor/include/fft_processor.hpp"

// Single GPU
drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, /*device_index=*/0);
gpu.Initialize();

fft_processor::FFTProcessor fft(&gpu.GetBackend());
// ... работа с fft ...
// gpu живёт дольше fft — RAII очищает автоматически
```

### OpenCL — через OpenCLBackend напрямую (в тестах/бенчмарках)

```cpp
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

// Если уже есть cl_context + cl_device_id + cl_command_queue:
auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
backend->InitializeFromExternalContext(context, device, queue);

fft_processor::FFTProcessor fft(backend.get());
```

### ROCm — через ROCmBackend (только AMD GPU, требует ENABLE_ROCM=1)

```cpp
#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "modules/fft_processor/include/fft_processor_rocm.hpp"

drv_gpu_lib::ROCmBackend backend;
backend.Initialize(/*device_index=*/0);

fft_processor::FFTProcessorROCm fft(&backend);
#endif
```

### Multi-GPU — через GPUManager

```cpp
#include "DrvGPU/include/gpu_manager.hpp"

drv_gpu_lib::GPUManager manager;
manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

// По одному FFTProcessor на GPU
fft_processor::FFTProcessor fft0(&manager.GetGPU(0)->GetBackend());
fft_processor::FFTProcessor fft1(&manager.GetGPU(1)->GetBackend());
```

---

## Типы данных

### FFTOutputMode

```cpp
enum class FFTOutputMode {
    COMPLEX,             // complex<float>[nFFT] — сырой FFT-спектр (re + im).
                         // Нормализация: совместимо с np.fft.fft() (без деления на N)
    MAGNITUDE_PHASE,     // magnitude[nFFT] + phase[nFFT] в радианах [-π, π]
                         // GPU-kernel: __fsqrt_rn(re²+im²) и atan2f(im, re)
    MAGNITUDE_PHASE_FREQ // То же + frequency[nFFT] в Гц: freq[k] = k × sample_rate / nFFT
                         // frequency[] вычисляется на CPU (без дополнительного kernel)
};
```

### FFTProcessorParams

```cpp
struct FFTProcessorParams {
    uint32_t beam_count = 1;                             // Число параллельных лучей
    uint32_t n_point = 0;                               // Реальных точек на луч (до zero-padding)
    float sample_rate = 1000.0f;                        // Частота дискретизации, Гц
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX; // Формат результата
    uint32_t repeat_count = 1;                          // Множитель nFFT сверх nextPow2
    float memory_limit = 0.80f;                         // Лимит GPU-памяти на батч (0.0–1.0)
};
```

**nFFT вычисление:**
```
nFFT = nextPowerOf2(n_point) × repeat_count
```
Пример: `n_point=1000, repeat_count=2` → `nFFT = 1024 × 2 = 2048`

| Параметр | Тип | Диапазон | Описание |
|----------|-----|----------|----------|
| `beam_count` | `uint32_t` | ≥ 1 | Число лучей; каждый луч — независимый FFT |
| `n_point` | `uint32_t` | ≥ 1 | Реальных точек на луч; nFFT = nextPow2(n_point) × repeat_count |
| `sample_rate` | `float` | > 0, Гц | Нужна для freq[k] = k × sample_rate / nFFT |
| `output_mode` | `FFTOutputMode` | — | Формат возвращаемых данных |
| `repeat_count` | `uint32_t` | ≥ 1 | Zero-padding коэффициент; 2 → интерполяция в частотной области |
| `memory_limit` | `float` | 0.0–1.0 | BatchManager делит лучи на батчи чтобы не превысить лимит |

### FFTBeamResult (базовый)

```cpp
struct FFTBeamResult {
    uint32_t beam_id = 0;     // Глобальный индекс луча (с учётом батчей)
    uint32_t nFFT = 0;        // Размер FFT: nextPow2(n_point) × repeat_count
    float sample_rate = 0.0f; // Копия из FFTProcessorParams
};
```

### FFTComplexResult

```cpp
struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum; // FFT-спектр [nFFT], ненормализован.
                                               // Физическая амплитуда: |spectrum[k]| / n_point
};
```

### FFTMagPhaseResult

```cpp
struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude;  // |X[k]| = sqrt(re²+im²), ненормализован [nFFT]
    std::vector<float> phase;      // arg(X[k]) = atan2(im, re), радианы [-π, π] [nFFT]
    std::vector<float> frequency;  // freq[k] = k × sample_rate / nFFT, Гц [nFFT]
                                   // ⚠️ ПУСТОЙ при output_mode == MAGNITUDE_PHASE
                                   //    Заполняется только при MAGNITUDE_PHASE_FREQ
};
```

### FFTProfilingData

```cpp
struct FFTProfilingData {
    double upload_time_ms = 0.0;           // CPU→GPU transfer (H2D или D2D)
    double fft_time_ms = 0.0;             // Чистое FFT-вычисление
    double post_processing_time_ms = 0.0; // Kernel complex→mag+phase (0 при COMPLEX)
    double download_time_ms = 0.0;        // GPU→CPU transfer
    double total_time_ms = 0.0;           // Суммарное время
};
```

### ROCmProfEvents (только ROCm backend)

```cpp
// Определён в fft_processor_rocm.hpp
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
// Стадии: "Upload", "Pad", "FFT", "MagPhase" (если MAGNITUDE_PHASE*), "Download"
```

---

## FFTProcessor (OpenCL backend)

**Файл**: `modules/fft_processor/include/fft_processor.hpp`
**Платформа**: OpenCL (NVIDIA, AMD — любая OpenCL-совместимая GPU)
**FFT библиотека**: clFFT (с pre-callback для zero-padding)

### Конструктор / Деструктор

```cpp
explicit FFTProcessor(drv_gpu_lib::IBackend* backend);
~FFTProcessor();

// Запрет копирования
FFTProcessor(const FFTProcessor&) = delete;
FFTProcessor& operator=(const FFTProcessor&) = delete;

// Перемещение
FFTProcessor(FFTProcessor&& other) noexcept;
FFTProcessor& operator=(FFTProcessor&& other) noexcept;
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `backend` | `IBackend*` | Указатель на OpenCL бэкенд (не владеет; должен жить дольше FFTProcessor) |

### Методы — Complex output

#### ProcessComplex() — CPU данные

```cpp
std::vector<FFTComplexResult> ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | `vector<complex<float>>` | Входные данные: `beam_count × n_point` элементов |
| `params` | `FFTProcessorParams` | Параметры FFT |
| `prof_events` | `vector<pair<...>>*` | `nullptr` = production (ноль overhead); не-`nullptr` = сбор `cl_event` для профилирования. Вызывающий отвечает за `clReleaseEvent`. |

**Возвращает:** `vector<FFTComplexResult>` — один результат на луч

#### ProcessComplex() — GPU данные

```cpp
std::vector<FFTComplexResult> ProcessComplex(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `gpu_data` | `cl_mem` | OpenCL буфер с данными на GPU (ownership не передаётся) |  // ownership - владение 
| `params` | `FFTProcessorParams` | Параметры FFT |
| `gpu_memory_bytes` | `size_t` | Размер буфера в байтах (0 = auto: `beam_count × n_point × sizeof(complex<float>)`) |
| `prof_events` | `vector<pair<...>>*` | Опциональный сбор событий профилирования |

### Методы — Magnitude + Phase output

#### ProcessMagPhase() — CPU данные

```cpp
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

#### ProcessMagPhase() — GPU данные

```cpp
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

Параметры аналогичны `ProcessComplex()`. `output_mode` в `params` определяет заполняется ли `frequency[]`.

### Вспомогательные методы

```cpp
FFTProfilingData GetProfilingData() const;  // Суммарные тайминги последнего вызова
uint32_t GetNFFT() const;                   // nFFT текущего/последнего запуска
```

---

## FFTProcessorROCm (ROCm/HIP backend)

**Файл**: `modules/fft_processor/include/fft_processor_rocm.hpp`
**Платформа**: AMD GPU с ROCm. Требует `ENABLE_ROCM=1`. На Windows файл полностью пропускается.
**FFT библиотека**: hipFFT (hiprtc-ядра для padding и mag/phase)

> ⚠️ **Отличие от OpenCL**: нет pre-callback → отдельный HIP-kernel для zero-padding (`pad_data`).
> Pipeline: `input_buffer_` → **pad_data** → `fft_input_` → **hipFFT** → `fft_output_` → **c2mp** → результат

### Конструктор / Деструктор

```cpp
explicit FFTProcessorROCm(drv_gpu_lib::IBackend* backend);
~FFTProcessorROCm();

FFTProcessorROCm(const FFTProcessorROCm&) = delete;
FFTProcessorROCm& operator=(const FFTProcessorROCm&) = delete;
FFTProcessorROCm(FFTProcessorROCm&& other) noexcept;
FFTProcessorROCm& operator=(FFTProcessorROCm&& other) noexcept;
```

### Методы — Complex output

#### ProcessComplex() — CPU данные

```cpp
std::vector<FFTComplexResult> ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events = nullptr);
// Стадии prof_events: "Upload", "Pad", "FFT", "Download"
```

#### ProcessComplex() — GPU данные

```cpp
std::vector<FFTComplexResult> ProcessComplex(
    void* gpu_data,             // Device pointer (hipMalloc)
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0);
```

> ⚠️ GPU-перегрузка ROCm **не имеет** `prof_events` параметра (в отличие от OpenCL).

### Методы — Magnitude + Phase output

#### ProcessMagPhase() — CPU данные

```cpp
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events = nullptr);
// Стадии prof_events: "Upload", "Pad", "FFT", "MagPhase", "Download"
```

#### ProcessMagPhase() — GPU данные

```cpp
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    void* gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0);
```

### Вспомогательные методы

```cpp
FFTProfilingData GetProfilingData() const;
uint32_t GetNFFT() const;
```

**Кеш планов ROCm** — двухслотовый LRU-2:
- При чередовании двух размеров батча план **не пересоздаётся** (swap кешей)
- При третьем размере — вытесняет старший кеш
- Для матричного бенчмарка с N разными `nFFT` → создавать отдельный `FFTProcessorROCm` на каждый размер

---

## Python API — FFTProcessorROCm

**Файл**: `python/py_fft_processor_rocm.hpp`
**Регистрация**: `register_fft_processor_rocm(m)` в `gpu_worklib_bindings.cpp`
**Требует**: `ENABLE_ROCM=1`, `ROCmGPUContext`

### Конструктор

```python
proc = gpuworklib.FFTProcessorROCm(ctx)
# ctx: ROCmGPUContext
```

### process_complex()

```python
spec = proc.process_complex(
    data,                  # numpy complex64: (n_point,) или (beam_count, n_point)
    sample_rate: float,    # Гц
    beam_count: int = 0,   # 0 = auto-detect из shape
    n_point: int = 0,      # 0 = auto-detect из shape
) -> np.ndarray            # complex64: (nFFT,) или (beam_count, nFFT)
```

### process_mag_phase()

```python
result = proc.process_mag_phase(
    data,                    # numpy complex64: (n_point,) или (beam_count, n_point)
    sample_rate: float,      # Гц
    beam_count: int = 0,
    n_point: int = 0,
    include_freq: bool = True,
) -> dict                    # ключи: magnitude, phase, frequency (если include_freq),
                             #        nFFT, sample_rate
```

Типы массивов в словаре:
- `magnitude`, `phase`, `frequency` — `numpy.float32` shape `(nFFT,)` или `(beam_count, nFFT)`
- `nFFT` — `int`
- `sample_rate` — `float`

### get_profiling()

```python
p = proc.get_profiling()
# dict: upload_ms, fft_ms, post_processing_ms, download_ms, total_ms
```

### Свойства

```python
proc.nfft  # int — текущий nFFT (nextPow2(n_point))
```

### Python API — Пример

```python
import numpy as np
import gpuworklib

ctx  = gpuworklib.ROCmGPUContext(0)
proc = gpuworklib.FFTProcessorROCm(ctx)

N = 1024
fs = 1e6
k  = 50
t  = np.arange(N)
sig = np.exp(1j * 2 * np.pi * k / N * t).astype(np.complex64)

# Complex output
spec = proc.process_complex(sig, sample_rate=fs)
peak_bin = int(np.argmax(np.abs(spec)))
print(f"peak_bin={peak_bin}, expected={k}")  # peak_bin=50

# Magnitude + Phase
mp = proc.process_mag_phase(sig, sample_rate=fs)
peak_hz = mp["frequency"][peak_bin]
print(f"peak_hz={peak_hz:.0f} Hz")           # peak_hz=48828 Hz (~k × fs/N)

# Multi-beam (4 лучей, 512 точек каждый)
data = np.random.randn(4, 512).astype(np.complex64)
spec2d = proc.process_complex(data, sample_rate=fs)  # shape (4, 512)
```

### Тесты Python

**Файл**: `Python_test/fft_processor/test_fft_processor_rocm.py`

| Класс | Тестов | Описание |
|-------|--------|----------|
| `TestNumPyReference` | 8 | NumPy-референс, всегда запускаются |
| `TestFFTProcessorROCm` | 7 | GPU тесты, skip если нет `FFTProcessorROCm` |

---

## Цепочки вызовов

### OpenCL — CPU данные → Complex FFT

```
IBackend::init()
  └─→ FFTProcessor::FFTProcessor(backend)   // создание + clfftSetup()
      └─→ ProcessComplex(data, params)       // CPU upload + FFT
          ├─→ UploadData()                   // H2D: vector → pre_callback_userdata_
          ├─→ ExecuteFFT()                   // clFFT с pre-callback (zero-padding inline)
          └─→ ReadComplexResults()           // D2H: nFFT complex<float> на луч
```

### OpenCL — GPU данные (cl_mem) → MagPhase FFT

```
SignalGenerator::GenerateGpu()             // cl_mem на GPU
  └─→ FFTProcessor::ProcessMagPhase(gpu_mem, params)
      ├─→ CopyGpuData()                    // D2D: cl_mem → pre_callback_userdata_
      ├─→ ExecuteFFT()                     // clFFT с pre-callback
      ├─→ ExecuteMagPhaseKernel()          // GPU: complex → mag + phase
      └─→ ReadMagPhaseResults()            // D2H: magnitude[], phase[], freq[]
```

### ROCm — CPU данные → MagPhase FFT

```
FFTProcessorROCm::FFTProcessorROCm(backend)
  └─→ ProcessMagPhase(data, params)
      ├─→ UploadData()                     // H2D: vector → input_buffer_
      ├─→ ExecutePadKernel()               // HIP: input_buffer_ → fft_input_ (zero-padding)
      ├─→ ExecuteFFT()                     // hipFFT: fft_input_ → fft_output_
      ├─→ ExecuteMagPhaseKernel()          // HIP: fft_output_ → mag_phase_interleaved_
      └─→ ReadMagPhaseResults()            // D2H: один D2H (интерлив mag+phase)
```

### С профилированием (OpenCL)

```
FFTProcessor fft(backend);
vector<pair<const char*, cl_event>> events;
fft.ProcessComplex(data, params, &events);
  // events содержит: {"Upload", ev1}, {"FFT", ev2}, {"Download", ev3}
  // Передать в GpuBenchmarkBase::CollectOrRelease() или GPUProfiler
```

---

## Примеры

### C++ — Single beam, COMPLEX output

```cpp
#include "modules/fft_processor/include/fft_processor.hpp"

// Генерация тестового сигнала
const float freq = 100.0f;
const float sample_rate = 1000.0f;
const size_t n_point = 1024;
std::vector<std::complex<float>> data(n_point);
for (size_t i = 0; i < n_point; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    data[i] = {std::cos(2 * M_PI * freq * t), std::sin(2 * M_PI * freq * t)};
}

// FFT
fft_processor::FFTProcessor fft(backend);  // backend — IBackend* из DrvGPU
fft_processor::FFTProcessorParams params;
params.beam_count = 1;
params.n_point = static_cast<uint32_t>(n_point);
params.sample_rate = sample_rate;
params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

auto results = fft.ProcessComplex(data, params);
const auto& r = results[0];
// r.beam_id, r.nFFT (= 1024), r.spectrum[i] — complex<float>
auto peak = std::max_element(r.spectrum.begin(), r.spectrum.end(),
    [](auto& a, auto& b) { return std::abs(a) < std::abs(b); });
```

### C++ — Multi-beam, MAGNITUDE_PHASE_FREQ

```cpp
fft_processor::FFTProcessor fft(backend);
fft_processor::FFTProcessorParams params;
params.beam_count = 8;
params.n_point = 2048;
params.sample_rate = 10000.0f;
params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE_FREQ;
params.repeat_count = 1;

auto results = fft.ProcessMagPhase(data, params);  // data: 8 × 2048 complex<float>
for (auto& r : results) {
    // r.magnitude[], r.phase[], r.frequency[] — все длиной nFFT
    auto peak_idx = std::distance(r.magnitude.begin(),
        std::max_element(r.magnitude.begin(), r.magnitude.end()));
    printf("Beam %u: peak at %.1f Hz, mag=%.4f\n",
        r.beam_id, r.frequency[peak_idx], r.magnitude[peak_idx]);
}
```

### C++ — GPU данные (SignalGenerator → FFTProcessor)

```cpp
// Генерация сигнала на GPU
signal_gen::SignalService service(backend);
cl_mem signal = service.GenerateGpu(cw_params, {1000.0, 4096}, 16);  // 16 лучей

// FFT из GPU памяти (без H2D копии входных данных)
fft_processor::FFTProcessor fft(backend);
fft_processor::FFTProcessorParams params;
params.beam_count = 16;
params.n_point = 4096;
params.sample_rate = 1000.0f;

size_t gpu_bytes = 16 * 4096 * sizeof(std::complex<float>);
auto results = fft.ProcessComplex(signal, params, gpu_bytes);

clReleaseMemObject(signal);
```

### C++ — С профилированием (для бенчмарка)

```cpp
fft_processor::FFTProcessor fft(backend);
fft_processor::FFTProcessorParams params{/*.beam_count=*/4, /*.n_point=*/1024, /*.sample_rate=*/1000.f};

std::vector<std::pair<const char*, cl_event>> events;
auto results = fft.ProcessComplex(data, params, &events);
// events: [{"Upload", ev0}, {"FFT", ev1}, {"Download", ev2}]

// Получить быстрые тайминги без профилирования:
auto prof = fft.GetProfilingData();
printf("FFT time: %.3f ms\n", prof.fft_time_ms);

// Освободить события (если не переданы в GPUProfiler)
for (auto& [name, ev] : events)
    clReleaseEvent(ev);
```

### C++ — ROCm backend

```cpp
#if ENABLE_ROCM
#include "modules/fft_processor/include/fft_processor_rocm.hpp"

fft_processor::FFTProcessorROCm fft(backend);  // backend — ROCm IBackend*
fft_processor::FFTProcessorParams params;
params.beam_count = 32;
params.n_point = 4096;
params.sample_rate = 10000.0f;
params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE_FREQ;

fft_processor::ROCmProfEvents events;
auto results = fft.ProcessMagPhase(data, params, &events);
// events: [{"Upload",...}, {"Pad",...}, {"FFT",...}, {"MagPhase",...}, {"Download",...}]
#endif
```

---

## Константы

| Константа | Значение | Описание |
|-----------|---------|----------|
| `PRE_CALLBACK_HEADER_SIZE` | 32 bytes | Заголовок в начале `pre_callback_userdata_` (только OpenCL) |

---

## Ограничения и нюансы

### FFT Plan Caching

FFTProcessor (OpenCL) кеширует clFFT plan. **При изменении `n_point` необходимо создать новый экземпляр:**

```cpp
// ПРАВИЛЬНО
FFTProcessor fft_4096(backend);  // для n_point = 4096
FFTProcessor fft_8192(backend);  // для n_point = 8192

// НЕПРАВИЛЬНО — CL_INVALID_VALUE при смене размера
FFTProcessor fft(backend);
fft.ProcessComplex(data_4096, params_4096);  // OK
fft.ProcessComplex(data_8192, params_8192);  // ОШИБКА!
```

### GPU Memory Ownership

- CPU данные (`vector<complex<float>>`) → копируются на GPU внутри `ProcessXxx`
- GPU данные (`cl_mem` / `void*`) → используются напрямую, ownership **не передаётся**
- Результаты всегда возвращаются на CPU

### frequency[] для MAGNITUDE_PHASE

```cpp
params.output_mode = FFTOutputMode::MAGNITUDE_PHASE;
auto results = fft.ProcessMagPhase(data, params);
results[0].frequency;  // ⚠️ ПУСТОЙ вектор — только для MAGNITUDE_PHASE_FREQ заполнен
```

### prof_events и production

- `prof_events = nullptr` (default) → **ноль overhead**, все cl_event освобождаются внутри
- `prof_events != nullptr` → cl_event собираются, вызывающий отвечает за `clReleaseEvent`

### ROCm: кеш планов

ROCm версия имеет двухслотовый LRU-2 кеш планов hipFFT. При матричном бенчмарке с N разными `nFFT` — создавать отдельный `FFTProcessorROCm` на каждый размер.

### Нормализация FFT

FFT **ненормализован** (как `np.fft.fft()`):
- Физическая амплитуда: `|spectrum[k]| / n_point`
- Для сравнения с `np.fft.fft()` — спектры совпадают напрямую

---

## См. также

- [Full.md](Full.md) — полная документация с математикой и описанием ядер
- [Quick.md](Quick.md) — шпаргалка
- Заголовочный файл OpenCL: `modules/fft_processor/include/fft_processor.hpp`
- Заголовочный файл ROCm: `modules/fft_processor/include/fft_processor_rocm.hpp`
- Типы данных: `modules/fft_processor/include/types/`

---

*Обновлено: 2026-03-10*
