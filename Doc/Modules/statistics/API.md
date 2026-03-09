# API Reference — StatisticsProcessor

> GPU-статистика комплексных сигналов по лучам: mean, median, variance, std (ROCm/HIP).
> Один класс, восемь методов, три группы.

**Namespace**: `statistics`
**Платформа**: ROCm only (AMD GPU, Linux). Требует `ENABLE_ROCM=1`.

---

## Подключение

```cpp
// C++ — только при ENABLE_ROCM=1
#include "modules/statistics/include/statistics_processor.hpp"
#include "modules/statistics/include/statistics_types.hpp"
```

```python
# Python
import gpuworklib
stats = gpuworklib.StatisticsProcessor(ctx)  # ctx — ROCmGPUContext
```

---

## Backend — откуда берётся и как получить

`StatisticsProcessor` принимает `IBackend*` — указатель на ROCm-бэкенд из DrvGPU.
**Не владеет указателем** — backend должен жить дольше StatisticsProcessor.

Если передать не-ROCm бэкенд — конструктор бросает `std::runtime_error`.

### Через ROCmBackend напрямую (тесты, бенчмарки)

```cpp
#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "modules/statistics/include/statistics_processor.hpp"

drv_gpu_lib::ROCmBackend backend;
backend.Initialize(/*device_index=*/0);

statistics::StatisticsProcessor stats(&backend);
// backend должен жить дольше stats
#endif
```

### Через DrvGPU (production pipeline)

```cpp
#include "DrvGPU/include/drv_gpu.hpp"
#include "modules/statistics/include/statistics_processor.hpp"

drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::ROCM, 0);
gpu.Initialize();

statistics::StatisticsProcessor stats(&gpu.GetBackend());
```

### Python — через ROCmGPUContext

```python
import gpuworklib

ctx   = gpuworklib.ROCmGPUContext(0)          # device_index=0
stats = gpuworklib.StatisticsProcessor(ctx)
```

---

## Типы данных

### StatisticsParams

```cpp
struct StatisticsParams {
    uint32_t beam_count   = 1;  // Число лучей (каналов)
    uint32_t n_point      = 0;  // Точек на луч (complex<float> или float)
    size_t   memory_limit = 0;  // GPU memory limit, байт (0 = авто)
};
```

| Параметр | Тип | Диапазон | Описание |
|----------|-----|----------|----------|
| `beam_count` | `uint32_t` | ≥ 1 | Число лучей; каждый луч обрабатывается независимо |
| `n_point` | `uint32_t` | ≥ 1 | Точек на луч; входной массив = `beam_count × n_point` элементов |
| `memory_limit` | `size_t` | 0 = авто | Ограничение GPU-памяти для буферов |

### MeanResult

```cpp
struct MeanResult {
    uint32_t beam_id = 0;
    std::complex<float> mean{0.0f, 0.0f};  // Re + j·Im среднее
};
```

### MedianResult

```cpp
struct MedianResult {
    uint32_t beam_id          = 0;
    float    median_magnitude = 0.0f;  // median(|z|) = sorted[N/2]
};
```

> ⚠️ **Медиана** = `sorted[N/2]`, не среднее двух средних элементов.
> При сравнении с NumPy: `np.sort(mags)[N//2]`, **не** `np.median(mags)`.

### StatisticsResult

```cpp
struct StatisticsResult {
    uint32_t            beam_id        = 0;
    std::complex<float> mean{0.0f, 0.0f};  // Комплексное среднее Re + j·Im
    float               variance       = 0.0f;  // Дисперсия |z| (population, ddof=0)
    float               std_dev        = 0.0f;  // СКО = sqrt(variance)
    float               mean_magnitude = 0.0f;  // Среднее модулей mean(|z|)
};
```

> ⚠️ Для `ComputeStatisticsFloat` / `ComputeMedianFloat` поле `mean` всегда `{0, 0}` —
> входные данные уже являются модулями (float), а не комплексным сигналом.

---

## StatisticsProcessor

**Файл**: `modules/statistics/include/statistics_processor.hpp`
**Платформа**: AMD GPU с ROCm. Требует `ENABLE_ROCM=1`. На Windows полностью пропускается.
**Kernel-компилятор**: hiprtc (JIT при первом вызове, затем HSACO-кеш на диске).

### Конструктор / Деструктор

```cpp
explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);
~StatisticsProcessor();

// Запрет копирования
StatisticsProcessor(const StatisticsProcessor&) = delete;
StatisticsProcessor& operator=(const StatisticsProcessor&) = delete;

// Перемещение
StatisticsProcessor(StatisticsProcessor&& other) noexcept;
StatisticsProcessor& operator=(StatisticsProcessor&& other) noexcept;
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `backend` | `IBackend*` | Указатель на ROCm-бэкенд (не владеет; должен жить дольше объекта). Не-ROCm бэкенд → `std::runtime_error`. |

---

### Группа 1 — ComputeStatistics (одно-проходный Уэлфорд)

Вычисляет за один проход: комплексное среднее, `mean(|z|)`, `variance(|z|)`, `std_dev(|z|)`.
Ядро `welford_fused` — нет отдельного буфера модулей, всё inline.

#### ComputeStatistics() — CPU данные

```cpp
std::vector<StatisticsResult> ComputeStatistics(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | `vector<complex<float>>` | Входные данные: `beam_count × n_point` элементов, beam-major layout |
| `params` | `StatisticsParams` | beam_count, n_point |

**Возвращает**: `vector<StatisticsResult>` — один результат на луч.

#### ComputeStatistics() — GPU данные

```cpp
std::vector<StatisticsResult> ComputeStatistics(
    void* gpu_data,
    const StatisticsParams& params);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `gpu_data` | `void*` | HIP device pointer на `complex<float>[beam_count × n_point]` (не владеет) |
| `params` | `StatisticsParams` | beam_count, n_point |

Выполняет `hipMemcpyDtoDAsync` вместо HtoD — без PCIe-пересылки.

#### ComputeStatisticsFloat() — GPU float данные *(только C++)*

```cpp
std::vector<StatisticsResult> ComputeStatisticsFloat(
    void* gpu_float_data,
    const StatisticsParams& params);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `gpu_float_data` | `void*` | HIP device pointer на `float[beam_count × n_point]` — уже вычисленные модули |
| `params` | `StatisticsParams` | beam_count, n_point |

Для интеграции с FFT-пайплайном: спектр уже как `float magnitude[beams × nFFT]`.
Использует ядро `welford_float`. Поле `mean` в результате всегда `{0, 0}`.
**Нет Python-биндинга.**

---

### Группа 2 — ComputeMedian (rocPRIM radix sort)

Вычисляет медиану `|z|` по каждому лучу:
`compute_magnitudes` → `rocprim::segmented_radix_sort_keys` → `extract_medians`.

#### ComputeMedian() — CPU данные

```cpp
std::vector<MedianResult> ComputeMedian(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);
```

#### ComputeMedian() — GPU данные

```cpp
std::vector<MedianResult> ComputeMedian(
    void* gpu_data,
    const StatisticsParams& params);
```

#### ComputeMedianFloat() — GPU float данные *(только C++)*

```cpp
std::vector<MedianResult> ComputeMedianFloat(
    void* gpu_float_data,
    const StatisticsParams& params);
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `gpu_float_data` | `void*` | HIP device pointer на `float` — уже вычисленные модули |
| `params` | `StatisticsParams` | beam_count, n_point |

Пропускает шаг `compute_magnitudes` — данные сразу идут в radix sort.
**Нет Python-биндинга.**

---

### Группа 3 — ComputeMean (иерархическая редукция)

Только комплексное среднее. Два прохода: `mean_reduce_phase1` (блочная сумма + double-load) → `mean_reduce_final` (финальная редукция, деление на N).

#### ComputeMean() — CPU данные

```cpp
std::vector<MeanResult> ComputeMean(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);
```

#### ComputeMean() — GPU данные

```cpp
std::vector<MeanResult> ComputeMean(
    void* gpu_data,
    const StatisticsParams& params);
```

---

## Python API

**Модуль**: `gpuworklib.StatisticsProcessor`
**Биндинг**: `python/py_statistics.hpp`
**Доступно только при ENABLE_ROCM=1.**

Python-биндинг предоставляет **3 метода** (нет биндинга для `ComputeStatisticsFloat` / `ComputeMedianFloat`).

### Конструктор

```python
stats = gpuworklib.StatisticsProcessor(ctx)
# ctx — gpuworklib.ROCmGPUContext(device_index=0)
```

### compute_mean()

```python
results = stats.compute_mean(data, beam_count=1)
# data: np.complex64, shape (beam_count * n_point,) или (beam_count, n_point)
```

**Возвращает** `list[dict]`:
```python
{'beam_id': int, 'mean_real': float, 'mean_imag': float}
```

### compute_median()

```python
results = stats.compute_median(data, beam_count=1)
# data: np.complex64
```

**Возвращает** `list[dict]`:
```python
{'beam_id': int, 'median_magnitude': float}
```

### compute_statistics()

```python
results = stats.compute_statistics(data, beam_count=1)
# data: np.complex64
```

**Возвращает** `list[dict]`:
```python
{
    'beam_id':        int,
    'mean_real':      float,   # Re(mean(z))
    'mean_imag':      float,   # Im(mean(z))
    'variance':       float,   # var(|z|), ddof=0
    'std_dev':        float,   # sqrt(variance)
    'mean_magnitude': float,   # mean(|z|)
}
```

### Сводная таблица

| Метод Python | Входные данные | Ключи словаря |
|-------------|----------------|---------------|
| `compute_mean(data, beam_count)` | `np.complex64` | `beam_id`, `mean_real`, `mean_imag` |
| `compute_median(data, beam_count)` | `np.complex64` | `beam_id`, `median_magnitude` |
| `compute_statistics(data, beam_count)` | `np.complex64` | `beam_id`, `mean_real`, `mean_imag`, `variance`, `std_dev`, `mean_magnitude` |

---

## Цепочки вызовов

### ComputeStatistics (CPU data)

```
ComputeStatistics(vector, params)
  → AllocateBuffers(beam_count, n_point)   [lazy, skip если размер не изменился]
  → CompileKernels()                        [lazy, hiprtc JIT → HSACO кеш]
  → UploadData(data, count)                hipMemcpyHtoDAsync
  → ExecuteWelfordFusedKernel(beams, N)    welford_fused: 1 проход, |z| inline
  → hipStreamSynchronize
  → hipMemcpyDtoH(results, beams × 20 байт)
```

### ComputeMedian (CPU data)

```
ComputeMedian(vector, params)
  → AllocateBuffers + CompileKernels
  → UploadData
  → ExecuteMagnitudesKernel                compute_magnitudes: complex → |z|
  → ExecuteMedianSort(beams, N)            rocprim::segmented_radix_sort_keys
  → ExecuteExtractMediansKernel            extract_medians: 1 thread per beam
  → hipStreamSynchronize
  → hipMemcpyDtoH(medians_compact, beams × 4 байта)
```

### ComputeStatisticsFloat (GPU float data)

```
ComputeStatisticsFloat(gpu_float, params)
  → AllocateBuffers + CompileKernels
  → CopyFloatGpuData(src, count)           hipMemcpyDtoDAsync → magnitudes_buf_
  → ExecuteWelfordFloatKernel(beams, N)    welford_float: float input, mean={0,0}
  → hipStreamSynchronize
  → hipMemcpyDtoH(results, beams × 20 байт)
```

### ComputeMedianFloat (GPU float data)

```
ComputeMedianFloat(gpu_float, params)
  → AllocateBuffers + CompileKernels
  → CopyFloatGpuData → magnitudes_buf_
  → ExecuteMedianSort                      sort magnitudes_buf_ → sort_buf_
  → ExecuteExtractMediansKernel
  → hipStreamSynchronize + hipMemcpyDtoH
```

### Python pipeline

```
ROCmGPUContext(0)
  └─→ StatisticsProcessor(ctx)
      ├─→ compute_statistics(data, beam_count)  → list[dict] (mean + variance + std)
      ├─→ compute_median(data, beam_count)       → list[dict] (median_magnitude)
      └─→ compute_mean(data, beam_count)         → list[dict] (mean_real + mean_imag)
```

---

## Примеры

### C++ — полный минимум

```cpp
#include "backends/rocm/rocm_backend.hpp"
#include "modules/statistics/include/statistics_processor.hpp"
#include "modules/statistics/include/statistics_types.hpp"

drv_gpu_lib::ROCmBackend backend;
backend.Initialize(0);

statistics::StatisticsProcessor stats(&backend);

statistics::StatisticsParams params;
params.beam_count = 4;
params.n_point    = 8192;

std::vector<std::complex<float>> data(params.beam_count * params.n_point);
// ... заполнение data ...

// Полная статистика (mean + variance + std + mean_mag)
auto results = stats.ComputeStatistics(data, params);
for (const auto& r : results) {
    printf("beam %u: mean=(%.4f,%.4f) mean_mag=%.4f std=%.6f\n",
           r.beam_id, r.mean.real(), r.mean.imag(),
           r.mean_magnitude, r.std_dev);
}

// Медиана модулей (GPU radix sort)
auto medians = stats.ComputeMedian(data, params);
for (const auto& r : medians)
    printf("beam %u: median_mag=%.4f\n", r.beam_id, r.median_magnitude);

// Только комплексное среднее
auto means = stats.ComputeMean(data, params);
for (const auto& r : means)
    printf("beam %u: mean=(%.6f+%.6fj)\n",
           r.beam_id, r.mean.real(), r.mean.imag());
```

### C++ — GPU данные (без PCIe HtoD)

```cpp
// Данные уже на GPU — прямая обработка без PCIe-пересылки
void* gpu_ptr = nullptr;
hipMalloc(&gpu_ptr, data.size() * sizeof(std::complex<float>));
hipMemcpy(gpu_ptr, data.data(), data.size() * sizeof(std::complex<float>),
          hipMemcpyHostToDevice);

auto gpu_results = stats.ComputeStatistics(gpu_ptr, params);
// Внутри: hipMemcpyDtoDAsync вместо HtoD

hipFree(gpu_ptr);
```

### C++ — float-input (после FFT)

```cpp
// gpu_spectrum — float magnitudes на GPU: beam_count * nFFT элементов
// Типичный сценарий: результат FFTProcessorROCm в режиме MAGNITUDE_PHASE
void* gpu_spectrum = /* hipDeviceptr к float[beam_count * nFFT] */;

statistics::StatisticsParams params;
params.beam_count = 4;
params.n_point    = 1024;  // nFFT

// Статистика спектра: mean={0,0}, остальные поля — корректны
auto spec_stats = stats.ComputeStatisticsFloat(gpu_spectrum, params);
for (const auto& r : spec_stats)
    printf("beam %u: mean_mag=%.4f std=%.4f\n",
           r.beam_id, r.mean_magnitude, r.std_dev);

// Медиана спектра (float input)
auto spec_medians = stats.ComputeMedianFloat(gpu_spectrum, params);
for (const auto& r : spec_medians)
    printf("beam %u: median_spec=%.4f\n", r.beam_id, r.median_magnitude);
```

### Python — полный минимум

```python
import sys
sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np

ctx   = gpuworklib.ROCmGPUContext(0)
stats = gpuworklib.StatisticsProcessor(ctx)

beam_count = 4
n_point    = 8192
data = (np.random.randn(beam_count * n_point) +
        1j * np.random.randn(beam_count * n_point)).astype(np.complex64)

# Полная статистика
results = stats.compute_statistics(data, beam_count=beam_count)
for r in results:
    print(f"beam {r['beam_id']}: "
          f"mean=({r['mean_real']:.4f}+{r['mean_imag']:.4f}j) "
          f"mean_mag={r['mean_magnitude']:.4f} "
          f"std={r['std_dev']:.4f}")

# Медиана
medians = stats.compute_median(data, beam_count=beam_count)
for r in medians:
    print(f"beam {r['beam_id']}: median_mag={r['median_magnitude']:.4f}")

# Только среднее
means = stats.compute_mean(data, beam_count=beam_count)
for r in means:
    print(f"beam {r['beam_id']}: mean=({r['mean_real']:.6f}+{r['mean_imag']:.6f}j)")

# NumPy-эталон (важно: ddof=0 и sorted[N//2]!)
beam0 = data[:n_point]
mags0 = np.abs(beam0)
print(f"NumPy mean_mag:  {np.mean(mags0):.6f}")
print(f"NumPy std:       {np.std(mags0, ddof=0):.6f}")        # ddof=0!
print(f"NumPy median:    {np.sort(mags0)[n_point // 2]:.6f}") # НЕ np.median()!
```

---

## Ограничения и нюансы

| Ограничение | Описание |
|-------------|----------|
| ROCm only | AMD GPU, Linux, `cmake -DENABLE_ROCM=ON`. На Windows файл пропускается. |
| Нет Python для float-input | `ComputeStatisticsFloat` и `ComputeMedianFloat` — только C++ |
| Медиана не стандартная | `sorted[N/2]`, не среднее двух средних. Сравнение: `np.sort(mags)[N//2]` |
| Дисперсия population | `ddof=0`. Сравнение: `np.std(mags, ddof=0)`, `np.var(mags, ddof=0)` |
| Первый вызов медленный | hiprtc JIT ~1-3 с. После HSACO-кеша — мгновенно |
| Re-allocation при смене размера | При изменении `beam_count` / `n_point` все GPU-буферы пересоздаются |
| Namespace | `statistics::`, не `drv_gpu_lib::`. Python: `gpuworklib.StatisticsProcessor` |
| float-input mean={0,0} | `ComputeStatisticsFloat` не вычисляет комплексное среднее (нет Im/Re данных) |

---

## См. также

- [Full.md](Full.md) — математика, pipeline, C4-диаграммы, все ядра, тесты
- [Quick.md](Quick.md) — шпаргалка
- [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) — оптимизация HIP-ядер

---

*Обновлено: 2026-03-09*
