# Statistics — API Reference

> Полный справочник публичного C++ и Python API модуля statistics

---

## C++ Namespace: `statistics`

---

### `StatisticsParams`

```cpp
// modules/statistics/include/statistics_types.hpp
struct StatisticsParams {
    uint32_t beam_count  = 1;  // число лучей
    uint32_t n_point     = 0;  // сэмплов на луч (complex float)
    size_t   memory_limit = 0; // GPU memory limit (0 = auto)
};
```

---

### `MeanResult`

```cpp
struct MeanResult {
    uint32_t beam_id = 0;
    std::complex<float> mean{0.0f, 0.0f};  // комплексное среднее
};
```

---

### `StatisticsResult`

```cpp
struct StatisticsResult {
    uint32_t beam_id = 0;
    std::complex<float> mean{0.0f, 0.0f};  // комплексное среднее
    float variance       = 0.0f;            // Var(|z|), ddof=0
    float std_dev        = 0.0f;            // sqrt(Var(|z|))
    float mean_magnitude = 0.0f;            // E[|z|]
};
```

---

### `MedianResult`

```cpp
struct MedianResult {
    uint32_t beam_id = 0;
    float median_magnitude = 0.0f;  // sorted_magnitudes[N/2]
};
```

---

### `FullStatisticsResult`

```cpp
// Результат ComputeAll / ComputeAllFloat — объединяет StatisticsResult + MedianResult
struct FullStatisticsResult {
    uint32_t beam_id = 0;
    std::complex<float> mean{0.0f, 0.0f};  // {0,0} для float-пути (ComputeAllFloat)
    float variance         = 0.0f;          // Var(|z|), ddof=0
    float std_dev          = 0.0f;          // sqrt(Var(|z|))
    float mean_magnitude   = 0.0f;          // E[|z|]
    float median_magnitude = 0.0f;          // sorted_magnitudes[N/2]
};
```

---

### `StatisticsROCmProfEvents`

```cpp
// Тип для сбора ROCm profiling events из ComputeAll/ComputeAllFloat
// Совместим с GpuBenchmarkBase::RecordROCmEvent(name, data)
using StatisticsROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

---

### `StatisticsProcessor`

```cpp
// modules/statistics/include/statistics_processor.hpp
// Требует: #if ENABLE_ROCM

namespace statistics {

class StatisticsProcessor {
public:
    // Конструктор (non-owning backend)
    explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);

    ~StatisticsProcessor();

    // Запрещено копирование, разрешено перемещение
    StatisticsProcessor(const StatisticsProcessor&) = delete;
    StatisticsProcessor& operator=(const StatisticsProcessor&) = delete;
    StatisticsProcessor(StatisticsProcessor&&) noexcept;
    StatisticsProcessor& operator=(StatisticsProcessor&&) noexcept;

    // ──────────────────────────────────────────────────────────
    // CPU data overloads (upload → compute → download)
    // ──────────────────────────────────────────────────────────

    std::vector<MeanResult> ComputeMean(
        const std::vector<std::complex<float>>& data,
        const StatisticsParams& params);

    std::vector<MedianResult> ComputeMedian(
        const std::vector<std::complex<float>>& data,
        const StatisticsParams& params);

    std::vector<StatisticsResult> ComputeStatistics(
        const std::vector<std::complex<float>>& data,
        const StatisticsParams& params);

    // ──────────────────────────────────────────────────────────
    // GPU data overloads (данные уже на устройстве)
    // ──────────────────────────────────────────────────────────

    std::vector<MeanResult> ComputeMean(
        void* gpu_data,
        const StatisticsParams& params);

    std::vector<MedianResult> ComputeMedian(
        void* gpu_data,
        const StatisticsParams& params);

    std::vector<StatisticsResult> ComputeStatistics(
        void* gpu_data,
        const StatisticsParams& params);

    // ──────────────────────────────────────────────────────────
    // Float input (модули уже вычислены, GPU data)
    // ──────────────────────────────────────────────────────────

    std::vector<StatisticsResult> ComputeStatisticsFloat(
        void* gpu_float_data,
        const StatisticsParams& params);

    std::vector<MedianResult> ComputeMedianFloat(
        void* gpu_float_data,
        const StatisticsParams& params);

    // ──────────────────────────────────────────────────────────
    // ComputeAll — статистика + медиана в одном GPU-вызове
    // Устраняет двойной PCIe upload (CPU path) / D2D copy (GPU path)
    // ──────────────────────────────────────────────────────────

    // CPU path: Upload → welford_fused → radix_sort → extract_medians
    std::vector<FullStatisticsResult> ComputeAll(
        const std::vector<std::complex<float>>& data,
        const StatisticsParams& params,
        StatisticsROCmProfEvents* prof_events = nullptr);

    // GPU path: D2D copy → welford_fused → radix_sort → extract_medians
    std::vector<FullStatisticsResult> ComputeAll(
        void* gpu_data,
        const StatisticsParams& params,
        StatisticsROCmProfEvents* prof_events = nullptr);

    // Float GPU path: D2D copy → welford_float → sort → extract_medians
    // ВАЖНО: mean.real() == 0, mean.imag() == 0 всегда
    std::vector<FullStatisticsResult> ComputeAllFloat(
        void* gpu_float_data,
        const StatisticsParams& params,
        StatisticsROCmProfEvents* prof_events = nullptr);

    // Float CPU path (convenience): hipMalloc → delegate к GPU overload
    std::vector<FullStatisticsResult> ComputeAllFloat(
        const std::vector<float>& data,
        const StatisticsParams& params);
};

}  // namespace statistics
```

---

### Исключения

| Метод | Условие | Исключение |
|-------|---------|------------|
| Конструктор | backend == nullptr или не инициализирован | `std::runtime_error` |
| Конструктор | backend не ROCm type | `std::runtime_error` |
| `ComputeMean(data, ...)` | data.size() != beam_count * n_point | `std::invalid_argument` |
| `ComputeMedian(data, ...)` | data.size() != beam_count * n_point | `std::invalid_argument` |
| `ComputeStatistics(data, ...)` | data.size() != beam_count * n_point | `std::invalid_argument` |
| `ComputeAll(data, ...)` | data.size() != beam_count * n_point | `std::invalid_argument` |
| `ComputeAllFloat(data, ...)` | data.size() != beam_count * n_point | `std::invalid_argument` |
| GPU overloads | gpu_data == nullptr | `std::invalid_argument` |
| `CompileKernels()` | hiprtc ошибка компиляции | `std::runtime_error` (с логом) |
| `AllocateBuffers()` | hipMalloc failed | `std::runtime_error` |

---

### Цепочка вызовов

```cpp
// Типичный сценарий:
StatisticsProcessor proc(&backend);                   // lazy: ничего не выделяется

auto r = proc.ComputeStatistics(data, params);
//   → CompileKernels()   [lazy, 1 раз; HSACO cache]
//   → AllocateBuffers()  [lazy resize]
//   → UploadData()       [H2D async]
//   → ExecuteWelfordFusedKernel()
//   → hipStreamSynchronize()
//   → hipMemcpyDtoH()
//   ← vector<StatisticsResult>
```

---

## Python API

```python
import gpuworklib

# Конструктор
proc = gpuworklib.StatisticsProcessor(ctx)
# ctx: ROCmGPUContext (НЕ GPUContext!)

# Методы
proc.compute_mean(data, beam_count=1)
proc.compute_median(data, beam_count=1)
proc.compute_statistics(data, beam_count=1)
proc.compute_all(data, beam_count=1)          # statistics + median, 1 GPU call
proc.compute_all_float(data, beam_count=1)    # float magnitudes, mean_real/imag=0
proc.compute_statistics_float(data, beam_count=1)
proc.compute_median_float(data, beam_count=1)
```

### compute_mean

```python
def compute_mean(
    data: np.ndarray,    # complex64, shape (B*N,) или (B,N)
    beam_count: int = 1
) -> list[dict]:
    ...

# Возврат: list[{'beam_id': int, 'mean_real': float, 'mean_imag': float}]
```

### compute_median

```python
def compute_median(
    data: np.ndarray,    # complex64, shape (B*N,) или (B,N)
    beam_count: int = 1
) -> list[dict]:
    ...

# Возврат: list[{'beam_id': int, 'median_magnitude': float}]
```

### compute_statistics

```python
def compute_statistics(
    data: np.ndarray,    # complex64, shape (B*N,) или (B,N)
    beam_count: int = 1
) -> list[dict]:
    ...

# Возврат:
# list[{
#   'beam_id':        int,
#   'mean_real':      float,   # Re(комплексного среднего)
#   'mean_imag':      float,   # Im(комплексного среднего)
#   'mean_magnitude': float,   # E[|z|]
#   'variance':       float,   # Var(|z|), ddof=0
#   'std_dev':        float,   # sqrt(variance)
# }]
```

---

### compute_all

```python
def compute_all(
    data: np.ndarray,    # complex64, shape (B*N,) или (B,N)
    beam_count: int = 1
) -> list[dict]:
    ...

# Возврат:
# list[{
#   'beam_id':          int,
#   'mean_real':        float,
#   'mean_imag':        float,
#   'variance':         float,   # Var(|z|), ddof=0
#   'std_dev':          float,
#   'mean_magnitude':   float,   # E[|z|]
#   'median_magnitude': float,   # sorted[N/2]
# }]
```

### compute_all_float

```python
def compute_all_float(
    data: np.ndarray,    # float32, shape (B*N,) или (B,N) — уже вычисленные модули
    beam_count: int = 1
) -> list[dict]:
    ...

# Возврат: те же 7 ключей, но mean_real и mean_imag всегда 0.0
```

---

### Полный Python пример

```python
import sys
sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np

# Context
ctx = gpuworklib.ROCmGPUContext(0)

# Processor
proc = gpuworklib.StatisticsProcessor(ctx)

# Data: beam-major layout
beam_count, n_point = 4, 4096
t = np.arange(n_point, dtype=np.float32) / 1000.0
freq = 100.0
data = np.tile(
    (np.cos(2 * np.pi * freq * t) + 1j * np.sin(2 * np.pi * freq * t))
    .astype(np.complex64),
    beam_count
)

params = {'beam_count': beam_count}

# Full statistics
stats = proc.compute_statistics(data, beam_count=beam_count)
for r in stats:
    print(f"Beam {r['beam_id']}: |mean|={r['mean_magnitude']:.4f}, std={r['std_dev']:.4f}")

# NumPy verification
for b in range(beam_count):
    beam = data[b*n_point:(b+1)*n_point]
    mags = np.abs(beam)
    np_mean_mag = np.mean(mags)
    np_std = np.std(mags, ddof=0)
    err_mag = abs(stats[b]['mean_magnitude'] - np_mean_mag)
    err_std = abs(stats[b]['std_dev'] - np_std)
    print(f"  Beam {b}: err_mean_mag={err_mag:.2e}, err_std={err_std:.2e}")

# Median
medians = proc.compute_median(data, beam_count=beam_count)
for m in medians:
    beam = data[m['beam_id']*n_point:(m['beam_id']+1)*n_point]
    np_median = float(np.sort(np.abs(beam))[n_point // 2])
    err = abs(m['median_magnitude'] - np_median)
    print(f"  Beam {m['beam_id']} median: GPU={m['median_magnitude']:.4f}, "
          f"NumPy={np_median:.4f}, err={err:.4f}")
```

---

### Сборка Python модуля

```bash
cmake -B build/debian-radeon9070 \
      -DBUILD_PYTHON=ON \
      -DENABLE_ROCM=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/debian-radeon9070 -j4

# .so файл:
# build/debian-radeon9070/python/gpuworklib.cpython-313-x86_64-linux-gnu.so

# Запуск с GPU (render group):
sg render -c "python3 my_script.py"
```

---

## Ссылки

- [Full.md](Full.md) — полная документация с математикой и pipeline
- [Quick.md](Quick.md) — краткий справочник
- [Doc/Python/rocm_modules_api.md](../../Python/rocm_modules_api.md) — все ROCm Python классы

---

*Обновлено: 2026-03-20*