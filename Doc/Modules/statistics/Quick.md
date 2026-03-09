# Statistics — Краткий справочник

> GPU-статистика комплексных сигналов по лучам: mean, median, variance, std (ROCm/HIP)

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Иногда нужно не фильтровать или трансформировать сигнал, а измерить его характеристики: насколько он сильный, насколько шумный, какова его типичная амплитуда. Модуль считает эти характеристики для каждого луча (канала) на GPU — всё параллельно.

---

### Три функции — когда что брать

**`ComputeStatistics`** — вычисляет сразу всё: среднее (комплексное), среднюю амплитуду, дисперсию и среднеквадратичное отклонение. Один проход по данным через алгоритм Велфорда (численно устойчив). Бери когда нужна полная картина — шумовой пол, уровень сигнала, стабильность.

**`ComputeMedian`** — вычисляет медиану амплитуд |z| по каждому лучу. Сначала вычисляет модули, потом сортирует через radix sort на GPU, берёт средний элемент. Медиана устойчивее к выбросам, чем среднее. Бери когда в данных есть редкие импульсные помехи, которые испортят среднее.

**`ComputeMean`** — только комплексное среднее, через иерархическое сложение (два прохода). Бери когда нужно только среднее и ничего лишнего — самый быстрый вариант.

---

### Важная деталь про медиану

Медиана здесь — это `sorted[N/2]`, а не среднее двух средних. В NumPy: `np.sort(mags)[N//2]`, **не** `np.median()`. Это важно при сравнении с эталоном.

---

### ROCm-only

Модуль работает только на AMD GPU с ROCm (Linux). Первый запуск медленный (~1-3 сек) — JIT-компиляция HIP ядер через hiprtc. После сохранения HSACO на диск — загружается мгновенно.

---

## Алгоритм

```
complex<float>[beams × N]
    ├─ ComputeStatistics      welford_fused kernel (1 проход)          → mean + variance + std per beam
    ├─ ComputeMedian          compute_magnitudes + rocPRIM sort         → median(|z|) per beam
    ├─ ComputeMean            hierarchical reduce (2-phase)             → complex mean per beam
    ├─ ComputeStatisticsFloat welford_float (float input, mean={0,0})  → variance + std per beam [C++ only]
    └─ ComputeMedianFloat     rocPRIM sort (float input)                → median per beam [C++ only]
```

---

## Быстрый старт

### C++

```cpp
#include "statistics_processor.hpp"
#include "statistics_types.hpp"
#include "backends/rocm/rocm_backend.hpp"

drv_gpu_lib::ROCmBackend backend;
backend.Initialize(0);
statistics::StatisticsProcessor stats(&backend);

statistics::StatisticsParams params;
params.beam_count = 4;
params.n_point    = 8192;

std::vector<std::complex<float>> data(4 * 8192, {1.0f, 0.0f});

// Полная статистика (mean + variance + std + mean_mag)
auto results = stats.ComputeStatistics(data, params);
for (const auto& r : results) {
    // r.beam_id, r.mean, r.mean_magnitude, r.variance, r.std_dev
}

// Медиана модулей (GPU radix sort)
auto medians = stats.ComputeMedian(data, params);
// medians[i].beam_id, medians[i].median_magnitude

// Комплексное среднее
auto means = stats.ComputeMean(data, params);
// means[i].beam_id, means[i].mean (complex<float>)
```

### Python

```python
import sys; sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib
import numpy as np

ctx   = gpuworklib.ROCmGPUContext(0)
stats = gpuworklib.StatisticsProcessor(ctx)

data = (np.random.randn(4 * 8192) +
        1j * np.random.randn(4 * 8192)).astype(np.complex64)

results = stats.compute_statistics(data, beam_count=4)
# results[i]: {'beam_id', 'mean_real', 'mean_imag',
#              'mean_magnitude', 'variance', 'std_dev'}

medians = stats.compute_median(data, beam_count=4)
# medians[i]: {'beam_id', 'median_magnitude'}

means = stats.compute_mean(data, beam_count=4)
# means[i]: {'beam_id', 'mean_real', 'mean_imag'}
```

---

## Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `beam_count` | `uint32_t` | Число лучей/каналов |
| `n_point` | `uint32_t` | Точек на луч (complex float) |

---

## Важно

- ROCm-only: AMD GPU + Linux + `-DENABLE_ROCM=ON`
- Медиана = `sorted[N/2]`, **не** среднее двух средних.
  NumPy-сравнение: `np.sort(mags)[N//2]`, **не** `np.median()`
- Дисперсия population (ddof=0): сравнивать `np.std(mags, ddof=0)`
- Первый вызов: JIT-компиляция hiprtc (~1-3 с), далее HSACO с диска
- GPU input: `stats.ComputeStatistics(void* gpu_ptr, params)` — без PCIe HtoD
- Float-input методы (`ComputeStatisticsFloat`, `ComputeMedianFloat`): только C++ API, `mean={0,0}`
- Namespace: `statistics::`, не `drv_gpu_lib::`

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4-диаграммы, kernels, все тесты
- [API.md](API.md) — полный API Reference (все 8 методов, типы, примеры)
- [Doc/Python/rocm_modules_api.md](../../Python/rocm_modules_api.md) — Python API

---

*Обновлено: 2026-03-09*
