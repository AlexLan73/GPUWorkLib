# Statistics — Краткий справочник

> GPU-статистика по лучам: среднее (complex), медиана, дисперсия, СКО за один проход

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Когда антенная решётка даёт данные одновременно по многим лучам, нужно быстро вычислить статистику по каждому. CPU делает это последовательно — медленно. Модуль обрабатывает все лучи параллельно на GPU за один вызов.

**Ключевая идея**: 256 лучей × 1.3M точек — GPU: ~30 мс, CPU: ~2000 мс (rocPRIM segmented sort на RDNA4).

---

### Что конкретно делает каждый метод

| Метод | Что делает | Когда брать |
|-------|------------|-------------|
| `ComputeStatistics` | Комплексное среднее + E[|z|] + Var(|z|) + STD(|z|) за **один проход** (Welford) | Основной метод. Всегда, когда нужна статистика |
| `ComputeMean` | Только комплексное среднее (Re + Im) — иерархическая редукция | Когда нужно только среднее, без дисперсии |
| `ComputeMedian` | Медиана модулей |z| — GPU radix sort + middle element | Робастная оценка уровня сигнала (шум, выбросы) |
| `ComputeStatisticsFloat` | Как `ComputeStatistics` но для float input (модули уже посчитаны) | После FFTProcessor: статистика |spectrum| |
| `ComputeMedianFloat` | Медиана для float input | Медиана |spectrum| после FFT |

---

### Аналогии и связь с другими модулями

- **strategies/** использует `ComputeStatisticsFloat(gpu_ptr, params)` для анализа |FFT-спектра|
- Может применяться после `FFTProcessor` для оценки уровня шума
- Результаты сравниваются с NumPy: `np.var(np.abs(z), ddof=0)`, `np.sort(np.abs(z))[N//2]`

**Ограничения**:
- ROCm-only: не работает на Windows, не работает с OpenCL backend
- `beam_count × n_point` должно делиться без остатка
- Первый вызов медленнее (~200-500 мс JIT компиляция ядер)

---

## Алгоритм

```
ComputeStatistics:  welford_fused kernel → E[z], E[|z|], E[|z|²] → Var, STD
ComputeMedian:      |z| → rocPRIM segmented_radix_sort → sorted[N/2]
ComputeMean:        2-phase reduction (phase1 block sum + final divide)
```

---

## Быстрый старт

### C++

```cpp
#include "statistics_processor.hpp"
#include "statistics_types.hpp"
#include "backends/rocm/rocm_backend.hpp"

#if ENABLE_ROCM
drv_gpu_lib::ROCmBackend backend;
backend.Initialize(0);

statistics::StatisticsProcessor proc(&backend);

std::vector<std::complex<float>> data(4 * 4096);  // 4 луча × 4096 сэмплов
// ... заполнить data (beam-major) ...

statistics::StatisticsParams params;
params.beam_count = 4;
params.n_point    = 4096;

// Полная статистика (рекомендуется)
auto stats = proc.ComputeStatistics(data, params);
for (const auto& r : stats) {
    // r.beam_id, r.mean (complex), r.mean_magnitude, r.variance, r.std_dev
}
#endif
```

### Python

```python
import sys; sys.path.insert(0, 'build/debian-radeon9070/python')
import gpuworklib, numpy as np

ctx = gpuworklib.ROCmGPUContext(0)    # НЕ GPUContext!
proc = gpuworklib.StatisticsProcessor(ctx)

# beam-major: beam0[0..N], beam1[0..N], ...
data = (np.random.randn(4 * 4096) + 1j * np.random.randn(4 * 4096)).astype(np.complex64)

results = proc.compute_statistics(data, beam_count=4)
# results[i]: {'beam_id', 'mean_real', 'mean_imag', 'mean_magnitude', 'variance', 'std_dev'}

means   = proc.compute_mean(data, beam_count=4)
# means[i]: {'beam_id', 'mean_real', 'mean_imag'}

medians = proc.compute_median(data, beam_count=4)
# medians[i]: {'beam_id', 'median_magnitude'}
```

---

## Ключевые параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `beam_count` | `uint32_t` | Число лучей (каналов) |
| `n_point` | `uint32_t` | Сэмплов на луч |
| Layout | — | beam-major: `data[b*N + i]` |

---

## Нюансы (важно)

- **Дисперсия**: population (ddof=0), NumPy: `np.var(x, ddof=0)`
- **Медиана**: `sorted[N/2]`, не `np.median()` (для чётного N отличается)
- **WARP_SIZE**: автоматически 64 для gfx9xx (CDNA), 32 для RDNA
- **Первый вызов**: ~200-500 мс JIT. После — из disk HSACO cache

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4, тесты с обоснованием, оптимизации
- [API.md](API.md) — полный справочник публичного API
- [Doc/Python/rocm_modules_api.md](../../Python/rocm_modules_api.md) — Python API

---

*Обновлено: 2026-03-09*