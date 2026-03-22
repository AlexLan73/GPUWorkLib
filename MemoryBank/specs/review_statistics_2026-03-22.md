# 🔍 Code Review: modules/statistics

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~19 файлов (.hpp/.cpp/.hip/.cl)
> **Методы анализа**: sequential-thinking, context7 (ROCm), web search (Welford parallel GPU variance)
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | **True Parallel Welford**: naive E[x²]-E[x]² → per-thread Welford accumulator + merge step (Bennett et al. 2009). Численно стабильно! | `statistics_kernels_rocm.hpp` (welford_fused, welford_stats, welford_float), `welford_fused_op.hpp` (shared_mem 4→5), `welford_float_op.hpp` (shared_mem 2→3) |
| 2 | 🟡→✅ | extract_medians: усреднение двух средних для чётного n_point | `statistics_kernels_rocm.hpp` |
| 3 | 🟡→✅ | MakeROCmDataFromEvents → shared `include/rocm_profiling_helpers.hpp` | `statistics_processor.cpp` |
| 4 | 🟡→✅ | Float vector wrappers: hipMalloc/hipMemcpy/hipFree → `UploadFloatData()` (RequireShared + async H2D, zero аллокаций) | `statistics_processor.hpp/cpp` |
| 5 | 🟡→✅ | mean_reduce LDS: `[256+1]` → `[BLOCK_SIZE+1]` + добавлен `#ifndef BLOCK_SIZE` guard | `statistics_kernels_rocm.hpp` |

---

## 📊 Общая оценка

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| **Архитектура** | ⭐⭐⭐⭐⭐ | **Эталонный** Ref03: GpuContext + 6 Ops + BufferSet + shared_buf |
| Kernel оптимизация | ⭐⭐⭐⭐⭐ | LDS +1 padding, warp shuffle, double-load, 2D grid, __launch_bounds__ |
| Алгоритмы | ⭐⭐⭐⭐ | welford_fused (1 pass), histogram median O(n), radix sort для малых n |
| Численная стабильность | ⭐⭐⭐ | Используется naive E[x²]-E[x]² (НЕ Welford!), годится для ЦОС |
| API | ⭐⭐⭐⭐⭐ | CPU/GPU/Float overloads, ComputeAll, profiling events |
| Тесты | ⭐⭐⭐⭐ | test_statistics_rocm + test_statistics_float_rocm + benchmark |
| Документация | ⭐⭐⭐⭐⭐ | Каждый kernel с описанием оптимизаций (P0-A, P1-B, P2-B...) |

**ВЕРДИКТ**: Лучший модуль проекта по архитектуре. Служит эталоном для миграции других модулей на Ref03.

---

## 📐 Архитектура (Ref03 Layer Map)

```
Layer 6: StatisticsProcessor (Facade)
  ├── EnsureCompiled → ctx_.CompileModule(10 kernels, one call)
  ├── ComputeMean → MeanReductionOp.Execute()
  ├── ComputeStatistics → WelfordFusedOp.Execute()
  ├── ComputeMedian → MedianRadixSortOp / MedianHistogramOp / MedianHistogramComplexOp
  └── ComputeAll → WelfordFusedOp + Median* (один upload)

Layer 5: Concrete Ops (6 классов)
  ├── MeanReductionOp          — BufferSet<1> (reduce_buf)
  ├── WelfordFusedOp           — BufferSet<0> (no private bufs)
  ├── WelfordFloatOp           — BufferSet<0>
  ├── MedianRadixSortOp        — BufferSet<3> (sort, temp, offsets)
  ├── MedianHistogramOp        — BufferSet<3> (hist, prefix, value)
  └── MedianHistogramComplexOp — BufferSet<3>

Layer 3: GpuKernelOp (base)
Layer 1: GpuContext ctx_ — stream, 10 compiled kernels, 4 shared slots
```

---

## 🔴 Критические проблемы (1)

### 1. "Welford" kernels используют naive variance формулу

**Файл**: `include/kernels/statistics_kernels_rocm.hpp:286-293` (welford_stats), `:364-374` (welford_fused)

**Описание**: Ядра называются "welford_*", но используют формулу `Var = E[x²] - (E[x])²`:

```hip
// Текущий код — НЕ Welford, а naive one-pass:
sum_mag += mag;         // accumulate sum
sum_sq  += mag * mag;   // accumulate sum of squares
...
r.mean_mag = vm * inv_n;           // E[x]
float mean_sq = vs * inv_n;       // E[x²]
r.variance = mean_sq - r.mean_mag * r.mean_mag;  // E[x²] - E[x]²  ← CATASTROPHIC CANCELLATION risk
```

**Проблема**: Когда дисперсия мала относительно среднего (σ² ≪ μ²), происходит [catastrophic cancellation](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Na%C3%AFve_algorithm). Пример:
- Сигнал: 1000.0 ± 0.001 → μ²=1e6, σ²=1e-6 → E[x²]-E[x]² вычитает два числа ~1e6, теряя 12 значащих цифр float32.

**Для ЦОС**: Обычно magnitude в диапазоне 0-100, variance ≠ 0 → формула работает. Но для сигналов с большим DC offset (постоянная составляющая + шум) — даст мусор.

**Истинный Welford для GPU** (параллельный merge, [Bennett et al. 2009](https://www.osti.gov/servlets/purl/1028931/)):
```
// Per-thread Welford accumulator:
struct WelfordAcc { float mean, M2; int count; };

// Merge step (для warp shuffle / LDS reduction):
WelfordAcc merge(WelfordAcc a, WelfordAcc b) {
  int n = a.count + b.count;
  float delta = b.mean - a.mean;
  float mean = a.mean + delta * b.count / n;
  float M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / n;
  return {mean, M2, n};
}
```

**Рекомендация**: Если для текущих задач ЦОС формула E[x²]-E[x]² достаточна — переименовать kernel'ы (`sum_of_squares_stats` вместо `welford_*`). Если нужна истинная стабильность — реализовать parallel Welford с merge step.

**Приоритет**: 🟡 Средний — для типичных сигналов ЦОС работает, но имя вводит в заблуждение.

---

## 🟡 Важные замечания (4)

### 2. extract_medians: нет усреднения для чётного n_point

**Файл**: `include/kernels/statistics_kernels_rocm.hpp:394`

```hip
medians[b] = sorted[b * n_point + n_point / 2];
```

Для **чётного** `n_point` математическая медиана = `(sorted[n/2-1] + sorted[n/2]) / 2`. Kernel берёт только `sorted[n/2]` (верхний из двух средних).

**Влияние**: Для больших `n_point` (>1000) разница пренебрежимо мала. Но для малых чётных (n=4, n=8) ошибка заметна.

**Исправление**:
```hip
if (n_point % 2 == 0) {
    medians[b] = (sorted[b * n_point + n_point / 2 - 1] +
                  sorted[b * n_point + n_point / 2]) * 0.5f;
} else {
    medians[b] = sorted[b * n_point + n_point / 2];
}
```

---

### 3. MakeROCmDataFromEvents — ещё одна копия

**Файл**: `src/statistics_processor.cpp:32-50`

Третья копия helper'а (fft_processor, spectrum_processor, statistics). Мы уже создали `utils/rocm_profiling_helpers.hpp` — нужно использовать и здесь.

**Исправление**: Заменить static helper → `#include "utils/rocm_profiling_helpers.hpp"` + `using fft_func_utils::MakeROCmDataFromEvents`.

---

### 4. Float vector wrappers — синхронный hipMemcpy + hipMalloc/hipFree

**Файлы**: `src/statistics_processor.cpp:399-426` (ComputeStatisticsFloat), `:451-478` (ComputeMedianFloat), `:680-708` (ComputeAllFloat)

Каждый вызов `ComputeXxxFloat(vector<float>)`:
1. `hipMalloc` — новый буфер
2. `hipMemcpy` — **синхронный** H2D
3. Вызов GPU overload
4. `hipFree` — освобождение

**Проблема**: При частых вызовах — лишние аллокации/освобождения. `hipMalloc` на AMD может занимать ~100μs.

**Рекомендация**: Использовать `RequireShared(kMagnitudes, ...)` + `hipMemcpyHtoDAsync` как в `UploadComplexData()`. Тогда буфер переиспользуется.

---

### 5. Hardcoded LDS size 256 в mean_reduce kernels

**Файл**: `include/kernels/statistics_kernels_rocm.hpp:105-106`

```hip
__shared__ float sdata_x[256 + 1];  // hardcoded
__shared__ float sdata_y[256 + 1];  // hardcoded
```

В welford kernels используется `extern __shared__` (динамический размер). В mean_reduce — static [256+1]. Если `kBlockSize` изменится на 512, mean_reduce сломается.

**Рекомендация**: Заменить на `extern __shared__` или `__shared__ float sdata_x[BLOCK_SIZE + 1]` (через #define при компиляции — уже передаётся `-DBLOCK_SIZE=256`).

---

## 🟢 Рекомендации (5)

### 6. Histogram median — великолепный алгоритм ✅

4-pass byte-wise histogram: O(4n) вместо O(n log n) sort. Trick: `__float_as_uint(val) ^ 0x80000000u` даёт order-preserving uint для положительных float. Каждый pass сужает до правильного байта медианного значения. Идеальное решение для n_point > 100K.

### 7. welford_fused — один pass без magnitudes buffer ✅

Вычисляет `|z| = sqrt(re² + im²)` на лету в grid-stride loop вместо отдельного compute_magnitudes kernel. Экономит: 1 kernel launch + n_point × 4 bytes буфер magnitudes.

### 8. Стратегия auto-select для median ✅

```cpp
if (params.n_point > kHistogramThreshold) {
    median_hist_complex_op_.Execute(...);  // O(n)
} else {
    median_sort_op_.Execute(...);          // O(n log n) but faster for small n
}
```
`kHistogramThreshold = 100'000` — разумный порог.

### 9. shared_buf namespace для slot assignments ✅

Чистое разделение: GpuContext не знает про семантику слотов, модуль statistics определяет свои `kInput`, `kMagnitudes`, `kResult`, `kMediansCompact`. Правильный подход для generic infrastructure.

### 10. ComputeAll — устранение double upload ✅

Один `UploadComplexData` → Welford + Median вместо двух отдельных `ComputeStatistics` + `ComputeMedian` (которые каждый делают свой upload). Правильная оптимизация.

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | Статус | Комментарий |
|----------|--------|-------------|
| **Ref03** | ✅ **Эталон** | GpuContext + 6 Ops + BufferSet + shared_buf |
| **DrvGPU** | ✅ | IBackend*, stream из backend |
| **Профилирование** | ✅ | ROCmProfEvents + GPUProfiler + benchmarks |
| **Консоль** | ✅ | ConsoleOutput::GetInstance() |
| **Стиль** | ✅ | Google C++, CamelCase, snake_case |
| **Kernel cache** | ✅ | KernelCacheService через GpuContext |
| **Move semantics** | ✅ | Все классы |
| **Batch support** | ✅ | ComputeAll с profiling events |
| **Multi-GPU** | ✅ | Per-module GpuContext, no globals |

---

## 📚 Источники

### Web Search
- [Welford's Online Algorithm — Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [Parallel Variance — Bennett et al. 2009](https://www.osti.gov/servlets/purl/1028931/)
- [Numerically Stable Parallel (Co-)Variance — Schubert 2018](https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf)
- [NVIDIA Forum: Welford on GPU](https://forums.developer.nvidia.com/t/welfords-algorithm/325669)

### Context7
- ROCm docs: hipFFT changelog, rocFFT optimizations

---

## 📋 Сводка задач

| # | Приоритет | Описание | Файл | Сложность |
|---|-----------|----------|------|-----------|
| 1 | 🟡 | Переименовать welford_* или реализовать parallel Welford merge | statistics_kernels_rocm.hpp | Высокая |
| 2 | 🟡 | extract_medians: усреднение для чётного n_point | statistics_kernels_rocm.hpp:394 | Низкая |
| 3 | 🟡 | MakeROCmDataFromEvents → shared utility | statistics_processor.cpp:32-50 | Низкая |
| 4 | 🟡 | Float vector wrappers → RequireShared + async upload | statistics_processor.cpp | Средняя |
| 5 | 🟡 | mean_reduce LDS: hardcoded 256 → BLOCK_SIZE define | statistics_kernels_rocm.hpp:105 | Низкая |

---

*Ревью подготовлено с: sequential-thinking (2 шага), context7 (ROCm), WebSearch (Welford parallel GPU)*
