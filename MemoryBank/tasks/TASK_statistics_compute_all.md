# TASK: ComputeAll — Statistics + Median за один вызов

> **Дата**: 2026-03-20
> **Модуль**: `modules/statistics`
> **Приоритет**: Medium
> **Статус**: BACKLOG
> **Оценка**: ~3.5 часа (Шаг1=10min | Шаг2=10min | Шаг3=45min | Шаг4=30min | Шаг5=20min | Шаг6=30min | Шаг7=20min | Шаг8=5min | Шаг9=15min)

---

## Обоснование

### Проблема

Сейчас для получения полной статистики (mean + variance + std + **median**) нужно 2 вызова:

```cpp
auto stats  = proc.ComputeStatistics(gpu_data, params);  // upload + welford + sync + D2H
auto median = proc.ComputeMedian(gpu_data, params);       // upload + median  + sync + D2H
```

Это приводит к:
- **Двойной upload/D2D** одних и тех же данных (kInput)
- **Два `hipStreamSynchronize`** вместо одного
- **Два `hipMemcpyDtoH`** вместо двух D2H в одной pipeline

### Почему НЕ async

Анализ показал (см. `research/statistics_async_recommendations.md`):
- Multi-GPU параллелизм **решается threads** (каждый thread обслуживает свой GPU)
- CPU idle 1.5ms на одном ядре при 80+ ядрах (2 Xeon + 10 MI100) = **незаметно**
- Begin/End event API: сложность +++, выигрыш ~0 для нашей архитектуры
- Pinned memory: results = 2KB, разница нулевая

### Выигрыш от ComputeAll

| Сценарий | Statistics + Median порознь | ComputeAll | Ускорение |
|----------|---------------------------|------------|-----------|
| CPU data, 100b × 65K (52MB) | ~13 ms | ~8 ms | **1.6x** |
| GPU data, 100b × 65K | ~3.0 ms | ~2.7 ms | ~10% |
| GPU data, 100b × 1M | ~23 ms | ~22 ms | ~5% |

Основной выигрыш — **убираем двойной upload** для CPU data.
Для GPU data (production) — экономим один sync + один D2D copy.

---

## Предварительный анализ: совместимость Operations

Shared buffer конфликт-анализ:

| Operation | Читает | Пишет |
|-----------|--------|-------|
| WelfordFusedOp | kInput | kResult |
| MedianRadixSortOp | kInput → kMagnitudes → sort | kMediansCompact |
| MedianHistogramComplexOp | kInput | kMediansCompact |

**Конфликтов нет**: обе операции ЧИТАЮТ kInput, пишут в РАЗНЫЕ буферы (kResult vs kMediansCompact). Можно запускать последовательно в одном stream.

---

## План реализации

### Шаг 1: Новый тип результата

**Файл**: `modules/statistics/include/statistics_types.hpp`

Добавить `FullStatisticsResult`:
```
struct FullStatisticsResult {
  uint32_t beam_id;
  complex<float> mean;        // из WelfordFused
  float variance;              // из WelfordFused
  float std_dev;               // из WelfordFused
  float mean_magnitude;        // из WelfordFused
  float median_magnitude;      // из Median
};
```

Почему отдельный тип, а не пара `{StatisticsResult, MedianResult}`:
- Один вектор проще для Python bindings
- Один beam_id вместо двух
- Чище API

### Шаг 2: Новые методы в Facade

**Файл**: `modules/statistics/include/statistics_processor.hpp`

Добавить в public секцию:
```
// GPU data (production path — данные уже на GPU)
vector<FullStatisticsResult> ComputeAll(void* gpu_data, const StatisticsParams& params);

// CPU data (тесты, Python)
vector<FullStatisticsResult> ComputeAll(const vector<complex<float>>& data, const StatisticsParams& params);

// GPU float data (magnitudes already computed)
vector<FullStatisticsResult> ComputeAllFloat(void* gpu_float_data, const StatisticsParams& params);

// CPU float data
vector<FullStatisticsResult> ComputeAllFloat(const vector<float>& data, const StatisticsParams& params);
```

### Шаг 3: Реализация

**Файл**: `modules/statistics/src/statistics_processor.cpp`

Логика `ComputeAll(void* gpu_data, params)`:
```
1. EnsureCompiled()
2. CopyComplexGpuData(gpu_data, count)          // D2D → kInput (ОДИН РАЗ)
3. welford_fused_op_.Execute(beam_count, n_point) // kernel → kResult
4. if n_point > threshold:
     median_hist_complex_op_.Execute(...)         // kernel → kMediansCompact
   else:
     median_sort_op_.Execute(...)                 // kernel → kMediansCompact
5. hipStreamSynchronize(stream)                   // ОДИН sync
6. ReadStatisticsResults(beam_count)              // D2H из kResult
7. ReadMedianResults(beam_count)                  // D2H из kMediansCompact
8. Объединить в vector<FullStatisticsResult>
```

Добавить private helper:
```
vector<FullStatisticsResult> MergeResults(
    const vector<StatisticsResult>& stats,
    const vector<MedianResult>& medians);
```

Логика `ComputeAll(vector<complex<float>>& data, params)`:
```
Аналогично, но вместо CopyComplexGpuData → UploadComplexData
```

Логика `ComputeAllFloat(void* gpu_float_data, params)`:
```
1. CopyFloatGpuData → kMagnitudes
2. welford_float_op_.Execute(...)           → kResult
3. if threshold: median_hist_op_ else median_sort_op_.ExecuteFloat(...)  → kMediansCompact
4. hipStreamSynchronize
5. Read + Merge
Примечание: WelfordFloat пишет mean_re=0, mean_im=0 → clear mean в результате
```

### Шаг 4: Профилирование ComputeAll

**Файлы**:
- `modules/statistics/src/statistics_processor.cpp` — добавить ROCm profiling events
- `modules/statistics/tests/statistics_compute_all_benchmark.hpp` — НОВЫЙ
- `modules/statistics/tests/test_statistics_compute_all_benchmark.hpp` — НОВЫЙ

**В реализации** (statistics_processor.cpp) — добавить по паттерну существующих методов:
```
hipEvent_t ev_start, ev_stop;
hipEventCreate(&ev_start); hipEventCreate(&ev_stop);
hipEventRecord(ev_start, stream);
// ... kernel launches ...
hipEventRecord(ev_stop, stream);
hipStreamSynchronize(stream);
ctx_.profiler_events.push_back(MakeROCmDataFromEvents(ev_start, ev_stop, "ComputeAll"));
```

**Benchmark файл** (`statistics_compute_all_benchmark.hpp`):
```
namespace test_statistics_compute_all_benchmark
class ComputeAllBenchmark : GpuBenchmarkBase
  - сравнение времени: ComputeAll vs (ComputeStatistics + ComputeMedian) раздельно
  - размеры: 4 beams × 65536 (CPU data path) + 4 beams × 1M (GPU data path)
  - вывод: ExportMarkdown() в Results/Profiler/
```

**Test runner** (`test_statistics_compute_all_benchmark.hpp`):
```
namespace test_statistics_compute_all_benchmark
run_compute_all_benchmark(IBackend* backend, int gpu_id)
```

### Шаг 5: Python binding

**Файл**: `python/py_statistics.hpp`

Добавить в `PyStatisticsProcessor`:
```
py::list compute_all(py::array complex64, uint32_t beam_count)
py::list compute_all_float(py::array float32, uint32_t beam_count)
```

Возвращает list of dict:
```python
[{
    'beam_id': 0,
    'mean_real': ..., 'mean_imag': ...,
    'variance': ..., 'std_dev': ...,
    'mean_magnitude': ...,
    'median_magnitude': ...
}, ...]
```

⚠️ **Обязательно**: обернуть GPU-вызов в `py::gil_scoped_release` — следовать паттерну
всех существующих методов в этом файле (строки 47-50).

Добавить в `register_statistics()`:
```
.def("compute_all", ...)
.def("compute_all_float", ...)
```

### Шаг 6: C++ тест

**Файл**: `modules/statistics/tests/test_statistics_rocm.hpp`

Новый тест (Test 12):
```
ComputeAll — verify all fields match ComputeStatistics + ComputeMedian called separately.
- Генерируем данные (4 beams × 65536 points, random)
- Вызываем ComputeStatistics + ComputeMedian порознь → reference
- Вызываем ComputeAll → actual
- Сравниваем все поля (tolerance 1e-5)
```

Новый тест (Test 13):
```
ComputeAll GPU data — verify void* path.
- hipMalloc + upload вручную
- ComputeAll(gpu_data, params)
- Сравнить с reference (tolerance 1e-5)
```

Новый тест (Test 14):
```
ComputeAllFloat — verify float path.
- Генерируем float magnitudes
- ComputeAllFloat + сравниваем (tolerance 1e-5)
- ЯВНО проверить: result.mean == {0.0f, 0.0f} для всех beam (float path не имеет complex mean)
```

Новый тест (Test 15):
```
ComputeAll edge cases:
- beam_count=1, n_point=100  (ниже threshold → radix sort path)
- beam_count=4, n_point=100000  (n_point == kHistogramThreshold, граница выбора стратегии)
- Сравнить с раздельными вызовами (tolerance 1e-5)
```

### Шаг 7: Python тест

**Файл**: `Python_test/statistics/test_compute_all.py`

```python
# Использовать DataValidator(tolerance=1e-5, metric="max_rel")
# из Python_test/common/validators.py — единая точка задания tolerance.

# Test 1: compute_all matches compute_statistics + compute_median
# Test 2: compute_all_float matches compute_statistics_float + compute_median_float
#         + проверить mean_real == 0, mean_imag == 0 для всех beam
# Test 3: compute_all performance vs separate calls (timing comparison)
```

### Шаг 8: Обновить all_test.hpp

**Файл**: `modules/statistics/tests/all_test.hpp`

Добавить include + вызовы:
```cpp
#include "statistics_compute_all_benchmark.hpp"
#include "test_statistics_compute_all_benchmark.hpp"
// test_statistics_compute_all_benchmark::run(backend, gpu_id);  // раскомментировать для бенчмарка
// + новые тесты (Test 12–15) вызываются из test_statistics_rocm::run()
```

### Шаг 9: Документация

- Обновить `Doc/Modules/statistics/Full.md` — секция 7 (API): добавить ComputeAll, ComputeAllFloat
- Обновить `Doc/Modules/statistics/API.md` — добавить новые сигнатуры
- Обновить `Doc/Python/statistics_api.md` — добавить compute_all, compute_all_float

---

## Изменяемые файлы (summary)

| Файл | Действие | Сложность |
|------|----------|-----------|
| `include/statistics_types.hpp` | +FullStatisticsResult | Тривиально |
| `include/statistics_processor.hpp` | +4 метода | Просто |
| `src/statistics_processor.cpp` | +4 метода + MergeResults + profiling events | Просто |
| `python/py_statistics.hpp` | +2 метода + binding | Просто |
| `tests/test_statistics_rocm.hpp` | +4 теста (Test 12–15) | Средне |
| `tests/statistics_compute_all_benchmark.hpp` | НОВЫЙ (benchmark) | Средне |
| `tests/test_statistics_compute_all_benchmark.hpp` | НОВЫЙ (test runner) | Просто |
| `tests/all_test.hpp` | +include + вызовы | Тривиально |
| `Python_test/statistics/test_compute_all.py` | +3 теста | Просто |

**Существующий API НЕ меняется** — полная обратная совместимость.

---

## Критерии приёмки

- [ ] `ComputeAll()` возвращает те же значения что `ComputeStatistics()` + `ComputeMedian()` по отдельности (tolerance ≤ 1e-5, метрика `max_rel`)
- [ ] Все 4 перегрузки работают (GPU complex, CPU complex, GPU float, CPU float)
- [ ] `ComputeAllFloat`: поле `mean == {0.0f, 0.0f}` для всех beam (float path не имеет complex mean — задокументированное поведение)
- [ ] C++ тесты (Test 12–15) проходят на AMD GPU (ROCm)
- [ ] Python тесты проходят с `DataValidator(tolerance=1e-5, metric="max_rel")`
- [ ] Существующие тесты (Test 1-11) не сломаны
- [ ] Python bindings: `compute_all()` и `compute_all_float()` доступны
- [ ] Профилирование: `ComputeAll` добавлен в ROCm profiling events; benchmark файл создан и включён в `all_test.hpp`

---

## Зависимости

- Нет внешних зависимостей
- Все Op классы уже реализованы и протестированы
- Shared buffers kResult и kMediansCompact — независимые слоты

---

## Риски

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| MedianSort перезаписывает kMagnitudes, Welford читает kInput — ОК | Нет конфликта | Проверено по коду |
| MedianSort.Execute() вызывает ExecuteMagnitudes() → пишет kMagnitudes ПЕРЕД sort | Нет проблемы | kMagnitudes != kInput, WelfordFused не трогает kMagnitudes |
| Порядок: Welford первый, Median второй — оба читают kInput | Нет проблемы | Оба только ЧИТАЮТ kInput |
