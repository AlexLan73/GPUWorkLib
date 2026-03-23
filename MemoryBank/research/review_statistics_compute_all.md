# Code Review: Statistics ComputeAll

> **⚠️ Статус документа**: первичный отчёт с замечаниями. **Актуальный итог после исправлений** — [`review_statistics_compute_all_v2.md`](review_statistics_compute_all_v2.md) (все CR закрыты ✅).

> **Дата**: 2026-03-20
> **Задача**: `MemoryBank/tasks/TASK_statistics_compute_all.md`
> **Файлы**: `modules/statistics/**` vs спецификация TASK_statistics_compute_all.md

---

## Критические проблемы 🔴

### CR1: `std::cout` вместо `ConsoleOutput` в test runner

**Файл**: `modules/statistics/tests/test_statistics_compute_all_benchmark.hpp`
**Строки**: 37–44 (заголовок), 103–105 (сравнение separate vs ComputeAll)

```cpp
// НАРУШЕНИЕ (строки 37-44):
std::cout << "\n"
          << "============================================================\n"
          << "  Statistics ComputeAll Benchmark — ROCm\n"
          ...

// НАРУШЕНИЕ (строки 103-105):
std::cout << "  Separate (avg 20 runs): " << sep_ms << " ms\n";
std::cout << "  ComputeAll (avg 20 runs): " << all_ms << " ms\n";
std::cout << "  Speedup: " << (sep_ms / all_ms) << "x\n";
```

**Проблема**: CLAUDE.md и архитектурное правило GPUWorkLib — "Консоль только через `ConsoleOutput::GetInstance()`".
При 10 GPU вывод через `std::cout` будет перемешиваться.

**Исправление**:
```cpp
auto& con = ConsoleOutput::GetInstance();
con.Print(0, "Stats Bench", "  Separate: " + std::to_string(sep_ms) + " ms");
con.Print(0, "Stats Bench", "  ComputeAll: " + std::to_string(all_ms) + " ms");
con.Print(0, "Stats Bench", "  Speedup: " + std::to_string(sep_ms / all_ms) + "x");
```

---

### CR2: Python тесты 1 и 2 — тавтологии (сравнивают NumPy с самим собой)

**Файл**: `Python_test/statistics/test_compute_all.py`
**Строки**: 153–164, 173–181

```python
# ТАВТОЛОГИЯ (Test 1, строки 155-157):
for field in ("mean_real", "mean_imag", "variance", "std_dev", "mean_magnitude"):
    result = validator.validate(stats[b][field], stats[b][field], name=field)
    # ^^^ ОБА аргумента одинаковые! Тест ВСЕГДА пройдёт!

# ТАВТОЛОГИЯ (Test 2, строки 177-179):
r = validator.validate(stats[b][field], stats[b][field], name=field)
# ^^^ Тоже сравниваем NumPy с NumPy
```

**Проблема**: `DataValidator.validate(x, x, ...)` всегда возвращает `passed=True` —
GPU реализация не верифицируется этими тестами совсем.

**Что должно быть** (смысл теста — проверить что MergeResults корректен):
```python
# Separate NumPy calculations
stats   = ref_statistics(data, BEAM_COUNT)   # → mean, var, std, mean_mag
medians = ref_median(data, BEAM_COUNT)       # → median

# "Combined" — simulate what ComputeAll should return
combined = {**stats[b], "median_magnitude": medians[b]}

# Validate combined vs separate (проверяем логику merge)
result = validator.validate(combined["variance"], stats[b]["variance"], name="variance")
result = validator.validate(combined["median_magnitude"], medians[b], name="median")
```
Либо вызывать реальный `proc.compute_all()` через gpuworklib и сравнивать с NumPy.

---

## Важные замечания 🟡

### IM1: Test 13 проверяет не все поля

**Файл**: `modules/statistics/tests/test_statistics_rocm.hpp`
**Строки**: 870–873

```cpp
// Test 13 проверяет ТОЛЬКО:
float e_mean_mag = std::fabs(full[b].mean_magnitude   - ref_stats[b].mean_magnitude);
float e_median   = std::fabs(full[b].median_magnitude - ref_medians[b].median_magnitude);

// НЕ ПРОВЕРЯЮТСЯ: mean.real, mean.imag, variance, std_dev
```

**Задача** (Шаг 6, Test 13) требует: *"verify void\* path"* — по аналогии с Test 12 все поля.
Test 12 (CPU path) проверяет все 6 полей, Test 13 (GPU path) — только 2.

**Исправление**: добавить проверку `e_mean_re, e_mean_im, e_var, e_std` по аналогии с Test 12.

---

### IM2: Несогласованность `beam_count` auto-detection в Python bindings

**Файл**: `python/py_statistics.hpp`

| Метод | Auto-detect beam_count из ndim=2 |
|-------|----------------------------------|
| `compute_all` (строки 123–153) | ❌ нет |
| `compute_all_float` (строки 155–162) | ✅ есть (`if (buf.ndim == 2 && beam_count == 0)`) |

Пользователь, привыкший к `compute_all_float` с 2D массивом и `beam_count=0`,
получит неожиданное поведение у `compute_all`.

**Исправление**: добавить аналогичный auto-detect в `compute_all`:
```cpp
auto buf_check = data.request();
if (buf_check.ndim == 2 && beam_count == 0)
    beam_count = static_cast<uint32_t>(buf_check.shape[0]);
if (beam_count == 0) beam_count = 1;
```

---

### IM3: `ComputeAllFloat(vector<float>)` не принимает `prof_events`

**Файл**: `modules/statistics/include/statistics_processor.hpp`, строка 138

```cpp
// Три метода имеют:
std::vector<FullStatisticsResult> ComputeAll(const vector<complex<float>>&, ...,
    StatisticsROCmProfEvents* prof_events = nullptr);

// Четвёртый — нет:
std::vector<FullStatisticsResult> ComputeAllFloat(const vector<float>& data,
    const StatisticsParams& params);  // ← нет prof_events
```

Незначительная несогласованность (CPU float overload всё равно делегирует в GPU overload), но нарушает симметрию API.

---

## Рекомендации 🟢

### REC1: Профилирование — реализация лучше чем в спеке

Задача предполагала `ctx_.profiler_events.push_back(...)` — встроенный профайлер.
Реализация использует `StatisticsROCmProfEvents* prof_events = nullptr` — **лучше**: гибко, не загрязняет `ctx_`, benchmark решает когда собирать. Одобряем.

### REC2: Benchmark тестирует только CPU path (65536)

**Файл**: `test_statistics_compute_all_benchmark.hpp`
Задача упоминала ещё `GPU data path (4 beams × 1M)` — его нет.
При желании — добавить второй benchmark с `ComputeAll(gpu_ptr, params)` и `n_point=1M`.
Сейчас не критично.

### REC3: Test 15 Edge Case B — уточнить путь в логе

**Файл**: `modules/statistics/tests/test_statistics_rocm.hpp`, строка 1021

`n_point=100000 == kHistogramThreshold` → уходит в **radix sort** (условие `> threshold`).
Рекомендация: добавить лог `"n=100000 → radix sort path (not histogram)"` для ясности.

---

## Соответствие спецификации и критериям приёмки

| Критерий приёмки | Статус | Примечание |
|------------------|--------|------------|
| ComputeAll() == ComputeStatistics + ComputeMedian | ✅ | Test 12, tolerance 1e-5f |
| Все 4 перегрузки работают | ✅ | CPU/GPU complex + CPU/GPU float |
| ComputeAllFloat: mean == {0,0} | ✅ | Test 14 явно проверяет |
| C++ тесты 12–15 проходят | ✅ | Реализованы и подключены в run() |
| Python тесты с DataValidator(1e-5) | ⚠️ | Тесты 1 и 2 — тавтологии (CR2) |
| Existing Tests 1–11 не сломаны | ✅ | |
| Python bindings compute_all/compute_all_float | ✅ | |
| Профилирование + benchmark в all_test.hpp | ✅ | Закомментирован (правильно) |
| Документация Full.md, API.md, statistics_api.md | ✅ | Все три обновлены |

---

## Соответствие стандартам GPUWorkLib

| Стандарт | Статус | Детали |
|----------|--------|--------|
| DrvGPU интеграция (GpuContext, IBackend) | ✅ | |
| Профилирование только через GPUProfiler | ✅ | GpuBenchmarkBase + RecordROCmEvent |
| **Консоль только через ConsoleOutput** | ❌ | std::cout в test_statistics_compute_all_benchmark.hpp |
| Ref03 архитектура (Facade + Ops) | ✅ | |
| Google C++ Style + 2-space indent | ✅ | |
| Lazy compilation (EnsureCompiled) | ✅ | |
| Multi-GPU safe | ✅ | per-module ctx_, ConsoleOutput |
| Обратная совместимость API | ✅ | Existing methods не тронуты |

---

## Итог

**Реализация на 95% соответствует спецификации.** Логика ComputeAll корректна, архитектура правильная, профилирование гибкое.

**Нужно исправить до merge**:
1. 🔴 CR1: заменить `std::cout` на `ConsoleOutput` в test runner
2. 🔴 CR2: исправить Python тесты 1 и 2 (убрать тавтологию)

**Желательно (не блокируют)**:
3. 🟡 IM1: дополнить Test 13 проверкой всех полей
4. 🟡 IM2: добавить beam_count auto-detect в `compute_all` Python binding
