# 🔍 Проверка реализации Heterodyne vs PLAN — v2

> **Дата проверки**: 2026-02-21 (обновлено после правок Alex)
> **Проверял**: Кодо (AI Assistant) — по фактическому коду на диске
> **Источники**:
>   - `PLAN_Heterodyne_LFM_Dechirp.md` v2.0 — что планировалось
>   - `ALGORITHM_Heterodyne_LFM_Dechirp.md` — описание первичной реализации
>   - Реальный код (прочитан с диска): все файлы модуля

---

## 📊 ИТОГОВАЯ ТАБЛИЦА (фактически проверено)

### C++ Модуль и инфраструктура

| Элемент | ПЛАН | ФАКТ (на диске) | Статус |
|---------|------|-----------------|--------|
| `modules/heterodyne/` директория | ✅ | ✅ | ✅ OK |
| `include/heterodyne_dechirp.hpp` | ✅ | ✅ с `conj_gen_`, `params_dirty_` | ✅ OK |
| `include/heterodyne_params.hpp` | ✅ | ✅ | ✅ OK |
| `include/i_heterodyne_processor.hpp` | ✅ | ✅ | ✅ OK |
| `include/processors/heterodyne_processor_opencl.hpp` | ✅ | ✅ с buf_rx_, buf_ref_, кешами | ✅ OK |
| `include/processors/heterodyne_processor_rocm.hpp` | ✅ заглушка | ✅ заглушка | ✅ OK |
| `src/heterodyne_dechirp.cpp` | ✅ | ✅ с EnsureConjugateGenerator() | ✅ OK |
| `src/heterodyne_processor_opencl.cpp` | ✅ | ✅ 477 строк | ✅ OK |
| `src/heterodyne_processor_rocm.cpp` | ✅ stub | ✅ stub | ✅ OK |
| `kernels/opencl/dechirp_multiply.cl` | ✅ | ✅ 1D gid, conj(rx×ref) | ✅ OK |
| `kernels/opencl/dechirp_correct.cl` | ✅ | ✅ с phase_step param | ✅ OK |
| `CMakeLists.txt` heterodyne | ✅ | ✅ add_subdirectory | ✅ OK |
| `src/main.cpp` включает heterodyne | ✅ | ✅ активен | ✅ OK |
| `LfmConjugateGenerator` hpp | ✅ | ✅ namespace signal_gen | ✅ OK |
| `lfm_conjugate.cl` | ✅ (имя план: ...generator.cl) | ✅ lfm_conjugate.cl | ⚠️ имя ≠ плану |

### C++ Тесты

| Элемент | ПЛАН | ФАКТ (на диске) | Статус |
|---------|------|-----------------|--------|
| `tests/all_test.hpp` | ✅ | ✅ вызывает тесты 1-7 | ✅ OK |
| `tests/README.md` | ✅ | ✅ | ✅ OK |
| Test 1: Single antenna (100μs) | ✅ | ✅ basic.hpp | ✅ OK |
| Test 2: 5 ant линейные [100..500]μs | ✅ | ✅ basic.hpp | ✅ OK |
| Test 3: Correction → DC | ✅ | ✅ basic.hpp | ✅ OK |
| Test 4: Full pipeline Process() | ✅ | ✅ pipeline.hpp | ✅ OK |
| Test 5: ProcessExternal() | ✅ | ✅ pipeline.hpp | ✅ OK |
| Test 6: Random delays seed=42 | ✅ | ✅ basic.hpp (seed=42) | ✅ OK |
| Test 7: AllMaxima контроль | ✅ | ✅ pipeline.hpp | ✅ OK |
| `test_heterodyne_external_ctx.hpp` (отд. файл) | ✅ план | ❌ слит в pipeline.hpp | ⚠️ приемлемо |

### Оптимизации

| OPT | Описание | ФАКТ (на диске) | Статус |
|-----|----------|-----------------|--------|
| OPT-1 | Cache cl_kernel (не создавать per-call) | ✅ kernel_multiply_, kernel_correct_ в hpp | ✅ OK |
| OPT-2 | Cache GPU buffers (EnsureBuffers) | ✅ buf_rx_, buf_ref_, buf_dc_, buf_corr_, buf_freq_ | ✅ OK |
| OPT-3 | GPU ref (no PCIe round-trip) | ✅ DechirpWithGPURef() — **только для ProcessExternal** | ⚠️ Частично |
| OPT-4 | Cache LfmConjugateGenerator | ✅ EnsureConjugateGenerator(), params_dirty_ | ✅ OK |
| OPT-5 | 1D kernel launch | ✅ `gid = ant * num_samples + n` | ✅ OK |
| OPT-6 | phase_step precomputed (без деления в ядре) | ✅ kernel принимает phase_step[] | ✅ OK |

### Python слой

| Элемент | ПЛАН | ФАКТ (на диске) | Статус |
|---------|------|-----------------|--------|
| Биндинги `py_heterodyne.hpp` | план: `bindings/heterodyne_bindings.cpp` | ✅ `python/py_heterodyne.hpp` (191 стр) | ✅ OK (путь другой) |
| `register_heterodyne()` | ✅ | ✅ | ✅ OK |
| `Python_test/heterodyne/` директория | ✅ | ✅ | ✅ OK |
| `test_heterodyne.py` (4 pytest теста) | — | ✅ 230 строк | ✅ OK |
| `test_heterodyne_step_by_step.py` (8 шагов) | ✅ план | ❌ нет | ❌ **Отсутствует** |
| `test_heterodyne_comparison.py` (GPU vs CPU) | ✅ план | ❌ нет | ❌ **Отсутствует** |
| `Results/Plots/heterodyne/` директория | ✅ | ✅ создана (пустая) | ✅ OK |
| 10 графиков в директории | ✅ план | ❌ пустая (тесты не запускались) | ⏳ Ждёт запуска |
| SNR вычисление в BuildResult | ✅ | ✅ `20*log10(peak/noise_est)` | ✅ OK |
| F_BEAT_TOL_HZ = 5000 Гц | ✅ | ✅ basic.hpp line 66 | ✅ OK |

### Параметры тестов

| Параметр | ПЛАН (Section 0) | ALGORITHM (описание) | КОД (факт) |
|----------|-----------------|----------------------|------------|
| fs | 12e6 | 12e6 | 12e6 ✅ |
| B (полоса) | 1e6 | 1e6 | **2e6** ⚠️ |
| N | 4000 | 4000 | **8000** ⚠️ |
| T (длит.) | 333.33 μs | 333.33 μs | **666.67 μs** ⚠️ |
| μ (chirp rate) | 3e9 Hz/s | 3e9 Hz/s | 3e9 Hz/s ✅ |
| Антенны | 5 | 5 | 5 ✅ |
| Задержки линейные | [100..500] μs | [20..200] μs | **[100..500] μs** ✅ (план) |
| F_BEAT_TOL_HZ | 5000 | — | 5000 ✅ |

---

## 🐛 Найденные проблемы в коде

### БАГ-1: OPT-3 в `Process()` неэффективен — ref_gpu генерируется и выбрасывается

**Файл**: `modules/heterodyne/src/heterodyne_dechirp.cpp`, метод `Process()`

**Проблема** (из реального кода):
```cpp
// OPT-3: Generate ref on GPU, keep it there
cl_mem ref_gpu = conj_gen_->GenerateToGpu();    // ← генерируем на GPU

// For CPU rx, we still need to upload, but ref stays on GPU (no double PCIe)
auto ref_cpu = conj_gen_->GenerateToCpu();      // ← ТАКЖЕ генерируем на CPU!
auto dc_data = processor_->Dechirp(rx_data, ref_cpu, params_);  // ← используем CPU!

clReleaseMemObject(ref_gpu);  // ← GPU ref выбрасывается без использования!
```

**Суть бага**: `ref_gpu` создаётся, ничем не используется, и сразу освобождается.
Реально вызывается старый `Dechirp()` с `ref_cpu`. Это **хуже чем до оптимизации** —
два генерации вместо одной + лишний GPU→CPU трансфер.

**Правильная реализация**:
```cpp
// Вариант A: Добавить DechirpWithCPURxAndGPURef() в processor
cl_mem ref_gpu = conj_gen_->GenerateToGpu();  // только на GPU
auto dc_data = processor_->DechirpWithGPURx(rx_data, ref_gpu, params_);
// ^ uploads rx to GPU, uses ref_gpu already on GPU
clReleaseMemObject(ref_gpu);

// Вариант B: Простое удаление лишнего (полная OPT-3 для CPU rx невозможна
//            без изменения processor interface)
// → убрать GenerateToGpu() и clReleaseMemObject(ref_gpu) из Process()
// → оставить только ref_cpu путь (без бесполезного GPU ref)
```

**Для ProcessExternal()**: OPT-3 работает ПРАВИЛЬНО:
```cpp
cl_mem ref_gpu = conj_gen.GenerateToGpu();
auto dc_data = processor_->DechirpWithGPURef(rx_cl_mem, ref_gpu, params);
clReleaseMemObject(ref_gpu);  // ← используется и корректно освобождается
```

---

### РАЗ-1: ALGORITHM файл устарел — параметры не совпадают с кодом

**Файл**: `ALGORITHM_Heterodyne_LFM_Dechirp.md`

| Что | В ALGORITHM | В КОДЕ (факт) |
|-----|------------|----------------|
| B (полоса) | 1 МГц | **2 МГц** |
| N | 4000 | **8000** |
| T | 333.33 μs | **666.67 μs** |
| Задержки | [20, 50, 100, 150, 200] мкс | **[100, 200, 300, 400, 500] мкс** |
| Тестов | 5 | **7** |
| Python биндинги | не упоминаются | `python/py_heterodyne.hpp` |

**Действие**: обновить ALGORITHM.md после стабилизации параметров.

---

### РАЗ-2: Два Python теста из плана не созданы

**ПЛАН** (Section 5) предполагал 3 Python файла, создан только 1:

| Файл | ПЛАН | ФАКТ |
|------|------|------|
| `test_heterodyne.py` (базовые pytest) | ✅ | ✅ 4 теста |
| `test_heterodyne_step_by_step.py` | ✅ 8 шагов, вывод, графики | ❌ **НЕТ** |
| `test_heterodyne_comparison.py` | ✅ GPU vs CPU отчёт | ❌ **НЕТ** |

`test_heterodyne_step_by_step.py` — самый ценный для отладки (8 шагов с промежуточными значениями и графиками). Создать, когда Alex подтвердит стабильность модуля.

---

### РАЗ-3: Параметры в ПЛАНЕ vs КОДЕ (B и N изменены)

ПЛАН зафиксировал `B=1MHz, N=4000`, а в коде `B=2MHz, N=8000`.

**Почему изменено**:
- `B=2MHz, N=8000` → μ = 2e6/(8000/12e6) = 3e9 Hz/s (тот же chirp rate!)
- T = 8000/12e6 = 666.67 μs (в 2 раза длиннее)
- Задержка 500μs << T=667μs (нет обрезания сигнала) ← это исправляет проблему плана
- При B=1MHz, N=4000, задержка 500μs ≈ 1.5×T → часть сигнала выходит за пределы — **физически некорректно**

**Вывод**: изменение B и N — **технически правильное решение**.
Нужно обновить ПЛАН и ALGORITHM с новыми параметрами.

---

## ✅ Что реализовано ПРАВИЛЬНО и ЛУЧШЕ чем в плане

| Что | Комментарий |
|-----|-------------|
| Namespace `signal_gen` | Соответствует паттерну проекта (план предлагал drv_gpu_lib) |
| `LfmParams + SystemSampling` | Переиспользует существующие структуры |
| `conj(rx×ref)` вместо `rx×conj(ref)` | Пик на +f_beat, корректно с SpectrumMaximaFinder |
| `py_heterodyne.hpp` в `python/` | Правильное место для проекта |
| OPT-2: буферы 5 кешированных | Включая `buf_corr_` и `buf_freq_` — план предлагал 3 |
| SNR из соседних точек | Реалистичная оценка, а не нулевое значение |
| N=8000, B=2MHz | Физически корректнее при delay до 500μs |

---

## 📋 Что делать дальше (приоритеты)

### 🔴 Критично (баги):
1. **Исправить БАГ-1** в `Process()` — убрать бесполезный `GenerateToGpu()` + `clReleaseMemObject(ref_gpu)`
   или реализовать `DechirpWithCPURxAndGPURef()`

### 🟡 Важно:
2. **Обновить `ALGORITHM_Heterodyne_LFM_Dechirp.md`** — новые параметры (B=2MHz, N=8000, 7 тестов)
3. **Создать `test_heterodyne_step_by_step.py`** — пошаговый тест с выводом и графиками
4. **Запустить Python тесты** — проверить что `test_heterodyne.py` проходит
5. **Обновить ПЛАН** — зафиксировать B=2MHz, N=8000 как принятые параметры

### 🟢 Желательно:
6. **Создать `test_heterodyne_comparison.py`** — детальный GPU vs CPU сравнительный отчёт
7. **Обновить `CLAUDE.md`** — Heterodyne: 🟡 → 🟢 Active (после прохождения Python тестов)

---

## 📐 Итог: что сделано из плана

```
C++ модуль:      ████████████████████ 100%  (все файлы, 7/7 тестов)
Оптимизации:     ████████████████░░░░  80%  (OPT-3 частично багованный)
Python биндинги: ████████████████████ 100%  (py_heterodyne.hpp + register)
Python тесты:    ████████████░░░░░░░░  40%  (1 из 3 файлов создан)
Графики:         ████░░░░░░░░░░░░░░░░  20%  (директория создана, пустая)
Документация:    ████████████░░░░░░░░  60%  (ALGORITHM устарел)
```

**Общая готовность**: ~80% от полного плана

---

*Обновлено: 2026-02-21 | Кодо (AI Assistant)*
*Предыдущая версия CHECK_Heterodyne_vs_Plan.md заменена этой (v2)*
