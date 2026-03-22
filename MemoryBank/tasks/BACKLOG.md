# BACKLOG — Очередь задач

> **Обновлено**: 2026-03-22
> **Главный план**: `MemoryBank/tasks/MODULES_WORK_PLAN.md`

---

## 🏗️ C++ test_utils — Модуль общей тестовой инфраструктуры

> **Индекс**: [`TASK_CppTest_INDEX.md`](TASK_CppTest_INDEX.md)
> **Исследование**: [`research/cpp_test_utils_full_plan.md`](../research/cpp_test_utils_full_plan.md)
> **Добавлено**: 2026-03-21

| # | Файл | Фаза | Приоритет | Что |
|---|------|------|-----------|-----|
| CppTest-01 | [TASK_CppTest_01](TASK_CppTest_01_result_configs.md) | 0 | 🔴 HIGH | `test_result.hpp` + `test_configs.hpp` (Value Objects) |
| CppTest-02 | [TASK_CppTest_02](TASK_CppTest_02_validators.md) | 0 | 🔴 HIGH | `validators/` — MaxRel, Abs, Rmse, Frequency, Composite |
| CppTest-03 | [TASK_CppTest_03](TASK_CppTest_03_references.md) | 0 | 🔴 HIGH | `references/` — CPU-эталоны signals, statistics, fft |
| CppTest-04 | [TASK_CppTest_04](TASK_CppTest_04_gpu_transfer.md) | 0 | 🔴 HIGH | `gpu_transfer.hpp` — ReadGpuBuffer (OpenCL + ROCm) |
| CppTest-05 | [TASK_CppTest_05](TASK_CppTest_05_runner_base.md) | 0 | 🔴 HIGH | `test_runner.hpp` + `gpu_test_base.hpp` + `reporters.hpp` + master |

**Порядок**: 01 → (02 ∥ 03 ∥ 04) → 05

---

## 🏗️ Python_test — Архитектурный рефакторинг v2

> **Индекс**: [`TASK_PythonArch_INDEX.md`](TASK_PythonArch_INDEX.md)
> **Исследование**: [`research/python_test_refactoring_plan.md`](../research/python_test_refactoring_plan.md)
> **Добавлено**: 2026-03-21

| # | Файл | Фаза | Приоритет | Что |
|---|------|------|-----------|-----|
| Arch-01 | [TASK_PythonArch_01](TASK_PythonArch_01_core_generators.md) | 1 | 🔴 HIGH | `Core/generators/` — CwGenerator, LfmGenerator, NoiseGenerator (Adapter+Factory) |
| Arch-02 | [TASK_PythonArch_02](TASK_PythonArch_02_core_processing.md) | 1 | 🔴 HIGH | `Core/processing/` — StatisticsAdapter, HeterodyneAdapter, FftAdapter |
| Arch-03 | [TASK_PythonArch_03](TASK_PythonArch_03_references.md) | 1 | 🔴 HIGH | `common/references/` — SignalReferences, FilterReferences (устранить дублирование) |
| Arch-04 | [TASK_PythonArch_04](TASK_PythonArch_04_validators.md) | 2 | 🟠 MED | `common/validators/` — иерархия + CompositeValidator (backward compat!) |
| Arch-05 | [TASK_PythonArch_05](TASK_PythonArch_05_io_store.md) | 2 | 🟠 MED | `common/io/` — ResultStore, NumpyStore, JsonStore |
| Arch-06 | [TASK_PythonArch_06](TASK_PythonArch_06_plotting.md) | 3 | 🟡 LOW | `common/plotting/` — PlotterFactory, SpectrumPlotter, TimePlotter |

**Начинать с Фазы 1**: Arch-03 ∥ Arch-01 → Arch-02 (03 и 01 параллельно, 02 после 01)

---

## Приоритет 🔴 Высокий

### TASK-process-magnitude-stats — ProcessMagnitude + Statistics pipeline (SVM тесты)

**Документ**: `MemoryBank/tasks/TASK_process_magnitude_statistics_pipeline.md`

**Суть**:
- `ComplexToMagPhaseROCm`: ProcessMagnitude / ProcessMagnitudeToGPU (magnitude-only, norm_coeff через умножение)
- `StatisticsProcessor`: vector<float> обёртки для тестов
- Тесты: малый объём, `hipMallocManaged` (SVM), `InputData<T>` — единообразие
- C++ и Python тесты обязательны

**Проверка**: другой AI как старший (чеклист в документе).

---

### TASK-REF01 — Выделить clFFT в тупиковую ветку + ROCm-only в main
**Дата планирования**: 2026-03-10
**Выполнять на**: AMD машина (Linux, ROCm)
**Оценка**: ~3-4 часа

**Шаги:**
1. `git checkout -b legacy/opencl-clfft && git push` — заморозить clFFT код
2. Вернуться на main, начать с `modules/fft_processor` (паттерн для остальных)
3. Убрать OpenCL вычислительные классы из модулей (по таблице):
   - `modules/fft_processor/` → убрать `FFTProcessor` (clFFT), оставить `FFTProcessorROCm`
   - `modules/fft_maxima/` → убрать `SpectrumProcessorOpenCL`
   - `modules/signal_generators/` → убрать OpenCL kernels + `*_opencl.cpp`
   - `modules/filters/` → убрать OpenCL kernels + OpenCL классы (без ROCm)
   - `modules/lch_farrow/` → убрать OpenCL kernels + `LchFarrow` (OpenCL)
   - `modules/heterodyne/` → убрать `HeterodyneProcessorOpenCL` + OpenCL kernels
4. CMake: убрать `clFFT` линковку из модулей (DrvGPU — НЕ ТРОГАТЬ!)
5. Закомментировать OpenCL-тесты в `all_test.hpp` каждого модуля
6. Проверка: `cmake .. -DENABLE_ROCM=ON && make`

**Что НЕ трогаем:**
- `DrvGPU/backends/opencl/` — обмен данных остаётся
- `BackendType` enum — не трогаем
- `DrvGPU/CMakeLists.txt` → OpenCL линковка остаётся

**Полный план**: `.claude/plans/memoized-wiggling-ritchie.md`



### TASK-fft-ref03 — fft_func: SpectrumProcessorROCm → Ref03 + namespace unification
- **Документ**: [`TASK_fft_func_ref03_remaining.md`](TASK_fft_func_ref03_remaining.md)
- **Добавлено**: 2026-03-22 (по результатам code review)
- **Задача A**: SpectrumProcessorROCm → GpuContext + BufferSet + Ops (~600 строк, 3 этапа)
- **Задача B**: Namespace `antenna_fft` → `fft_processor` (30 файлов, cross-module)
- Задача A не зависит от B. B лучше делать после A.

### TASK-02 — fft_processor: Python Binding
- Нет `py_fft_processor.hpp` и `py_fft_processor_rocm.hpp`
- Нет Python тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-02`

### TASK-03 — fft_maxima: Python Binding
- Есть `Doc/Python/spectrum_maxima_api.md` (план), но нет реального binding
- Нет `py_spectrum_maxima_finder.hpp` / `_rocm.hpp`
- Зависит от TASK-02
- Детали: `MODULES_WORK_PLAN.md#TASK-03`

---

## Приоритет 🟠 Выше среднего

### TASK-05 — fm_correlator: Тесты + API.md
- Все тесты в `all_test.hpp` закомментированы
- Нет `Doc/Modules/fm_correlator/API.md`
- Нет Python тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-05`

---

## Приоритет 🟡 Средний

### TASK-01 — drvgpu: Раскомментировать External Context тесты
- 18 тестов написаны, закомментированы в `all_test.hpp`
- Быстро (30 мин)
- Детали: `MODULES_WORK_PLAN.md#TASK-01`

### TASK-04 — signal_generators: FormSignalROCm Python Binding
- Нет `py_form_signal_rocm.hpp`
- FormSignalROCm используется в strategies — нужен Python доступ для тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-04`

---

## Перспективные задачи

- `strategies`: подробный task-пакет на реализацию ROCm архитектуры
  - Главный файл: `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`
  - Статус: 85% готово, осталось SetExternalWeights + GPU запуск + графики

---

## Готовые модули (не требуют работы)

| Модуль | C++ | Python | Docs |
|--------|-----|--------|------|
| statistics | ✅ | ✅ | ✅ |
| lch_farrow | ✅ | ✅ | ✅ |
| vector_algebra | ✅ | ✅ | ✅ |
| filters | ✅ | ✅ | ✅ |
| heterodyne | ✅ | ✅ | ✅ |

*Последнее обновление: 2026-03-10*
