# 📋 MODULES WORK PLAN — Полный план работ по модулям

> **Порядок**: по `config/tests_order.txt`
> **Обновлено**: 2026-03-10
> **Принцип**: каждый таск самодостаточен — можно прерваться и продолжить

---

## Статус модулей (сводка)

| # | Модуль | C++ | Тесты | Python | Docs | Статус |
|---|--------|-----|-------|--------|------|--------|
| 1 | drvgpu | ✅ | ✅ | — | ✅ | 🟢 Готово (GPU-запуск ждёт железо) |
| 2 | fft_processor | ✅ | ✅ | ✅ FFTProcessorROCm | ✅ | 🟢 Готово |
| 3 | fft_maxima | ✅ | ✅ | ✅ SpectrumMaximaFinderROCm | ✅ | 🟢 Готово |
| 4 | signal_generators | ✅ | ✅ | ✅ FormSignalGeneratorROCm | ✅ | 🟢 Готово |
| 5 | statistics | ✅ | ✅ | ✅ | ✅ | 🟢 Готово |
| 6 | lch_farrow | ✅ | ✅ | ✅ | ✅ | 🟢 Готово |
| 7 | vector_algebra | ✅ | ✅ | ✅ | ✅ | 🟢 Готово |
| 8 | filters | ✅ | ✅ | ✅ | ✅ | 🟢 Готово |
| 9 | heterodyne | ✅ | ✅ | ✅ | ✅ | 🟢 Готово |
| 10 | fm_correlator | ✅ | ✅ 4 теста | ✅ | ✅ API.md | 🟢 Готово (GPU-запуск ждёт железо) |
| 11 | strategies | ✅ | ✅ | ✅ | ✅ | 🟢 Готово (GPU-запуск ждёт железо) |

---

## TASK-01 — DrvGPU: Раскомментировать тесты External Context

**Модуль**: `DrvGPU/`
**Приоритет**: 🟡 Средний
**Зависимости**: Нет
**Предполагаемый размер**: ~30 мин

### Контекст

В `DrvGPU/tests/all_test.hpp` тесты External Context были добавлены но закомментированы.
Задача: раскомментировать и убедиться что проходят.

### Что сделано
- [x] `test_rocm_external_context.hpp` — 6 тестов (COMPLETED 2026-03-09)
- [x] `test_hybrid_external_context.hpp` — 6 тестов
- [x] `test_drv_gpu_external.hpp` — 6 тестов

### Задачи
- [ ] **T01-1**: Открыть `DrvGPU/tests/all_test.hpp`
- [ ] **T01-2**: Раскомментировать include для `test_rocm_external_context.hpp`
- [ ] **T01-3**: Раскомментировать include для `test_hybrid_external_context.hpp`
- [ ] **T01-4**: Раскомментировать include для `test_drv_gpu_external.hpp`
- [ ] **T01-5**: Собрать (`cmake --build`) и запустить `./GPUWorkLib drvgpu`
- [ ] **T01-6**: Убедиться что все 18 тестов PASSED

### Критерий приёмки
```
DrvGPU: All tests PASSED (включая external context)
```

---

## TASK-02 — fft_processor: Python Binding

**Модуль**: `modules/fft_processor/`
**Приоритет**: 🔴 Высокий (нет Python API совсем)
**Зависимости**: Нет
**Предполагаемый размер**: ~2-3 часа

### Контекст

`fft_processor` — базовый модуль FFT, используется в strategies и других местах.
Сейчас нет `py_fft_processor.hpp` и нет регистрации в `gpu_worklib_bindings.cpp`.

### Задачи

#### C++ Binding файл
- [ ] **T02-1**: Прочитать `modules/fft_processor/include/` — найти публичный API
- [ ] **T02-2**: Прочитать `python/py_lch_farrow.hpp` как образец биндинга (OpenCL)
- [ ] **T02-3**: Создать `python/py_fft_processor.hpp`:
  ```cpp
  // register_fft_processor(m) — OpenCL backend
  // Классы: FFTProcessor
  // Методы: process(), set_params(), get_result()
  ```
- [ ] **T02-4**: Прочитать `python/py_lch_farrow_rocm.hpp` как образец ROCm биндинга
- [ ] **T02-5**: Создать `python/py_fft_processor_rocm.hpp`:
  ```cpp
  // register_fft_processor_rocm(m) — ROCm/hipFFT backend
  // Классы: FFTProcessorROCm, ComplexToMagPhaseROCm
  ```
- [ ] **T02-6**: Добавить в `python/gpu_worklib_bindings.cpp`:
  - include для py_fft_processor.hpp (блок OpenCL)
  - include для py_fft_processor_rocm.hpp (блок ROCm)
  - вызовы register_fft_processor(m) и register_fft_processor_rocm(m)
- [ ] **T02-7**: Добавить в `python/CMakeLists.txt` если нужно

#### Python тесты
- [ ] **T02-8**: Создать `Python_test/fft_processor/test_fft_processor.py`:
  - Тест 1: FFT единичного синуса → пик на нужной частоте (сравнение с NumPy FFT)
  - Тест 2: FFT шума → средняя амплитуда (статистика)
  - Тест 3 (ROCm): FFTProcessorROCm vs numpy.fft.fft — max error < 1e-4
  - Тест 4 (ROCm): ComplexToMagPhaseROCm — сравнение abs/angle с numpy

#### Документация
- [ ] **T02-9**: Создать `Doc/Python/fft_processor_api.md`
- [ ] **T02-10**: Добавить ссылку в `Doc/Modules/fft_processor/Full.md`

#### Сборка и проверка
- [ ] **T02-11**: `cmake --build` — проверить что компилируется
- [ ] **T02-12**: `python3 -m pytest Python_test/fft_processor/ -v`

### Критерий приёмки
```python
import gpuworklib
fft = gpuworklib.FFTProcessorROCm(ctx)
result = fft.process(signal)
# max_error vs numpy < 1e-4
```

---

## TASK-03 — fft_maxima: Python Binding

**Модуль**: `modules/fft_maxima/`
**Приоритет**: 🔴 Высокий (документация есть, binding нет)
**Зависимости**: TASK-02 (fft_processor binding нужен для интеграционных тестов)
**Предполагаемый размер**: ~3-4 часа

### Контекст

`fft_maxima` (SpectrumMaximaFinder) — поиск пиков в спектре.
Есть документация `Doc/Python/spectrum_maxima_api.md` но нет реального binding.
Используется в strategies через C++ напрямую.

### Задачи

#### C++ Binding файл
- [ ] **T03-1**: Прочитать `modules/fft_maxima/include/` — публичный API
- [ ] **T03-2**: Прочитать `Doc/Python/spectrum_maxima_api.md` — плановый Python API
- [ ] **T03-3**: Создать `python/py_spectrum_maxima_finder.hpp`:
  ```cpp
  // register_spectrum_maxima_finder(m) — OpenCL backend
  // Классы: SpectrumMaximaFinder
  // Методы: find_all_maxima(), find_one_max()
  ```
- [ ] **T03-4**: Создать `python/py_spectrum_maxima_finder_rocm.hpp`:
  ```cpp
  // register_spectrum_maxima_finder_rocm(m) — ROCm backend
  // Классы: AllMaximaPipelineROCm, OneMaxNoPhaseROCm, GlobalMinMaxROCm
  ```
- [ ] **T03-5**: Добавить регистрации в `python/gpu_worklib_bindings.cpp`
- [ ] **T03-6**: Добавить в `python/CMakeLists.txt` если нужно

#### Python тесты
- [ ] **T03-7**: Создать `Python_test/fft_maxima/test_spectrum_maxima_rocm.py`:
  - Тест 1: `AllMaximaPipelineROCm` — сигнал с двумя частотами → 2 пика найдены
  - Тест 2: `OneMaxNoPhaseROCm` — одиночный синус → правильная частота
  - Тест 3: `GlobalMinMaxROCm` — min/max в спектре
  - Тест 4: интеграция с FFTProcessorROCm (если T02 готов)
  - Тест 5: сравнение с `scipy.signal.find_peaks`
- [ ] **T03-8**: Обновить существующий `Python_test/fft_maxima/test_spectrum_find_all_maxima.py` если нужно

#### Документация
- [ ] **T03-9**: Обновить `Doc/Python/spectrum_maxima_api.md` с реальными сигнатурами
- [ ] **T03-10**: Создать `Doc/Modules/fft_maxima/API.md` если не существует

#### Сборка и проверка
- [ ] **T03-11**: `cmake --build`
- [ ] **T03-12**: `python3 -m pytest Python_test/fft_maxima/ -v`

### Критерий приёмки
```python
import gpuworklib
finder = gpuworklib.AllMaximaPipelineROCm(ctx, fft_size=8192, limit=100)
maxima = finder.find(spectrum_gpu_buffer)
# len(maxima) == 2 для двухтонального сигнала
```

---

## TASK-04 — signal_generators: FormSignalROCm Python Binding

**Модуль**: `modules/signal_generators/`
**Приоритет**: 🟡 Средний
**Зависимости**: Нет
**Предполагаемый размер**: ~2 часа

### Контекст

`FormSignalROCm` — основной генератор сигналов для strategies pipeline.
Используется в C++ тестах strategies, но недоступен из Python.
Уже есть: `register_lfm_analytical_delay` (только LFM с задержкой).
Нет: `FormSignalROCm`, `FormSignalGenerator`.

### Задачи

#### C++ Binding файл
- [ ] **T04-1**: Прочитать `modules/signal_generators/include/form_signal_rocm.hpp` — API
- [ ] **T04-2**: Прочитать `python/py_lfm_analytical_delay.hpp` как образец
- [ ] **T04-3**: Создать `python/py_form_signal_rocm.hpp`:
  ```cpp
  // register_form_signal_rocm(m)
  // Классы: FormSignalGeneratorROCm
  // Параметры: FormParams (antennas, points, fs, f0, amplitude, noise_amplitude, tau_base, tau_step)
  // Методы: generate() → numpy array [antennas × points], get_gpu_buffer()
  ```
- [ ] **T04-4**: Добавить `register_form_signal_rocm(m)` в `gpu_worklib_bindings.cpp`

#### Python тесты
- [ ] **T04-5**: Создать `Python_test/signal_generators/test_form_signal_rocm.py`:
  - Тест 1: генерация CW сигнала → частота f0 в спектре (scipy.fft)
  - Тест 2: задержки tau_step → проверить задержку между антеннами (кросс-корреляция)
  - Тест 3: шум noise_amplitude=0 vs >0 — проверить SNR
  - Тест 4: параметры из `STRATEGIES_ROCM_EXECUTION.md` (5 антенн, 8000 точек)

#### Документация
- [ ] **T04-6**: Добавить FormSignalROCm в `Doc/Python/signal_generators_api.md` (или создать)

#### Сборка и проверка
- [ ] **T04-7**: `cmake --build`
- [ ] **T04-8**: `python3 -m pytest Python_test/signal_generators/test_form_signal_rocm.py -v`

### Критерий приёмки
```python
import gpuworklib
gen = gpuworklib.FormSignalGeneratorROCm(ctx)
gen.set_params(antennas=5, points=8000, fs=12e6, f0=2e6, tau_step=100e-6)
signal = gen.generate()  # numpy [5 × 8000], complex64
```

---

## TASK-05 — fm_correlator: Раскомментировать тесты + API.md

**Модуль**: `modules/fm_correlator/`
**Приоритет**: 🟠 Выше среднего
**Зависимости**: Нет
**Предполагаемый размер**: ~1.5 часа

### Контекст

`fm_correlator` — FM коррелятор (ROCm only).
Python binding `register_fm_correlator_rocm` зарегистрирован.
Но все тесты в `all_test.hpp` закомментированы + нет `API.md`.
Нет Python тестов.

### Задачи

#### C++ тесты
- [ ] **T05-1**: Открыть `modules/fm_correlator/tests/all_test.hpp`
- [ ] **T05-2**: Раскомментировать базовые тесты:
  ```cpp
  fm_correlator::tests::run_test_msequence();
  fm_correlator::tests::run_test_basic_pipeline();
  ```
  (НЕ раскомментировать benchmark — они долгие)
- [ ] **T05-3**: Запустить `./GPUWorkLib fm_correlator` — проверить PASSED
- [ ] **T05-4**: Если тест падает — разобраться и починить

#### Python тесты
- [ ] **T05-5**: Прочитать `python/py_fm_correlator_rocm.hpp` — Python API
- [ ] **T05-6**: Создать `Python_test/fm_correlator/test_fm_correlator.py`:
  - Тест 1: базовая корреляция M-sequence с собой → пик в 0
  - Тест 2: корреляция со сдвигом → пик на нужном лаге
  - Тест 3: сравнение с `numpy.correlate`

#### Документация
- [ ] **T05-7**: Прочитать `Doc/Modules/fm_correlator/Full.md` — существующая документация
- [ ] **T05-8**: Создать `Doc/Modules/fm_correlator/API.md`:
  - Все публичные классы/методы с сигнатурами
  - Пример C++ (из тестов)
  - Пример Python (из T05-5)
- [ ] **T05-9**: Добавить ссылку на API.md в Full.md

### Критерий приёмки
```
./GPUWorkLib fm_correlator → test_msequence PASSED, test_basic PASSED
python3 -m pytest Python_test/fm_correlator/ -v → 3 тесты PASSED
Doc/Modules/fm_correlator/API.md создан
```

---

## TASK-06 — strategies: Завершение (SetExternalWeights + GPU запуск + Графики)

**Модуль**: `modules/strategies/`
**Приоритет**: 🔴 Высокий (финальный этап большой задачи)
**Зависимости**: GPU AMD 9070 должен быть доступен
**Предполагаемый размер**: ~4-6 часов
**Спецификация**: `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`

### Контекст

Основная реализация готова (Phases A-F почти завершены).
Остались: внешняя матрица весов, графики, и реальный запуск на GPU.

### Задача 6.1 — SetExternalWeights C++ + Python

- [ ] **T06-1**: Прочитать `modules/strategies/include/antenna_processor.hpp` — текущий API весов
- [ ] **T06-2**: Реализовать `SetExternalWeights(std::vector<std::vector<std::complex<float>>>)` в C++:
  - Принять матрицу W [beams × antennas] с CPU
  - Загрузить в GPU буфер `d_W_`
  - Установить флаг `use_external_weights_ = true`
- [ ] **T06-3**: Добавить тест в C++ (`test_external_weights.hpp`):
  - Создать W вручную на CPU → SetExternalWeights → Process → проверить результат
- [ ] **T06-4**: Добавить в Python binding `py_strategies_rocm.hpp`:
  ```python
  def set_external_weights(self, W: np.ndarray):
      # W — numpy array [beams × antennas], complex64
  ```
- [ ] **T06-5**: Тест Python: `set_external_weights(W_numpy)` → process → сравнить с NumPy эталоном

### Задача 6.2 — Запуск на GPU AMD 9070

- [ ] **T06-6**: Собрать с `cmake .. -DENABLE_ROCM=ON`
- [ ] **T06-7**: Запустить `./GPUWorkLib strategies` — проверить PASSED
- [ ] **T06-8**: Запустить Python шаг за шагом:
  ```bash
  python3 -m pytest Python_test/strategies/test_strategies_step_by_step.py -v
  ```
- [ ] **T06-9**: Запустить ScenarioBuilder тесты:
  ```bash
  python3 -m pytest Python_test/strategies/test_scenario_builder.py -v
  ```
- [ ] **T06-10**: Запустить FarrowPipeline тесты:
  ```bash
  python3 -m pytest Python_test/strategies/test_farrow_pipeline.py -v
  ```
- [ ] **T06-11**: Зафиксировать результаты — что прошло, что упало

### Задача 6.3 — Визуализация

- [ ] **T06-12**: Создать `Python_test/strategies/plot_strategies_results.py`:
  - График 1: спектр Pipeline A vs Pipeline B (5 антенн, 8000 точек)
  - График 2: debug points 2.1 / 2.2 / 2.3 — сигнал на каждом этапе
  - График 3: найденные пики Step2.1 / Step2.2 / Step2.3
  - Сохранить в `Results/Plots/strategies/`
- [ ] **T06-13**: Запустить и проверить графики

### Критерий приёмки
```
./GPUWorkLib strategies → ALL PASSED
pytest test_strategies_step_by_step.py → 10 тестов PASSED
set_external_weights работает в C++ и Python
Графики сохранены в Results/Plots/strategies/
```

---

## TASK-07 — ScenarioBuilder + FarrowPipeline: Запуск тестов (IN_PROGRESS)

**Модуль**: `Python_test/strategies/`
**Приоритет**: 🟡 Средний
**Зависимости**: Python + numpy должны быть установлены
**Предполагаемый размер**: ~1-2 часа

### Контекст

Код написан, тесты написаны, но не запускались (ждут машины с Python/numpy).

### Задачи
- [ ] **T07-1**: Запустить ScenarioBuilder тесты:
  ```bash
  python3 -m pytest Python_test/strategies/test_scenario_builder.py -v
  ```
- [ ] **T07-2**: Зафиксировать результат — что прошло
- [ ] **T07-3**: Запустить FarrowPipeline тесты:
  ```bash
  python3 -m pytest Python_test/strategies/test_farrow_pipeline.py -v
  ```
- [ ] **T07-4**: Если тест падает — починить
- [ ] **T07-5**: Создать визуализацию `plot_farrow_vs_phase.py`:
  - Спектры Pipeline A vs B на одном графике
  - Сохранить `Results/Plots/strategies/farrow_vs_phase.png`
- [ ] **T07-6**: Перенести COMPLETED (если всё прошло)

---

## Порядок выполнения (рекомендуемый)

```
TASK-07  →  ScenarioBuilder/Farrow тесты (быстро, numpy-only)
TASK-01  →  DrvGPU external context тесты (быстро)
TASK-05  →  fm_correlator тесты + API.md (средне)
TASK-04  →  signal_generators FormSignalROCm binding (средне)
TASK-02  →  fft_processor Python binding (сложно)
TASK-03  →  fft_maxima Python binding (сложно, зависит от T02)
TASK-06  →  strategies финализация (зависит от GPU)
```

---

## Готовые модули (не требуют работы)

| Модуль | Статус |
|--------|--------|
| statistics | 🟢 C++ ✅ Python ✅ Docs ✅ |
| lch_farrow | 🟢 C++ ✅ Python ✅ Docs ✅ |
| vector_algebra | 🟢 C++ ✅ Python ✅ Docs ✅ |
| filters | 🟢 C++ ✅ Python ✅ Docs ✅ |
| heterodyne | 🟢 C++ ✅ Python ✅ Docs ✅ |

---

*Создано: 2026-03-10*
*На основе: review всех модулей в порядке tests_order.txt*
