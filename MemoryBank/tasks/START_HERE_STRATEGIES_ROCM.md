# START HERE — STRATEGIES ROCm IMPLEMENTATION

> Этот файл предназначен для другой AI.
> Если ты другая AI, начинай работу **с этого файла**.
> Не переизобретай архитектуру. Реализуй уже согласованный план.
> Основная цель: довести `modules/strategies/` до рабочей ROCm-реализации по зафиксированной архитектуре.

---

## 1. Что нужно сделать

Нужно реализовать ROCm-only pipeline для `strategies` с такой схемой:

1. входной `d_S` уже находится на GPU
2. debug point `2.1`: stats/save/python по `d_S`
3. GEMM: `d_X = W × S`
4. debug point `2.2`: stats/save/python по `d_X`
5. общий reusable блок `Window + FFT`: `d_X -> d_spectrum`
6. debug point `2.3`: stats/save/python по `|d_spectrum|`
7. на одном `d_spectrum` посчитать все 3 post-FFT сценария:
   - `Step2.1`: `OneMax + 3-point Parabola`, без фазы
   - `Step2.2`: `AllMaxima`, default limit = `1000`
   - `Step2.3`: `GlobalMinMax`, default limit = `1000`

Главный критерий: **минимальное время выполнения**.

Главное архитектурное правило:

- `Window + FFT` считается **ровно один раз**
- `d_spectrum` повторно используется всеми consumers
- нельзя дублировать FFT под каждый post-FFT сценарий

---

## 2. Что нельзя менять

Не меняй следующие договорённости:

- `Window + FFT` — это отдельный общий блок
- post-FFT поиск живёт в `modules/fft_maxima/`
- post-FFT statistics живёт в `modules/statistics/`
- `strategies` только оркестрирует pipeline
- `AntennaProcessorTest` наследуется от `AntennaProcessor_v1`
- матрица `W` — **Delay-and-sum**, не `Identity`
- post-FFT statistics считаются **только по `|spectrum|`**
- production-path должен уметь считать **все 3 сценария**

Если тебе кажется, что архитектуру надо поменять, не меняй её самовольно. Сначала зафиксируй это как вопрос.

---

## 3. Что прочитать перед началом

Прочитай файлы в таком порядке:

1. `MemoryBank/tasks/START_HERE_STRATEGIES_ROCM.md`
2. `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`
3. `Doc_Addition/PLAN/strategies_test_architecture.md`
4. `Doc/Modules/strategies/AP_INDEX.md`
5. `Doc/Modules/strategies/Full.md`
6. `Doc/Modules/strategies/AP_C2_Container.md`
7. `Doc/Modules/strategies/AP_C3_Component.md`
8. `Doc/Modules/strategies/AP_C4_Code.md`
9. `Doc/Modules/strategies/AP_Seq.md`

Если встретишь в старых ASCII-диаграммах legacy-имена:

- `Branch 3` читать как `Step2.1`
- `Branch 4` читать как `Step2.2`
- `Branch 2` читать как `Step2.3`

---

## 4. Границы ответственности по модулям

### `modules/strategies/`

Здесь должно быть:

- orchestration pipeline
- работа со stream/event
- вызов GEMM
- вызов `Window + FFT`
- вызов debug points `2.1 / 2.2 / 2.3`
- вызов всех post-FFT consumers
- сбор `AntennaResult`

Здесь не должно быть:

- новой FFT-логики для post-FFT поиска
- дублирования kernels из `fft_maxima`
- дублирования statistics kernels

### `modules/fft_maxima/`

Здесь реализовать:

- `Step2.1`: `OneMax + Parabola` без фазы
- `Step2.2`: `AllMaxima`
- `Step2.3`: `GlobalMinMax`

Важно:

- входом должен быть **готовый** `d_spectrum`
- `Window + FFT` не должен быть зашит внутрь post-FFT алгоритмов для `strategies`

### `modules/statistics/`

Здесь реализовать:

- stats по `d_S`
- stats по `d_X`
- stats по `|d_spectrum|`

Использовать и расширять существующую инфраструктуру:

- `welford_fused`
- `extract_medians`
- radix sort

---

## 5. Обязательные параметры тестовых данных

Используй `FormSignalGeneratorROCm`.

```cpp
FormParams p;
p.antennas = 5;
p.points = 8000;
p.fs = 12.0e6;
p.f0 = 2.0e6;
p.amplitude = 1.0;
p.noise_amplitude = 0.0;
p.tau_base = 0.0;
p.tau_step = 100e-6;
```

Проверка:

- длительность сигнала: `8000 / 12e6 = 666.7 us`
- задержки: `0, 100, 200, 300, 400 us`
- `tau_step = 100e-6` допустим

---

## 6. Матрица W

Нужен вариант **Delay-and-sum**.

Базовая формула:

```cpp
W[beam][ant] = exp(-j * 2*pi * f0 * tau_ant) / sqrt(N_ant)
```

Нужно реализовать два режима и для C++, и для Python:

### C++

- auto-generate
- external load из `std::vector<std::vector<std::complex<float>>>`

### Python

- auto-generate
- external load из `np.ndarray` или `list[list[complex]]`

---

## 7. Обязательные классы и API

### C++

- `AntennaProcessor`
- `AntennaProcessor_v1`
- `AntennaProcessorTest`
- `PostFftScenarioMode`

### Требование к тестовому классу

`AntennaProcessorTest` должен наследоваться от `AntennaProcessor_v1` и использовать protected-step API, а не копировать pipeline отдельно.

Нужные protected шаги:

- `do_gemm()`
- `do_window_fft()`
- `do_debug_point_21()`
- `do_debug_point_22()`
- `do_debug_point_23()`
- `do_run_post_fft_scenarios()`

### Python

Нужен step-by-step API:

- `step_0_prepare_input()`
- `step_1_debug_input()`
- `step_2_gemm()`
- `step_3_debug_post_gemm()`
- `step_4_window_fft()`
- `step_5_debug_post_fft()`
- `step_6_1_one_max_parabola()`
- `step_6_2_all_maxima()`
- `step_6_3_global_minmax()`
- `process_full()`

После каждого шага, где это нужно:

- делать `hipMemcpyDtoH`
- возвращать данные в Python
- сравнивать с NumPy/SciPy reference

---

## 8. Конфигурация режима post-FFT

Нужен enum:

```cpp
enum class PostFftScenarioMode : uint8_t {
  ALL_REQUIRED = 0,
  ONE_MAX_PARABOLA = 1,
  ALL_MAXIMA = 2,
  GLOBAL_MINMAX = 3
};
```

Правило:

- `ALL_REQUIRED` — основной production-режим
- остальные — для отладки, изолированного теста и бенчмарка

---

## 9. Потоки и синхронизация

Используй такую схему:

- `Stream 1`: debug `2.1` по `d_S`
- `Stream 2`: main (`GEMM -> Window + FFT`)
- `Stream 3`: debug `2.2` по `d_X`
- `Stream 4`: debug `2.3` + post-FFT consumers

События:

- `event_c1_done`
- `event_gemm_done`
- `event_c2_done`
- `event_fft_done`
- `event_c3_done`

---

## 10. Порядок реализации

Работай в таком порядке.

### Phase 1 — каркас `strategies`

- создать/обновить конфиги
- добавить `PostFftScenarioMode`
- добавить каркас `AntennaProcessor_v1`
- выделить общий блок `Window + FFT`
- подготовить `AntennaResult`

### Phase 2 — `fft_maxima`

- добавить `Step2.1` без фазы
- добавить `Step2.2`
- добавить `Step2.3`
- убедиться, что всё работает от внешнего `d_spectrum`

### Phase 3 — `statistics`

- сделать API для `2.1`
- сделать API для `2.2`
- сделать API для `2.3` по `|spectrum|`

### Phase 4 — `W` generation

- C++ auto-generate
- C++ external load
- Python auto-generate
- Python external load

### Phase 5 — `AntennaProcessorTest`

- наследование от `AntennaProcessor_v1`
- пошаговые методы
- CPU-copy на границе шагов

### Phase 6 — tests

- C++ tests
- Python tests
- сравнение с NumPy/SciPy

---

## 11. Критерии готовности

Считать задачу выполненной только если:

- FFT считается один раз
- `Step2.1`, `Step2.2`, `Step2.3` работают от одного `d_spectrum`
- `strategies` не содержит дублирования логики `fft_maxima`
- `statistics` содержит post-FFT stats по `|spectrum|`
- `AntennaProcessorTest` работает пошагово
- Python step-by-step тесты проходят
- C++ тесты проходят

---

## 12. Формат отчёта после каждой фазы

После каждой завершённой фазы пиши короткий отчёт:

```md
## Phase N Report

Сделано:
- ...

Изменённые файлы:
- `path/to/file`

Проверка:
- что собрано
- что запущено
- что ещё не проверено

Риски:
- ...
```

Если что-то не реализовано, пиши прямо:

- `Не сделано`
- `Не проверено`
- `Есть блокер`

---

## 13. Если нужно принять решение

Если найдёшь несколько вариантов реализации, выбирай тот, который:

1. не дублирует FFT
2. минимизирует количество новых сущностей
3. сохраняет границы между `strategies`, `fft_maxima`, `statistics`
4. даёт минимальное время выполнения на ROCm

---

## 14. Главный рабочий файл

После чтения этого файла основной рабочий документ:

- `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`

Он содержит подробный execution checklist.

---

## 15. Старт

Если ты другая AI, начинай так:

1. прочитай файлы из раздела `Что прочитать перед началом`
2. составь короткий implementation plan по Phase 1
3. начни с `strategies` orchestrator и `PostFftScenarioMode`
4. после этого переходи к `fft_maxima`

Не начинай с мелких косметических правок. Сначала собирай рабочий каркас pipeline.
