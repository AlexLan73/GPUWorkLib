# STRATEGIES ROCm EXECUTION TASK

> Дата: 2026-03-07
> Модуль: `modules/strategies/`
> Платформа: ROCm only
> GPU: AMD Radeon 9070 / MI100
> ROCm: 7.2 / 7.5
> Цель: реализовать `strategies` по финальному архитектурному плану
> Точка входа для другой AI: `MemoryBank/tasks/START_HERE_STRATEGIES_ROCM.md`

---

## 1. Итоговая архитектура

### Базовый pipeline

1. `d_S` уже лежит на GPU, в модуль передаются GPU pointer + metadata
2. Debug point `2.1`: статистика / сохранение / Python-copy по `d_S`
3. GEMM: `d_X = W × S`
4. Debug point `2.2`: статистика / сохранение / Python-copy по `d_X`
5. Общий блок `Window + FFT`: `d_X -> d_spectrum`
6. Debug point `2.3`: статистика / сохранение / Python-copy по `|d_spectrum|`
7. На одном `d_spectrum` запускаются все 3 post-FFT сценария:
   - `Step2.1` One MAX + 3-point parabola, без фазы
   - `Step2.2` All maxima, limit по умолчанию `1000`
   - `Step2.3` Global MAX/MIN, limit по умолчанию `1000`

### Ключевой принцип производительности

- `Window + FFT` выполняется ровно **один раз**
- `d_spectrum` используется повторно всеми post-FFT consumers
- нельзя дублировать FFT для `Step2.1`, `Step2.2`, `Step2.3`

---

## 2. Разделение по модулям

### `modules/strategies/`

Оставить только orchestration:

- приём `d_S` и metadata
- вызов GEMM
- запуск debug points `2.1 / 2.2 / 2.3`
- запуск общего блока `Window + FFT`
- запуск post-FFT сценариев
- сбор `AntennaResult`

### `modules/fft_maxima/`

Добавить ROCm-only post-FFT обработчики:

- `Step2.1`: `OneMax + 3-point Parabola` без фазы
- `Step2.2`: `AllMaxima` с limit=`1000`
- `Step2.3`: `GlobalMinMax` с limit=`1000`

Требования:

- не смешивать `Window + FFT` с post-FFT поиском
- post-FFT обработчики работают от уже готового `d_spectrum`
- для `Step2.1` сделать новую совместимую структуру результата без фазы
- обеспечить совместимость по API и структурам результата

### `modules/statistics/`

Добавить:

- статистику по `d_S`
- статистику по `d_X`
- post-FFT статистику по `|d_spectrum|`
- загружаемые / переиспользуемые ROCm kernels для `mean`, `median`, `std`, `var`, `min`, `max`

---

## 3. Конфигурация и enum

Нужно ввести enum для post-FFT сценариев:

```cpp
enum class PostFftScenarioMode : uint8_t {
  ALL_REQUIRED = 0,
  ONE_MAX_PARABOLA = 1,
  ALL_MAXIMA = 2,
  GLOBAL_MINMAX = 3
};
```

Поведение:

- `ALL_REQUIRED` — production path по ТЗ: считать все 3 сценария
- остальные режимы — для selective debug / isolated benchmark / step-by-step tests

---

## 4. Тестовые данные

Использовать `FormSignalGeneratorROCm`.

### Параметры

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

- длительность сигнала `T = 8000 / 12e6 = 666.7 us`
- задержки `0, 100, 200, 300, 400 us` допустимы

---

## 5. Матрица W

Нельзя использовать `Identity` как основной вариант.

Нужно реализовать **Delay-and-sum**:

```cpp
W[beam][ant] = exp(-j * 2*pi * f0 * tau_ant) / sqrt(N_ant)
```

### Обязательные API

#### C++

- auto-generate:
  - `GenerateDelayAndSumWeights(signal_params, array_params)`
- external load:
  - `SetExternalWeights(std::vector<std::vector<std::complex<float>>>)`

#### Python

- auto-generate:
  - `generate_delay_and_sum_weights(...)`
- external load:
  - `set_external_weights(np.ndarray | list[list[complex]])`

---

## 6. C++ классы

### Основные

- `AntennaProcessor`
- `AntennaProcessor_v1`
- `AntennaProcessorTest`

### Архитектурное требование

`AntennaProcessorTest` наследуется от `AntennaProcessor_v1` и использует те же protected шаги:

- `do_gemm()`
- `do_window_fft()`
- `do_debug_point_21()`
- `do_debug_point_22()`
- `do_debug_point_23()`
- `do_run_post_fft_scenarios()`

Не делать отдельную независимую реализацию pipeline для тестов.

---

## 7. Python API

Нужен пошаговый API для Python.

### Класс

- `AntennaProcessorTest`

### Методы

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

### Принцип

После каждого шага:

- по необходимости `hipMemcpyDtoH`
- вернуть данные в Python
- сравнить с NumPy/SciPy reference

---

## 8. Потоки и синхронизация

### Stream layout

- `Stream 1`: debug `2.1` по `d_S`
- `Stream 2`: main (`GEMM -> Window + FFT`)
- `Stream 3`: debug `2.2` по `d_X`
- `Stream 4`: debug `2.3` + post-FFT consumers

### Events

- `event_c1_done`
- `event_gemm_done`
- `event_c2_done`
- `event_fft_done`
- `event_c3_done`

### Важно

- `Step2.1`, `Step2.2`, `Step2.3` запускаются от одного `d_spectrum`
- post-FFT statistics по `|spectrum|` запускаются в том же post-FFT слое

---

## 9. Точки debug / checkpoint

### `2.1`

- stats по `d_S`
- save `d_S`
- python copy

### `2.2`

- stats по `d_X`
- save `d_X`
- python copy

### `2.3`

- stats по `|d_spectrum|`
- save `d_spectrum`
- python copy

---

## 10. Реализационные задачи

### Phase A — strategies orchestrator

- [x] Описать и реализовать `AntennaProcessorConfig`
- [x] Добавить `PostFftScenarioMode`
- [x] Реализовать `AntennaProcessor_v1`
- [x] Реализовать debug points `2.1 / 2.2 / 2.3`
- [x] Реализовать `Window + FFT` как единый reusable блок
- [x] Реализовать вызов всех 3 post-FFT сценариев

### Phase B — fft_maxima extension

- [x] Добавить ROCm-only `OneMax + Parabola` без фазы (hiprtc kernel one_max_no_phase)
- [x] Добавить ROCm-only `AllMaxima` с limit (AllMaximaPipelineROCm integration)
- [x] Добавить ROCm-only `GlobalMinMax` (hiprtc kernel global_minmax)
- [x] Переиспользовать существующие kernels и структуры, где это возможно
- [x] Не дублировать FFT внутри `fft_maxima` для strategies pipeline

### Phase C — statistics extension

- [x] Добавить API для статистики по `d_S` (ComputeStatistics on complex GPU data)
- [x] Добавить API для статистики по `d_X` (ComputeStatistics on complex GPU data)
- [x] Добавить API для статистики по `|d_spectrum|` (ComputeStatisticsFloat on float GPU data)
- [x] Переиспользовать `welford_fused`, `extract_medians`, radix sort
- [x] Добавить `welford_float` kernel + `ComputeStatisticsFloat` / `ComputeMedianFloat`

### Phase D — weight generation

- [x] Реализовать C++ auto-generate Delay-and-sum weights (WeightGenerator::generate_delay_and_sum)
- [ ] Реализовать C++ external weights input (SetExternalWeights)
- [x] Реализовать Python auto-generate Delay-and-sum weights (generate_delay_and_sum_weights)
- [ ] Реализовать Python external weights input (set_external_weights)

### Phase E — Python tests

- [x] Создать `Python_test/strategies/test_strategies_step_by_step.py`
- [x] Сравнить `2.1 / 2.2 / 2.3` с NumPy/SciPy
- [x] Проверить `Step2.1` (test_gpu_one_max_frequency)
- [x] Проверить `Step2.2` (test_gpu_all_maxima)
- [x] Проверить `Step2.3` (test_gpu_global_minmax)
- [ ] Построить графики для `5 x 8000`

### Phase F — C++ tests

- [x] Создать header-only тесты в `modules/strategies/tests/`
- [x] Добавить `all_test.hpp`
- [x] Проверить Delay-and-sum matrix generation
- [x] Проверить `Window + FFT`
- [x] Проверить все 3 post-FFT сценария
- [x] Проверить debug copy/save/stat flows

---

## 11. Критерии приёмки

- [x] FFT считается один раз на кадр
- [x] Все 3 post-FFT сценария запускаются от одного `d_spectrum`
- [x] Debug точки `2.1 / 2.2 / 2.3` работают независимо
- [x] Python step-by-step тесты проходят (10 тестов: 5 NumPy + 5 GPU)
- [x] C++ тесты проходят (test_strategies_pipeline.hpp)
- [ ] ROCm only path работает на AMD 9070 / MI100 (нужен запуск на GPU)
- [x] Нет дублирования логики FFT в `strategies`
- [x] Новые post-FFT алгоритмы лежат в `fft_maxima` (AllMaximaPipelineROCm)
- [x] Новая статистика лежит в `statistics` (ComputeStatisticsFloat/ComputeMedianFloat)

---

## 12. Документы-источники

- `Doc_Addition/PLAN/strategies_test_architecture.md`
- `Doc/Modules/strategies/AP_INDEX.md`
- `Doc/Modules/strategies/Full.md`
- `Doc/Modules/strategies/AP_C2_Container.md`
- `Doc/Modules/strategies/AP_C3_Component.md`
- `Doc/Modules/strategies/AP_C4_Code.md`
- `Doc/Modules/strategies/AP_Seq.md`
