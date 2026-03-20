# Разногласия: план vs реальность — Python_test/strategies/

**Дата ревью**: 2026-03-20
**Источники**:
- `MemoryBank/specs/python_test_refactoring.md` — план рефакторинга
- `MemoryBank/tasks/TASK_Python_06_test_strategies.md` — задача на pipeline тест
- `MemoryBank/tasks/TASK_Python_07_remove_pytest_strategies.md` — задача на удаление pytest
- `Python_test/strategies/` — реальный код

---

## 🔴 Критические разногласия

### Р1. Два конфликтующих `SignalVariant` enum

**Где**: `test_params.py:31` и `signal_factory.py:33`

**Проблема**: В папке существуют ДВА класса с одинаковым именем `SignalVariant` в разных файлах:

| Файл | Варианты |
|------|----------|
| `test_params.py` | `SIN`, `LFM_NO_DELAY`, `LFM_WITH_DELAY`, `LFM_FARROW` |
| `signal_factory.py` | `V1_CW_CLEAN`, `V2_CW_NOISE`, `V3_CW_PHASE_DELAY`, `V4_CW_PHASE_NOISE`, `V5_FROM_FILE` |

**Используются разными тестами**:
- `test_base_pipeline.py`, `test_debug_steps.py`, `signal_generators_strategy.py` → `from test_params import SignalVariant`
- `test_strategies_pipeline.py`, `signal_factory.py` → `from signal_factory import SignalVariant`

**Риск**: При случайном `from strategies import *` или неправильном импорте — тихая замена enum.
Два разных enum с одним именем вводят разработчика в заблуждение.

**Предложение**: Переименовать один из enum:
- `test_params.py` → `LfmSignalVariant` (он про LFM-режимы)
- `signal_factory.py` → `PipelineSignalVariant` (он про варианты pipeline)
- ИЛИ: объединить оба в один `signal_factory.py` с расширенным набором вариантов

---

### Р2. Два независимых интерфейса Strategy для генерации сигналов

**Где**: `signal_generators_strategy.py:28` и `signal_factory.py:76`

**Проблема**: Существуют ДВА разных интерфейса с разными сигнатурами:

```python
# signal_generators_strategy.py — СТАРЫЙ
class ISignalStrategy(ABC):
    def generate(self, params: AntennaTestParams) -> np.ndarray:
        ...  # возвращает только S [n_ant, n_samples]

# signal_factory.py — НОВЫЙ (из рефакторинга)
class ISignalSource(ABC):
    def generate(self, cfg: SignalConfig) -> SignalData:
        ...  # возвращает S + W + эталоны
```

**Дублирование**: `LfmFarrowStrategy` в `signal_generators_strategy.py` и `GpuCwDelayedSignalSource` в `signal_factory.py` делают похожие вещи разными способами.

**Предложение**: Явно разграничить области применения в docstring каждого файла, или мигрировать старые тесты на новый интерфейс.

---

### Р3. `GpuCw*` классы не используют GPU — вводящее в заблуждение название

**Где**: `signal_factory.py:100-215`

**Проблема**: Классы `GpuCwCleanSignalSource`, `GpuCwNoiseSignalSource` и др. содержат **чистый NumPy** внутри:

```python
class GpuCwCleanSignalSource(ISignalSource):
    def generate(self, cfg: SignalConfig) -> SignalData:
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs   # NumPy!
        cw = np.exp(1j * 2 * np.pi * cfg.f0 * t)                   # NumPy!
```

**Что планировалось** (из `python_test_refactoring.md`):
```
V1: gw.CwGenerator(fs, f0, n_samples, n_ant, noise=False)   ← GPU gpuworklib!
V2: gw.CwGenerator(fs, f0, n_samples, n_ant, snr_db=20)     ← GPU gpuworklib!
```

**Реальность**: GPU-генерация не реализована. Класс называется `GpuCw*` но работает на CPU.

**Предложение**: Переименовать в `CwCleanSignalSource`, `CwNoiseSignalSource` и т.д. (убрать `Gpu` из имени).

---

## 🟡 Важные расхождения

### Р4. Статусы задач TASK_06 и TASK_07 устарели — помечены TODO, фактически DONE

**Где**: `TASK_Python_06_test_strategies.md:2`, `TASK_Python_07_remove_pytest_strategies.md:2`

**Проблема**: Оба файла имеют статус `🔲 TODO`, хотя:
- `test_strategies_pipeline.py` создан и содержит все 5 тестов V1–V5 ✅
- pytest удалён из всех файлов strategies/ ✅ (выполнено в сессии 2026-03-20)

**Предложение**: Обновить статус на `✅ DONE` в обоих файлах.

---

### Р5. CHECK-6.3c из спеки отсутствует в коде

**Где**: `python_test_refactoring.md:278` vs `pipeline_step_validator.py:281-307`

**Проблема**: Спека указывает три проверки для STEP 6.3:
```
CHECK-6.3a: min < max per beam        ← РЕАЛИЗОВАНО ✅
CHECK-6.3b: dyn_range > 0 dB          ← РЕАЛИЗОВАНО ✅
CHECK-6.3c: (V3/V4) dyn_range NumPy≈GPU  ← ОТСУТСТВУЕТ ❌
```

`pipeline_step_validator.py` реализует только `run_step_6_3()` с CHECK-6.3a и 6.3b.
CHECK-6.3c (сравнение GPU dynamic_range с NumPy эталоном) не реализован.

**Предложение**: Добавить CHECK-6.3c или удалить его из спеки как избыточный
(NumpyReference уже имеет `compute_dynamic_range_db()` — данные есть).

---

### Р6. Два параллельных NumPy pipeline — не разграничены в документации

**Где**: `pipeline_runner.py` и `pipeline_step_validator.py`

**Проблема**: В папке существуют два разных pipeline-движка:

| | `pipeline_runner.py` | `pipeline_step_validator.py` |
|-|---------------------|------------------------------|
| Назначение | Чистый NumPy beamforming (FarrowDelay/ScenarioBuilder) | Валидация GPU proc.step_N() |
| Входные данные | ScenarioBuilder.build() → dict | AntennaProcessorTest pybind11 |
| Тесты | `test_farrow_pipeline.py`, `test_scenario_builder.py` | `test_strategies_pipeline.py` |
| Создан | 2026-03-08 (до рефакторинга) | После рефакторинга |

Спека `python_test_refactoring.md` описывает только второй. Первый существовал до рефакторинга и не упоминается в TASK_06/07. Это не ошибка, но создаёт путаницу для нового разработчика.

**Предложение**: Добавить в README (если будет создан) или в docstring двух файлов явное разграничение: "это NumPy-only тест" vs "это GPU-валидация".

---

### Р7. TASK_Python_06 описывает устаревший API `AntennaProcessorConfig`

**Где**: `TASK_Python_06_test_strategies.md:131-137`

**Проблема**: Задача описывает предполагаемый API:
```python
ap_cfg = gw.AntennaProcessorConfig()   # <-- проверить реальное имя!
ap_cfg.n_ant = self._cfg.n_ant
proc = gw.AntennaProcessorTest(ctx, ap_cfg)
```

**Реальный API** в `test_strategies_pipeline.py`:
```python
self._proc = gw.AntennaProcessorTest(
    ctx,
    n_ant               = self._cfg.n_ant,
    n_samples           = self._cfg.n_samples,
    sample_rate         = float(self._cfg.fs),
    signal_frequency_hz = float(self._cfg.f0),
    debug_mode          = True,
)
```

`AntennaProcessorConfig` не существует — конструктор принимает именованные параметры напрямую.
Задача описывала "надо выяснить" — выяснили и реализовали, но задачу не обновили.

---

### Р8. `test_params.py` назван как тест, но тестов не содержит

**Где**: `test_params.py:1`

**Проблема**: Файл называется `test_params.py`, но содержит только `dataclass AntennaTestParams` и `enum SignalVariant`. Это конфигурационный файл, не тестовый. TestRunner при сканировании папки может попытаться его "запустить" как тест.

**Предложение**: Переименовать в `antenna_test_params.py` (убрать префикс `test_`).

---

## 🟢 Рекомендации

### Р9. `__pycache__` содержит pytest-артефакты

**Где**: `strategies/__pycache__/`

Файлы вида `conftest.cpython-312-pytest-8.4.2.pyc` и `conftest.cpython-313-pytest-9.0.2.pyc` указывают что pytest всё ещё иногда используется для запуска тестов (или использовался в прошлом). Это нормально, но:

**Предложение**: Добавить `__pycache__/` в `.gitignore` (если ещё не добавлено).

---

### Р10. `conftest.py` содержит `make_pipeline_runner()` без связи с TestRunner

**Где**: `conftest.py:74-77`

```python
def make_pipeline_runner():
    """PipelineRunner без checkpoint'ов (нет вывода на диск)."""
    from pipeline_runner import PipelineRunner
    return PipelineRunner(output_dir=None)
```

`PipelineRunner` — это "старый" NumPy pipeline (Р6), не связанный с `TestRunner` из `common/runner.py`. Функция не используется в новых тестах, но присутствует в фабрике.

**Предложение**: Добавить docstring-комментарий что это для `test_farrow_pipeline.py` и `test_scenario_builder.py`.

---

### Р11. Спека описывает stream-параллельность STEP1 и STEP2

**Где**: `python_test_refactoring.md:171-199` (диаграмма со Stream debug1 / Stream main)

Диаграмма показывает STEP1 и STEP2 запускаемыми параллельно в разных GPU-стримах. В `pipeline_step_validator.py` они вызываются **последовательно** в Python:
```python
steps.extend([
    self.run_step_1(),
    self.run_step_2(),
    ...
])
```

Это не ошибка Python-кода — параллельность находится на уровне GPU (внутри C++/HIP). Но диаграмма в спеке вводит в заблуждение читателя Python-кода.

---

## 📋 Сводная таблица

| ID | Тип | Описание | Файл | Приоритет |
|----|-----|----------|------|-----------|
| Р1 | 🔴 | Два конфликтующих `SignalVariant` enum | `test_params.py`, `signal_factory.py` | Высокий |
| Р2 | 🔴 | Два ISignalStrategy с разными сигнатурами | `signal_generators_strategy.py`, `signal_factory.py` | Высокий |
| Р3 | 🔴 | `GpuCw*` не использует GPU — вводящее имя | `signal_factory.py:100-215` | Средний |
| Р4 | 🟡 | TASK_06/07 статус TODO вместо DONE | `TASK_Python_06.md`, `TASK_Python_07.md` | Низкий |
| Р5 | 🟡 | CHECK-6.3c отсутствует в коде | `pipeline_step_validator.py:281` | Средний |
| Р6 | 🟡 | Два NumPy pipeline без разграничения | `pipeline_runner.py`, `pipeline_step_validator.py` | Средний |
| Р7 | 🟡 | TASK_06 описывает несуществующий API | `TASK_Python_06.md:131` | Низкий |
| Р8 | 🟡 | `test_params.py` содержит не тесты | `test_params.py:1` | Средний |
| Р9 | 🟢 | pytest-артефакты в `__pycache__` | `strategies/__pycache__/` | Низкий |
| Р10 | 🟢 | `make_pipeline_runner()` без связи с TestRunner | `conftest.py:74` | Низкий |
| Р11 | 🟢 | Диаграмма stream-параллельности вводит в заблуждение | `python_test_refactoring.md:171` | Низкий |

---

## ✅ Что соответствует плану

- `NumpyReference` — реализован точно по спеке ✅
- `PipelineStepValidator` — все CHECK-0/1/2/3/4/5/6.1/6.2/6.3a/6.3b реализованы ✅
- `SignalSourceFactory.create(variant)` — Factory Method по GoF ✅
- `TestStrategiesPipeline` — 5 методов V1–V5, Template Method ✅
- `SignalData` dataclass — содержит d_S, d_W, S_ref, W_ref ✅
- `FileSignalSource` — бросает SkipTest как заглушка ✅
- `GPUContextManager` используется правильно ✅
- pytest полностью убран из кода ✅
- Структура ООП/SOLID/GRASP/GoF соответствует плану ✅


## Ответ
**Предложение**: Переименовать один из enum:
- `test_params.py` → `LfmSignalVariant` (он про LFM-режимы)
- `signal_factory.py` → `PipelineSignalVariant` (он про варианты pipeline)
- ИЛИ: объединить оба в один `signal_factory.py` с расширенным набором вариантов

считаю это лучший вариант
 объединить оба в один `signal_factory.py` с расширенным набором вариантов

### Р2. Два независимых интерфейса Strategy для генерации сигналов
**Предложение**: Мигрировать старые тесты на новый интерфейс. - Лучший вариант

### Р3. `GpuCw*` классы не используют GPU — вводящее в заблуждение название
Согласен

### Р5. CHECK-6.3c из спеки отсутствует в коде
удалить только в том случае если дублируется. Если нет давай разбираться. что он должен проверять

совсем остальным согласен
