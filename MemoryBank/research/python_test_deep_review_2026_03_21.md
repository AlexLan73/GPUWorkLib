# Code Review: Python_test/ — Глубокое ревью инфраструктуры и тестов

**Дата**: 2026-03-21
**Reviewer**: Кодо (AI Assistant)
**Scope**: 106 файлов, ~25K строк, 16 модулей + common/
**Источники**: Context7 (pybind11, numpy), sequential-thinking, codebase analysis

---

## Критические проблемы 🔴

### 1. ImportError: NumericValidator / SpectralValidator удалены, но импортируются

**Суть**: В `validators.py` (строка 7) написано:
> "Убраны: NumericValidator, RMSEValidator, SpectralValidator, EnergyValidator"

Но **3 файла** всё ещё импортируют удалённые классы:

| Файл | Строка | Импорт | Статус |
|------|--------|--------|--------|
| `filters/filter_test_base.py` | 42 | `from common.validators import NumericValidator` | ImportError |
| `signal_generators/signal_test_base.py` | 43 | `from common.validators import NumericValidator` | ImportError |
| `heterodyne/heterodyne_test_base.py` | 28 | `from common.validators import NumericValidator, SpectralValidator` | ImportError |

**Последствия**: Все тесты, использующие эти базовые классы, **крашатся при импорте**. Это затрагивает:
- Все фильтры через `FilterTestBase`
- Все генераторы через `SignalTestBase`
- Все тесты гетеродина через `HeterodyneTestBase`

Также `test_base.py:25` ссылается на `NumericValidator` в docstring.

**Исправление**:
```python
# БЫЛО:
from common.validators import NumericValidator
validator = NumericValidator(tolerance=0.01)

# СТАЛО:
from common.validators import DataValidator
validator = DataValidator(tolerance=0.01, metric="max_rel")
```

Для `SpectralValidator` — аналогично заменить на `DataValidator(metric="max_rel")`.

**Приоритет**: БЛОКЕР — базовые классы нерабочие.

---

### 2. Обход GPULoader — хардкод путей к билду

**4 файла** обходят `GPULoader` и хардкодят путь к `gpuworklib`:

| Файл | Строка | Хардкод |
|------|--------|---------|
| `filters/test_fir_filter_rocm.py` | 24 | `build/debian-radeon9070/python` |
| `filters/test_iir_filter_rocm.py` | 27 | `build/debian-radeon9070/python` |
| `lch_farrow/test_lch_farrow_rocm.py` | 34 | `build/debian-radeon9070/python` |
| `fm_correlator/test_fm_correlator.py` | 169 | `build/python` |

**Проблемы**:
- Не работает на Windows (nvidia ветка, другой путь билда)
- Не работает с MSVC Release/Debug сборками
- Игнорирует `GPUWORKLIB_BUILD_DIR` env var
- Дублирует логику, которую GPULoader уже решает

**Исправление**:
```python
# БЫЛО:
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'debian-radeon9070', 'python'))
import gpuworklib
HAS_GPU = True

# СТАЛО:
_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.gpu_loader import GPULoader
gw = GPULoader.get()
HAS_GPU = gw is not None
```

**Приоритет**: КРИТИЧНО — ломает кросс-платформенность.

---

## Важные замечания 🟡

### 3. Дублирование PipelineRunner (старый + новый API)

**Файл**: `strategies/pipeline_runner.py` (843 строки)

В одном файле сосуществуют:
- **Старый API**: `PipelineRunner.run_pipeline_a()` / `run_pipeline_b()` (строки 338-598) — содержат полную логику
- **Новый API**: `PipelineBase` / `PipelineA` / `PipelineB` (строки 639-843) — абстракция через Template Method

Оба API делают одно и то же. ~200 строк дублированного кода.

**Рекомендация**: Завершить миграцию — `PipelineRunner` должен делегировать в `PipelineA/PipelineB`, а не содержать собственную реализацию. Или удалить `PipelineBase/A/B` если они не используются.

---

### 4. Core/ DSL — мёртвый код (908 строк, 14 файлов)

**Каталог**: `Python_test/Core/` (generators/, processing/)

Core/ содержит DSL-обёртки (ISignalGenerator, IProcessor, GeneratorFactory), но **ни один реальный тест их не использует**. Единственные пользователи — свои же smoke-тесты (`test_generators_smoke.py`, `test_processing_smoke.py`).

```
# Кто импортирует Core?
from Core.generators import GeneratorFactory  # только Core/__init__.py и smoke тесты
from Core.processing import StatisticsAdapter  # только Core/__init__.py и smoke тесты
```

**Рекомендация**: Либо интегрировать Core/ в реальные тесты, либо удалить как dead code.

---

### 5. DRY-нарушение: NumPy-референсы в 3+ местах

Одни и те же формулы (CW, LFM, dechirp) реализованы в нескольких местах:

| Формула | Локации |
|---------|---------|
| CW sinusoid | `common/references/signal_refs.py` (SignalReferences.cw), `signal_generators/test_form_signal_rocm.py` (generate_cw_numpy), `statistics/test_statistics_rocm.py` (make_sinusoid) |
| LFM chirp | `common/references/signal_refs.py` (SignalReferences.lfm), `signal_generators/test_form_signal_rocm.py` (generate_lfm_numpy) |
| Dechirp | `common/references/signal_refs.py` (SignalReferences.dechirp), `heterodyne/test_heterodyne_rocm.py` (ref_dechirp), `heterodyne/heterodyne_test_base.py` (dechirp_numpy) |

**Рекомендация**: Все тесты должны использовать `common/references/signal_refs.py` как единую точку истины. Удалить дублирующие локальные реализации.

---

### 6. statistics_rocm.py — subprocess + regex вместо Python bindings

**Файл**: `statistics/test_statistics_rocm.py` (681 строк)

GPU-тесты (6-9) запускают C++ бинарник через `subprocess.run(["sg", "render", "-c", BINARY_PATH])` и парсят stdout регулярками. Это:
- Хрупко (формат вывода может измениться)
- Зависит от `sg render` команды
- Не использует Python bindings (`gpuworklib.StatisticsProcessor`)
- Нельзя отладить отдельные тесты

NumPy-тесты (1-5) в том же файле — отличные, автономные.

**Рекомендация**: Перевести GPU-тесты на `gpuworklib.StatisticsProcessor` (по аналогии с `test_compute_all.py` который уже использует bindings).

---

### 7. Неконсистентные test runners

Два паттерна запуска тестов сосуществуют:

**Паттерн A** (TestRunner — правильный):
```python
# test_form_signal_rocm.py — использует TestRunner
runner = TestRunner()
results = runner.run(TestNumPyReference())
runner.print_summary(results)
```

**Паттерн B** (Custom loop — старый, 11 файлов):
```python
# test_fir_filter_rocm.py, test_heterodyne_rocm.py, test_statistics_rocm.py и др.
def run(label, fn):
    try:
        fn()
        passed += 1
    except AssertionError as e:  # <-- не ловит SkipTest!
        failed += 1
```

Файлы с паттерном B (11 штук): `test_fir_filter_rocm.py`, `test_iir_filter_rocm.py`, `test_kalman_rocm.py`, `test_kaufman_rocm.py`, `test_moving_average_rocm.py`, `test_heterodyne_rocm.py`, `test_lch_farrow_rocm.py`, `test_statistics_rocm.py`, `test_compute_all.py`, `test_zero_copy.py`, `test_hybrid_backend.py`.

**Проблемы паттерна B**:
- Не ловит `SkipTest` → тест считается FAILED вместо SKIPPED (в некоторых файлах)
- Использует `global passed, failed` — не thread-safe
- Нет `ValidationResult` — просто `assert` без деталей

**Рекомендация**: Мигрировать все файлы на `TestRunner`.

---

### 8. pipeline_runner.py — bare imports без sys.path

```python
# pipeline_runner.py:59-60
from scenario_builder import ULAGeometry, ScenarioBuilder, EmitterSignal
from farrow_delay import FarrowDelay
```

Эти импорты работают **только** если CWD = `Python_test/strategies/`. При запуске из корня проекта — `ModuleNotFoundError`.

Затронутые файлы: `pipeline_runner.py`, `plot_strategies_results.py`, `conftest.py`, `test_scenario_builder.py`, `test_farrow_pipeline.py`, `signal_generators_strategy.py`.

**Исправление**: Добавить `strategies/` в `sys.path` в conftest.py, или использовать relative imports.

---

## Рекомендации 🟢

### 9. DataValidator: cast to complex128 для всех данных

```python
# validators.py:113-114
a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.complex128)
r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.complex128)
```

Для float32 массивов это удваивает потребление памяти и замедляет вычисления. Для больших массивов (500K+ samples в statistics) может быть заметно.

**Рекомендация**: Использовать `np.promote_types(actual.dtype, reference.dtype)` для минимально необходимого повышения точности.

### 10. TestResult.passed = False когда validations пустой

```python
# result.py:61-62
if not self.validations:
    return False
```

Тест без валидаций считается проваленным. Это может удивить в тестах, которые проверяют только "не упало с ошибкой".

**Рекомендация**: Добавить документацию или метод `TestResult.no_checks_is_pass()`.

### 11. numpy.testing.assert_allclose (Context7)

Некоторые тесты уже используют `np.testing.assert_allclose` (test_form_signal_rocm.py:232), другие — самописный `np.allclose` + `assert`. Context7 подтверждает: `assert_allclose(actual, desired, rtol=1e-5, atol=1e-8)` — стандарт для научных вычислений.

**Рекомендация**: Для простых тестов (без custom metrics) использовать `np.testing.assert_allclose` с параметром `strict=True` для проверки dtype/shape.

### 12. HeterodyneConfig vs DechirpParams — дублирование

`common/configs.py` содержит `HeterodyneConfig` (наследует SignalConfig), а `heterodyne/conftest.py` содержит `DechirpParams` — оба описывают одно и то же.

**Рекомендация**: Унифицировать — использовать один тип.

---

## Соответствие стандартам GPUWorkLib

| Критерий | Оценка | Комментарий |
|----------|--------|-------------|
| **pytest запрещён** | PASS | Нигде не используется. TestRunner + SkipTest |
| **DrvGPU интеграция** | PASS | GPULoader + GPUContextManager корректно |
| **Google Style (naming)** | PASS | snake_case методы, CamelCase классы |
| **GoF/GRASP паттерны** | PASS | Template Method, Strategy, Singleton, Coordinator, Factory — корректно |
| **Отдельные файлы для классы** | PASS | Один класс = один файл |
| **Кросс-платформенность** | WARN | 4 файла с хардкодом путей |
| **DRY** | WARN | Дублирование референсов в 3+ местах |
| **Консистентность** | WARN | 2 стиля тест-раннеров (TestRunner vs custom loop) |

---

## Архитектура — что сделано хорошо

1. **common/ инфраструктура** — образцовая: чистые Value Objects, Strategy validators, Singleton GPU loader, Template Method test base
2. **conftest.py** — правильная организация: каждый модуль имеет свой conftest с factory functions
3. **GPULoader** — отличный Singleton с 7 стратегиями поиска, env override, кросс-платформенность
4. **DataValidator** — элегантная замена 4 специализированных валидаторов одним универсальным
5. **reporters.py** — Observer + Composite (MultiReporter) — расширяемый и чистый
6. **configs.py** — ISP: мелкие dataclasses вместо одного большого dict, computed properties
7. **references/** — DRY: единая точка истины для NumPy-эталонов (хотя не все тесты используют)

---

## План действий (приоритет)

| # | Задача | Приоритет | Усилие |
|---|--------|-----------|--------|
| 1 | Заменить NumericValidator/SpectralValidator на DataValidator | БЛОКЕР | 30 мин |
| 2 | Заменить хардкод путей на GPULoader | КРИТИЧНО | 1 час |
| 3 | Обновить docstring в test_base.py (NumericValidator -> DataValidator) | Средний | 5 мин |
| 4 | Мигрировать custom loops -> TestRunner (11 файлов) | Средний | 3 часа |
| 5 | Унифицировать NumPy-референсы (использовать signal_refs.py) | Средний | 2 часа |
| 6 | Удалить старый PipelineRunner.run_pipeline_a/b | Средний | 30 мин |
| 7 | Добавить strategies/ в sys.path для bare imports | Средний | 15 мин |
| 8 | Перевести statistics GPU тесты на Python bindings | Низкий | 2 часа |
| 9 | Решить судьбу Core/ DSL (интегрировать или удалить) | Низкий | 1 час |
| 10 | Унифицировать HeterodyneConfig / DechirpParams | Низкий | 30 мин |

---

## Источники

- **Context7 / pybind11**: numpy array passing, buffer protocol, py::vectorize patterns
- **Context7 / numpy**: `np.testing.assert_allclose(actual, desired, rtol, atol, strict=True)` — стандарт для научных тестов
- **sequential-thinking**: разбор архитектуры, классификация багов
- **Codebase analysis**: 106 файлов, grep по паттернам, чтение 25+ файлов

---

*Reviewed by: Кодо (AI Assistant), 2026-03-21*
