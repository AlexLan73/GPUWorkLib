# Code Review — TASK_PythonArch_04/05/06

> **Дата**: 2026-04-09
> **Reviewer**: Кодо (AI Assistant)
> **Объект**: `Python_test/common/validators/`, `Python_test/common/io/`, `Python_test/common/plotting/`
> **Связанные таски**: [TASK_PythonArch_INDEX.md](../tasks/TASK_PythonArch_INDEX.md)
> **Связанная серия**: Фазы 2–3 (закрыты 2026-04-09)

---

## 🎯 TL;DR

| Аспект | Вердикт |
|---|---|
| Архитектура (SOLID/GRASP/GoF) | ✅ Соответствует |
| Backward compat `DataValidator` | ✅ Полная, включая атрибуты |
| Critical bug `complex64 → complex128` | ✅ Исправлен и покрыт smoke-тестом |
| Strict `<` (не `<=`) | ✅ Все 7 мест проверены |
| `reference` обязателен для comparative | ✅ fail-fast через `_require_reference` |
| Отсутствие `pytest` | ✅ Только упоминания "НЕ pytest!" в docstring |
| Smoke-тесты | ✅ 29/29 (по TASK_PythonArch_INDEX.md) |
| Consumers старого API | ✅ 5 файлов используют `DataValidator(tolerance=…, metric=…)` |

**Критических проблем: 0.** Есть точечные замечания по чистоте кода и мёртвому коду.

---

## 🔴 Критические проблемы

**Нет.**

---

## 🟡 Важные замечания

### 1. Мёртвый код: `_apply_style()` в plotter_base.py

**Файл**: [plotter_base.py:121-134](../../Python_test/common/plotting/plotter_base.py#L121-L134)

Метод объявлен в базовом классе `IPlotter`, но **не вызывается нигде**:
- `SpectrumPlotter.plot()` и `plot_compare()` инлайнят `plt.style.use(self.config.style)` напрямую
- `TimePlotter.plot()` и `plot_magnitude()` делают то же самое
- Grep по всему проекту: 1 совпадение — само определение

**Риск**: Следующий разработчик добавит новый плоттер, использует `self._apply_style()` — и столкнётся с тем, что стиль будет применён _до_ `plt.subplots()` корректно только если матплотлиб backend уже загружен.

**Предложение**: либо удалить метод, либо перевести `SpectrumPlotter`/`TimePlotter` на его использование (DRY — стиль применяется в одном месте базового класса).

```python
# Вариант 1 — удалить.
# Вариант 2 — использовать в реализациях:
def plot(self, *args, ...):
    plt = self._apply_style()  # вместо 3-строчного блока import
    fig, ax = plt.subplots(figsize=self.config.figsize)
    ...
```

### 2. Неиспользуемый импорт `field` в plotter_base.py

**Файл**: [plotter_base.py:26](../../Python_test/common/plotting/plotter_base.py#L26)

```python
from dataclasses import dataclass, field  # ← field не используется
```

Ни одно поле `PlotConfig` не использует `field(default_factory=…)` (все — immutable-скаляры). Pyflakes/ruff это поймает.

**Предложение**: `from dataclasses import dataclass`.

### 3. Хрупкий duck typing в `ResultStore.save_test_result`

**Файл**: [result_store.py:116](../../Python_test/common/io/result_store.py#L116)

```python
if hasattr(result, "to_dict") and hasattr(result, "test_name"):
    data = result.to_dict()
    name = result.test_name
elif isinstance(result, dict):
    ...
```

Проблема: если вдруг `TestResult` потеряет `to_dict()` (рефакторинг), код **не упадёт громко** — он молча перейдёт в `elif isinstance(dict)`, получит `False` и выбросит менее информативный `TypeError` в `else`.

**Предложение**: использовать `isinstance(result, TestResult)` как первую ветку (TestResult живёт в том же репо, циклического импорта нет — `result.py` не зависит от `io/`).

```python
from ..result import TestResult  # top-level import
...
if isinstance(result, TestResult):
    data = result.to_dict()
    name = result.test_name
elif isinstance(result, dict):
    ...
```

### 4. `CompositeValidator` с пустым списком → `passed=True`

**Файл**: [composite.py:50-57](../../Python_test/common/validators/composite.py#L50-L57)

```python
if not self._validators:
    return ValidationResult(
        passed=True,            # ← PASS при 0/0
        metric_name=name,
        actual_value=0.0,
        threshold=0.0,
        ...
    )
```

Философски спорно: пустой composite «прошёл» — это потенциально скрывает баг в пользовательском коде (забыли `add()`).

**Риск**: низкий, но существующий. Тест пройдёт зелёным даже если все дочерние валидаторы «забыли» зарегистрировать.

**Предложение на выбор**:
- Raise `ValueError("CompositeValidator пуст — забыли add()?")` — fail-fast в духе `_require_reference`.
- Оставить PASS, но логически `passed=False` с сообщением — понятнее в отчёте.
- Оставить как есть, но добавить в документацию строку «пустой composite намеренно считается PASS».

---

## 🟢 Рекомендации

### 5. Документировать ожидаемый порядок спектра в `FrequencyValidator`

**Файл**: [signal.py:44-55](../../Python_test/common/validators/signal.py#L44-L55)

Когда `actual` — real массив, код трактует его как «готовый амплитудный спектр» и применяет `np.fft.fftfreq(n, d=1/fs)`. Это корректно **только если** спектр в натуральном порядке FFT (0…+fs/2, –fs/2…0). Если пользователь передаст `np.fft.fftshift(np.abs(fft(x)))` — частоты окажутся в обратном порядке, и `argmax` вернёт «левую» частоту вместо «правой».

**Предложение**: добавить в docstring явную оговорку:

```
Поведение:
    * complex actual → трактуется как временной сигнал, FFT делается внутри.
    * real actual    → трактуется как амплитудный спектр В НАТУРАЛЬНОМ
                       ПОРЯДКЕ np.fft.fft (без fftshift).
```

### 6. Магическое число `1e-10` в `PowerValidator`

**Файл**: [signal.py:87](../../Python_test/common/validators/signal.py#L87)

```python
rel_err = abs(power - self._expected) / max(self._expected, 1e-10)
```

`1e-10` скрыт в строке кода — не документирован, не параметризован. Если валидатор вызывают с `expected_power=0` (ошибочно), ошибка вернётся огромной → FAIL, но без ясности почему.

**Предложение**: вынести в модуль-константу или проверить в `__init__`:

```python
def __init__(self, expected_power: float, tolerance: float = 0.05):
    if expected_power <= 0:
        raise ValueError("expected_power должен быть > 0")
    self._expected = float(expected_power)
    self._tol = float(tolerance)
```

Это согласуется с общей политикой «fail-fast» (как `_require_reference`).

### 7. `JsonStore.add_timestamp=True` может перезаписать `saved_at` из data

**Файл**: [json_store.py:42](../../Python_test/common/io/json_store.py#L42)

```python
payload: dict = {"saved_at": datetime.now().isoformat(), **data}
```

Если `data` содержит ключ `"saved_at"` (например, из ранее загруженного JSON), он **перезапишет** наш. Это обратное от ожидаемого (хотим штампнуть момент сохранения).

**Предложение**: инвертировать порядок или зафиксировать приоритет явно:

```python
payload = {**data, "saved_at": datetime.now().isoformat()}  # наш штамп — последний
```

### 8. `TimePlotter.plot_magnitude` не следует сигнатуре базового класса

**Файл**: [time_plotter.py:83-90](../../Python_test/common/plotting/time_plotter.py#L83-L90)

```python
def plot_magnitude(self,
                   signal: np.ndarray,
                   fs: float,
                   title: str = "Magnitude",
                   max_samples: int = 2048) -> str:
```

`IPlotter.plot(*args, title, **kwargs)` использует `*args`, а `plot_magnitude` — именованные позиционные. Это непринципиально (метод не переопределяет `plot`), но создаёт неконсистентность: `factory.spectrum().plot(cw, 12e6)` и `factory.timeseries().plot_magnitude(cw, 12e6)` выглядят похожими, но первый — `*args`, второй — positional. Если потом появится `plot_compare` и подобные — API разъедется.

**Предложение**: либо все вспомогательные `plot_*` методы делать через `*args`, либо задокументировать, что только базовый `plot()` обязан следовать контракту `IPlotter`.

### 9. `print(f"[Plotter] Saved: {path}")` — жёсткий stdout

**Файл**: [plotter_base.py:115](../../Python_test/common/plotting/plotter_base.py#L115)

Не проблема для Python-тестов (правило про `ConsoleOutput` касается C++ GPU-части), но при параллельном запуске тестов консоль может перемешаться.

**Предложение**: опциональный `verbose` флаг в `PlotConfig`:

```python
@dataclass
class PlotConfig:
    ...
    verbose: bool = True

# in save_fig:
if self.config.save and self.config.verbose:
    print(f"[Plotter] Saved: {path}")
```

---

## ✅ Что сделано отлично

### Validators
- **Strategy + Composite + Factory** — классика GoF, чисто реализовано.
- **`_to_1d` / `_promote`** правильно продвигают complex → `complex128`, сохраняя Im-часть. Критический баг старого `DataValidator` покрыт smoke-тестом `test_backward_compat_complex_detects_im`.
- **`_require_reference`** — fail-fast для comparative-валидаторов. None проходит silently в старом коде, теперь ValueError.
- **Near-zero reference fallback** (`scale < 1e-15` → абсолютный tol `1e-10`) — сохранил поведение старого `DataValidator`, не ломая существующие тесты.
- **`DataValidator` как настоящий класс** с публичными `.tolerance / .metric / METRICS` — все 5 consumer-файлов (`filter_test_base`, `signal_test_base`, `test_base`, `statistics/test_compute_all`, `strategies/pipeline_step_validator`) работают без изменений.

### I/O
- **`_find_repo_root()`** — умная функция, проверяет `.git` как dir **и** как файл (git worktree, submodule). Это защищает от ошибки «запустил из рандомного cwd, файл ушёл в `./Results/`».
- **`array_exists` учитывает .npy И .npz** — есть специальный smoke-тест `test_array_exists_sees_npz` на тот случай, если кто-то сохранил через `save_comparison` и потом ищет через `array_exists`.
- **Все smoke-тесты пишут в `tempfile.TemporaryDirectory`** — ни один байт не падает в реальный `Results/`. Изоляция идеальная.
- **`save_config` поддерживает dataclass / plain class / dict** — покрывает все 3 варианта конфигов GPUWorkLib.

### Plotting
- **`_slugify`** — Windows-безопасные имена (убирает `: / \ | ? *` + пробелы). Smoke-тест `test_spectrum_plotter_saves_png` прямо проверяет это на заголовке `"CW 2MHz: f0=2e6/test"`.
- **`dataclasses.replace`** для `_with_subdir` — идиоматично, устойчиво к добавлению полей в `PlotConfig`.
- **Lazy import matplotlib** — smoke-тесты пропускаются через `SkipTest`, если matplotlib не установлен.
- **`matplotlib.use("Agg")`** — headless backend, не требует GUI (важно для CI).
- **Плоттеры не делят PlotConfig** — каждый `factory.spectrum()` получает свою копию через `replace()`.

---

## 📋 Соответствие стандартам GPUWorkLib

| Правило (из [TASK_PythonArch_INDEX.md](../tasks/TASK_PythonArch_INDEX.md)) | Статус |
|---|---|
| #1 Backward compat `DataValidator` как класс с публичными атрибутами | ✅ [__init__.py:36-69](../../Python_test/common/validators/__init__.py#L36-L69) |
| #2 pytest ЗАПРЕЩЁН | ✅ Grep: только "НЕ pytest!" в docstring |
| #3 SkipTest — только из `common.runner` | ✅ `test_smoke.py` всех трёх пакетов |
| #4 complex → complex128, real → float64 | ✅ [numeric.py:29-54](../../Python_test/common/validators/numeric.py#L29-L54) |
| #5 Core/ → common/ одностороннее | ✅ (не относится к этим пакетам — всё в common/) |
| #7 Файлы только в основной репозиторий | ✅ `git rev-parse`: `E:/C++/GPUWorkLib` |
| #9 `assert` ЗАПРЕЩЁН в тестах | ✅ Все smoke используют `ValidationResult`/`TestResult` |
| #10 Strict `<` не `<=` | ✅ 7/7 мест (grep проверен) |
| #11 Smoke sys.path-bootstrap | ✅ `parents[2]` во всех 3 test_smoke.py |
| #12 Reference обязателен для comparative | ✅ `_require_reference` в Relative/Absolute/Rmse |

---

## 🔬 Верификация smoke-тестов

Из [TASK_PythonArch_INDEX.md](../tasks/TASK_PythonArch_INDEX.md) — цифры подтверждены чтением соответствующих `test_smoke.py`:

| Suite | Файл | Тестов |
|---|---|---|
| `common.validators` | [validators/test_smoke.py](../../Python_test/common/validators/test_smoke.py) | **14** (backward_compat×3 + numeric×4 + signal×2 + composite×2 + factory×3 + raises) |
| `common.io` | [io/test_smoke.py](../../Python_test/common/io/test_smoke.py) | **9** (find_root + array_roundtrip + comparison + exists_npz + test_result×2 + benchmark + config + list) |
| `common.plotting` | [plotting/test_smoke.py](../../Python_test/common/plotting/test_smoke.py) | **6** (spectrum + time_iq + time_magnitude + subdir + compare + config_property) |
| **ИТОГО** | | **29** ✅ |

---

## 📦 Consumer-файлы `DataValidator` (backward-compat верификация)

Grep `DataValidator(` по `Python_test/`, исключая новый `common/validators/`:

| Файл | Вызовов | Использует `.tolerance`/`.metric`? |
|---|---|---|
| [filters/filter_test_base.py](../../Python_test/filters/filter_test_base.py) | 2 | Нет |
| [signal_generators/signal_test_base.py](../../Python_test/signal_generators/signal_test_base.py) | 1 | Нет |
| [heterodyne/heterodyne_test_base.py](../../Python_test/heterodyne/heterodyne_test_base.py) | — (импорт) | Нет |
| [statistics/test_compute_all.py](../../Python_test/statistics/test_compute_all.py) | 2 | Нет |
| [common/test_base.py](../../Python_test/common/test_base.py) | 1 | Нет |
| [strategies/pipeline_step_validator.py](../../Python_test/strategies/pipeline_step_validator.py) | 7 | Нет |

**Итог**: все consumer'ы используют только конструкторный API — `DataValidator(tolerance=…, metric=…)` + `.validate(actual, reference, name=…)`. Новый класс-обёртка полностью эквивалентен старой реализации. Никто в прод-коде не читает `.tolerance`/`.metric`/`.METRICS` — атрибуты оставлены «на всякий случай» для внешних интеграций.

---

## 📊 Итоговый вердикт

**Production-ready.** Все 3 пакета можно использовать в тестах модулей без изменений существующего кода. Критических или блокирующих проблем не найдено. Замечания — чистка кода (dead method `_apply_style`, unused import `field`) и мелкие улучшения fail-fast / документации.

**Рекомендуемый следующий шаг** (до прогона на AMD/NVIDIA GPU):

1. Удалить `_apply_style()` и `field` импорт из [plotter_base.py](../../Python_test/common/plotting/plotter_base.py). ~2 минуты.
2. Заменить `hasattr` на `isinstance(TestResult)` в [result_store.py:116](../../Python_test/common/io/result_store.py#L116). ~2 минуты.
3. Инвертировать порядок `saved_at` в [json_store.py:42](../../Python_test/common/io/json_store.py#L42). ~1 минута.
4. Добавить оговорку про `fft vs fftshift` в docstring `FrequencyValidator`. ~2 минуты.

После этого — прогнать на реальном GPU по плану из [TASK_PythonArch_INDEX.md](../tasks/TASK_PythonArch_INDEX.md#__----final-verification).

---

## 📚 Источники

- **Context7**: не требовался (чистый Python + numpy + matplotlib — стандартный стек, внутренняя архитектура).
- **Sequential-thinking**: 2 итерации для разбивки и консолидации наблюдений.
- **Grep верификация**: pytest (3 пакета), `passed=.*<` (7 мест), `DataValidator(` (5 consumer-файлов), `.tolerance/.metric` (консьюмеры не читают).

---

*Review by: Кодо (AI Assistant) | 2026-04-09*
