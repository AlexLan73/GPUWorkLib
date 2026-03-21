# 🔍 РЕВЬЮ: Python_test Refactoring Plan — Расхождения и баги

> **Дата**: 2026-03-21
> **Ревьюер**: Кодо
> **Файлы**: `research/python_test_refactoring_plan.md`, `tasks/BACKLOG.md`, `TASK_PythonArch_01..06`, `Python_test/`
> **Метод**: sequential-thinking + cross-reference с существующим кодом

---

## 📊 Сводка

| Критичность | Кол-во | Описание |
|-------------|--------|----------|
| 🔴 Критические | 5 | Баги, ломающие backward compat или нарушающие заявленные принципы |
| 🟠 Важные | 7 | Пропуски в плане, потеря функциональности |
| 🟡 Средние | 3 | Дублирование, неточности |
| 🔵 Минорные | 5 | Стиль, хрупкость, мелкие несоответствия |

---

## 🔴 КРИТИЧЕСКИЕ

### R-01. GeneratorFactory._build() нарушает OCP

**Файл**: `TASK_PythonArch_01_core_generators.md` → `factory.py`

**Проблема**: План декларирует _"OCP — добавить новый тип = register(), не менять Factory"_, но метод `_build()` содержит:

```python
if type_name == "cw":
    return cls_(ctx, fs=params.fs, f0=params.f0_hz, ...)
if type_name == "lfm":
    return cls_(ctx, fs=params.fs, f_start=params.f0_hz, ...)
if type_name == "noise":
    return cls_(ctx, fs=params.fs, ...)
```

Каждый новый тип **требует изменения Factory** → прямое нарушение OCP.

**Причина**: Конструкторы CwGenerator/LfmGenerator/NoiseGenerator имеют **разные** сигнатуры.

**Решение**: Ввести classmethod `from_config(cls, ctx, params: SignalConfig) -> ISignalGenerator` в `ISignalGenerator`. Каждый адаптер сам извлекает нужные поля из SignalConfig. Factory вызывает `cls_.from_config(ctx, params)` без if/elif.

```python
# ISignalGenerator — добавить:
@classmethod
@abstractmethod
def from_config(cls, ctx, params: "SignalConfig") -> "ISignalGenerator": ...

# CwGenerator:
@classmethod
def from_config(cls, ctx, params):
    return cls(ctx, fs=params.fs, f0=params.f0_hz, amplitude=params.amplitude)

# Factory._build() → одна строка:
return cls._registry[type_name].from_config(ctx, params)
```

---

### R-02. RelativeValidator `<=` vs DataValidator `<` — тихое изменение поведения

**Файлы**: `TASK_PythonArch_04` → `numeric.py` vs `common/validators.py`

**Проблема**:
- Существующий `DataValidator._max_rel()`: `err < self.tolerance` (strict `<`)
- Новый `RelativeValidator.validate()`: `metric <= self._tol` (non-strict `<=`)

Backward-compat wrapper `DataValidator(tol, "max_rel") → ValidatorFactory.create() → RelativeValidator(tol)` **изменит поведение**: тесты с `err == tolerance` раньше были FAIL, теперь станут PASS.

**Решение**: Использовать `<` (strict) в RelativeValidator/AbsoluteValidator/RmseValidator для совместимости с существующим DataValidator.

---

### R-03. CompositeValidator — бессмысленная "worst" метрика

**Файл**: `TASK_PythonArch_04` → `composite.py`

**Проблема**:
```python
worst = max(results, key=lambda r: r.actual_value / max(r.threshold, 1e-15))
```

Для **гетерогенных** валидаторов это бессмысленно:
- `FrequencyValidator`: actual=2000000 Hz, threshold=1000 Hz → ratio = **2000**
- `RelativeValidator`: actual=0.001, threshold=0.01 → ratio = **0.1**

FrequencyValidator **всегда** "worst" даже при PASS, потому что частоты в Гц >> безразмерные ошибки.

**Решение**: Использовать нормированное превышение только для FAILED:
```python
failed = [r for r in results if not r.passed]
if failed:
    worst = max(failed, key=lambda r: (r.actual_value - r.threshold) / max(abs(r.threshold), 1e-15))
else:
    worst = results[0]  # все прошли — любой
```

---

### R-04. FftReferences.magnitude_db() — баг приоритета `.astype()`

**Файл**: `TASK_PythonArch_03` → `fft_refs.py`

**Проблема**:
```python
return 20 * np.log10(mag / ref + 1e-12).astype(np.float32)
```

Python парсит как: `20 * np.log10(mag / ref + np.float32(1e-12))` — `.astype()` применяется к **скаляру** `1e-12`, а не к результату `20 * np.log10(...)`.

**Решение**: Добавить скобки:
```python
return (20 * np.log10(mag / ref + 1e-12)).astype(np.float32)
```

---

### R-05. Новые валидаторы не приводят к complex128

**Файлы**: `TASK_PythonArch_04` → `numeric.py` vs `common/validators.py`

**Проблема**: Существующий `DataValidator.validate()` приводит данные к `complex128` для точности:
```python
a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.complex128)
```

Новый `RelativeValidator.validate()` использует:
```python
a = np.asarray(actual)
```

Для `float32` данных (основной тип GPU) разница в точности может давать **другой результат** сравнения, особенно при tolerance ~1e-3.

**Решение**: Добавить `.astype(np.float64)` для вычислений в `RelativeValidator` и `AbsoluteValidator` (как уже сделано в `RmseValidator`).

---

## 🟠 ВАЖНЫЕ

### R-06. getX_numpy() не включена в SignalReferences

**Файлы**: `TASK_PythonArch_03` vs `signal_generators/conftest.py:79-92`

**Проблема**: Функция `getX_numpy()` — CPU-эталон для FormSignal (формула с окном, центрированной фазой, нормировкой). Используется в `test_form_signal.py`, `test_delayed_form_signal.py`. В `SignalReferences` она **отсутствует**.

**Решение**: Добавить `SignalReferences.form_signal(fs, points, f0, amplitude, phase, fdev, norm_val, tau=0.0)` в `signal_refs.py`.

---

### R-07. DechirpParams → SignalConfig — потеря вычисляемых свойств

**Файлы**: `TASK_PythonArch_02` vs `heterodyne/conftest.py:29-54`

**Проблема**: `DechirpParams` имеет важные вычисляемые свойства:
- `bandwidth` → `f_end - f_start`
- `chirp_rate` → `bandwidth / duration`
- `range_from_delay(delay_s)` → `c_light * delay_s / 2`
- `fbeat_from_delay(delay_s)` → `chirp_rate * delay_s`

`HeterodyneAdapter` принимает `SignalConfig`, которая **не имеет** этих свойств. Тесты гетеродина активно их используют.

**Решение**: Либо (A) расширить `SignalConfig` свойствами для LFM/dechirp, либо (B) оставить `DechirpParams` и типизировать `HeterodyneAdapter.__init__(ctx, params: DechirpParams | SignalConfig, ...)`.

---

### R-08. LfmParams ↔ SignalConfig — разная семантика полей

**Файлы**: `signal_generators/conftest.py:29-36` vs `common/configs.py:22-56`

| Поле | LfmParams | SignalConfig |
|------|-----------|-------------|
| Длина | `length: int` | `n_samples: int` |
| Начальная частота | `f_start: float` | `f0_hz: float` |
| Конечная частота | `f_end: float` | `f0_hz + fdev_hz` |

**Проблема**: План не описывает миграцию. `GeneratorFactory.create("lfm", ctx, params)` использует `params.f0_hz` как `f_start` и `params.f0_hz + params.fdev_hz` как `f_end`. Но `SignalConfig.f0_hz` по умолчанию `2e6`, а `fdev_hz=0` → `f_start == f_end` = не ЛЧМ!

**Решение**: Явно документировать маппинг в TASK_Arch_01:
```
SignalConfig для LFM:
  f0_hz  = f_start (начальная частота ЛЧМ)
  fdev_hz = bandwidth (f_end - f_start)
```

---

### R-09. IProcessor нарушает LSP

**Файл**: `TASK_PythonArch_02` → `base.py`

**Проблема**: `IProcessor.process() -> np.ndarray | dict`
- `StatisticsAdapter.process()` → `dict`
- `HeterodyneAdapter.process()` → `np.ndarray`
- `FftAdapter.process()` → `np.ndarray`

Код `result = proc.process(data); np.abs(result)` работает для Heterodyne/FFT но **падает** для Statistics.

**Решение варианты**:
- **(A) Лучший**: Убрать общий `IProcessor` — адаптеры достаточно разные (YAGNI)
- (B) Разделить на `IArrayProcessor(→ndarray)` и `IMetricsProcessor(→dict)`
- (C) StatisticsAdapter возвращает structured ndarray вместо dict

---

### R-10. BACKLOG vs TASK_Arch_02 — противоречие в зависимостях

**Файлы**: `BACKLOG.md:23` vs `TASK_PythonArch_02:4`

- BACKLOG: _"не зависят друг от друга, можно параллельно"_
- TASK_Arch_02: _"Зависимости: TASK_Arch_01 (Core уже создан)"_

**Решение**: TASK_Arch_02 зависит от TASK_Arch_01 только наличием `Core/__init__.py`. Можно создать его заранее или первым. Убрать "параллельно" из BACKLOG для Arch-01/Arch-02.

---

### R-11. FftAdapter hardcodes `FFTProcessorROCm`

**Файл**: `TASK_PythonArch_02` → `fft.py:232`

```python
self._proc = gw.FFTProcessorROCm(ctx, n_fft)
```

На ветке `nvidia` (OpenCL) `FFTProcessorROCm` **не существует** — нужен `FFTProcessor`.

**Решение**: Автоопределение по типу контекста или параметр `backend`:
```python
if hasattr(gw, 'FFTProcessorROCm'):
    self._proc = gw.FFTProcessorROCm(ctx, n_fft)
else:
    self._proc = gw.FFTProcessor(ctx, n_fft)
```

---

### R-12. ResultStore — относительные пути

**Файл**: `TASK_PythonArch_05` → `result_store.py`

**Проблема**: `ResultStore(base_dir="Results")` — относительный путь. Если тест запущен не из корня проекта (`cd Python_test && python statistics/test_statistics_rocm.py`), файлы окажутся в `Python_test/Results/` вместо `GPUWorkLib/Results/`.

Существующий код использует `_PROJECT_ROOT`:
```python
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Решение**: Default path через `_PROJECT_ROOT / "Results"`:
```python
def __init__(self, base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).parents[2] / "Results"
    ...
```

---

## 🟡 СРЕДНИЕ

### R-13. SignalReferences.noise() никогда не совпадёт с GPU

**Файл**: `TASK_PythonArch_03` → `signal_refs.py:128-136`

GPU использует **Philox PRNG + Box-Muller**, NumPy — **PCG64**. Одинаковый `seed` → **разные** числа. Эталон полезен только для статистических свойств (mean≈0, std≈amplitude), не для поэлементного сравнения.

**Решение**: Добавить docstring-предупреждение:
```python
"""⚠️ GPU (Philox) и NumPy (PCG64) дают разные числа при одном seed.
Использовать для статистической проверки (mean, std), НЕ для поэлементного сравнения."""
```

---

### R-14. save_fig() дублируется в SpectrumPlotter и TimePlotter

**Файл**: `TASK_PythonArch_06` → `spectrum_plotter.py` + `time_plotter.py`

Оба класса реализуют **идентичный** `save_fig()` с обработкой show/close. Базовый `IPlotter.save_fig()` уже существует, но без show/close.

**Решение**: Расширить `IPlotter.save_fig()` обработкой show/close, убрать дублирование.

---

### R-15. conftest.py — pytest-артефактное имя

**Файлы**: 10× `conftest.py` в подкаталогах Python_test/

Имя `conftest.py` — pytest-конвенция. pytest **ЗАПРЕЩЁН**. Файлы не используют pytest магию — просто модули с helpers/params.

**Решение**: Не блокирующее, но при рефакторинге рассмотреть переименование в `helpers.py` или `test_params.py` для ясности.

---

## 🔵 МИНОРНЫЕ

### R-16. Мини-тесты в TASK файлах используют assert вместо TestRunner

TASK_Arch_01..06 содержат `assert sig.dtype == np.complex64` — это правильно для быстрых smoke-тестов, но не соответствует собственной инфраструктуре (TestRunner + ValidationResult). Для consistency можно оставить assert в примерах.

### R-17. IValidator.validate() — 3-й аргумент name

Старый `IValidator.validate(actual, reference)` — 2 аргумента. Новый добавляет `name=""`. Существующий DataValidator уже принимает name — OK, не ломает.

### R-18. StatisticsReferences — формула для complex данных

`np.abs(data)**2 if np.iscomplexobj(data) else data` — нужно верифицировать что GPU StatisticsProcessor считает так же (power vs amplitude). Может отличаться.

### R-19. PlotterFactory.dataclasses.replace()

Используется `from dataclasses import replace` внутри метода — lazy import ОК, но может удивить. Не критично.

### R-20. common/__init__.py — порядок обновления

После каждого TASK нужно обновлять `common/__init__.py`. Порядок не описан, но зависимости между TASK делают это неявным.

---

## 🎯 Рекомендации по приоритету выполнения

```
ПЕРЕД НАЧАЛОМ РЕАЛИЗАЦИИ:
  1. Исправить R-01 (from_config в ISignalGenerator) — меняет архитектуру Factory
  2. Исправить R-02 (< vs <=) — backward compat
  3. Исправить R-04 (astype баг) — копипаста бага
  4. Решить R-09 (IProcessor LSP) — архитектурное решение

ФАЗА 1 РЕАЛИЗАЦИИ:
  Arch-03 (references) + R-06 (добавить form_signal) + R-13 (docstring noise)
  → Arch-01 (generators) с fix R-01
  → Arch-02 (processing) с fix R-09, R-07, R-11

ФАЗА 2 РЕАЛИЗАЦИИ:
  Arch-04 (validators) с fix R-02, R-03, R-05
  → Arch-05 (io) с fix R-12

ФАЗА 3:
  Arch-06 (plotting) с fix R-14
```

---

*Ревью: Кодо | 2026-03-21 | 20 расхождений найдено*

---

## ✅ Решения Alex (2026-03-21) — ВСЕ TASK ФАЙЛЫ ОБНОВЛЕНЫ

| # | Решение Alex | Что сделано в TASK |
|---|-------------|-------------------|
| R-01 | ✅ Согласен | `from_config()` classmethod в ISignalGenerator, Factory без if/elif |
| R-02 | ✅ Strict `<` | `<` вместо `<=` в RelativeValidator/Absolute/Rmse |
| R-03 | ✅ Согласен | `worst` = первый FAILED, не ratio-based |
| R-04 | ✅ Согласен | `(20 * np.log10(...)).astype(np.float32)` — скобки |
| R-05 | ✅ float64 | `.astype(np.float64)` в Relative/Absolute (как Rmse). GPU=float32, вычисления=float64 |
| R-06 | ✅ Да | `SignalReferences.form_signal()` добавлена |
| R-07 | ✅ HeterodyneConfig | Наследник SignalConfig с chirp_rate, fbeat_from_delay и т.д. |
| R-08 | ✅ Привести к одной | Маппинг LfmParams→SignalConfig документирован в TASK_03 |
| R-09 | ✅ GpuProcessorMixin | Убран IProcessor с union return. Миксин + типизированные адаптеры (масштабируемо) |
| R-10 | ✅ Да | BACKLOG: 03∥01→02. INDEX: зависимости уточнены |
| R-11 | ✅ ROCm only | FftAdapter: комментарий "только ROCm в main" |
| R-12 | ✅ Абсолютные пути | `_PROJECT_ROOT / "Results"` в ResultStore |
| R-13 | ✅ Сценарий GPU→CPU→compare | Docstring в noise() + StatisticsReferences описывает сценарий |
| R-14 | ✅ Согласен | save_fig() расширен в IPlotter, убрано дублирование |
| R-15 | ✅ Сделать! | Правило в INDEX: новые файлы = helpers.py, не conftest.py |
| R-16 | ✅ assert→TestRunner | Все мини-тесты переписаны на TestRunner + ValidationResult |
| R-17 | ✅ OK | Совместимо |
| R-18 | ✅ Учесть | Предупреждение в docstring StatisticsReferences |
| R-19 | ✅ Без удивлений | Явный PlotConfig() вместо dataclasses.replace() |
| R-20 | ✅ Делай | Порядок обновления common/__init__.py в INDEX |
