# Python_test — Архитектурный рефакторинг v2 (INDEX)

> **Цель**: Привести `Python_test/` к единой архитектуре ООП + SOLID + GRASP + GoF
> **Создан**: 2026-03-21
> **Исследование**: `MemoryBank/research/python_test_refactoring_plan.md`
> **Предыдущая серия**: `TASK_Python_01..08` — убрали pytest, добавили DataValidator ✅

---

## 🎯 Главная идея

```
БЫЛО:                          СТАНЕТ:
─────────────────────          ───────────────────────────────────────
Дублирование эталонов  →       common/references/  (единая точка истины)
Разрозненный I/O       →       common/io/          (ResultStore)
DataValidator = 3в1    →       common/validators/  (иерархия + Composite)
Ручное создание GPU    →       Core/generators/    (Adapter + Factory)
Нет переиспользования  →       Core/processing/    (StatisticsAdapter...)
```

---

## 📋 Цепочка выполнения

```
Фаза 1 — Быстрый результат (приоритет HIGH):
  TASK_Arch_03 (references)  →  все тесты могут использовать SignalReferences
        ↓ (параллельно с Arch-03)
  TASK_Arch_01 (Core/generators)  →  CwGenerator, LfmGenerator, NoiseGenerator
        ↓ (нужен Core/__init__.py из Arch-01)
  TASK_Arch_02 (Core/processing)  →  StatisticsAdapter, HeterodyneAdapter
        + HeterodyneConfig в common/configs.py

Фаза 2 — Архитектура (приоритет MEDIUM):
  TASK_Arch_04 (validators)  →  иерархия + CompositeValidator
        ↓
  TASK_Arch_05 (io/store)    →  ResultStore, NumpyStore, JsonStore

Фаза 3 — Полировка (приоритет LOW):
  TASK_Arch_06 (plotting)    →  PlotterFactory, SpectrumPlotter

⚠️ Arch-03 и Arch-01 можно параллельно (независимы).
⚠️ Arch-02 ПОСЛЕ Arch-01 (нужен Core/__init__.py).
```

---

## 📁 Таски

| # | Файл | Что делает | Фаза | Статус |
|---|------|-----------|------|--------|
| 01 | [TASK_PythonArch_01_core_generators.md](TASK_PythonArch_01_core_generators.md) | `Core/generators/` — ISignalGenerator + CW/LFM/Noise Adapters + Factory | 1 | ⬜ TODO |
| 02 | [TASK_PythonArch_02_core_processing.md](TASK_PythonArch_02_core_processing.md) | `Core/processing/` — IProcessor + Statistics/Heterodyne/FFT Adapters | 1 | ⬜ TODO |
| 03 | [TASK_PythonArch_03_references.md](TASK_PythonArch_03_references.md) | `common/references/` — SignalReferences, FilterReferences, FftReferences | 1 | ⬜ TODO |
| 04 | [TASK_PythonArch_04_validators.md](TASK_PythonArch_04_validators.md) | `common/validators/` — иерархия + Composite + Factory (backward compat!) | 2 | ⬜ TODO |
| 05 | [TASK_PythonArch_05_io_store.md](TASK_PythonArch_05_io_store.md) | `common/io/` — IDataStore + NumpyStore + JsonStore + ResultStore | 2 | ⬜ TODO |
| 06 | [TASK_PythonArch_06_plotting.md](TASK_PythonArch_06_plotting.md) | `common/plotting/` — PlotterFactory + SpectrumPlotter + TimePlotter | 3 | ⬜ TODO |

---

## 🏗️ Целевая структура файлов

```
Python_test/
├── common/
│   ├── references/             ← TASK_03 (DRY NumPy эталоны)
│   │   ├── __init__.py
│   │   ├── signal_refs.py      ← SignalReferences: cw(), lfm(), noise()
│   │   ├── filter_refs.py      ← FilterReferences: fir_filter(), iir_filter()
│   │   ├── statistics_refs.py  ← StatisticsReferences: mean(), std(), median()
│   │   └── fft_refs.py         ← FftReferences: fft(), magnitude()
│   │
│   ├── validators/             ← TASK_04 (Strategy + Composite)
│   │   ├── __init__.py         ← DataValidator backward compat + новый API
│   │   ├── base.py             ← IValidator (ABC)
│   │   ├── numeric.py          ← RelativeValidator, AbsoluteValidator, RmseValidator
│   │   ├── signal.py           ← FrequencyValidator, PhaseValidator
│   │   ├── composite.py        ← CompositeValidator
│   │   └── factory.py          ← ValidatorFactory
│   │
│   ├── io/                     ← TASK_05 (Repository I/O)
│   │   ├── __init__.py
│   │   ├── base.py             ← IDataStore (ABC)
│   │   ├── numpy_store.py      ← NumpyStore (.npy, .npz)
│   │   ├── json_store.py       ← JsonStore (.json)
│   │   └── result_store.py     ← ResultStore (координатор)
│   │
│   └── plotting/               ← TASK_06 (расширение)
│       ├── factory.py          ← PlotterFactory(module_name)
│       ├── spectrum_plotter.py ← SpectrumPlotter
│       └── time_plotter.py     ← TimePlotter
│
├── Core/                       ← TASK_01 + TASK_02 (готовые GPU-объекты)
│   ├── __init__.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py             ← ISignalGenerator (ABC)
│   │   ├── cw.py               ← CwGenerator (Adapter)
│   │   ├── lfm.py              ← LfmGenerator (Adapter)
│   │   ├── noise.py            ← NoiseGenerator (Adapter)
│   │   └── factory.py          ← GeneratorFactory (Registry)
│   └── processing/
│       ├── __init__.py
│       ├── base.py             ← IProcessor (ABC)
│       ├── statistics.py       ← StatisticsAdapter
│       ├── heterodyne.py       ← HeterodyneAdapter
│       └── fft.py              ← FftAdapter
```

---

## ⚠️ Критические правила

1. **Backward compatibility** — `DataValidator` остаётся как функция-alias (не ломать старые тесты!)
2. **pytest ЗАПРЕЩЁН** — TestRunner + SkipTest + `if __name__ == "__main__"`
3. **SkipTest** — только из `common.runner`, не из unittest
4. **float32/complex64** — все GPU данные; вычисления ошибок в **float64**
5. **Core/ зависит от common/** — но не наоборот (Low Coupling GRASP)
6. **Тесты не переписываем сразу** — новый код используется по мере удобства
7. **Файлы только в основной репозиторий** — НЕ в `.claude/worktrees/*/`
8. **conftest.py → helpers.py** — при создании новых файлов НЕ использовать имя `conftest.py` (pytest-артефакт). Существующие переименовать при удобном случае.
9. **assert ЗАПРЕЩЁН** в тестах — использовать `ValidationResult` + `TestResult` + `TestRunner`
10. **Strict `<`** в валидаторах — НЕ `<=`, для совместимости с DataValidator

## 📋 Порядок обновления common/__init__.py

После каждого TASK обновлять `common/__init__.py`:

```
Фаза 1:
  Arch-03 → добавить: from .references import SignalReferences, ...
  Arch-01 → ничего (Core/ — отдельный пакет)
  Arch-02 → добавить: from .configs import HeterodyneConfig

Фаза 2:
  Arch-04 → заменить: from .validators import IValidator, DataValidator, ...
  Arch-05 → добавить: from .io import ResultStore, NumpyStore, JsonStore

Фаза 3:
  Arch-06 → обновить: from .plotting import PlotterFactory, ...
```

---

## 🔑 Паттерны (карта)

```
GoF Creational:
  Singleton     → GPULoader, GPUContextManager (уже есть ✅)
  Factory       → GeneratorFactory, ValidatorFactory, PlotterFactory
  Registry      → GeneratorFactory._registry (OCP без изменения Factory)

GoF Structural:
  Adapter       → CwGenerator, LfmGenerator, StatisticsAdapter
  Composite     → CompositeValidator

GoF Behavioural:
  Template Method → TestBase, SignalTestBase (уже есть ✅)
  Strategy        → IValidator, IDataStore, IPlotter
  Observer        → IReporter (уже есть ✅)

GoF Arch:
  Repository    → ResultStore
  Value Object  → TestResult, ValidationResult (уже есть ✅)

GRASP:
  Information Expert → SignalReferences, SignalConfig, ResultStore
  Creator            → ValidatorFactory, GeneratorFactory
  Controller         → TestRunner (уже есть ✅)
  Low Coupling       → Core → common (односторонняя зависимость)
  High Cohesion      → каждый пакет = одна ответственность

SOLID:
  SRP → каждый класс = одна метрика / один формат
  OCP → register() вместо if/elif в Factory
  LSP → все Adapters взаимозаменяемы через интерфейс
  ISP → ISignalGenerator ≠ IProcessor ≠ IDataStore
  DIP → тесты зависят от IValidator, не от RelativeValidator
```

---

*Создан: 2026-03-21 | Кодо*
