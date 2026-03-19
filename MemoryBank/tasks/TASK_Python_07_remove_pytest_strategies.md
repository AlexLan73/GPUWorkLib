# TASK Python-07: Удалить pytest из strategies/

**Статус**: 🔲 TODO
**Приоритет**: 🟠 СРЕДНИЙ — после TASK 01–06
**Файлы**: все `*.py` в `Python_test/strategies/` и `Python_test/conftest.py`
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Убрать `import pytest` и все pytest-зависимости из:
1. `Python_test/conftest.py` (корневой)
2. `Python_test/strategies/conftest.py`
3. `Python_test/strategies/test_*.py` (6 файлов)

---

## 📖 Что прочитать перед реализацией

- `Python_test/conftest.py` — текущий код (прочитать полностью)
- `Python_test/strategies/conftest.py` — текущий код
- Каждый `Python_test/strategies/test_*.py` — прочитать каждый
- `TASK_Python_01` — `TestRunner`, `SkipTest`

---

## 📋 Таблица замен

| pytest | Замена |
|--------|--------|
| `import pytest` | Удалить строку |
| `pytest.skip(reason)` | `raise SkipTest(reason)` |
| `@pytest.fixture(scope="session")` | Убрать декоратор, метод → в Singleton |
| `@pytest.fixture(scope="module")` | Убрать декоратор, lazy-init в setUp() |
| `@pytest.fixture` | Убрать декоратор, вызывать напрямую |
| `@pytest.mark.parametrize(params)` | Явный цикл `for param in params:` |
| `def test_xxx(fixture1, fixture2):` | `def test_xxx(self):` в TestBase/классе |

---

## 🔧 Файл 1: `Python_test/conftest.py`

**Текущее состояние**: `import pytest` + 5 `@pytest.fixture`

**Что сделать**:

```python
# conftest.py — ПОСЛЕ (убрать pytest, оставить логику)
"""
conftest.py — инициализация Python_test/
=========================================
ВАЖНО: Этот файл больше НЕ использует pytest.
       Инициализация через GPUContextManager (Singleton).
       SkipTest из common.runner — вместо pytest.skip().
"""

import os
import sys

# Добавить Python_test/ в sys.path
_PT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

# Инициализация GPU-контекстов при импорте
# GPUContextManager — Singleton, создаётся один раз
from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager

# Предзагрузить библиотеку (если установлена)
_loader = GPULoader.get()

# Пути (для использования в тестах напрямую)
PROJECT_ROOT = os.path.dirname(_PT_DIR)
PLOT_DIR = os.path.join(PROJECT_ROOT, "Results", "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)
```

**Убрать**: `@pytest.fixture(scope="session")` декораторы — функции `gw()`, `gpu_ctx()`, `rocm_ctx()` и т.д.
Логика инициализации переходит в `GPUContextManager.get()` / `GPUContextManager.get_rocm()`.

---

## 🔧 Файл 2: `Python_test/strategies/conftest.py`

**Текущее состояние**: `@pytest.fixture` для `farrow`, `scenario_8ant`, `pipeline_runner`

**Что сделать**:

```python
# strategies/conftest.py — ПОСЛЕ
"""
strategies/conftest.py — вспомогательные функции для strategies тестов
======================================================================
Не использует pytest. Fixture заменены на функции-фабрики.
"""

import os
import sys
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from farrow_delay import FarrowDelay
from scenario_builder import make_single_target, make_multi_target


# Вместо @pytest.fixture — обычные функции
def make_farrow() -> FarrowDelay:
    """Создать FarrowDelay."""
    return FarrowDelay()


def make_scenario_8ant():
    """Создать сценарий: 8 антенн, одна цель @ 30°."""
    return make_single_target(
        n_ant=8, theta_deg=30.0, fs=12e6,
        n_samples=4096, fdev_hz=1e6
    )


def make_scenario_multi():
    """Создать сценарий: 3 цели @ 15°/30°/45°."""
    return make_multi_target(
        n_ant=8, thetas=[15, 30, 45],
        f0s=[2e6, 3e6, 4e6], fdevs=[1e6, 1e6, 1e6],
        fs=12e6, n_samples=4096
    )


def get_strategy_plot_dir() -> str:
    """Путь к Results/Plots/strategies/."""
    from conftest import PLOT_DIR
    path = os.path.join(PLOT_DIR, "strategies")
    os.makedirs(path, exist_ok=True)
    return path
```

---

## 🔧 Файл 3–8: `Python_test/strategies/test_*.py`

Для каждого файла выполнить:

### Шаг A — Прочитать файл полностью

### Шаг B — Применить замены

**Замена `def test_xxx(fixture1, fixture2)` → метод класса**:

```python
# ДО (pytest):
def test_identity(farrow):
    result = farrow.apply_single(signal, delay_samples=0)
    assert np.allclose(result, signal, atol=1e-6)

# ПОСЛЕ (TestRunner):
class TestFarrowDelay:
    def setUp(self):
        self._farrow = make_farrow()
        self._signal = np.exp(1j * np.linspace(0, 10*np.pi, 1024)).astype(np.complex64)

    def test_identity(self) -> TestResult:
        result = self._farrow.apply_single(self._signal, delay_samples=0)
        v = DataValidator(tolerance=1e-6, metric="abs")
        tr = TestResult(test_name="test_identity")
        tr.add(v.validate(result, self._signal, name="identity_check"))
        return tr
```

**Замена `pytest.skip()`**:
```python
# ДО:
def test_something(gpu_ctx):
    if gpu_ctx is None:
        pytest.skip("нет GPU")

# ПОСЛЕ:
def test_something(self):
    ctx = GPUContextManager.get_rocm()
    if ctx is None:
        raise SkipTest("нет ROCm GPU")
```

**Замена `@pytest.mark.parametrize`**:
```python
# ДО:
@pytest.mark.parametrize("delay", [0, 1.5, 3.0, -2.5])
def test_delay(farrow, delay):
    result = farrow.apply_single(signal, delay)
    ...

# ПОСЛЕ:
def test_delays(self) -> TestResult:
    tr = TestResult(test_name="test_delays")
    for delay in [0, 1.5, 3.0, -2.5]:
        result = self._farrow.apply_single(self._signal, delay)
        v = DataValidator(tolerance=1e-4, metric="abs")
        tr.add(v.validate(..., name=f"delay_{delay}"))
    return tr
```

**Замена `assert` → `DataValidator.validate()`**:
```python
# ДО:
assert np.allclose(result, reference, atol=1e-3)

# ПОСЛЕ:
v = DataValidator(tolerance=1e-3, metric="abs")
tr.add(v.validate(result, reference, name="check_name"))
```

### Шаг C — Добавить точку входа

В конце каждого `test_*.py`:
```python
if __name__ == "__main__":
    from common.runner import TestRunner
    runner = TestRunner()
    results = runner.run(TestClassName())
    runner.print_summary(results)
```

---

## 📁 Список файлов для обработки

| Файл | Что там сейчас | Приоритет |
|------|----------------|-----------|
| `conftest.py` (корень) | 5 pytest.fixture | 🔴 |
| `strategies/conftest.py` | 4 pytest.fixture | 🔴 |
| `strategies/test_farrow_pipeline.py` | 19 pytest тестов | 🟡 |
| `strategies/test_scenario_builder.py` | 27 pytest тестов | 🟡 |
| `strategies/test_debug_steps.py` | pytest тесты | 🟡 |
| `strategies/test_base_pipeline.py` | pytest тесты | 🟡 |
| `strategies/test_strategies_step_by_step.py` | pytest тесты | 🟡 |
| `strategies/test_strategies_step_by_step_01.py` | pytest тесты | 🟡 |
| `strategies/test_timing_analysis.py` | pytest тесты | 🟠 |

---

## ✅ Критерии готовности

1. `grep -r "import pytest" Python_test/strategies/ Python_test/conftest.py` — пусто!
2. `grep -r "pytest.skip\|pytest.fixture\|pytest.mark" Python_test/strategies/` — пусто!
3. Каждый тест-файл запускается напрямую: `python test_farrow_pipeline.py` — без ошибок импорта
4. Все тест-классы работают через `TestRunner`
5. `conftest.py` не импортирует pytest, но логика инициализации GPU сохранена

---

## ❌ Что НЕ делать

- НЕ удалять бизнес-логику тестов — только убирать pytest обёртки
- НЕ менять алгоритмы сравнения (FarrowDelay тесты — точные допуски)
- НЕ забывать добавить `setUp()` для создания farrow/scenario/proc
- НЕ использовать `unittest.TestCase` вместо pytest — нам не нужен unittest тоже!
