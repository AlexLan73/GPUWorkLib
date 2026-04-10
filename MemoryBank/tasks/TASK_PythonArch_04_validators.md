# TASK_PythonArch_04 — common/validators/ (иерархия)

> **Фаза**: 2 (приоритет MEDIUM)
> **Зависимости**: — (не зависит от Фазы 1, но лучше после TASK_03)
> **Статус**: ✅ DONE 2026-04-09
> **Оценка**: ~2 часа
> **Паттерны**: GoF Strategy, GoF Composite, GRASP Creator, SOLID SRP/OCP/DIP

## ✅ Итог реализации (2026-04-09)

- 6 файлов созданы в `Python_test/common/validators/`: `base.py`, `numeric.py`,
  `signal.py`, `composite.py`, `factory.py`, `__init__.py`.
- Старый `common/validators.py` удалён.
- **DataValidator — настоящий класс** с `.tolerance / .metric / .METRICS`
  (backward-compat без сюрпризов).
- **Complex128-promotion** в numeric-валидаторах — мнимая часть
  complex64 больше НЕ теряется (главный баг ревью).
- Strict `<` во всех валидаторах, включая Frequency/Power.
- Fail-fast: `ValueError` при `reference=None` в comparative-валидаторах.
- `CompositeValidator` возвращает `actual_value=n_pass, threshold=n_total`.
- Smoke-тесты: `Python_test/common/validators/test_smoke.py` — **14/14 PASS**.
- Все 5 существующих consumer-файлов (`strategies/pipeline_step_validator.py`,
  `filters/filter_test_base.py`, `signal_generators/signal_test_base.py`,
  `heterodyne/heterodyne_test_base.py`, `statistics/test_compute_all.py`)
  компилируются; `strategies.pipeline_step_validator` импортируется без ошибок.

---

## 🎯 Цель

Рефакторинг `common/validators.py` → пакет `common/validators/` с иерархией.

**Проблема сейчас**:
- `DataValidator` делает 3 вещи сразу (metric="max_rel"/"abs"/"rmse") — нарушение SRP
- Нет специализированных валидаторов: проверка пика FFT, фазы
- Нет Composite для "все условия AND"
- Тесты зависят напрямую от DataValidator, а не от интерфейса (нарушение DIP)

**Решение**: Иерархия Strategy + Composite + Factory.
`DataValidator` остаётся как backward-compat alias — существующие тесты не ломаются!

---

## 📁 Создаваемые файлы (6 штук)

```
Python_test/common/validators/
├── __init__.py      ← 1. DataValidator backward compat + новый API
├── base.py          ← 2. IValidator (ABC)
├── numeric.py       ← 3. RelativeValidator, AbsoluteValidator, RmseValidator
├── signal.py        ← 4. FrequencyValidator, PhaseValidator, PowerValidator
├── composite.py     ← 5. CompositeValidator (AND)
└── factory.py       ← 6. ValidatorFactory
```

**ВАЖНО**: старый файл `common/validators.py` → удалить после теста.

---

## 📝 Детальное ТЗ

### 2. `common/validators/base.py` — IValidator (ABC)

```python
from abc import ABC, abstractmethod
import numpy as np
from ..result import ValidationResult

class IValidator(ABC):
    """
    GoF Strategy: интерфейс для всех валидаторов.
    SOLID DIP: тесты зависят от IValidator, не от конкретной реализации.

    Два вида валидаторов:
      1. Comparative  — RelativeValidator/AbsoluteValidator/RmseValidator:
                         reference ОБЯЗАТЕЛЕН, иначе ValueError.
      2. Standalone   — FrequencyValidator/PowerValidator:
                         reference игнорируется (можно передать None).

    Такой контракт делает невалидные вызовы громкими (fail-fast), вместо
    молчаливого сравнения «чего-то с None».
    """

    @abstractmethod
    def validate(self,
                 actual: np.ndarray | float | list,
                 reference: np.ndarray | float | list | None = None,
                 name: str = "") -> ValidationResult:
        """
        Выполняет проверку actual против reference (или standalone).

        Args:
            actual:    Фактический результат GPU
            reference: Эталон (NumPy/SciPy), или None для standalone-проверок
            name:      Имя метрики для отчёта

        Returns:
            ValidationResult(passed, metric_name, actual_value, threshold)

        Raises:
            ValueError: если comparative-валидатор получил reference=None
        """

    def __call__(self, actual, reference=None, name="") -> ValidationResult:
        """Позволяет вызывать валидатор как функцию: vr = validator(actual, ref)"""
        return self.validate(actual, reference, name)
```

---

### 3. `common/validators/numeric.py` — Числовые метрики (SRP)

```python
import numpy as np
from .base import IValidator
from ..result import ValidationResult


def _to_1d(x) -> np.ndarray:
    """
    Приводит вход к 1D np.ndarray с безопасным dtype для вычислений.

    Правило:
        - complex вход  → complex128 (сохраняем мнимую часть!)
        - вещественный  → float64
    Это повторяет поведение старого DataValidator:
    он использовал complex128 для всего. float64 здесь тоже ок,
    потому что np.abs(complex) корректно вычисляет модуль.
    """
    arr = np.atleast_1d(np.asarray(x)).ravel()
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    return arr.astype(np.float64)


def _promote(a: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Приводит оба массива к общему dtype (complex128, если хотя бы один complex)."""
    if np.iscomplexobj(a) or np.iscomplexobj(r):
        return a.astype(np.complex128), r.astype(np.complex128)
    return a.astype(np.float64), r.astype(np.float64)


class RelativeValidator(IValidator):
    """
    max|actual - ref| / max|ref| < tolerance.
    SRP: только эта метрика. Применение: сигналы, спектры, GEMM, статистика.

    ⚠️ Используем strict < (не <=) для совместимости с DataValidator.
    ⚠️ Complex входы обрабатываются через complex128 — мнимая часть НЕ теряется.
       Для real-входа используется float64.
    """

    def __init__(self, tolerance: float, name: str = "relative_error"):
        if tolerance is None:
            raise ValueError("tolerance не может быть None")
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        if reference is None:
            raise ValueError(
                "RelativeValidator требует reference (эталон для сравнения)"
            )
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        err   = float(np.max(np.abs(a - r)))
        scale = float(np.max(np.abs(r)))
        if scale < 1e-15:
            # reference ≈ 0 → абсолютный допуск (как в DataValidator)
            return ValidationResult(
                passed=err < 1e-10,
                metric_name=name or self._name,
                actual_value=err,
                threshold=1e-10,
                message="(near-zero reference, absolute tolerance)"
            )
        metric = err / scale
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=name or self._name,
            actual_value=metric,
            threshold=self._tol,
        )


class AbsoluteValidator(IValidator):
    """
    max|actual - ref| < tolerance.
    SRP: только абсолютная погрешность. Применение: частоты (Гц), индексы бинов.
    Complex-безопасен: np.abs(complex) = модуль.
    """

    def __init__(self, tolerance: float, name: str = "absolute_error"):
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        if reference is None:
            raise ValueError(
                "AbsoluteValidator требует reference (эталон для сравнения)"
            )
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        metric = float(np.max(np.abs(a - r)))
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=name or self._name,
            actual_value=metric,
            threshold=self._tol,
        )


class RmseValidator(IValidator):
    """
    rms(|actual - ref|) / rms(|ref|) < tolerance.
    SRP: только RMSE. Применение: шумные данные, фильтры.
    Complex-безопасен: |diff|² корректно считается и для complex.
    """

    def __init__(self, tolerance: float, name: str = "rmse"):
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        if reference is None:
            raise ValueError(
                "RmseValidator требует reference (эталон для сравнения)"
            )
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        rms_err = float(np.sqrt(np.mean(np.abs(a - r) ** 2)))
        rms_ref = float(np.sqrt(np.mean(np.abs(r) ** 2)))
        if rms_ref < 1e-15:
            return ValidationResult(
                passed=rms_err < 1e-10,
                metric_name=name or self._name,
                actual_value=rms_err,
                threshold=1e-10,
                message="(near-zero reference)"
            )
        metric = rms_err / rms_ref
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=name or self._name,
            actual_value=metric,
            threshold=self._tol,
        )
```

---

### 4. `common/validators/signal.py` — Сигнальные метрики

```python
import numpy as np
from .base import IValidator
from ..result import ValidationResult

class FrequencyValidator(IValidator):
    """
    Проверяет что пик FFT-спектра находится в диапазоне
    (expected_hz - tolerance_hz, expected_hz + tolerance_hz).

    Standalone: reference игнорируется.
    Применение: проверка несущей частоты CW/LFM.

    ⚠️ Strict < (правило #10 в TASK_PythonArch_INDEX).
    """

    def __init__(self, expected_hz: float, tolerance_hz: float, fs: float):
        self._expected = float(expected_hz)
        self._tol = float(tolerance_hz)
        self._fs  = float(fs)

    def validate(self, actual: np.ndarray, reference=None,
                 name: str = "peak_freq_hz") -> ValidationResult:
        """
        Args:
            actual:    временной сигнал (complex64) или готовый амп. спектр (float)
            reference: игнорируется (standalone-валидатор)
        """
        arr = np.asarray(actual)
        # Если это комплексный временной сигнал — берём FFT. Если уже спектр
        # (real-массив) — используем как есть.
        if np.iscomplexobj(arr):
            spec = np.abs(np.fft.fft(arr))
        else:
            spec = arr
        n = len(spec)
        freqs = np.fft.fftfreq(n, d=1.0 / self._fs)
        peak_idx  = int(np.argmax(spec))
        actual_hz = float(freqs[peak_idx])
        err = abs(actual_hz - self._expected)
        return ValidationResult(
            passed=err < self._tol,
            metric_name=name,
            actual_value=actual_hz,
            threshold=self._tol,
            message=f"expected={self._expected/1e6:.3f}MHz ±{self._tol/1e3:.1f}kHz, "
                    f"got={actual_hz/1e6:.3f}MHz, err={err/1e3:.2f}kHz",
        )


class PowerValidator(IValidator):
    """
    Проверяет мощность сигнала: |mean(|x|²) - expected_power| / expected_power < tol.
    Применение: проверка амплитуды генераторов.

    ⚠️ Strict < (правило #10 в TASK_PythonArch_INDEX).
    """

    def __init__(self, expected_power: float, tolerance: float = 0.05):
        self._expected = float(expected_power)
        self._tol = float(tolerance)

    def validate(self, actual: np.ndarray, reference=None,
                 name: str = "power") -> ValidationResult:
        # Не касуем в complex64 насильно — np.abs(..)**2 валиден и для real
        arr  = np.asarray(actual)
        power = float(np.mean(np.abs(arr) ** 2))
        rel_err = abs(power - self._expected) / max(self._expected, 1e-10)
        return ValidationResult(
            passed=rel_err < self._tol,
            metric_name=name,
            actual_value=power,
            threshold=self._tol,
            message=f"power={power:.4f}, expected={self._expected:.4f}, "
                    f"rel_err={rel_err:.4f}",
        )
```

---

### 5. `common/validators/composite.py` — CompositeValidator

```python
import numpy as np
from .base import IValidator
from ..result import ValidationResult

class CompositeValidator(IValidator):
    """
    GoF Composite: AND-логика. Все вложенные должны пройти.

    Позволяет строить сложные многоуровневые проверки:
        v = CompositeValidator(
            RelativeValidator(0.01, "amplitude"),
            FrequencyValidator(2e6, 1e3, 12e6),
        )
        result = v.validate(gpu_signal, ref_signal)
        # PASS только если И амплитуда И частота верны

    Fluent API:
        v = CompositeValidator()
        v.add(RelativeValidator(0.01)).add(FrequencyValidator(2e6, 1e3, 12e6))
    """

    def __init__(self, *validators: IValidator):
        self._validators = list(validators)

    def add(self, validator: IValidator) -> "CompositeValidator":
        """Fluent API: добавляет валидатор в цепочку."""
        self._validators.append(validator)
        return self

    def validate(self, actual, reference=None,
                 name: str = "composite") -> ValidationResult:
        if not self._validators:
            return ValidationResult(
                passed=True, metric_name=name,
                actual_value=0.0, threshold=0.0,
                message="(пустой CompositeValidator)"
            )

        results = []
        for v in self._validators:
            try:
                r = v.validate(actual, reference, name)
            except Exception as e:
                r = ValidationResult(
                    passed=False, metric_name=name,
                    actual_value=float("nan"), threshold=0.0,
                    message=f"Ошибка в {v.__class__.__name__}: {e}"
                )
            results.append(r)

        passed = all(r.passed for r in results)
        msgs   = " | ".join(str(r) for r in results)

        # Гетерогенные метрики нельзя агрегировать напрямую (Гц vs безразмерные),
        # поэтому actual_value = "сколько пройдено", threshold = "сколько всего".
        # Это даёт однозначный отчёт для Composite.
        n_pass  = sum(1 for r in results if r.passed)
        n_total = len(results)

        return ValidationResult(
            passed=passed,
            metric_name=name,
            actual_value=float(n_pass),
            threshold=float(n_total),
            message=msgs,
        )

    def __len__(self) -> int:
        return len(self._validators)
```

---

### 6. `common/validators/factory.py` — ValidatorFactory

```python
from .base import IValidator
from .numeric import RelativeValidator, AbsoluteValidator, RmseValidator
from .signal import FrequencyValidator, PowerValidator
from .composite import CompositeValidator

class ValidatorFactory:
    """
    GRASP Creator: создаёт нужный тип валидатора по строковому ключу.
    OCP: новые метрики добавляются регистрацией, не изменением этого класса.
    """

    # Базовые метрики
    _NUMERIC = {
        "max_rel":  RelativeValidator,
        "abs":      AbsoluteValidator,
        "rmse":     RmseValidator,
    }

    @classmethod
    def create(cls, metric: str = "max_rel",
               tolerance: float = 0.01,
               name: str = "") -> IValidator:
        """
        Создаёт простой валидатор по метрике.

        Args:
            metric:    "max_rel" | "abs" | "rmse"
            tolerance: порог (зависит от метрики)
            name:      имя для отчёта

        Returns:
            IValidator
        """
        if metric not in cls._NUMERIC:
            available = list(cls._NUMERIC)
            raise ValueError(
                f"Неизвестная метрика: '{metric}'. Доступные: {available}"
            )
        return cls._NUMERIC[metric](tolerance, name)

    @classmethod
    def create_for_signal(cls,
                          expected_hz: float,
                          fs: float,
                          tolerance_hz: float = 1e3,
                          rel_tolerance: float = 0.01) -> CompositeValidator:
        """
        Комплексная проверка сигнала: корреляция + пик частоты.
        Удобный метод для тестов генераторов.
        """
        return CompositeValidator(
            RelativeValidator(rel_tolerance, "amplitude"),
            FrequencyValidator(expected_hz, tolerance_hz, fs),
        )

    @classmethod
    def create_for_filter(cls, rel_tolerance: float = 0.01) -> CompositeValidator:
        """
        Комплексная проверка фильтра: RMSE.
        """
        return CompositeValidator(
            RmseValidator(rel_tolerance, "filter_rmse"),
        )
```

---

### 1. `common/validators/__init__.py` — BACKWARD COMPAT

```python
"""
Пакет валидаторов.

Backward compatibility: DataValidator остаётся как НАСТОЯЩИЙ класс
с публичными атрибутами .tolerance / .metric / .METRICS, чтобы не ломать
внешний код, который к ним обращается. Внутри делегирует новым классам.

Старый API (работает без изменений):
    from common import DataValidator
    v = DataValidator(tolerance=0.01, metric="max_rel")
    v.tolerance   # 0.01
    v.metric      # "max_rel"
    DataValidator.METRICS   # ("max_rel", "abs", "rmse")

Новый API (предпочтительный):
    from common.validators import RelativeValidator, CompositeValidator
    from common.validators import ValidatorFactory
"""

from .base import IValidator
from .numeric import RelativeValidator, AbsoluteValidator, RmseValidator
from .signal import FrequencyValidator, PowerValidator
from .composite import CompositeValidator
from .factory import ValidatorFactory
from ..result import ValidationResult


# ── BACKWARD COMPAT ───────────────────────────────────────────────────────────
class DataValidator(IValidator):
    """
    [Backward-compat] Универсальный валидатор, метрика задаётся при создании.

    Реализован как тонкий класс-обёртка над RelativeValidator/AbsoluteValidator/
    RmseValidator. Сохраняет публичный API старого common/validators.py:

        v = DataValidator(tolerance=0.01, metric="max_rel")
        v.tolerance  # 0.01         ← публичный атрибут
        v.metric     # "max_rel"    ← публичный атрибут
        DataValidator.METRICS       # tuple валидных метрик

    Новый код должен использовать RelativeValidator / ValidatorFactory напрямую.
    """

    METRICS = ("max_rel", "abs", "rmse")

    def __init__(self, tolerance: float,
                 metric: str = "max_rel",
                 name: str = ""):
        if metric not in self.METRICS:
            raise ValueError(
                f"metric должен быть одним из {self.METRICS}, получено: {metric!r}"
            )
        self.tolerance = float(tolerance)
        self.metric = metric
        self._default_name = name
        self._impl: IValidator = ValidatorFactory.create(metric, tolerance, name)

    def validate(self, actual, reference,
                 name: str = "") -> ValidationResult:
        return self._impl.validate(
            actual, reference, name or self._default_name
        )


__all__ = [
    # Новый API
    "IValidator",
    "RelativeValidator", "AbsoluteValidator", "RmseValidator",
    "FrequencyValidator", "PowerValidator",
    "CompositeValidator",
    "ValidatorFactory",
    # Backward compat (реальный класс, не функция!)
    "DataValidator",
]
```

---

## 📋 Что обновить после создания файлов

### `common/__init__.py`:
```python
# БЫЛО:
from .validators import IValidator, DataValidator

# СТАЛО:
from .validators import (
    IValidator, DataValidator,  # backward compat
    RelativeValidator, AbsoluteValidator, RmseValidator,
    FrequencyValidator, PowerValidator,
    CompositeValidator, ValidatorFactory,
)
```

### `common/validators.py` → **удалить** (после убеждения что тесты работают)

---

## ✅ Критерии завершения

- [ ] Все 6 файлов созданы
- [ ] `from common import DataValidator` работает (старый код не ломается)
- [ ] `DataValidator` — настоящий класс, `v.tolerance / v.metric / v.METRICS` доступны
- [ ] `DataValidator(tolerance=1e-6, metric="max_rel").validate(complex_a, complex_r)` — **мнимая часть НЕ теряется**
- [ ] `from common.validators import RelativeValidator, CompositeValidator` работает
- [ ] `RelativeValidator(0.01).validate(gpu, ref)` → ValidationResult (поддерживает complex64)
- [ ] `RelativeValidator(0.01).validate(x, None)` → ValueError (reference обязателен)
- [ ] `FrequencyValidator(2e6, 1e3, 12e6).validate(cw_signal)` → PASS (strict `<`)
- [ ] `CompositeValidator(v1, v2).validate(a, r)` → `actual_value = n_pass`, `threshold = n_total`
- [ ] `ValidatorFactory.create("max_rel", 0.01)` → RelativeValidator
- [ ] `ValidatorFactory.create_for_signal(2e6, 12e6)` → CompositeValidator(2 шт)
- [ ] `common/validators.py` старый файл — удалён (после теста всего)
- [ ] **Прогнать ВСЕ существующие тесты**, которые используют DataValidator:
      - [ ] `python Python_test/strategies/test_strategies_pipeline.py`
      - [ ] `python Python_test/filters/test_*.py` (все)
      - [ ] `python Python_test/signal_generators/test_*.py` (все)
      - [ ] `python Python_test/statistics/test_compute_all.py`
      - Ни один не должен упасть по причине изменения валидаторов.

## 🧪 Быстрая проверка

```python
# python Python_test/common/validators/test_validators_smoke.py
import sys
from pathlib import Path
# Bootstrap: добавляем Python_test/ в sys.path (файл живёт внутри пакета common/validators/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from common.runner import TestRunner
from common.result import TestResult, ValidationResult

class TestValidatorsSmoke:
    """Smoke-тест валидаторов — работает без GPU."""

    def test_backward_compat_real(self):
        """DataValidator (старый API) работает с real-массивами."""
        from common import DataValidator
        result = TestResult(test_name="backward_compat_real")
        v = DataValidator(tolerance=0.01, metric="max_rel")
        # Публичные атрибуты из старого API должны быть доступны
        assert_ok = (v.tolerance == 0.01 and v.metric == "max_rel")
        a = np.ones(100, dtype=np.float32)
        r = np.ones(100, dtype=np.float32) * 1.005
        vr = v.validate(a, r, "rel_0.5%")
        result.add(vr)
        result.metadata["public_attrs_ok"] = assert_ok
        return result

    def test_backward_compat_complex(self):
        """CRITICAL: DataValidator не должен терять мнимую часть complex64."""
        from common import DataValidator
        result = TestResult(test_name="backward_compat_complex")
        v = DataValidator(tolerance=1e-6, metric="max_rel")
        # Только мнимая часть различается — max_rel должен это увидеть
        a = np.ones(100, dtype=np.complex64) * (1 + 0j)
        r = np.ones(100, dtype=np.complex64) * (1 + 0.5j)
        vr = v.validate(a, r, "complex_diff")
        # passed должно быть False: разница 0.5 в imag-части — это ~50% rel_err
        vr_inverted = ValidationResult(
            passed=(not vr.passed),   # инвертируем: ждали FAIL — получили PASS-на-FAIL
            metric_name="complex_im_detected",
            actual_value=vr.actual_value,
            threshold=1e-6,
            message=f"исходный: {vr}",
        )
        result.add(vr_inverted)
        return result

    def test_relative_validator_complex(self):
        """RelativeValidator должен сохранять мнимую часть complex128."""
        from common.validators import RelativeValidator
        result = TestResult(test_name="relative_complex")
        v = RelativeValidator(1e-6)
        a = np.array([1 + 0j, 2 + 0j], dtype=np.complex64)
        vr = v.validate(a, a, "self")   # сравнение с собой → PASS
        result.add(vr)
        return result

    def test_composite_validator(self):
        """CompositeValidator: RelativeValidator + FrequencyValidator."""
        from common.validators import (
            CompositeValidator, RelativeValidator, FrequencyValidator,
        )
        result = TestResult(test_name="composite_validator")
        fs = 12e6
        n  = 4096
        f0 = 2e6
        t  = np.arange(n) / fs
        cw = np.exp(2j * np.pi * f0 * t).astype(np.complex64)
        v_cmp = CompositeValidator(
            RelativeValidator(1e-3),
            FrequencyValidator(expected_hz=f0, tolerance_hz=1e3, fs=fs),
        )
        vr = v_cmp.validate(cw, cw)   # сравнение с самим собой → PASS
        result.add(vr)
        return result

    def test_factory(self):
        """ValidatorFactory создаёт правильные типы."""
        from common.validators import ValidatorFactory, RelativeValidator
        result = TestResult(test_name="factory")
        v = ValidatorFactory.create("max_rel", 0.01)
        ok = isinstance(v, RelativeValidator)
        result.add(ValidationResult(
            passed=ok,
            metric_name="factory_type",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
        ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestValidatorsSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 2*
