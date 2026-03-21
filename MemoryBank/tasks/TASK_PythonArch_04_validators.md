# TASK_PythonArch_04 — common/validators/ (иерархия)

> **Фаза**: 2 (приоритет MEDIUM)
> **Зависимости**: — (не зависит от Фазы 1, но лучше после TASK_03)
> **Статус**: ⬜ TODO
> **Оценка**: ~2 часа
> **Паттерны**: GoF Strategy, GoF Composite, GRASP Creator, SOLID SRP/OCP/DIP

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

class RelativeValidator(IValidator):
    """
    max|actual - ref| / max|ref| < tolerance.
    SRP: только эта метрика. Применение: сигналы, спектры.

    ⚠️ Используем strict < (не <=) для совместимости с DataValidator.
    ⚠️ Вычисления в float64 для точности (GPU данные = float32).
    """

    def __init__(self, tolerance: float, name: str = "relative_error"):
        self._tol = tolerance
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.float64)
        r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.float64)
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
    """

    def __init__(self, tolerance: float, name: str = "absolute_error"):
        self._tol = tolerance
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.float64)
        r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.float64)
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
    """

    def __init__(self, tolerance: float, name: str = "rmse"):
        self._tol = tolerance
        self._name = name

    def validate(self, actual, reference, name="") -> ValidationResult:
        a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.float64)
        r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.float64)
        rms_err = float(np.sqrt(np.mean(np.abs(a - r)**2)))
        rms_ref = float(np.sqrt(np.mean(np.abs(r)**2)))
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
    [expected_hz - tolerance_hz, expected_hz + tolerance_hz].

    Standalone: reference не нужен.
    Применение: проверка несущей частоты CW/LFM.
    """

    def __init__(self, expected_hz: float, tolerance_hz: float, fs: float):
        self._expected = expected_hz
        self._tol = tolerance_hz
        self._fs  = fs

    def validate(self, actual: np.ndarray, reference=None,
                 name: str = "peak_freq_hz") -> ValidationResult:
        """
        Args:
            actual: сигнал или спектр (complex64/float32, 1D)
        """
        arr = np.asarray(actual)
        # Если это временной сигнал — берём FFT
        if np.iscomplexobj(arr):
            spec = np.abs(np.fft.fft(arr))
        else:
            spec = arr
        n = len(spec)
        freqs = np.fft.fftfreq(n, d=1.0/self._fs)
        peak_idx  = int(np.argmax(spec))
        actual_hz = float(freqs[peak_idx])
        err = abs(actual_hz - self._expected)
        return ValidationResult(
            passed=err <= self._tol,
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
    """

    def __init__(self, expected_power: float, tolerance: float = 0.05):
        self._expected = expected_power
        self._tol = tolerance

    def validate(self, actual: np.ndarray, reference=None,
                 name: str = "power") -> ValidationResult:
        arr  = np.asarray(actual, dtype=np.complex64)
        power = float(np.mean(np.abs(arr)**2))
        rel_err = abs(power - self._expected) / max(self._expected, 1e-10)
        return ValidationResult(
            passed=rel_err <= self._tol,
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

        # Worst = первый FAILED (нормированное сравнение бессмысленно
        # для гетерогенных метрик: Гц vs безразмерные ошибки).
        failed = [r for r in results if not r.passed]
        representative = failed[0] if failed else results[0]

        return ValidationResult(
            passed=passed,
            metric_name=name,
            actual_value=representative.actual_value,
            threshold=representative.threshold,
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

Backward compatibility: DataValidator остаётся как функция-alias.
Существующие тесты работают без изменений!

Старый API (работает):
    from common import DataValidator
    v = DataValidator(tolerance=0.01, metric="max_rel")

Новый API (предпочтительный):
    from common.validators import RelativeValidator, CompositeValidator
    from common.validators import ValidatorFactory
"""

# Новый API
from .base import IValidator
from .numeric import RelativeValidator, AbsoluteValidator, RmseValidator
from .signal import FrequencyValidator, PowerValidator
from .composite import CompositeValidator
from .factory import ValidatorFactory

# ── BACKWARD COMPAT ───────────────────────────────────────────────────────────
# DataValidator(tolerance, metric, name) → IValidator
# Существующие тесты НЕ трогаем.
def DataValidator(tolerance: float, metric: str = "max_rel",
                  name: str = "") -> IValidator:
    """
    [DEPRECATED] Используй ValidatorFactory.create() или конкретный класс.
    Оставлен для backward compatibility.
    """
    return ValidatorFactory.create(metric, tolerance, name)

__all__ = [
    # Новый API
    "IValidator",
    "RelativeValidator", "AbsoluteValidator", "RmseValidator",
    "FrequencyValidator", "PowerValidator",
    "CompositeValidator",
    "ValidatorFactory",
    # Backward compat
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
- [ ] `DataValidator(0.01, "max_rel")` → RelativeValidator(0.01)
- [ ] `from common.validators import RelativeValidator, CompositeValidator` работает
- [ ] `RelativeValidator(0.01).validate(gpu, ref)` → ValidationResult
- [ ] `FrequencyValidator(2e6, 1e3, 12e6).validate(cw_signal)` → PASS
- [ ] `CompositeValidator(v1, v2).validate(a, r)` → AND результат
- [ ] `ValidatorFactory.create("max_rel", 0.01)` → RelativeValidator
- [ ] `ValidatorFactory.create_for_signal(2e6, 12e6)` → CompositeValidator(2 шт)
- [ ] `common/validators.py` старый файл — удалён (после теста всего)
- [ ] Запустить существующие тесты — всё должно работать без изменений

## 🧪 Быстрая проверка

```python
# python Python_test/common/validators/test_validators_smoke.py
import numpy as np
from common.runner import TestRunner
from common.result import TestResult, ValidationResult

class TestValidatorsSmoke:
    """Smoke-тест валидаторов — работает без GPU."""

    def test_backward_compat(self):
        """DataValidator (старый API) должен работать без изменений."""
        from common import DataValidator
        result = TestResult(test_name="backward_compat")
        v = DataValidator(tolerance=0.01, metric="max_rel")
        a = np.ones(100, dtype=np.float32)
        r = np.ones(100, dtype=np.float32) * 1.005
        vr = v.validate(a, r, "test")
        result.add(vr)
        return result

    def test_composite_validator(self):
        """CompositeValidator: RelativeValidator + FrequencyValidator."""
        from common.validators import CompositeValidator, RelativeValidator, FrequencyValidator
        from common.references import SignalReferences

        result = TestResult(test_name="composite_validator")
        cw = SignalReferences.cw(12e6, 4096, 2e6)
        v_cmp = CompositeValidator(
            RelativeValidator(0.001),
            FrequencyValidator(2e6, 1e3, 12e6),
        )
        vr = v_cmp.validate(cw, cw)  # validate against itself — should PASS
        result.add(vr)
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestValidatorsSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 2*
