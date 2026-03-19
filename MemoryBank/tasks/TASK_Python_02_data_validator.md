# TASK Python-02: common/validators.py — DataValidator

**Статус**: 🔲 TODO
**Приоритет**: 🔴 КРИТИЧЕСКИЙ — используется во всех тестах
**Файл**: `Python_test/common/validators.py` (переписать!)
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Заменить 4 разных валидатора (`NumericValidator`, `RMSEValidator`, `SpectralValidator`, `EnergyValidator`) на **один универсальный** `DataValidator`.

Все данные — скаляр/вектор/матрица — сводятся к одной логике через `np.atleast_1d().ravel()`.

---

## 📖 Что прочитать перед реализацией

- `Python_test/common/validators.py` — ТЕКУЩИЙ КОД (прочитать внимательно!)
- `Python_test/common/result.py` — классы `ValidationResult`, `TestResult`
- `MemoryBank/specs/python_test_refactoring.md` — раздел "Типы данных"

---

## ⚠️ Типы данных в проекте (везде float32!)

| Данные | NumPy тип | Примечание |
|--------|-----------|-----------|
| Сигнал d_S, d_W, d_X, d_spectrum | `np.complex64` | complex<float> |
| Магнитуды d_magnitudes | `np.float32` | |
| Statistics.mean | `np.complex64` | комплексное среднее |
| Statistics.variance, std_dev, mean_mag | `np.float32` | от |z| |
| Частоты refined_freq_hz | `np.float32` | |
| dynamic_range_dB | `np.float32` | |

**DataValidator должен корректно работать со всеми этими типами!**

---

## 🏗️ Реализация

### Сохранить из старого кода

`IValidator` (ABC) — **оставить без изменений**:
```python
class IValidator(ABC):
    @abstractmethod
    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        ...
```

### Удалить из старого кода

Полностью удалить (заменяются DataValidator):
- `NumericValidator`
- `RMSEValidator`
- `SpectralValidator`
- `EnergyValidator`

### Создать `DataValidator`

```python
class DataValidator(IValidator):
    """Универсальный валидатор: скаляр / вектор / матрица.

    Strategy (GoF) — метрика выбирается при создании.
    Information Expert (GRASP) — знает как сравнивать данные.

    Все данные приводятся через np.atleast_1d(x).ravel()
    к одномерному массиву, затем применяется одна из трёх метрик.

    Метрики:
        "max_rel" → max(|actual - ref|) / max(|ref|) < tolerance
                    Для сигналов, спектров, статистики.
                    При ref ≈ 0 переключается на абсолютный допуск 1e-10.

        "abs"     → max(|actual - ref|) < tolerance
                    Для частот в Гц, индексов бинов FFT.
                    reference может быть нулём — не нормируется.

        "rmse"    → rms(|actual - ref|) / rms(|ref|) < tolerance
                    Для шумных данных где нужна среднеквадратичная метрика.
                    При ref ≈ 0 переключается на абсолютный допуск 1e-10.

    Usage:
        # Сравнить комплексные векторы (сигнал):
        v = DataValidator(tolerance=0.01, metric="max_rel")
        r = v.validate(gpu_mean, numpy_mean, name="mean_stats")
        print(r)   # [PASS] mean_stats: 0.003421 (tol=0.01)

        # Сравнить частоту пика (абсолютная погрешность в Гц):
        v = DataValidator(tolerance=50e3, metric="abs")
        r = v.validate(refined_freq_hz, expected_f0=2e6, name="peak_freq")

        # Сравнить матрицы float32:
        v = DataValidator(tolerance=0.001, metric="max_rel")
        r = v.validate(gemm_output_gpu, gemm_output_numpy, name="gemm")
    """

    METRICS = ("max_rel", "abs", "rmse")

    def __init__(self, tolerance: float,
                 metric: str = "max_rel",
                 name: str = ""):
        """
        Args:
            tolerance: допустимый порог (зависит от metric)
            metric:    "max_rel" | "abs" | "rmse"
            name:      имя метрики для отчёта (если не передано в validate())
        """
        if metric not in self.METRICS:
            raise ValueError(f"metric должен быть одним из {self.METRICS}, получено: {metric!r}")
        self.tolerance = tolerance
        self.metric = metric
        self._default_name = name

    def validate(self, actual, reference,
                 name: str = "") -> ValidationResult:
        """Сравнить actual с reference.

        Args:
            actual:    GPU-результат (скаляр, list, np.ndarray любой формы)
            reference: эталон (скаляр, list, np.ndarray любой формы)
            name:      имя метрики для ValidationResult (переопределяет self._default_name)

        Returns:
            ValidationResult

        Note:
            Оба аргумента приводятся к complex128 для вычислений (точность),
            но исходные типы могут быть float32/complex64 — это нормально.
        """
        metric_name = name or self._default_name or self.metric
        # Привести к 1D complex128 для вычислений
        a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.complex128)
        r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.complex128)

        if self.metric == "max_rel":
            return self._max_rel(a, r, metric_name)
        elif self.metric == "abs":
            return self._abs(a, r, metric_name)
        elif self.metric == "rmse":
            return self._rmse(a, r, metric_name)

    # ── Приватные методы вычисления метрик ──────────────────────────────────

    def _max_rel(self, a, r, name) -> ValidationResult:
        """max(|a-r|) / max(|r|) < tolerance"""
        diff = np.abs(a - r)
        ref_norm = np.max(np.abs(r))
        if ref_norm < 1e-15:
            # reference ≈ 0 → абсолютный допуск
            err = float(np.max(diff))
            return ValidationResult(
                passed=err < 1e-10,
                metric_name=name,
                actual_value=err,
                threshold=1e-10,
                message="(near-zero reference, using absolute tolerance)"
            )
        err = float(np.max(diff) / ref_norm)
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )

    def _abs(self, a, r, name) -> ValidationResult:
        """max(|a-r|) < tolerance"""
        err = float(np.max(np.abs(a - r)))
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )

    def _rmse(self, a, r, name) -> ValidationResult:
        """rms(|a-r|) / rms(|r|) < tolerance"""
        diff = a - r
        rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        ref_rms = float(np.sqrt(np.mean(np.abs(r) ** 2)))
        if ref_rms < 1e-15:
            return ValidationResult(
                passed=rmse < 1e-10,
                metric_name=name,
                actual_value=rmse,
                threshold=1e-10,
                message="(near-zero reference)"
            )
        err = rmse / ref_rms
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )
```

---

## 📋 Итоговая структура файла

```python
# validators.py — DataValidator (Strategy GoF)
# Один универсальный класс вместо 6 специализированных

from abc import ABC, abstractmethod
import numpy as np
from .result import ValidationResult

class IValidator(ABC):          # ← оставить без изменений
    ...

class DataValidator(IValidator): # ← новый, заменяет всё остальное
    ...
```

---

## ✅ Критерии готовности

1. Файл `validators.py` содержит только `IValidator` и `DataValidator`
2. Старые классы (`NumericValidator`, `RMSEValidator` и т.д.) — **удалены**
3. `DataValidator` работает корректно с:
   - `np.complex64` вектором (статистика mean)
   - `np.float32` вектором (variance, std)
   - `np.complex64` матрицей (GEMM output, spectrum)
   - Python `float` скаляром (refined_freq_hz)
   - Python `int` скаляром (peak_bin)
4. Метрика `"max_rel"`: при `ref ≈ 0` → абсолютный допуск 1e-10
5. Метрика `"abs"`: не нормирует, просто `max(|a-r|)`
6. При неверной метрике — `ValueError` с понятным сообщением
7. `validate()` принимает необязательный аргумент `name` (переопределяет default)
8. Обновить `Python_test/common/__init__.py`:
   ```python
   from .validators import IValidator, DataValidator
   ```

---

## ❌ Что НЕ делать

- НЕ оставлять `NumericValidator` etc. — даже закомментированными
- НЕ кастить к float64 заранее в `__init__` — кастить только внутри `validate()`
- НЕ добавлять новые метрики без обсуждения
- НЕ менять `IValidator` (другие классы могут его использовать)
- НЕ трогать `result.py`

---

## 🔍 Где используется (нужно обновить после создания DataValidator)

После создания DataValidator нужно будет найти и заменить все упоминания старых валидаторов:
```bash
grep -r "NumericValidator\|RMSEValidator\|SpectralValidator\|EnergyValidator" Python_test/
```
Это задача TASK_Python_07 (замена в тестах).
