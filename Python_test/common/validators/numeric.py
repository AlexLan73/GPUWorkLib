"""
numeric.py — числовые валидаторы (Strategy GoF, SRP)
======================================================

Три метрики сравнения, по одной на класс:

    RelativeValidator  — max|a-r| / max|r| < tol
    AbsoluteValidator  — max|a-r|          < tol
    RmseValidator      — rms|a-r| / rms|r| < tol

⚠️ Критично: complex-входы обрабатываются через complex128 (мнимая часть НЕ теряется).
              Real-входы через float64 — безопасно и точно.
              `np.abs(complex)` корректно вычисляет модуль, поэтому все метрики
              одинаково работают и для complex64, и для float32.

⚠️ Strict `<` (не `<=`), правило #10 в TASK_PythonArch_INDEX.
"""

from __future__ import annotations

import numpy as np

from ..result import ValidationResult
from .base import IValidator


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_1d(x) -> np.ndarray:
    """Привести вход к 1D np.ndarray с безопасным dtype.

    Правило:
      * complex-вход → complex128 (мнимая часть СОХРАНЯЕТСЯ).
      * real-вход    → float64.

    Это повторяет поведение старого `DataValidator`, который делал полный
    cast в complex128. Smart-promotion здесь экономит память и работу,
    но сохраняет совместимость для всех существующих тестов.
    """
    arr = np.atleast_1d(np.asarray(x)).ravel()
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    return arr.astype(np.float64)


def _promote(a: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Привести оба массива к общему dtype.

    Если хотя бы один массив complex — оба приводятся к complex128.
    Иначе оба приводятся к float64.
    """
    if np.iscomplexobj(a) or np.iscomplexobj(r):
        return a.astype(np.complex128), r.astype(np.complex128)
    return a.astype(np.float64), r.astype(np.float64)


def _require_reference(validator_name: str, reference) -> None:
    """Fail-fast для comparative-валидаторов без reference."""
    if reference is None:
        raise ValueError(
            f"{validator_name} требует reference (эталон для сравнения), "
            f"получено None."
        )


# ── Валидаторы ──────────────────────────────────────────────────────────────

class RelativeValidator(IValidator):
    """max|actual - ref| / max|ref| < tolerance.

    SRP: только эта метрика. Применение: сигналы, спектры, GEMM, статистика.

    Особенность: если reference ≈ 0 (max|ref| < 1e-15), переключается на
    абсолютный допуск 1e-10 — так поступал старый DataValidator.
    """

    def __init__(self, tolerance: float, name: str = "relative_error"):
        if tolerance is None:
            raise ValueError("tolerance не может быть None")
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference=None, name: str = "") -> ValidationResult:
        _require_reference("RelativeValidator", reference)
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        err   = float(np.max(np.abs(a - r)))
        scale = float(np.max(np.abs(r)))
        metric_name = name or self._name
        if scale < 1e-15:
            return ValidationResult(
                passed=err < 1e-10,
                metric_name=metric_name,
                actual_value=err,
                threshold=1e-10,
                message="(near-zero reference, absolute tolerance)",
            )
        metric = err / scale
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=metric_name,
            actual_value=metric,
            threshold=self._tol,
        )


class AbsoluteValidator(IValidator):
    """max|actual - ref| < tolerance.

    SRP: только абсолютная погрешность. Применение: частоты (Гц), индексы бинов,
    любые величины где нормировка не имеет смысла (или reference может быть 0).

    Complex-безопасен: np.abs(complex) корректно возвращает модуль.
    """

    def __init__(self, tolerance: float, name: str = "absolute_error"):
        if tolerance is None:
            raise ValueError("tolerance не может быть None")
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference=None, name: str = "") -> ValidationResult:
        _require_reference("AbsoluteValidator", reference)
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        metric = float(np.max(np.abs(a - r)))
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=name or self._name,
            actual_value=metric,
            threshold=self._tol,
        )


class RmseValidator(IValidator):
    """rms|actual - ref| / rms|ref| < tolerance.

    SRP: только нормированная среднеквадратичная ошибка.
    Применение: шумные данные, фильтры, статистика где max слишком чувствителен.

    Особенность: если reference ≈ 0 (rms|ref| < 1e-15), переключается на
    абсолютный допуск 1e-10.
    """

    def __init__(self, tolerance: float, name: str = "rmse"):
        if tolerance is None:
            raise ValueError("tolerance не может быть None")
        self._tol = float(tolerance)
        self._name = name

    def validate(self, actual, reference=None, name: str = "") -> ValidationResult:
        _require_reference("RmseValidator", reference)
        a, r = _promote(_to_1d(actual), _to_1d(reference))
        rms_err = float(np.sqrt(np.mean(np.abs(a - r) ** 2)))
        rms_ref = float(np.sqrt(np.mean(np.abs(r) ** 2)))
        metric_name = name or self._name
        if rms_ref < 1e-15:
            return ValidationResult(
                passed=rms_err < 1e-10,
                metric_name=metric_name,
                actual_value=rms_err,
                threshold=1e-10,
                message="(near-zero reference, absolute tolerance)",
            )
        metric = rms_err / rms_ref
        return ValidationResult(
            passed=metric < self._tol,
            metric_name=metric_name,
            actual_value=metric,
            threshold=self._tol,
        )
