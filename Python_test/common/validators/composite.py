"""
composite.py — CompositeValidator (GoF Composite)
===================================================

AND-агрегатор: все вложенные валидаторы должны пройти, чтобы Composite был PASS.

Сценарии использования:

    # Одиночный вызов через конструктор:
    v = CompositeValidator(
        RelativeValidator(0.01, "amplitude"),
        FrequencyValidator(2e6, 1e3, 12e6),
    )
    vr = v.validate(gpu_signal, ref_signal)
    # PASS только если И амплитуда И частота верны.

    # Fluent-стиль:
    v = CompositeValidator()
    v.add(RelativeValidator(0.01)).add(FrequencyValidator(2e6, 1e3, 12e6))
"""

from __future__ import annotations

from ..result import ValidationResult
from .base import IValidator


class CompositeValidator(IValidator):
    """AND-логика: passed = all(child.passed for child in validators).

    Поля результата:
        actual_value = сколько вложенных валидаторов прошли
        threshold    = сколько всего вложенных валидаторов
        message      = соединение str-представлений всех дочерних результатов

    Такая агрегация однозначно интерпретируется, даже если метрики разные
    (Гц vs безразмерные ошибки — их нельзя нормировать друг к другу).
    """

    def __init__(self, *validators: IValidator):
        self._validators: list[IValidator] = list(validators)

    def add(self, validator: IValidator) -> "CompositeValidator":
        """Добавить валидатор в цепочку (fluent API)."""
        self._validators.append(validator)
        return self

    def validate(self, actual, reference=None,
                 name: str = "composite") -> ValidationResult:
        # Fail-fast: пустой composite почти всегда — забытый add().
        # Молчаливый PASS скрывает этот баг. Согласовано с общей философией
        # "громкие ошибки" (_require_reference, PowerValidator и др.).
        if not self._validators:
            raise ValueError(
                "CompositeValidator пуст — добавьте валидаторы через "
                "конструктор CompositeValidator(v1, v2, …) или .add()."
            )

        results: list[ValidationResult] = []
        for v in self._validators:
            try:
                r = v.validate(actual, reference, name)
            except Exception as exc:  # pragma: no cover - защита от падений
                r = ValidationResult(
                    passed=False,
                    metric_name=name,
                    actual_value=float("nan"),
                    threshold=0.0,
                    message=f"Ошибка в {v.__class__.__name__}: {exc}",
                )
            results.append(r)

        passed = all(r.passed for r in results)
        msgs   = " | ".join(str(r) for r in results)
        n_pass = sum(1 for r in results if r.passed)
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
