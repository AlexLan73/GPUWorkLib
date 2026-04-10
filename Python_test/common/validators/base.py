"""
base.py — IValidator (Strategy interface)
==========================================

GoF Strategy: единый интерфейс для всех валидаторов.
SOLID DIP: тесты зависят от IValidator, не от конкретной реализации.

Два вида валидаторов:
  1. Comparative  — RelativeValidator/AbsoluteValidator/RmseValidator:
                    reference ОБЯЗАТЕЛЕН, иначе ValueError.
  2. Standalone   — FrequencyValidator/PowerValidator:
                    reference игнорируется (можно передать None).

Такой контракт делает невалидные вызовы громкими (fail-fast), вместо
молчаливого сравнения «чего-то с None».
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..result import ValidationResult


class IValidator(ABC):
    """Абстрактный валидатор — Strategy interface."""

    @abstractmethod
    def validate(self,
                 actual,
                 reference=None,
                 name: str = "") -> ValidationResult:
        """Выполнить проверку actual против reference (или standalone).

        Args:
            actual:    Фактический результат GPU (scalar / list / np.ndarray)
            reference: Эталон (NumPy/SciPy); None для standalone-проверок.
            name:      Имя метрики для отчёта (опционально).

        Returns:
            ValidationResult(passed, metric_name, actual_value, threshold)

        Raises:
            ValueError: если comparative-валидатор получил reference=None.
        """

    def __call__(self, actual, reference=None, name: str = "") -> ValidationResult:
        """Позволяет вызывать валидатор как функцию: ``vr = validator(actual, ref)``."""
        return self.validate(actual, reference, name)
