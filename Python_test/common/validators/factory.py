"""
factory.py — ValidatorFactory (GRASP Creator)
===============================================

Фабрика валидаторов по строковому ключу.

    ValidatorFactory.create("max_rel", tolerance=0.01)
    ValidatorFactory.create_for_signal(expected_hz=2e6, fs=12e6)

OCP: новые метрики добавляются регистрацией в `_NUMERIC`, без изменения
кода потребителей.
"""

from __future__ import annotations

from .base import IValidator
from .composite import CompositeValidator
from .numeric import AbsoluteValidator, RelativeValidator, RmseValidator
from .signal import FrequencyValidator, PowerValidator


class ValidatorFactory:
    """Создаёт валидатор по метрике-строке.

    Ключи соответствуют старому API `DataValidator`:
        "max_rel" → RelativeValidator
        "abs"     → AbsoluteValidator
        "rmse"    → RmseValidator
    """

    _NUMERIC: dict[str, type[IValidator]] = {
        "max_rel": RelativeValidator,
        "abs":     AbsoluteValidator,
        "rmse":    RmseValidator,
    }

    @classmethod
    def create(cls,
               metric: str = "max_rel",
               tolerance: float = 0.01,
               name: str = "") -> IValidator:
        """Создать простой числовой валидатор.

        Args:
            metric:    "max_rel" | "abs" | "rmse"
            tolerance: допустимый порог (смысл зависит от метрики)
            name:      имя для отчёта
        """
        if metric not in cls._NUMERIC:
            available = list(cls._NUMERIC)
            raise ValueError(
                f"Неизвестная метрика: {metric!r}. Доступные: {available}"
            )
        return cls._NUMERIC[metric](tolerance, name)

    @classmethod
    def create_for_signal(cls,
                          expected_hz: float,
                          fs: float,
                          tolerance_hz: float = 1e3,
                          rel_tolerance: float = 0.01) -> CompositeValidator:
        """Composite для проверки генерируемого сигнала: форма + несущая частота.

        Возвращает CompositeValidator(Relative, Frequency).
        """
        return CompositeValidator(
            RelativeValidator(rel_tolerance, "amplitude"),
            FrequencyValidator(expected_hz, tolerance_hz, fs),
        )

    @classmethod
    def create_for_filter(cls, rel_tolerance: float = 0.01) -> CompositeValidator:
        """Composite для проверки фильтра — нормированный RMSE."""
        return CompositeValidator(
            RmseValidator(rel_tolerance, "filter_rmse"),
        )
