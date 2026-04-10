"""
common.validators — пакет валидаторов
=======================================

Backward compatibility: `DataValidator` остаётся НАСТОЯЩИМ классом
с публичными атрибутами ``.tolerance``, ``.metric`` и класс-атрибутом
``.METRICS``. Существующие тесты работают без изменений.

Старый API (работает):

    from common.validators import DataValidator
    v = DataValidator(tolerance=0.01, metric="max_rel")
    v.tolerance           # 0.01
    v.metric              # "max_rel"
    DataValidator.METRICS # ("max_rel", "abs", "rmse")

Новый API (предпочтительный):

    from common.validators import RelativeValidator, CompositeValidator
    from common.validators import ValidatorFactory
    v = RelativeValidator(0.01).validate(gpu_out, ref_out)
"""

from __future__ import annotations

from ..result import ValidationResult
from .base import IValidator
from .composite import CompositeValidator
from .factory import ValidatorFactory
from .numeric import AbsoluteValidator, RelativeValidator, RmseValidator
from .signal import FrequencyValidator, PowerValidator


# ── BACKWARD COMPAT ─────────────────────────────────────────────────────────

class DataValidator(IValidator):
    """[Backward-compat] Универсальный валидатор, метрика задаётся при создании.

    Реализован как тонкая обёртка поверх RelativeValidator/AbsoluteValidator/
    RmseValidator. Сохраняет публичный API старого `common/validators.py`:

        v = DataValidator(tolerance=0.01, metric="max_rel")
        v.tolerance  # 0.01         ← публичный атрибут
        v.metric     # "max_rel"    ← публичный атрибут
        DataValidator.METRICS       # tuple валидных метрик

    Новый код должен использовать `RelativeValidator` / `ValidatorFactory`
    напрямую.
    """

    METRICS: tuple[str, ...] = ("max_rel", "abs", "rmse")

    def __init__(self,
                 tolerance: float,
                 metric: str = "max_rel",
                 name: str = ""):
        if metric not in self.METRICS:
            raise ValueError(
                f"metric должен быть одним из {self.METRICS}, "
                f"получено: {metric!r}"
            )
        self.tolerance = float(tolerance)
        self.metric = metric
        self._default_name = name
        self._impl: IValidator = ValidatorFactory.create(metric, tolerance, name)

    def validate(self, actual, reference=None,
                 name: str = "") -> ValidationResult:
        return self._impl.validate(actual, reference, name or self._default_name)


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
