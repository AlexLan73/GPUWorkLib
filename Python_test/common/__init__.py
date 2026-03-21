"""
common — общая инфраструктура для Python_test/
=================================================

Пакеты:
    result      — TestResult, ValidationResult (value objects)
    configs     — SignalConfig, FilterConfig, HeterodyneConfig (dataclasses)
    validators  — IValidator + DataValidator (Strategy GoF)
    runner      — TestRunner + SkipTest
    reporters   — IReporter + ConsoleReporter / JSONReporter
    gpu_loader  — GPULoader (Singleton) — находит .so один раз
    gpu_context — GPUContextManager (Singleton) — хранит GPU-контекст
    test_base   — TestBase (Template Method)
    plotting    — IPlotter ABC + реализации
    references  — SignalReferences, FilterReferences, StatisticsReferences, FftReferences (DRY)
"""

from .result import TestResult, ValidationResult
from .configs import SignalConfig, FilterConfig, HeterodyneConfig
from .gpu_loader import GPULoader
from .gpu_context import GPUContextManager
from .validators import IValidator, DataValidator
from .runner import TestRunner, SkipTest

# Референсные реализации (NumPy/SciPy эталоны)
from .references import SignalReferences, FilterReferences
from .references import StatisticsReferences, FftReferences

__all__ = [
    "TestResult", "ValidationResult",
    "SignalConfig", "FilterConfig", "HeterodyneConfig",
    "GPULoader", "GPUContextManager",
    "IValidator", "DataValidator",
    "TestRunner", "SkipTest",
    # References (DRY NumPy/SciPy эталоны)
    "SignalReferences", "FilterReferences",
    "StatisticsReferences", "FftReferences",
]
