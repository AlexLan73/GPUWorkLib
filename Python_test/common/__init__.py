"""
common — общая инфраструктура для Python_test/
=================================================

Пакеты:
    result      — TestResult, ValidationResult (value objects)
    configs     — SignalConfig, FilterConfig (dataclasses)
    validators  — IValidator + NumericValidator / SpectralValidator / EnergyValidator
    reporters   — IReporter + ConsoleReporter / JSONReporter
    gpu_loader  — GPULoader (Singleton) — находит .so один раз
    gpu_context — GPUContextManager (Singleton) — хранит GPU-контекст
    test_base   — TestBase (Template Method)
    plotting    — IPlotter ABC + реализации
"""

from .result import TestResult, ValidationResult
from .configs import SignalConfig, FilterConfig
from .gpu_loader import GPULoader
from .gpu_context import GPUContextManager

__all__ = [
    "TestResult", "ValidationResult",
    "SignalConfig", "FilterConfig",
    "GPULoader", "GPUContextManager",
]
