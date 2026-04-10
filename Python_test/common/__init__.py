"""
common — общая инфраструктура для Python_test/
=================================================

Пакеты:
    result      — TestResult, ValidationResult (value objects)
    configs     — SignalConfig, FilterConfig, HeterodyneConfig (dataclasses)
    validators  — IValidator + DataValidator + новая иерархия (Strategy GoF)
    runner      — TestRunner + SkipTest
    reporters   — IReporter + ConsoleReporter / JSONReporter
    gpu_loader  — GPULoader (Singleton) — находит .so один раз
    gpu_context — GPUContextManager (Singleton) — хранит GPU-контекст
    test_base   — TestBase (Template Method)
    plotting    — IPlotter ABC + SpectrumPlotter/TimePlotter + PlotterFactory
    references  — SignalReferences, FilterReferences, StatisticsReferences, FftReferences (DRY)
    io          — ResultStore + NumpyStore + JsonStore (Repository)
"""

from .result import TestResult, ValidationResult
from .configs import SignalConfig, FilterConfig, HeterodyneConfig
from .gpu_loader import GPULoader
from .gpu_context import GPUContextManager
from .runner import TestRunner, SkipTest

# Validators — новая иерархия + backward compat
from .validators import (
    IValidator,
    DataValidator,                # backward compat (настоящий класс)
    RelativeValidator, AbsoluteValidator, RmseValidator,
    FrequencyValidator, PowerValidator,
    CompositeValidator,
    ValidatorFactory,
)

# Референсные реализации (NumPy/SciPy эталоны)
from .references import (
    SignalReferences, FilterReferences,
    StatisticsReferences, FftReferences,
)

# I/O слой (Repository pattern)
from .io import ResultStore, NumpyStore, JsonStore, IDataStore

__all__ = [
    "TestResult", "ValidationResult",
    "SignalConfig", "FilterConfig", "HeterodyneConfig",
    "GPULoader", "GPUContextManager",
    "TestRunner", "SkipTest",
    # Validators
    "IValidator", "DataValidator",
    "RelativeValidator", "AbsoluteValidator", "RmseValidator",
    "FrequencyValidator", "PowerValidator",
    "CompositeValidator", "ValidatorFactory",
    # References
    "SignalReferences", "FilterReferences",
    "StatisticsReferences", "FftReferences",
    # I/O
    "ResultStore", "NumpyStore", "JsonStore", "IDataStore",
]
