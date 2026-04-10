"""
result.py — Value Objects для результатов тестов
==================================================

Value Objects (GoF) — неизменяемые объекты, идентифицируются по значению.

Classes:
    ValidationResult — результат одной проверки (passed/failed + метрика)
    TestResult       — результат всего теста (имя + список ValidationResult)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ValidationResult:
    """Результат одной валидации (Strategy pattern output).

    Attributes:
        passed:       прошла ли проверка
        metric_name:  название метрики (например "RMSE", "peak_freq_hz")
        actual_value: фактическое значение метрики
        threshold:    допустимый порог
        message:      человеко-читаемое сообщение
    """
    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    message: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"[{status}] {self.metric_name}: "
                f"{self.actual_value:.6g} (threshold={self.threshold:.6g}) "
                f"{self.message}")


@dataclass
class TestResult:
    """Сводный результат теста.

    Attributes:
        test_name:   имя теста
        validations: список ValidationResult
        error:       исключение (если тест упал с ошибкой)
        metadata:    произвольные доп. данные (dict)
    """
    test_name: str
    validations: List[ValidationResult] = field(default_factory=list)
    error: Optional[Exception] = None
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True если все валидации прошли и нет ошибки."""
        if self.error is not None:
            return False
        if not self.validations:
            # assert-style test: passed if explicitly marked
            return bool(self.metadata.get("assert_passed"))
        return all(v.passed for v in self.validations)

    def add(self, v: ValidationResult) -> "TestResult":
        """Добавить ValidationResult (fluent API)."""
        self.validations.append(v)
        return self

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        n_pass = sum(1 for v in self.validations if v.passed)
        n_total = len(self.validations)
        lines = [f"[{status}] {self.test_name} ({n_pass}/{n_total} checks passed)"]
        for v in self.validations:
            lines.append(f"  {v}")
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Сериализация TestResult → plain dict (для JSON / ResultStore).

        Используется в:
            common/io/result_store.py ResultStore.save_test_result()
            common/reporters.py        JSONReporter._add_record() (косвенно)
        """
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "validations": [
                {
                    "metric": v.metric_name,
                    "passed": v.passed,
                    "actual": v.actual_value,
                    "threshold": v.threshold,
                    "message": v.message,
                }
                for v in self.validations
            ],
            "error": str(self.error) if self.error is not None else None,
            "metadata": self.metadata,
        }
