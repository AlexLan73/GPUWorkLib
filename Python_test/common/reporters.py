"""
reporters.py — Observer pattern для отчётности о результатах тестов
====================================================================

Observer (GoF):
    IReporter     — абстрактный интерфейс наблюдателя
    ConsoleReporter — вывод в консоль
    JSONReporter    — сохранение в JSON-файл

Usage:
    reporter = ConsoleReporter()
    reporter.on_test_started("test_fir_lowpass")
    reporter.on_passed("test_fir_lowpass", result)
    # или
    reporter.on_failed("test_fir_lowpass", exception)
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import List

from .result import TestResult, ValidationResult


class IReporter(ABC):
    """Абстрактный Observer для событий тестирования."""

    @abstractmethod
    def on_test_started(self, name: str) -> None:
        """Вызывается в начале теста."""
        ...

    @abstractmethod
    def on_passed(self, name: str, result: TestResult) -> None:
        """Вызывается при успешном прохождении теста."""
        ...

    @abstractmethod
    def on_failed(self, name: str, result: TestResult) -> None:
        """Вызывается при провале теста."""
        ...

    def on_suite_finished(self, results: List[TestResult]) -> None:
        """Вызывается после завершения всего набора тестов (опционально)."""
        pass


class ConsoleReporter(IReporter):
    """Структурированный вывод в консоль.

    Использует ANSI-цвета если терминал их поддерживает.
    """

    _GREEN = "\033[92m"
    _RED   = "\033[91m"
    _RESET = "\033[0m"
    _BOLD  = "\033[1m"

    def __init__(self, use_colors: bool = True, verbose: bool = True):
        """
        Args:
            use_colors: использовать ANSI-цвета
            verbose:    выводить детали каждой проверки
        """
        self.use_colors = use_colors
        self.verbose = verbose
        self._start_times: dict = {}

    def _color(self, text: str, color: str) -> str:
        if self.use_colors:
            return f"{color}{text}{self._RESET}"
        return text

    def on_test_started(self, name: str) -> None:
        self._start_times[name] = time.time()
        print(f"\n{'─' * 60}")
        print(self._color(f"  RUN  {name}", self._BOLD))

    def on_passed(self, name: str, result: TestResult) -> None:
        elapsed = time.time() - self._start_times.get(name, time.time())
        print(self._color(f"  PASS {name} ({elapsed*1000:.1f} ms)", self._GREEN))
        if self.verbose:
            for v in result.validations:
                print(f"    ✓ {v}")

    def on_failed(self, name: str, result: TestResult) -> None:
        elapsed = time.time() - self._start_times.get(name, time.time())
        print(self._color(f"  FAIL {name} ({elapsed*1000:.1f} ms)", self._RED))
        for v in result.validations:
            marker = "✓" if v.passed else "✗"
            print(f"    {marker} {v}")
        if result.error:
            print(self._color(f"  ERROR: {result.error}", self._RED))

    def on_suite_finished(self, results: List[TestResult]) -> None:
        n_pass = sum(1 for r in results if r.passed)
        n_total = len(results)
        print(f"\n{'═' * 60}")
        msg = f"  {n_pass}/{n_total} tests passed"
        if n_pass == n_total:
            print(self._color(msg, self._GREEN))
        else:
            print(self._color(msg, self._RED))
        print('═' * 60)


class JSONReporter(IReporter):
    """Сохраняет результаты тестов в JSON-файл.

    Формат файла: Results/JSON/{suite_name}.json
    """

    def __init__(self, output_path: str):
        """
        Args:
            output_path: путь к выходному JSON-файлу
        """
        self.output_path = output_path
        self._records: list = []
        self._start_times: dict = {}
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def on_test_started(self, name: str) -> None:
        self._start_times[name] = time.time()

    def on_passed(self, name: str, result: TestResult) -> None:
        self._add_record(name, result, True)

    def on_failed(self, name: str, result: TestResult) -> None:
        self._add_record(name, result, False)

    def _add_record(self, name: str, result: TestResult, passed: bool) -> None:
        elapsed = time.time() - self._start_times.get(name, time.time())
        record = {
            "name": name,
            "passed": passed,
            "elapsed_ms": round(elapsed * 1000, 2),
            "validations": [
                {
                    "metric": v.metric_name,
                    "passed": v.passed,
                    "actual": v.actual_value,
                    "threshold": v.threshold,
                    "message": v.message,
                }
                for v in result.validations
            ],
            "error": str(result.error) if result.error else None,
            "metadata": result.metadata,
        }
        self._records.append(record)

    def on_suite_finished(self, results: List[TestResult]) -> None:
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "tests": self._records,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[JSONReporter] Saved to {self.output_path}")


class MultiReporter(IReporter):
    """Делегирует события нескольким репортерам одновременно (Composite)."""

    def __init__(self, *reporters: IReporter):
        self._reporters = list(reporters)

    def on_test_started(self, name: str) -> None:
        for r in self._reporters:
            r.on_test_started(name)

    def on_passed(self, name: str, result: TestResult) -> None:
        for r in self._reporters:
            r.on_passed(name, result)

    def on_failed(self, name: str, result: TestResult) -> None:
        for r in self._reporters:
            r.on_failed(name, result)

    def on_suite_finished(self, results: List[TestResult]) -> None:
        for r in self._reporters:
            r.on_suite_finished(results)
