"""
validators.py — DataValidator (Strategy GoF)
=============================================

Один универсальный класс вместо 4 специализированных.

Убраны: NumericValidator, RMSEValidator, SpectralValidator, EnergyValidator.
Добавлен: DataValidator — работает со скалярами, векторами и матрицами.

Classes:
    IValidator    — абстрактный интерфейс (Strategy GoF)
    DataValidator — универсальный валидатор, метрика задаётся при создании
"""

from abc import ABC, abstractmethod
import numpy as np

from .result import ValidationResult


class IValidator(ABC):
    """Абстрактный валидатор — Strategy interface."""

    @abstractmethod
    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        """Сравнить actual с reference, вернуть ValidationResult.

        Args:
            actual:    результат GPU-обработки
            reference: эталонный результат (scipy/numpy)

        Returns:
            ValidationResult с passed=True если проверка пройдена
        """
        ...


class DataValidator(IValidator):
    """Универсальный валидатор: скаляр / вектор / матрица.

    Strategy (GoF) — метрика выбирается при создании.
    Information Expert (GRASP) — знает как сравнивать данные.

    Все данные приводятся через np.atleast_1d(x).ravel()
    к одномерному массиву, затем применяется одна из трёх метрик.

    Метрики:
        "max_rel" → max(|actual - ref|) / max(|ref|) < tolerance
                    Для сигналов, спектров, статистики.
                    При ref ≈ 0 переключается на абсолютный допуск 1e-10.

        "abs"     → max(|actual - ref|) < tolerance
                    Для частот в Гц, индексов бинов FFT.
                    reference может быть нулём — не нормируется.

        "rmse"    → rms(|actual - ref|) / rms(|ref|) < tolerance
                    Для шумных данных где нужна среднеквадратичная метрика.
                    При ref ≈ 0 переключается на абсолютный допуск 1e-10.

    Usage:
        # Сравнить комплексные векторы (сигнал):
        v = DataValidator(tolerance=0.01, metric="max_rel")
        r = v.validate(gpu_mean, numpy_mean, name="mean_stats")
        print(r)   # [PASS] mean_stats: 0.003421 (tol=0.01)

        # Сравнить частоту пика (абсолютная погрешность в Гц):
        v = DataValidator(tolerance=50e3, metric="abs")
        r = v.validate(refined_freq_hz, expected_f0=2e6, name="peak_freq")

        # Сравнить матрицы float32:
        v = DataValidator(tolerance=0.001, metric="max_rel")
        r = v.validate(gemm_output_gpu, gemm_output_numpy, name="gemm")
    """

    METRICS = ("max_rel", "abs", "rmse")

    def __init__(self, tolerance: float,
                 metric: str = "max_rel",
                 name: str = ""):
        """
        Args:
            tolerance: допустимый порог (зависит от metric)
            metric:    "max_rel" | "abs" | "rmse"
            name:      имя метрики для отчёта (если не передано в validate())
        """
        if metric not in self.METRICS:
            raise ValueError(
                f"metric должен быть одним из {self.METRICS}, получено: {metric!r}"
            )
        self.tolerance = tolerance
        self.metric = metric
        self._default_name = name

    def validate(self, actual, reference,
                 name: str = "") -> ValidationResult:
        """Сравнить actual с reference.

        Args:
            actual:    GPU-результат (скаляр, list, np.ndarray любой формы)
            reference: эталон (скаляр, list, np.ndarray любой формы)
            name:      имя метрики для ValidationResult (переопределяет self._default_name)

        Returns:
            ValidationResult

        Note:
            Оба аргумента приводятся к complex128 для вычислений (точность),
            но исходные типы могут быть float32/complex64 — это нормально.
        """
        metric_name = name or self._default_name or self.metric
        # Привести к 1D complex128 для вычислений
        a = np.atleast_1d(np.asarray(actual)).ravel().astype(np.complex128)
        r = np.atleast_1d(np.asarray(reference)).ravel().astype(np.complex128)

        if self.metric == "max_rel":
            return self._max_rel(a, r, metric_name)
        elif self.metric == "abs":
            return self._abs(a, r, metric_name)
        else:  # "rmse"
            return self._rmse(a, r, metric_name)

    # ── Приватные методы вычисления метрик ──────────────────────────────────

    def _max_rel(self, a, r, name) -> ValidationResult:
        """max(|a-r|) / max(|r|) < tolerance"""
        diff = np.abs(a - r)
        ref_norm = np.max(np.abs(r))
        if ref_norm < 1e-15:
            # reference ≈ 0 → абсолютный допуск
            err = float(np.max(diff))
            return ValidationResult(
                passed=err < 1e-10,
                metric_name=name,
                actual_value=err,
                threshold=1e-10,
                message="(near-zero reference, using absolute tolerance)"
            )
        err = float(np.max(diff) / ref_norm)
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )

    def _abs(self, a, r, name) -> ValidationResult:
        """max(|a-r|) < tolerance"""
        err = float(np.max(np.abs(a - r)))
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )

    def _rmse(self, a, r, name) -> ValidationResult:
        """rms(|a-r|) / rms(|r|) < tolerance"""
        diff = a - r
        rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        ref_rms = float(np.sqrt(np.mean(np.abs(r) ** 2)))
        if ref_rms < 1e-15:
            return ValidationResult(
                passed=rmse < 1e-10,
                metric_name=name,
                actual_value=rmse,
                threshold=1e-10,
                message="(near-zero reference)"
            )
        err = rmse / ref_rms
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=name,
            actual_value=err,
            threshold=self.tolerance,
        )
