"""
validators.py — Strategy pattern для валидации результатов GPU
===============================================================

Strategy (GoF) + Polymorphism (GRASP):
    IValidator — абстрактный интерфейс
    NumericValidator  — |actual - ref| / |ref| < tolerance
    SpectralValidator — пик в пределах freq_tol_hz и mag_tol_db
    EnergyValidator   — энергия в полосе >= min_ratio * total_energy

Usage:
    v = NumericValidator(tolerance=0.01)
    result = v.validate(gpu_output, scipy_reference)
    assert result.passed
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


class NumericValidator(IValidator):
    """Проверка численной точности: max(|actual - ref|) / max(|ref|) < tolerance.

    Используется для прямого сравнения GPU-выхода с scipy/numpy эталоном.
    Работает с вещественными и комплексными массивами.
    """

    def __init__(self, tolerance: float = 0.01,
                 metric_name: str = "relative_error"):
        """
        Args:
            tolerance:   допустимая относительная погрешность (0..1)
            metric_name: название метрики в отчёте
        """
        self.tolerance = tolerance
        self.metric_name = metric_name

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        actual_f = actual.ravel().astype(np.complex128)
        ref_f = reference.ravel().astype(np.complex128)

        diff = np.abs(actual_f - ref_f)
        ref_norm = np.max(np.abs(ref_f))

        if ref_norm < 1e-15:
            error = float(np.max(diff))
            passed = error < 1e-10
            return ValidationResult(
                passed=passed,
                metric_name=self.metric_name,
                actual_value=error,
                threshold=1e-10,
                message="(near-zero reference, using absolute tolerance)"
            )

        rel_error = float(np.max(diff) / ref_norm)
        return ValidationResult(
            passed=rel_error < self.tolerance,
            metric_name=self.metric_name,
            actual_value=rel_error,
            threshold=self.tolerance,
        )


class RMSEValidator(IValidator):
    """Проверка RMSE: rms(actual - ref) / rms(ref) < tolerance."""

    def __init__(self, tolerance: float = 0.01,
                 metric_name: str = "rmse_relative"):
        self.tolerance = tolerance
        self.metric_name = metric_name

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        diff = actual.ravel() - reference.ravel()
        rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        ref_rms = float(np.sqrt(np.mean(np.abs(reference.ravel()) ** 2)))

        if ref_rms < 1e-15:
            return ValidationResult(
                passed=rmse < 1e-10,
                metric_name=self.metric_name,
                actual_value=rmse,
                threshold=1e-10,
                message="(near-zero reference)"
            )

        rel = rmse / ref_rms
        return ValidationResult(
            passed=rel < self.tolerance,
            metric_name=self.metric_name,
            actual_value=rel,
            threshold=self.tolerance,
        )


class SpectralValidator(IValidator):
    """Проверка спектрального пика: частота и амплитуда в допуске.

    Ищет максимум спектра в actual и reference,
    проверяет что они отличаются не более чем на freq_tol_hz и mag_tol_db.
    """

    def __init__(self, fs: float,
                 freq_tol_hz: float = 10.0,
                 mag_tol_db: float = 1.0):
        """
        Args:
            fs:           частота дискретизации (Гц)
            freq_tol_hz:  допуск по частоте пика (Гц)
            mag_tol_db:   допуск по амплитуде пика (дБ)
        """
        self.fs = fs
        self.freq_tol_hz = freq_tol_hz
        self.mag_tol_db = mag_tol_db

    def _find_peak(self, signal: np.ndarray):
        """Найти пик спектра. Возвращает (freq_hz, mag_db)."""
        n = len(signal)
        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(n, d=1.0 / self.fs)
        idx = np.argmax(spectrum)
        return float(freqs[idx]), float(20 * np.log10(spectrum[idx] + 1e-30))

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        act_freq, act_mag = self._find_peak(actual.ravel())
        ref_freq, ref_mag = self._find_peak(reference.ravel())

        freq_err = abs(act_freq - ref_freq)
        mag_err = abs(act_mag - ref_mag)
        passed = (freq_err <= self.freq_tol_hz) and (mag_err <= self.mag_tol_db)

        return ValidationResult(
            passed=passed,
            metric_name="spectral_peak",
            actual_value=freq_err,
            threshold=self.freq_tol_hz,
            message=(f"peak_freq: {act_freq:.1f} Hz (ref={ref_freq:.1f} Hz), "
                     f"mag_err={mag_err:.2f} dB (tol={self.mag_tol_db:.1f} dB)")
        )


class EnergyValidator(IValidator):
    """Проверка энергии в полосе: E(band) / E(total) >= min_ratio.

    Полезно для проверки фильтров — убедиться что полезная полоса
    содержит достаточную долю энергии сигнала.
    """

    def __init__(self, fs: float,
                 band_hz: tuple,
                 min_ratio: float = 0.8):
        """
        Args:
            fs:        частота дискретизации (Гц)
            band_hz:   (f_low, f_high) — интересующая полоса
            min_ratio: минимально допустимая доля энергии в полосе
        """
        self.fs = fs
        self.band_hz = band_hz
        self.min_ratio = min_ratio

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult:
        sig = actual.ravel()
        n = len(sig)
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(n, d=1.0 / self.fs)

        f_low, f_high = self.band_hz
        mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)

        e_band = float(np.sum(spectrum[mask]))
        e_total = float(np.sum(spectrum))

        ratio = e_band / (e_total + 1e-30)

        return ValidationResult(
            passed=ratio >= self.min_ratio,
            metric_name="energy_ratio",
            actual_value=ratio,
            threshold=self.min_ratio,
            message=f"band=[{f_low:.0f}, {f_high:.0f}] Hz"
        )
