"""
signal.py — сигнальные валидаторы (standalone)
================================================

Standalone — reference игнорируется; проверяем свойства самого сигнала.

    FrequencyValidator — пик FFT-спектра попадает в [f0 - Δ, f0 + Δ]
    PowerValidator     — средняя мощность |x|² в пределах tol от ожидаемой

⚠️ Strict `<` (правило #10 в TASK_PythonArch_INDEX).
"""

from __future__ import annotations

import numpy as np

from ..result import ValidationResult
from .base import IValidator


class FrequencyValidator(IValidator):
    """Проверяет что пик FFT-спектра близок к ожидаемой частоте.

    Применение: валидация несущей частоты CW / центральной частоты LFM.

    Args:
        expected_hz:  Ожидаемая частота пика в Гц.
        tolerance_hz: Допустимое отклонение в Гц.
        fs:           Частота дискретизации сигнала в Гц.

    Поведение:
        * Если `actual` — complex-массив, считается как временной сигнал,
          FFT делается внутри.
        * Если `actual` — real-массив, считается как готовый амплитудный
          спектр В НАТУРАЛЬНОМ ПОРЯДКЕ ``np.fft.fft`` (без ``fftshift``):
          ``[0, df, 2df, …, -df]``. Если у вас уже применён ``fftshift`` —
          обратно разверните через ``np.fft.ifftshift`` перед передачей,
          иначе ``argmax`` вернёт частоту с противоположным знаком.
    """

    def __init__(self, expected_hz: float, tolerance_hz: float, fs: float):
        self._expected = float(expected_hz)
        self._tol = float(tolerance_hz)
        self._fs  = float(fs)

    def validate(self, actual, reference=None,
                 name: str = "peak_freq_hz") -> ValidationResult:
        arr = np.asarray(actual)
        # Complex → временной сигнал; real → готовый спектр
        if np.iscomplexobj(arr):
            spec = np.abs(np.fft.fft(arr))
        else:
            spec = np.asarray(arr, dtype=np.float64)
        n = len(spec)
        freqs = np.fft.fftfreq(n, d=1.0 / self._fs)
        peak_idx  = int(np.argmax(spec))
        actual_hz = float(freqs[peak_idx])
        err = abs(actual_hz - self._expected)
        return ValidationResult(
            passed=err < self._tol,
            metric_name=name,
            actual_value=actual_hz,
            threshold=self._tol,
            message=(
                f"expected={self._expected/1e6:.3f}MHz ±{self._tol/1e3:.1f}kHz, "
                f"got={actual_hz/1e6:.3f}MHz, err={err/1e3:.2f}kHz"
            ),
        )


class PowerValidator(IValidator):
    """Проверяет среднюю мощность сигнала: |mean(|x|²) - P₀| / P₀ < tol.

    Применение: валидация амплитуды генераторов, нормировки данных.

    Args:
        expected_power: Ожидаемая мощность (P₀). Должна быть > 0.
        tolerance:      Относительная погрешность (по умолчанию 5%).

    Raises:
        ValueError: если ``expected_power <= 0`` — такой валидатор
                    бессмысленен (относительная ошибка не определена).
    """

    def __init__(self, expected_power: float, tolerance: float = 0.05):
        if expected_power <= 0:
            raise ValueError(
                f"PowerValidator: expected_power должен быть > 0, "
                f"получено: {expected_power}"
            )
        self._expected = float(expected_power)
        self._tol = float(tolerance)

    def validate(self, actual, reference=None,
                 name: str = "power") -> ValidationResult:
        # Не касуем в complex64 насильно — np.abs(..)**2 корректно и для real
        arr  = np.asarray(actual)
        power = float(np.mean(np.abs(arr) ** 2))
        rel_err = abs(power - self._expected) / self._expected
        return ValidationResult(
            passed=rel_err < self._tol,
            metric_name=name,
            actual_value=power,
            threshold=self._tol,
            message=(
                f"power={power:.4f}, expected={self._expected:.4f}, "
                f"rel_err={rel_err:.4f}"
            ),
        )
