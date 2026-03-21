"""
heterodyne_test_base.py — HeterodyneTestBase (Template Method)
==============================================================

Специализированный базовый класс для тестов HeterodyneDechirp.

Добавляет к TestBase:
  - numpy-эталон дечирпа: dechirp_numpy()
  - хелперы: validate_beat_frequency(), validate_range_estimation()
  - Template hook: compute_expected_fbeats()

Пример использования:
    class TestDechirpBasic(HeterodyneTestBase):
        def get_params(self): return DechirpParams()

        def process(self, data, ctx):
            return self.het_proc.dechirp(data)

        def validate(self, result, params):
            return self.validate_beat_frequency(result, params)
"""

import numpy as np
from abc import abstractmethod

from common.test_base import TestBase
from common.result import TestResult, ValidationResult
from common.validators import DataValidator
from heterodyne.conftest import DechirpParams


class HeterodyneTestBase(TestBase):
    """Базовый класс для тестов GPU-гетеродина.

    Template Method hooks (наследуемые от TestBase):
        get_params()    → DechirpParams
        generate_data() → S_rx [n_ant, n_samples]
        process()       → результат дечирпа
        validate()      → TestResult

    Добавленные методы:
        dechirp_numpy()         — numpy эталон дечирпа
        validate_beat_freq()    — проверка beat frequency
        _find_peak_freq()       — пик спектра для канала
    """

    def generate_data(self, params: DechirpParams) -> np.ndarray:
        """Генерация ЛЧМ приёмного сигнала с задержками.

        По умолчанию: 5 антенн, линейные задержки 100..500 мкс.
        Переопределить для специфичных тестов.
        """
        delays_s = np.arange(1, params.n_antennas + 1) * 100e-6
        S = np.zeros((params.n_antennas, params.n_samples), dtype=np.complex64)
        for ant in range(params.n_antennas):
            tau = delays_s[ant]
            t = np.arange(params.n_samples) / params.fs
            mask = t >= tau
            t_local = t[mask] - tau
            phase = 2 * np.pi * (params.f_start * t_local +
                                  0.5 * params.chirp_rate * t_local ** 2)
            S[ant, mask] = np.exp(1j * phase).astype(np.complex64)
        self._last_delays_s = delays_s
        return S

    @staticmethod
    def dechirp_numpy(s_rx: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
        """Numpy-эталон дечирпа: s_dc = s_rx * conj(s_ref).

        Args:
            s_rx:   приёмный ЛЧМ [n_samples] или [n_ant, n_samples]
            s_ref:  опорный ЛЧМ [n_samples]

        Returns:
            Дечирпированный сигнал той же формы что s_rx
        """
        if s_rx.ndim == 1:
            return (s_rx * np.conj(s_ref)).astype(np.complex64)
        return (s_rx * np.conj(s_ref)[np.newaxis, :]).astype(np.complex64)

    def _find_peak_freq(self, signal_1d: np.ndarray,
                        fs: float) -> float:
        """Найти частоту пика спектра.

        Args:
            signal_1d: [n_samples] комплексный сигнал
            fs:        частота дискретизации

        Returns:
            Частота пика (Гц), всегда положительная
        """
        n = len(signal_1d)
        spectrum = np.abs(np.fft.fft(signal_1d))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        # Ищем в положительных частотах
        half = n // 2
        idx = np.argmax(spectrum[:half])
        return float(freqs[idx])

    def validate_beat_frequency(self, dechirped: np.ndarray,
                                params: DechirpParams,
                                delays_s: np.ndarray,
                                tolerance_hz: float = 1e3) -> TestResult:
        """Проверить что beat frequency совпадает с ожидаемой.

        Beat freq = chirp_rate * delay_s

        Args:
            dechirped:    дечирпированный сигнал [n_ant, n_samples]
            params:       параметры дечирпа
            delays_s:     истинные задержки антенн [n_ant]
            tolerance_hz: допустимое отклонение (Гц)

        Returns:
            TestResult с ValidationResult для каждой антенны
        """
        tr = TestResult(self.name)
        for ant in range(min(params.n_antennas, len(delays_s))):
            expected_fbeat = params.chirp_rate * delays_s[ant]
            actual_fbeat = self._find_peak_freq(dechirped[ant], params.fs)
            err = abs(actual_fbeat - expected_fbeat)
            tr.add(ValidationResult(
                passed=err <= tolerance_hz,
                metric_name=f"ant{ant}_fbeat_hz",
                actual_value=actual_fbeat,
                threshold=expected_fbeat + tolerance_hz,
                message=f"expected={expected_fbeat:.0f} Hz, err={err:.0f} Hz"
            ))
        return tr
