"""
NumPy-эталоны для сигналов. Единая точка истины (DRY).
GRASP Information Expert: знает все формулы генерации сигналов.
"""

import numpy as np
from math import pi


class SignalReferences:
    """
    Статические NumPy-реализации GPU-генераторов.

    Использование:
        from common.references import SignalReferences

        ref_cw  = SignalReferences.cw(fs=12e6, n_samples=4096, f0=2e6)
        ref_lfm = SignalReferences.lfm(fs=12e6, n_samples=4096, f_start=0, f_end=2e6)
    """

    @staticmethod
    def cw(fs: float, n_samples: int, f0: float,
           amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        CW сигнал (непрерывная синусоида).

        Returns: complex64, shape=(n_samples,)
        """
        t = np.arange(n_samples) / fs
        return (amplitude * np.exp(1j * (2*pi*f0*t + phase))).astype(np.complex64)

    @staticmethod
    def lfm(fs: float, n_samples: int, f_start: float, f_end: float,
            amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        ЛЧМ (линейная частотная модуляция).

        Returns: complex64, shape=(n_samples,)
        """
        t = np.arange(n_samples) / fs
        duration = n_samples / fs
        rate = (f_end - f_start) / duration
        phi = 2*pi * (f_start*t + 0.5*rate*t**2) + phase
        return (amplitude * np.exp(1j * phi)).astype(np.complex64)

    @staticmethod
    def lfm_with_delay(fs: float, n_samples: int, f_start: float, f_end: float,
                       delay_s: float, amplitude: float = 1.0) -> np.ndarray:
        """
        ЛЧМ с задержкой (для тестов гетеродина/дечирпа).

        Returns: complex64, shape=(n_samples,)
        Сигнал начинается с t=delay_s, до этого -- нули.
        """
        t = np.arange(n_samples) / fs
        duration = n_samples / fs
        rate = (f_end - f_start) / duration
        result = np.zeros(n_samples, dtype=np.complex64)
        mask = t >= delay_s
        t_local = t[mask] - delay_s
        phi = 2*pi * (f_start*t_local + 0.5*rate*t_local**2)
        result[mask] = (amplitude * np.exp(1j * phi)).astype(np.complex64)
        return result

    @staticmethod
    def lfm_multi_antenna(fs: float, n_samples: int, f_start: float, f_end: float,
                          delays_s: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Несколько ЛЧМ с разными задержками (массив антенн).

        Args:
            delays_s: задержки по каждой антенне, shape=(n_antennas,)

        Returns: complex64, shape=(n_antennas, n_samples)
        """
        n_ant = len(delays_s)
        result = np.zeros((n_ant, n_samples), dtype=np.complex64)
        for i, tau in enumerate(delays_s):
            result[i] = SignalReferences.lfm_with_delay(
                fs, n_samples, f_start, f_end, tau, amplitude
            )
        return result

    @staticmethod
    def noise(n_samples: int, seed: int = 42, amplitude: float = 1.0) -> np.ndarray:
        """
        Гауссов шум (воспроизводимый через seed).

        GPU (Philox PRNG) и NumPy (PCG64) дают РАЗНЫЕ числа при одном seed!
        Этот метод полезен для чисто Python-тестов без GPU.

        Для валидации GPU statistics:
            1. GPU генерирует данные -> копия на CPU (ndarray)
            2. GPU StatisticsProcessor считает stats из этих данных
            3. NumPy считает stats из ТЕХ ЖЕ скопированных данных
            4. Сравниваем GPU stats vs NumPy stats (DataValidator)

        Returns: complex64, shape=(n_samples,)
        """
        rng = np.random.default_rng(seed)
        sig = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        return (sig * amplitude / 2**0.5).astype(np.complex64)

    @staticmethod
    def form_signal(fs: float, points: int, f0: float, amplitude: float,
                    phase: float, fdev: float, norm_val: float,
                    tau: float = 0.0) -> np.ndarray:
        """
        CPU reference FormSignal (формула getX без шума).

        Воспроизводит GPU FormSignalGenerator: окно + центрированная фаза.

        Args:
            fs:        частота дискретизации (Гц)
            points:    число отсчётов
            f0:        несущая частота (Гц)
            amplitude: амплитуда
            phase:     начальная фаза (рад)
            fdev:      девиация частоты (Гц)
            norm_val:  нормировочный коэффициент
            tau:       задержка (с), default=0.0

        Returns: complex64, shape=(points,)
        """
        dt = 1.0 / fs
        ti = points * dt
        t = np.arange(points, dtype=np.float64) * dt + tau
        in_window = (t >= 0.0) & (t <= ti - dt)
        t_centered = t - ti / 2.0
        ph = 2.0 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase
        X = amplitude * norm_val * np.exp(1j * ph)
        X[~in_window] = 0.0
        return X.astype(np.complex64)

    @staticmethod
    def dechirp(s_rx: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
        """
        Дечирп: s_dc = s_rx * conj(s_ref).
        NumPy-эталон для HeterodyneAdapter.

        Returns: complex64, shape=s_rx.shape
        """
        return (s_rx * np.conj(s_ref)).astype(np.complex64)
