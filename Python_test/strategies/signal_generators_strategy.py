"""
signal_generators_strategy.py — ISignalStrategy + 4 реализации (GoF Strategy)
==============================================================================

Паттерн Strategy (GoF): алгоритмы генерации сигнала взаимозаменяемы.
OCP: добавить новый тип = новый класс, не меняя клиентский код.
DIP: клиенты зависят от ISignalStrategy, не от конкретных классов.

Реализации:
    SinSignalStrategy      — синус (exp(j·2π·f0·t))
    LfmNoDelayStrategy     — ЛЧМ без задержек
    LfmWithDelayStrategy   — ЛЧМ + линейные задержки (integer shift)
    LfmFarrowStrategy      — ЛЧМ + дробные задержки (Farrow интерполяция)
    SignalStrategyFactory  — Factory Method
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from test_params import AntennaTestParams
from signal_factory import SignalVariant


# ─────────────────────────────────────────────────────────────────────────────
# ISignalStrategy — интерфейс стратегии
# ─────────────────────────────────────────────────────────────────────────────

class ISignalStrategy(ABC):
    """Strategy (GoF): абстракция генерации сигнала."""

    @abstractmethod
    def generate(self, params: AntennaTestParams) -> np.ndarray:
        """Генерировать сигнал на NumPy.

        Args:
            params: AntennaTestParams с n_ant, n_samples, fs, f0_hz, fdev_hz

        Returns:
            complex64 [n_ant, n_samples] — входной сигнал
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Читаемое имя стратегии."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_time_axis(params: AntennaTestParams) -> np.ndarray:
    """Временная ось [0, T), dt = 1/fs."""
    return np.arange(params.n_samples, dtype=np.float64) / params.fs


def _gen_lfm(params: AntennaTestParams, delay_s: float = 0.0) -> np.ndarray:
    """ЛЧМ сигнал с задержкой delay_s (секунды).

    Формула: exp(j·(2π·f0·(t-τ) + π·fdev/T·(t-τ)²))
    """
    T = params.n_samples / params.fs
    t = _make_time_axis(params) - delay_s
    phase = 2.0 * np.pi * params.f0_hz * t
    if params.fdev_hz > 0:
        phase += np.pi * (params.fdev_hz / T) * t ** 2
    return np.exp(1j * phase).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SinSignalStrategy
# ─────────────────────────────────────────────────────────────────────────────

class SinSignalStrategy(ISignalStrategy):
    """Синусоидальный сигнал (fdev=0), одинаковый на всех антеннах."""

    def generate(self, params: AntennaTestParams) -> np.ndarray:
        t = _make_time_axis(params)
        s = np.exp(1j * 2.0 * np.pi * params.f0_hz * t).astype(np.complex64)
        return np.tile(s, (params.n_ant, 1))

    @property
    def name(self) -> str:
        return "SIN"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LfmNoDelayStrategy
# ─────────────────────────────────────────────────────────────────────────────

class LfmNoDelayStrategy(ISignalStrategy):
    """ЛЧМ без задержек (tau_step=0), одинаковый на всех антеннах."""

    def generate(self, params: AntennaTestParams) -> np.ndarray:
        s = _gen_lfm(params, delay_s=0.0)
        return np.tile(s, (params.n_ant, 1))

    @property
    def name(self) -> str:
        return "LFM_NO_DELAY"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LfmWithDelayStrategy — целочисленные задержки
# ─────────────────────────────────────────────────────────────────────────────

class LfmWithDelayStrategy(ISignalStrategy):
    """ЛЧМ + линейные задержки (сдвиг целых отсчётов, без интерполяции).

    delay[i] = i * tau_step_us мкс → samples = round(delay * fs)
    """

    def generate(self, params: AntennaTestParams) -> np.ndarray:
        result = np.zeros((params.n_ant, params.n_samples), dtype=np.complex64)
        for i in range(params.n_ant):
            delay_s     = i * params.tau_step_us * 1e-6
            shift_samp  = int(round(delay_s * params.fs))
            s           = _gen_lfm(params, delay_s=0.0)
            if shift_samp == 0:
                result[i] = s
            elif shift_samp < params.n_samples:
                result[i, shift_samp:] = s[:params.n_samples - shift_samp]
            # else: нулевая антенна (задержка > длины сигнала)
        return result

    @property
    def name(self) -> str:
        return "LFM_WITH_DELAY"


# ─────────────────────────────────────────────────────────────────────────────
# 4. LfmFarrowStrategy — дробные задержки через Farrow
# ─────────────────────────────────────────────────────────────────────────────

class LfmFarrowStrategy(ISignalStrategy):
    """ЛЧМ + дробные задержки через FarrowDelay (scipy/numpy интерполяция).

    Использует FarrowDelay из farrow_delay.py (тот же класс что в
    test_farrow_pipeline.py). Если FarrowDelay недоступен — фоллбек
    на LfmWithDelayStrategy с предупреждением.
    """

    def generate(self, params: AntennaTestParams) -> np.ndarray:
        try:
            from farrow_delay import FarrowDelay
            farrow = FarrowDelay()
        except ImportError:
            import warnings
            warnings.warn(
                "FarrowDelay not available, falling back to LfmWithDelayStrategy",
                RuntimeWarning,
                stacklevel=2,
            )
            return LfmWithDelayStrategy().generate(params)

        result = np.zeros((params.n_ant, params.n_samples), dtype=np.complex64)
        s0 = _gen_lfm(params, delay_s=0.0)   # базовый сигнал без задержки

        for i in range(params.n_ant):
            delay_s       = i * params.tau_step_us * 1e-6
            delay_samples = delay_s * params.fs
            delayed   = farrow.apply_single(s0.astype(np.complex128), delay_samples)
            result[i] = delayed[:params.n_samples].astype(np.complex64)

        return result

    @property
    def name(self) -> str:
        return "LFM_FARROW"


# ─────────────────────────────────────────────────────────────────────────────
# SignalStrategyFactory — Factory Method (GoF)
# ─────────────────────────────────────────────────────────────────────────────

class SignalStrategyFactory:
    """Фабрика сигнальных стратегий (Factory Method GoF).

    Использование:
        strategy = SignalStrategyFactory.create(SignalVariant.LFM_FARROW)
        S = strategy.generate(params)   # [n_ant, n_samples] complex64
    """

    _registry: dict = {
        SignalVariant.SIN:            SinSignalStrategy,
        SignalVariant.LFM_NO_DELAY:   LfmNoDelayStrategy,
        SignalVariant.LFM_WITH_DELAY: LfmWithDelayStrategy,
        SignalVariant.LFM_FARROW:     LfmFarrowStrategy,
    }

    @classmethod
    def create(cls, variant: SignalVariant) -> ISignalStrategy:
        """Создать стратегию по варианту.

        Args:
            variant: SignalVariant (SIN, LFM_NO_DELAY, ...)

        Returns:
            ISignalStrategy — конкретная реализация

        Raises:
            ValueError: при неизвестном варианте
        """
        klass = cls._registry.get(variant)
        if klass is None:
            raise ValueError(f"Unknown SignalVariant: {variant}")
        return klass()

    @classmethod
    def all_variants(cls) -> list:
        """Вернуть список всех SignalVariant."""
        return list(cls._registry.keys())
