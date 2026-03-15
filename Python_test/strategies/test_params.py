"""
test_params.py — AntennaTestParams dataclass для Python тестов стратегий
========================================================================

GRASP Information Expert: содержит все параметры теста.
Соответствует C++ AntennaTestParams из tests/antenna_test_params.hpp.

Параметры по TestStrategia.md:
    n_ant=2500, n_samples=5000, fs=0.5e6, fdev=90кГц, W=2500×100
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class SignalVariant(Enum):
    """Варианты входного сигнала (Strategy GoF).

    Соответствует C++ enum class SignalVariant.
    """
    SIN            = auto()   # синус (fdev=0)
    LFM_NO_DELAY   = auto()   # ЛЧМ без задержек (2.1)
    LFM_WITH_DELAY = auto()   # ЛЧМ + целочисленные задержки (2.2.1)
    LFM_FARROW     = auto()   # ЛЧМ + дробные задержки (2.2.2)


@dataclass
class AntennaTestParams:
    """Параметры теста антенной стратегии.

    Использование:
        params = AntennaTestParams.small()      # быстрый тест
        params = AntennaTestParams.full_spec()  # полный тест (2500×5000)
    """
    # Размеры
    n_ant:     int   = 100         # число антенн (small по умолчанию)
    n_samples: int   = 5000        # отсчётов на антенну
    n_beams:   int   = 100         # столбцы матрицы W (=n_ant → квадратная)

    # Частоты
    fs:        float = 0.5e6       # частота дискретизации, Гц
    fdev_hz:   float = 90e3        # девиация ЛЧМ, Гц
    f0_hz:     float = 100e3       # целевая частота для валидации

    # Задержки
    tau_step_us: float = 2.0       # шаг задержки на антенну, мкс

    # Вариант сигнала
    signal_variant: SignalVariant = SignalVariant.SIN

    # Опции вывода
    save_to_files: bool = False
    output_dir:    str  = "Results/strategies/"

    # ── Фабричные методы ─────────────────────────────────────────────────

    @classmethod
    def small(cls, variant: SignalVariant = SignalVariant.SIN) -> "AntennaTestParams":
        """Быстрый тест: 100 антенн, квадратная матрица."""
        return cls(
            n_ant=100, n_beams=100, n_samples=5000,
            fs=0.5e6, fdev_hz=90e3, f0_hz=100e3,
            signal_variant=variant,
        )

    @classmethod
    def full_spec(cls, variant: SignalVariant = SignalVariant.SIN) -> "AntennaTestParams":
        """Полный тест по TestStrategia.md: 2500 антенн × 5000 отсчётов.

        Note:
            n_beams=100 → НЕ квадратная матрица (2500×100).
            Для GPU теста требует n_beams в AntennaProcessorConfig (TODO).
        """
        return cls(
            n_ant=2500, n_beams=100, n_samples=5000,
            fs=0.5e6, fdev_hz=90e3, f0_hz=100e3,
            signal_variant=variant,
        )

    @classmethod
    def debug(cls, variant: SignalVariant = SignalVariant.SIN) -> "AntennaTestParams":
        """Debug тест с записью в файлы."""
        p = cls.small(variant)
        p.save_to_files = True
        p.output_dir    = "Results/strategies/debug/"
        return p

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def bin_hz(self) -> float:
        """Ширина одного FFT бина, Гц."""
        import math
        nfft = 2 ** math.ceil(math.log2(self.n_samples))
        return self.fs / nfft

    @property
    def expected_peak_bin(self) -> int:
        """Ожидаемый бин пика для f0_hz."""
        import math
        nfft = 2 ** math.ceil(math.log2(self.n_samples))
        return round(self.f0_hz / (self.fs / nfft))
