"""
ISignalGenerator -- интерфейс GPU-генератора сигнала.

SOLID ISP: только генерация сигнала, ничего лишнего.
GoF: общий интерфейс для всех Adapter-реализаций.
GRASP LSP: CwGenerator/LfmGenerator/NoiseGenerator взаимозаменяемы.
OCP: каждый адаптер реализует from_config() -- Factory не знает
     про конкретные параметры, вызывает cls.from_config(ctx, params).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.configs import SignalConfig


class ISignalGenerator(ABC):
    """
    Интерфейс GPU-генератора сигнала.

    Контракт:
        - from_config(ctx, params) -> создать из SignalConfig (OCP)
        - generate(n_samples) -> complex64 ndarray
        - set_params(**kwargs) -> обновить параметры без пересоздания
        - sample_rate -> float
        - generator_type -> str ("cw", "lfm", "noise")
    """

    @classmethod
    @abstractmethod
    def from_config(cls, ctx, params: "SignalConfig") -> "ISignalGenerator":
        """
        Создаёт генератор из SignalConfig (OCP-точка расширения).
        Каждый адаптер сам извлекает нужные поля из params.

        Args:
            ctx:    GPU контекст (GPUContext или ROCmGPUContext)
            params: SignalConfig с fs, f0_hz, fdev_hz, amplitude, seed
        """

    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Генерирует сигнал длиной n_samples на GPU.

        Returns:
            np.ndarray: dtype=complex64, shape=(n_samples,)
        """

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Обновляет параметры без пересоздания GPU-объекта.
        Конкретные kwargs зависят от типа генератора.
        """

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Частота дискретизации (Гц)."""

    @property
    @abstractmethod
    def generator_type(self) -> str:
        """Строковый тип: 'cw', 'lfm', 'noise'."""
