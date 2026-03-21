"""
CwGenerator -- GPU CW-генератор (непрерывная синусоида).
GoF Adapter: оборачивает gpuworklib.SignalGenerator в ISignalGenerator.
"""

from __future__ import annotations
import numpy as np
from .base import ISignalGenerator
from common import GPULoader


class CwGenerator(ISignalGenerator):
    """
    GPU CW-генератор (непрерывная синусоида).

    GoF Adapter: оборачивает gpuworklib.SignalGenerator
    в стандартный интерфейс ISignalGenerator.

    Args:
        ctx:        GPU контекст (GPUContext или ROCmGPUContext)
        fs:         Частота дискретизации (Гц)
        f0:         Несущая частота (Гц)
        amplitude:  Амплитуда (default=1.0)
        phase:      Начальная фаза (рад, default=0.0)
    """

    def __init__(self, ctx, fs: float, f0: float,
                 amplitude: float = 1.0, phase: float = 0.0):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._f0 = f0
        self._amplitude = amplitude
        self._phase = phase

    @classmethod
    def from_config(cls, ctx, params) -> "CwGenerator":
        """OCP: Factory вызывает этот метод, не зная про f0/amplitude."""
        return cls(ctx, fs=params.fs, f0=params.f0_hz,
                   amplitude=params.amplitude)

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_cw(
            freq=self._f0,
            fs=self._fs,
            length=n_samples,
            amplitude=self._amplitude,
        )

    def set_params(self, **kwargs) -> None:
        if "f0" in kwargs:         self._f0 = kwargs["f0"]
        if "amplitude" in kwargs:  self._amplitude = kwargs["amplitude"]
        if "phase" in kwargs:      self._phase = kwargs["phase"]

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "cw"
