"""
LfmGenerator -- GPU LFM-генератор (линейная частотная модуляция).
GoF Adapter: gpuworklib.SignalGenerator.generate_lfm -> ISignalGenerator.

Маппинг полей SignalConfig -> LFM:
    f0_hz  = f_start (начальная частота ЛЧМ)
    fdev_hz = bandwidth (f_end - f_start)
"""

from __future__ import annotations
import numpy as np
from .base import ISignalGenerator
from common import GPULoader


class LfmGenerator(ISignalGenerator):
    """
    GPU LFM-генератор (линейная частотная модуляция).

    GoF Adapter: gpuworklib.SignalGenerator.generate_lfm -> ISignalGenerator.

    Args:
        ctx:        GPU контекст
        fs:         Частота дискретизации (Гц)
        f_start:    Начальная частота (Гц)
        f_end:      Конечная частота (Гц)
        amplitude:  Амплитуда (default=1.0)
        phase:      Начальная фаза (рад, default=0.0)
    """

    def __init__(self, ctx, fs: float, f_start: float, f_end: float,
                 amplitude: float = 1.0, phase: float = 0.0):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._f_start = f_start
        self._f_end = f_end
        self._amplitude = amplitude
        self._phase = phase

    @classmethod
    def from_config(cls, ctx, params) -> "LfmGenerator":
        """OCP: SignalConfig.f0_hz=f_start, fdev_hz=bandwidth (f_end-f_start)."""
        return cls(ctx, fs=params.fs,
                   f_start=params.f0_hz,
                   f_end=params.f0_hz + params.fdev_hz,
                   amplitude=params.amplitude)

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_lfm(
            f_start=self._f_start,
            f_end=self._f_end,
            fs=self._fs,
            length=n_samples,
            amplitude=self._amplitude,
        )

    def set_params(self, **kwargs) -> None:
        if "f_start" in kwargs:   self._f_start = kwargs["f_start"]
        if "f_end" in kwargs:     self._f_end = kwargs["f_end"]
        if "amplitude" in kwargs: self._amplitude = kwargs["amplitude"]

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "lfm"
