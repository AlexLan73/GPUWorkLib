"""
NoiseGenerator -- GPU Noise-генератор (Гауссов шум, Philox PRNG).
GoF Adapter: gpuworklib.SignalGenerator.generate_noise -> ISignalGenerator.
"""

from __future__ import annotations
import numpy as np
from .base import ISignalGenerator
from common import GPULoader


class NoiseGenerator(ISignalGenerator):
    """
    GPU Noise-генератор (Гауссов шум, Philox PRNG).

    GoF Adapter: gpuworklib.SignalGenerator.generate_noise -> ISignalGenerator.

    Args:
        ctx:        GPU контекст
        fs:         Частота дискретизации (Гц)
        amplitude:  Амплитуда (default=1.0)
        seed:       Начальное значение PRNG (default=42, воспроизводимость)
    """

    def __init__(self, ctx, fs: float,
                 amplitude: float = 1.0, seed: int = 42):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._amplitude = amplitude
        self._seed = seed

    @classmethod
    def from_config(cls, ctx, params) -> "NoiseGenerator":
        """OCP: seed из SignalConfig.seed (default=42)."""
        return cls(ctx, fs=params.fs, amplitude=params.amplitude,
                   seed=getattr(params, "seed", 42))

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_noise(
            length=n_samples,
            amplitude=self._amplitude,
            seed=self._seed,
        )

    def set_params(self, **kwargs) -> None:
        if "amplitude" in kwargs: self._amplitude = kwargs["amplitude"]
        if "seed" in kwargs:      self._seed = kwargs["seed"]

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "noise"
