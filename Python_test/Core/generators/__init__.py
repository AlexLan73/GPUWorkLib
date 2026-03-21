"""
GPU-генераторы сигналов как Adapter'ы над gpuworklib.

GoF Adapter: оборачивает gpuworklib.SignalGenerator в ISignalGenerator.
GoF Factory Method + Registry: GeneratorFactory.create("cw", ctx, params).
SOLID OCP: добавить новый генератор = register() здесь, не менять Factory.
GRASP Creator: Factory знает как создать ISignalGenerator.

Использование:
    from Core.generators import GeneratorFactory
    from common.configs import SignalConfig

    params = SignalConfig(fs=12e6, f0_hz=2e6, fdev_hz=1e6, n_samples=4096)
    gen = GeneratorFactory.create("cw", ctx, params)
    signal = gen.generate(4096)  # -> ndarray complex64
"""

from .base import ISignalGenerator
from .cw import CwGenerator
from .lfm import LfmGenerator
from .noise import NoiseGenerator
from .factory import GeneratorFactory

# OCP: добавить новый генератор = register() здесь, не менять Factory
GeneratorFactory.register("cw", CwGenerator)
GeneratorFactory.register("lfm", LfmGenerator)
GeneratorFactory.register("noise", NoiseGenerator)

__all__ = [
    "ISignalGenerator",
    "CwGenerator", "LfmGenerator", "NoiseGenerator",
    "GeneratorFactory",
]
