"""
Core -- репозиторий готовых GPU-объектов GPUWorkLib.

Протестированные адаптеры над gpuworklib.
Low Coupling (GRASP): Core -> common (односторонняя зависимость).

Использование:
    from Core.generators import GeneratorFactory
    from Core.processing import StatisticsAdapter, FftAdapter

    gen = GeneratorFactory.create("cw", ctx, params)
    signal = gen.generate(n_samples)

    stats = StatisticsAdapter(ctx)
    result = stats.process(signal.reshape(1, -1))
"""
from . import generators
from . import processing
