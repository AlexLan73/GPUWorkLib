"""
GPU-процессоры как переиспользуемые Adapter'ы.

GoF Adapter: оборачивает gpuworklib процессоры в удобный Python-интерфейс.
GpuProcessorMixin: общий __repr__ и проверка GPULoader (без навязывания сигнатуры).

Использование:
    from Core.processing import StatisticsAdapter, FftAdapter, HeterodyneAdapter

    stats = StatisticsAdapter(ctx)
    result = stats.process(data)  # -> dict

    fft = FftAdapter(ctx, n_fft=4096)
    spectrum = fft.process(signal)  # -> ndarray complex64
"""

from .base import GpuProcessorMixin
from .statistics import StatisticsAdapter
from .heterodyne import HeterodyneAdapter
from .fft import FftAdapter

__all__ = [
    "GpuProcessorMixin",
    "StatisticsAdapter",
    "HeterodyneAdapter",
    "FftAdapter",
]
