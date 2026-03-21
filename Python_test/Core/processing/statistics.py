"""
StatisticsAdapter -- GPU Статистика: mean, std, median.
GoF Adapter: gpuworklib.StatisticsProcessor -> удобный Python-интерфейс.
"""

import numpy as np
from .base import GpuProcessorMixin


class StatisticsAdapter(GpuProcessorMixin):
    """
    GPU Статистика: mean, std, median.

    GoF Adapter: gpuworklib.StatisticsProcessor -> удобный Python-интерфейс.

    Входные данные:
        np.ndarray shape=(n_channels, n_samples), dtype=complex64 или float32

    Результат process():
        dict с ключами: "mean", "std", "median"
        Каждый ключ -> np.ndarray shape=(n_channels,)

    Args:
        ctx: GPU контекст (ROCmGPUContext)
    """

    _processor_name = "StatisticsAdapter"

    def __init__(self, ctx):
        gw = self._load_gw()
        self._proc = gw.StatisticsProcessor(ctx)

    def process(self, data: np.ndarray) -> dict:
        """
        Вычисляет mean, std, median по каналам.

        Returns:
            {"mean": np.ndarray, "std": np.ndarray, "median": np.ndarray}
            Каждый массив shape=(n_channels,)
        """
        raw = self._proc.compute_statistics(data)
        # raw -- List[(mean, std, median)] по каналам
        n = len(raw)
        means   = np.array([raw[i][0] for i in range(n)], dtype=np.float32)
        stds    = np.array([raw[i][1] for i in range(n)], dtype=np.float32)
        medians = np.array([raw[i][2] for i in range(n)], dtype=np.float32)
        return {"mean": means, "std": stds, "median": medians}

    def compute_mean_std(self, data: np.ndarray) -> tuple:
        """Упрощённый вызов -- только mean и std (без median)."""
        result = self.process(data)
        return result["mean"], result["std"]
