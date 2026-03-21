"""
HeterodyneAdapter -- GPU Гетеродин/Дечирп для ЛЧМ.
GoF Adapter: gpuworklib.HeterodyneDechirp -> удобный Python-интерфейс.
"""

import numpy as np
from .base import GpuProcessorMixin
from common.configs import HeterodyneConfig


class HeterodyneAdapter(GpuProcessorMixin):
    """
    GPU Гетеродин/Дечирп для ЛЧМ.

    GoF Adapter: gpuworklib.HeterodyneDechirp -> удобный Python-интерфейс.

    Входные данные:
        np.ndarray shape=(n_antennas, n_samples), dtype=complex64

    Результат:
        np.ndarray shape=(n_antennas, n_samples), dtype=complex64
        Дечирпованный сигнал: s_dc = s_rx * conj(s_ref)

    Args:
        ctx:     GPU контекст
        params:  HeterodyneConfig (расширяет SignalConfig + chirp_rate, n_antennas)
    """

    _processor_name = "HeterodyneAdapter"

    def __init__(self, ctx, params: HeterodyneConfig):
        gw = self._load_gw()
        self._proc = gw.HeterodyneDechirp(
            ctx, params.fs, params.f_start, params.f_end,
            params.n_samples, params.n_antennas
        )
        self._params = params

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Выполняет дечирп на GPU.

        Args:
            data: shape=(n_antennas, n_samples), dtype=complex64

        Returns:
            np.ndarray: shape=(n_antennas, n_samples), dtype=complex64
        """
        return self._proc.dechirp(data)

    @property
    def n_antennas(self) -> int:
        return self._params.n_antennas

    @property
    def params(self) -> HeterodyneConfig:
        """Параметры дечирпа (chirp_rate, fbeat_from_delay и т.д.)."""
        return self._params
