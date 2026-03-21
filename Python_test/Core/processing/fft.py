"""
FftAdapter -- GPU FFT-процессор.
GoF Adapter: gpuworklib.FFTProcessorROCm -> удобный Python-интерфейс.

Ветка main -- только ROCm (FFTProcessorROCm).
OpenCL/clFFT вынесен в legacy/opencl-clfft.
"""

import numpy as np
from .base import GpuProcessorMixin


class FftAdapter(GpuProcessorMixin):
    """
    GPU FFT-процессор.

    Входные данные:
        np.ndarray shape=(n_samples,) или (n_channels, n_samples), dtype=complex64

    Результат (зависит от mode):
        mode="complex"   -> complex64 ndarray (FFT bins)
        mode="magnitude" -> float32 ndarray (|FFT|)

    Args:
        ctx:     GPU контекст (ROCmGPUContext)
        n_fft:   Размер FFT (степень 2, например 4096)
        mode:    "complex" | "magnitude" (default="complex")
    """

    def __init__(self, ctx, n_fft: int, mode: str = "complex"):
        if mode not in ("complex", "magnitude"):
            raise ValueError(f"mode должен быть 'complex' или 'magnitude', не '{mode}'")
        gw = self._load_gw()

        # Ветка main: только ROCm
        self._proc = gw.FFTProcessorROCm(ctx, n_fft)
        self._n_fft = n_fft
        self._mode = mode
        self._processor_name = f"FftAdapter(n_fft={n_fft}, mode={mode})"

    def process(self, data: np.ndarray) -> np.ndarray:
        """Выполняет FFT на GPU. Возвращает complex64 или float32."""
        if self._mode == "complex":
            return self._proc.process_complex(data)
        else:
            return self._proc.process_magnitude(data)

    @property
    def n_fft(self) -> int:
        return self._n_fft
