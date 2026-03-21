"""
NumPy FFT-эталоны. GRASP Information Expert.
"""

import numpy as np


class FftReferences:
    """
    NumPy-реализации GPU-FFT.

    Использование:
        from common.references import FftReferences

        spec     = FftReferences.fft(data)
        mag      = FftReferences.magnitude(data)
        peak_hz  = FftReferences.peak_freq(data, fs=12e6)
    """

    @staticmethod
    def fft(data: np.ndarray, n_fft: int = None) -> np.ndarray:
        """
        FFT через numpy.

        Returns: complex64, shape=(n_fft,) или (n_channels, n_fft)
        """
        if data.ndim == 1:
            return np.fft.fft(data, n=n_fft).astype(np.complex64)
        return np.fft.fft(data, n=n_fft, axis=-1).astype(np.complex64)

    @staticmethod
    def magnitude(data: np.ndarray, n_fft: int = None) -> np.ndarray:
        """
        |FFT| -- амплитудный спектр.

        Returns: float32, shape=(n_fft,) или (n_channels, n_fft)
        """
        return np.abs(FftReferences.fft(data, n_fft)).astype(np.float32)

    @staticmethod
    def magnitude_db(data: np.ndarray, n_fft: int = None,
                     ref: float = 1.0) -> np.ndarray:
        """Амплитудный спектр в дБ."""
        mag = FftReferences.magnitude(data, n_fft)
        return (20 * np.log10(mag / ref + 1e-12)).astype(np.float32)

    @staticmethod
    def peak_freq(data: np.ndarray, fs: float,
                  n_fft: int = None) -> float:
        """
        Частота пика спектра (Гц).

        Returns: float -- частота максимального бина
        """
        mag = FftReferences.magnitude(data, n_fft)
        n = mag.shape[-1]
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        peak_idx = int(np.argmax(mag))
        return float(freqs[peak_idx])

    @staticmethod
    def freq_axis(n_fft: int, fs: float) -> np.ndarray:
        """Ось частот (Гц) для построения графиков."""
        return np.fft.fftfreq(n_fft, d=1.0 / fs).astype(np.float32)
