"""
NumPy/SciPy-эталоны для фильтров. GRASP Information Expert.
"""

import numpy as np

try:
    from scipy import signal as scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class FilterReferences:
    """
    SciPy-реализации GPU-фильтров.

    Использование:
        from common.references import FilterReferences

        ref = FilterReferences.fir_lowpass(data, fs=50e3, cutoff_hz=1e3, n_taps=64)
    """

    @staticmethod
    def _check_scipy():
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy требуется для FilterReferences. pip install scipy")

    @staticmethod
    def fir_lowpass(data: np.ndarray, fs: float, cutoff_hz: float,
                    n_taps: int = 64, window: str = "hamming") -> np.ndarray:
        """
        FIR lowpass фильтр через scipy.

        Returns: complex64, shape=data.shape
        """
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        coeffs = scipy_signal.firwin(n_taps, cutoff_hz / nyq, window=window)
        result = scipy_signal.lfilter(coeffs, [1.0], data)
        return result.astype(np.complex64)

    @staticmethod
    def fir_bandpass(data: np.ndarray, fs: float,
                     f_low: float, f_high: float,
                     n_taps: int = 64) -> np.ndarray:
        """FIR bandpass фильтр через scipy."""
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        coeffs = scipy_signal.firwin(
            n_taps, [f_low / nyq, f_high / nyq],
            pass_zero=False
        )
        return scipy_signal.lfilter(coeffs, [1.0], data).astype(np.complex64)

    @staticmethod
    def fir_from_coeffs(data: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """FIR с произвольными коэффициентами."""
        FilterReferences._check_scipy()
        return scipy_signal.lfilter(coeffs, [1.0], data).astype(np.complex64)

    @staticmethod
    def iir_lowpass(data: np.ndarray, fs: float, cutoff_hz: float,
                    order: int = 4, ftype: str = "butter") -> np.ndarray:
        """
        IIR lowpass фильтр через scipy.sosfilt (численно стабильный).
        """
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        sos = scipy_signal.iirfilter(
            order, cutoff_hz / nyq,
            btype="low", ftype=ftype, output="sos"
        )
        return scipy_signal.sosfilt(sos, data).astype(np.complex64)
