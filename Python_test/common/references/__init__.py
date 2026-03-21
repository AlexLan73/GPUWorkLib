"""
NumPy/SciPy -- единое место для CPU-реализаций GPU-алгоритмов.
DRY: каждая формула живёт в одном месте.

GRASP Information Expert: каждый класс знает формулы своей предметной области.
SOLID SRP: один класс = одна категория эталонов.

Использование:
    from common.references import SignalReferences, FilterReferences
    from common.references import StatisticsReferences, FftReferences

    ref_cw  = SignalReferences.cw(fs=12e6, n_samples=4096, f0=2e6)
    ref_lfm = SignalReferences.lfm(fs=12e6, n_samples=4096, f_start=0, f_end=2e6)
    ref_fft = FftReferences.magnitude(ref_cw)
    peak_hz = FftReferences.peak_freq(ref_cw, fs=12e6)
"""

from .signal_refs import SignalReferences
from .filter_refs import FilterReferences
from .statistics_refs import StatisticsReferences
from .fft_refs import FftReferences

__all__ = [
    "SignalReferences",
    "FilterReferences",
    "StatisticsReferences",
    "FftReferences",
]
