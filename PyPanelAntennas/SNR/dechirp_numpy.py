"""
dechirp_numpy.py — numpy реализация гетеродина (дечирпа) для LFM.

Основная операция: умножение принятого сигнала на комплексно-сопряжённый
опорный LFM сигнал. Результат — CW (тональный) на доплеровской частоте.

    s_dechirped(t) = s_rx(t) * conj(s_ref(t))

Используется в Эксп.1 для проверки полного pipeline:
    LFM → dechirp → CW + noise → ComputeSnrDb

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import numpy as np

from lfm_signal_generator import make_lfm_chirp


def dechirp(
    rx_signal: np.ndarray,
    f_start: float,
    f_end: float,
    sample_rate: float = 1.0,
) -> np.ndarray:
    """Дечирпировать принятый LFM через умножение на conj(опорный_LFM).

    Args:
        rx_signal:    принятый сигнал [n_samples] (complex)
        f_start:      начальная частота опорного chirp (Гц)
        f_end:        конечная частота опорного chirp (Гц)
        sample_rate:  частота дискретизации (Гц)

    Returns:
        dechirped signal [n_samples] (complex64)
    """
    n = len(rx_signal)
    ref = make_lfm_chirp(n, f_start, f_end, sample_rate, amplitude=1.0)
    return (rx_signal * np.conj(ref)).astype(np.complex64)
