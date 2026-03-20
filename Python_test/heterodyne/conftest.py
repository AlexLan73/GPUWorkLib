"""
conftest.py — фабричные функции для Python_test/heterodyne/
============================================================

Предоставляет factory functions для тестов гетеродина/дечирпа.
"""

import os
from dataclasses import dataclass
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Параметры по умолчанию (из C++ тестов)
# ─────────────────────────────────────────────────────────────────────────────

_FS = 12e6
_F_START = 0.0
_F_END = 2e6
_N = 8000
_N_ANTENNAS = 5

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
het_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "heterodyne")
os.makedirs(het_plot_dir, exist_ok=True)


@dataclass
class DechirpParams:
    """Параметры LFM Dechirp тестов."""
    fs: float = _FS
    f_start: float = _F_START
    f_end: float = _F_END
    n_samples: int = _N
    n_antennas: int = _N_ANTENNAS
    c_light: float = 3e8

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def chirp_rate(self) -> float:
        return self.bandwidth / self.duration

    def range_from_delay(self, delay_s: float) -> float:
        return self.c_light * delay_s / 2.0

    def fbeat_from_delay(self, delay_s: float) -> float:
        return self.chirp_rate * delay_s


# ─────────────────────────────────────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────────────────────────────────────

def make_dechirp_params() -> DechirpParams:
    """Параметры дечирпа по умолчанию."""
    return DechirpParams()


def make_het_proc(gw, gpu_ctx, params: DechirpParams = None):
    """HeterodyneDechirp — GPU процессор."""
    if params is None:
        params = DechirpParams()
    return gw.HeterodyneDechirp(
        gpu_ctx,
        params.fs, params.f_start, params.f_end, params.n_samples, params.n_antennas
    )


def make_delays_linear_us(n_antennas: int = _N_ANTENNAS) -> np.ndarray:
    """Линейные задержки (100..500 мкс)."""
    return np.array([100., 200., 300., 400., 500.])[:n_antennas]


def make_lfm_srx(params: DechirpParams = None,
                 delays_us: np.ndarray = None) -> np.ndarray:
    """Приёмный ЛЧМ сигнал [n_antennas, n_samples] с задержками."""
    if params is None:
        params = DechirpParams()
    if delays_us is None:
        delays_us = make_delays_linear_us(params.n_antennas)
    delays_s = delays_us * 1e-6
    S = np.zeros((params.n_antennas, params.n_samples), dtype=np.complex64)
    for ant in range(params.n_antennas):
        tau = delays_s[ant]
        t = np.arange(params.n_samples) / params.fs
        mask = t >= tau
        t_local = t[mask] - tau
        phase = 2 * np.pi * (params.f_start * t_local +
                              0.5 * params.chirp_rate * t_local ** 2)
        S[ant, mask] = np.exp(1j * phase).astype(np.complex64)
    return S


def make_s_ref(params: DechirpParams = None) -> np.ndarray:
    """Опорный ЛЧМ сигнал [n_samples] с нулевой задержкой."""
    if params is None:
        params = DechirpParams()
    t = np.arange(params.n_samples) / params.fs
    phase = 2 * np.pi * (params.f_start * t + 0.5 * params.chirp_rate * t ** 2)
    return np.exp(1j * phase).astype(np.complex64)
