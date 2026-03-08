"""
conftest.py — pytest fixtures для Python_test/heterodyne/
==========================================================

Наследует gpu_ctx, gw из корневого conftest.py.
Fixtures специфичны для тестов гетеродина/дечирпа.

Fixtures:
    het_plot_dir   — Results/Plots/heterodyne/
    dechirp_params — параметры ЛЧМ дечирпа (FS, B, N, MU, C)
    het_proc       — HeterodyneDechirp (GPU, scope=session)
    lfm_srx        — тестовый приёмный ЛЧМ сигнал
"""

import os
from dataclasses import dataclass
import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Параметры по умолчанию (из C++ тестов)
# ─────────────────────────────────────────────────────────────────────────────

_FS = 12e6          # 12 МГц
_F_START = 0.0
_F_END = 2e6        # 2 МГц
_N = 8000           # число отсчётов
_N_ANTENNAS = 5


@dataclass
class DechirpParams:
    """Параметры LFM Dechirp тестов.

    Attributes:
        fs:          частота дискретизации (Гц)
        f_start:     начальная частота ЛЧМ (Гц)
        f_end:       конечная частота ЛЧМ (Гц)
        n_samples:   число отсчётов
        n_antennas:  число антенн
        c_light:     скорость света (м/с)
    """
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
        """Скорость изменения частоты (Гц/с)."""
        return self.bandwidth / self.duration

    def range_from_delay(self, delay_s: float) -> float:
        """Дальность по задержке."""
        return self.c_light * delay_s / 2.0

    def fbeat_from_delay(self, delay_s: float) -> float:
        """Биение (beat frequency) по задержке."""
        return self.chirp_rate * delay_s


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def het_plot_dir(plot_dir) -> str:
    """Директория Results/Plots/heterodyne/."""
    path = os.path.join(plot_dir, "heterodyne")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def dechirp_params() -> DechirpParams:
    """Параметры дечирпа (фиксированные для всех тестов сессии)."""
    return DechirpParams()


@pytest.fixture(scope="session")
def het_proc(gw, gpu_ctx, dechirp_params):
    """HeterodyneDechirp — GPU процессор (skip если нет GPU)."""
    p = dechirp_params
    return gw.HeterodyneDechirp(
        gpu_ctx,
        p.fs, p.f_start, p.f_end, p.n_samples, p.n_antennas
    )


@pytest.fixture(scope="module")
def delays_linear_us(dechirp_params) -> np.ndarray:
    """Линейные задержки для 5 антенн (100..500 мкс)."""
    return np.array([100., 200., 300., 400., 500.])[:dechirp_params.n_antennas]


@pytest.fixture(scope="module")
def lfm_srx(dechirp_params, delays_linear_us) -> np.ndarray:
    """Приёмный ЛЧМ сигнал [n_antennas, n_samples] с задержками."""
    p = dechirp_params
    delays_s = delays_linear_us * 1e-6
    S = np.zeros((p.n_antennas, p.n_samples), dtype=np.complex64)

    for ant in range(p.n_antennas):
        tau = delays_s[ant]
        t = np.arange(p.n_samples) / p.fs
        mask = t >= tau
        t_local = t[mask] - tau
        phase = 2 * np.pi * (p.f_start * t_local +
                              0.5 * p.chirp_rate * t_local ** 2)
        S[ant, mask] = np.exp(1j * phase).astype(np.complex64)

    return S


@pytest.fixture(scope="module")
def s_ref(dechirp_params) -> np.ndarray:
    """Опорный ЛЧМ сигнал [n_samples] с нулевой задержкой."""
    p = dechirp_params
    t = np.arange(p.n_samples) / p.fs
    phase = 2 * np.pi * (p.f_start * t + 0.5 * p.chirp_rate * t ** 2)
    return np.exp(1j * phase).astype(np.complex64)
