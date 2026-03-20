"""
conftest.py — фабричные функции для Python_test/signal_generators/
===================================================================

Предоставляет factory functions для тестов генераторов сигналов.
NumPy-эталонные функции также здесь (Information Expert).
"""

import os
import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Параметры по умолчанию
# ─────────────────────────────────────────────────────────────────────────────

_FS = 12e6        # 12 МГц
_LENGTH = 8192
_F0 = 2e6         # 2 МГц несущая
_FDEV = 1e6       # девиация ЛЧМ

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sig_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "signal_generators")
os.makedirs(sig_plot_dir, exist_ok=True)


@dataclass
class LfmParams:
    """Параметры ЛЧМ-сигнала для тестов."""
    fs: float = _FS
    length: int = _LENGTH
    f_start: float = 0.0
    f_end: float = 2e6
    amplitude: float = 1.0
    phase: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────────────────────────────────────

def make_sig_gen(gw, gpu_ctx):
    """SignalGenerator (создаётся один раз)."""
    return gw.SignalGenerator(gpu_ctx)


def make_fft_proc(gw, gpu_ctx):
    """FFTProcessor (создаётся один раз)."""
    return gw.FFTProcessor(gpu_ctx)


def make_lfm_params() -> LfmParams:
    """Типичные параметры ЛЧМ-сигнала."""
    return LfmParams()


# ─────────────────────────────────────────────────────────────────────────────
# NumPy эталонные функции
# ─────────────────────────────────────────────────────────────────────────────

def cw_numpy(fs: float, length: int, f0: float,
             amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Эталонный CW сигнал (numpy)."""
    t = np.arange(length) / fs
    return (amplitude * np.exp(1j * (2 * np.pi * f0 * t + phase))).astype(np.complex64)


def lfm_numpy(fs: float, length: int, f_start: float, f_end: float,
              amplitude: float = 1.0) -> np.ndarray:
    """Эталонный ЛЧМ сигнал (numpy)."""
    t = np.arange(length) / fs
    duration = length / fs
    chirp_rate = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def getX_numpy(fs: float, points: int, f0: float, amplitude: float,
               phase: float, fdev: float, norm_val: float, tau: float = 0.0) -> np.ndarray:
    """CPU reference FormSignal (формула getX без шума)."""
    dt = 1.0 / fs
    ti = points * dt
    t = np.arange(points, dtype=np.float64) * dt + tau

    in_window = (t >= 0.0) & (t <= ti - dt)
    t_centered = t - ti / 2.0
    ph = 2.0 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase

    X = amplitude * norm_val * np.exp(1j * ph)
    X[~in_window] = 0.0
    return X.astype(np.complex64)
