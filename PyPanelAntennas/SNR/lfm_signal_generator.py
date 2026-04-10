"""
lfm_signal_generator.py — генератор синтетических данных для SNR-модели.

Создаёт:
  - Комплексный CW (tone) — эмулирует "дечирпированный" LFM (после гетеродина).
    Это основной тип данных который придёт в `ComputeSnrDb`: LFM → dechirp → CW + шум.
  - AWGN (complex): σ² = noise_power, равномерно распределено между re/im.
  - Полный LFM chirp (для проверки pipeline dechirp → CW).

Принцип SNR:
    SNR_in (dB) = 10 * log10(|A|² / σ²)
    где A — амплитуда CW, σ² — мощность комплексного шума.

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ============================================================================
# CW (dechirped tone) — основной тип для тестов SNR
# ============================================================================

def make_cw(
    n_samples: int,
    freq_norm: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Создать комплексный CW.

    Args:
        n_samples:  число отсчётов
        freq_norm:  нормированная частота f_d/f_s ∈ (-0.5, 0.5)
        amplitude:  A (реальная амплитуда)
        phase:      начальная фаза (радианы)

    Returns:
        complex64[n_samples] — сигнал A*exp(j*(2π*f_norm*n + phase))
    """
    n = np.arange(n_samples, dtype=np.float64)
    theta = 2.0 * math.pi * freq_norm * n + phase
    signal = amplitude * (np.cos(theta) + 1j * np.sin(theta))
    return signal.astype(np.complex64)


# ============================================================================
# AWGN (complex white Gaussian noise)
# ============================================================================

def make_awgn(
    n_samples: int,
    noise_power: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Комплексный AWGN с дисперсией σ² = noise_power.

    Каждая компонента (re, im) имеет дисперсию σ²/2 — суммарная мощность σ².

    Args:
        n_samples:    число отсчётов
        noise_power:  σ² (полная мощность комплексного шума)
        rng:          np.random.Generator (для воспроизводимости)

    Returns:
        complex64[n_samples] ~ CN(0, σ²)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigma = math.sqrt(noise_power / 2.0)
    re = rng.standard_normal(n_samples) * sigma
    im = rng.standard_normal(n_samples) * sigma
    return (re + 1j * im).astype(np.complex64)


# ============================================================================
# CW + AWGN c заданным SNR_in
# ============================================================================

def make_cw_with_snr(
    n_samples: int,
    freq_norm: float,
    snr_in_db: float,
    noise_power: float = 1.0,
    rng: np.random.Generator | None = None,
    phase: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Создать CW + AWGN с заданным SNR_in.

    SNR_in (dB) = 10*log10(A² / σ²)
    →  A = σ * 10^(SNR_dB / 20)
    →  A² = σ² * 10^(SNR_dB / 10)

    Returns:
        (signal, amplitude)
    """
    amplitude = math.sqrt(noise_power * 10.0 ** (snr_in_db / 10.0))
    cw = make_cw(n_samples, freq_norm, amplitude, phase)
    noise = make_awgn(n_samples, noise_power, rng)
    return (cw + noise).astype(np.complex64), amplitude


# ============================================================================
# Полный LFM chirp (для проверки pipeline LFM → dechirp → CW)
# ============================================================================

def make_lfm_chirp(
    n_samples: int,
    f_start: float,
    f_end: float,
    sample_rate: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Линейно-частотно-модулированный (LFM) сигнал.

    Мгновенная частота: f(t) = f_start + (f_end - f_start) * t / T
    Фаза:              φ(t) = 2π * integral(f(τ) dτ)
                             = 2π * (f_start*t + (f_end-f_start)/(2T) * t²)

    Args:
        n_samples:    длительность (отсчёты)
        f_start:      начальная частота (Гц)
        f_end:        конечная частота (Гц)
        sample_rate:  частота дискретизации (Гц)
        amplitude:    A

    Returns:
        complex64[n_samples]
    """
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    T = n_samples / sample_rate
    K = (f_end - f_start) / T
    phase = 2.0 * math.pi * (f_start * t + 0.5 * K * t * t)
    return (amplitude * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)
