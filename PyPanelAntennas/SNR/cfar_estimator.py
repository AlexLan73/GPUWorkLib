"""
cfar_estimator.py — CA-CFAR SNR оценщик на numpy.

Реплицирует точную логику C++ `peak_cfar_kernel` (план v4, раздел 2.2.7):
  1. Вычисление |X[k]|^2 из FFT complex output (square-law detector).
  2. argmax по всему спектру (или [0..N/2] если search_full_spectrum=False).
  3. CA-CFAR: mean(|X[k_ref]|^2) по guard+ref окну с wraparound по модулю nFFT.
  4. SNR_fft_dB = 10 * log10(peak_sq / noise_mean).

Также реплицирует логику PadDataOp:
  - step_samples = ceil(n_samples / target_n_fft) если step_samples=0
  - N_actual = n_samples // step_samples
  - nFFT = next_power_of_2(N_actual)   (как в FFTProcessorROCm::NextPowerOf2)
  - zero-pad входа с N_actual до nFFT

Это эталонная численная модель — результаты здесь должны совпадать
с C++ `SnrEstimatorOp` в пределах единиц ULP (после учёта float32 округления).

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Параметры (соответствуют SnrEstimationConfig из statistics_types.hpp)
# ============================================================================

@dataclass
class CfarConfig:
    """Аналог C++ SnrEstimationConfig."""
    target_n_fft: int = 0            # 0 = auto → 2048
    step_samples: int = 0            # 0 = auto из target_n_fft
    step_antennas: int = 0           # 0 = auto (ceil(n_ant / 50))
    guard_bins: int = 3
    ref_bins: int = 8
    search_full_spectrum: bool = True
    window: str = "hann"             # "rect", "hann", "hamming", "blackman", "flattop"
    cfar_estimator: str = "mean"     # "mean" (CA-CFAR) или "median" (OS-CFAR)


# Default константы из snr_defaults (плана)
K_TARGET_N_FFT = 2048
K_TARGET_ANTENNAS_MEDIAN = 50


# ============================================================================
# Вспомогательные функции — повторяем логику FFTProcessorROCm
# ============================================================================

def next_power_of_2(n: int) -> int:
    """Реплика FFTProcessorROCm::NextPowerOf2 (fft_processor_rocm.cpp:559).

    Возвращает минимальную степень 2, >= n.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_pipeline_sizes(
    n_samples: int,
    target_n_fft: int,
    step_samples: int,
) -> tuple[int, int, int]:
    """Вычислить (step_samples, n_actual, n_fft) по auto-правилам Config.

    Повторяет auto-логику из SnrEstimatorOp.Execute:
      1. target_n_fft = 0 → K_TARGET_N_FFT (2048)
      2. step_samples = 0 → ceil(n_samples / target_n_fft)
      3. n_actual    = n_samples // step_samples
      4. n_fft       = next_power_of_2(n_actual)
    """
    if target_n_fft == 0:
        target_n_fft = K_TARGET_N_FFT
    if step_samples == 0:
        step_samples = (n_samples + target_n_fft - 1) // target_n_fft
    n_actual = n_samples // step_samples
    n_fft = next_power_of_2(n_actual)
    return step_samples, n_actual, n_fft


def zero_pad(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """Zero-pad complex сигнала от len(signal) до n_fft.

    Реплицирует PadDataOp: копирует n_actual сэмплов, остаток — нули.
    """
    if len(signal) >= n_fft:
        return signal[:n_fft]
    padded = np.zeros(n_fft, dtype=signal.dtype)
    padded[: len(signal)] = signal
    return padded


# ============================================================================
# Window functions (справочник см. Harris 1978 "On the Use of Windows...")
# ============================================================================
#
# Sidelobe level (peak) и processing loss по Harris 1978:
#   Rect      : −13.3 dB, 0.00 dB loss,  main lobe width 0.89 bins
#   Hann      : −31.5 dB, 1.76 dB loss,  main lobe width 1.44 bins
#   Hamming   : −42.7 dB, 1.34 dB loss,  main lobe width 1.30 bins
#   Blackman  : −58.1 dB, 2.37 dB loss,  main lobe width 1.68 bins
#   Flat-top  : −93.0 dB, 3.86 dB loss,  main lobe width 3.72 bins
#
# Выбор: Hann — хороший компромисс (достаточно подавления + малые потери).
# Hamming — лучше по peak sidelobe но медленнее спад (не идеально для близких пиков).
# Blackman — очень чистый, но +0.6 dB loss.
# Flat-top — даёт точную амплитуду пика, но очень широкий main lobe (плохо для SNR).

def make_window(name: str, n: int) -> np.ndarray:
    """Создать оконную функцию по имени.

    Returns:
        np.float32[n]
    """
    name = name.lower()
    if name in ("rect", "rectangular", "none"):
        return np.ones(n, dtype=np.float32)

    k = np.arange(n, dtype=np.float64)
    nm1 = n - 1

    if name == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * math.pi * k / nm1))
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * math.pi * k / nm1)
    elif name == "blackman":
        # Exact Blackman coefficients (not the "classic" approximation)
        w = (0.42 - 0.5 * np.cos(2.0 * math.pi * k / nm1)
             + 0.08 * np.cos(4.0 * math.pi * k / nm1))
    elif name == "flattop":
        # SR785 flat-top (Matlab default)
        a0, a1, a2, a3, a4 = (
            0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
        )
        x = 2.0 * math.pi * k / nm1
        w = (a0 - a1 * np.cos(x) + a2 * np.cos(2 * x)
             - a3 * np.cos(3 * x) + a4 * np.cos(4 * x))
    else:
        raise ValueError(f"Unknown window: {name}")

    return w.astype(np.float32)


# ============================================================================
# CFAR детектор
# ============================================================================

@dataclass
class CfarResult:
    """Результат одной антенны."""
    snr_db: float
    k_peak: int
    peak_sq: float          # |X[k_peak]|^2
    noise_mean: float       # mean(|X[k_ref]|^2)
    n_actual: int
    n_fft: int


def estimate_snr_one_antenna(
    signal: np.ndarray,
    config: CfarConfig,
) -> CfarResult:
    """Оценить SNR_fft для одной антенны (по одному signal'у).

    Pipeline (повторяет C++ `SnrEstimatorOp`):
      1. Decimation: signal[::step_samples][:n_actual]
      2. Zero-pad до n_fft
      3. FFT
      4. |X|^2 (square-law)
      5. argmax по [0..n_fft-1] (или [0..n_fft//2])
      6. CA-CFAR: mean по guard+ref окну с wraparound
      7. SNR_dB = 10*log10(peak² / noise_mean)
    """
    n_samples = len(signal)
    step, n_actual, n_fft = compute_pipeline_sizes(
        n_samples, config.target_n_fft, config.step_samples
    )

    # 1. Decimation (берём каждый step-й, до n_actual)
    decimated = signal[::step][:n_actual].astype(np.complex64)

    # 1.5 Window (применяется ПОСЛЕ децимации ДО zero-padding)
    #     Это будущий C++ WindowedPadDataOp / параметр window в PadDataOp.
    if config.window != "rect":
        w = make_window(config.window, n_actual)
        decimated = (decimated * w).astype(np.complex64)

    # 2. Zero-pad до n_fft (как PadDataOp)
    padded = zero_pad(decimated, n_fft)

    # 3. FFT
    spectrum = np.fft.fft(padded, n=n_fft)

    # 4. |X|^2 (square-law) — реплика complex_to_magnitude_squared
    mag_sq = (spectrum.real * spectrum.real +
              spectrum.imag * spectrum.imag).astype(np.float32)

    # 5. argmax — либо по всему спектру, либо по [0..n_fft/2)
    search_end = n_fft if config.search_full_spectrum else n_fft // 2
    k_peak = int(np.argmax(mag_sq[:search_end]))
    peak_sq = float(mag_sq[k_peak])

    # 6. CFAR ref-window: k_peak ± (guard+1 .. guard+ref) mod n_fft
    # Собираем ref bins в массив для median/mean
    ref_values = np.empty(2 * config.ref_bins, dtype=np.float32)
    for i in range(config.ref_bins):
        offset = config.guard_bins + 1 + i
        k_left = (k_peak - offset) % n_fft
        k_right = (k_peak + offset) % n_fft
        ref_values[2 * i] = mag_sq[k_left]
        ref_values[2 * i + 1] = mag_sq[k_right]

    if config.cfar_estimator == "median":
        # OS-CFAR: устойчив к выбросам (sinc sidelobes не портят оценку)
        noise_est = float(np.median(ref_values))
    else:
        # CA-CFAR: классическое среднее
        noise_est = float(ref_values.mean())

    noise_est = max(noise_est, 1e-30)  # защита от log10(0)

    # 7. SNR_dB = 10*log10(peak / noise_est)
    ratio = max(peak_sq / noise_est, 1e-30)
    snr_db = 10.0 * math.log10(ratio)

    # Для совместимости с CfarResult оставляем поле noise_mean
    noise_mean = noise_est

    return CfarResult(
        snr_db=snr_db,
        k_peak=k_peak,
        peak_sq=peak_sq,
        noise_mean=noise_mean,
        n_actual=n_actual,
        n_fft=n_fft,
    )


def estimate_snr_many_antennas(
    data: np.ndarray,
    config: CfarConfig,
) -> tuple[float, np.ndarray, CfarResult]:
    """Оценить SNR для 2D массива [n_antennas × n_samples].

    Применяет step_antennas: берёт каждую step_antennas-ю антенну.
    Возвращает (median_snr_db, per_antenna_snr_db_array, last_cfar_result_for_debug).
    """
    n_antennas, n_samples = data.shape

    step_ant = config.step_antennas
    if step_ant == 0:
        step_ant = max(1, (n_antennas + K_TARGET_ANTENNAS_MEDIAN - 1)
                          // K_TARGET_ANTENNAS_MEDIAN)

    used_indices = list(range(0, n_antennas, step_ant))
    n_ant_used = len(used_indices)

    snr_per_ant = np.zeros(n_ant_used, dtype=np.float32)
    last_result: Optional[CfarResult] = None

    for i, ant_idx in enumerate(used_indices):
        result = estimate_snr_one_antenna(data[ant_idx], config)
        snr_per_ant[i] = result.snr_db
        last_result = result

    median_snr_db = float(np.median(snr_per_ant))
    return median_snr_db, snr_per_ant, last_result  # type: ignore[return-value]


# ============================================================================
# Удобный класс-обёртка для e2e тестов Python_test/statistics/
# ============================================================================

class CfarEstimator:
    """Simple wrapper for e2e tests — matches Python bindings API."""

    def __init__(
        self,
        target_n_fft: int = 0,
        guard_bins: int = 3,
        ref_bins: int = 8,
        search_full_spectrum: bool = True,
        window: str = "hann",
    ) -> None:
        self.config = CfarConfig(
            target_n_fft=target_n_fft,
            guard_bins=guard_bins,
            ref_bins=ref_bins,
            search_full_spectrum=search_full_spectrum,
            window=window,
        )

    def estimate(self, signal: np.ndarray) -> float:
        """Вернуть snr_db для одной антенны."""
        return estimate_snr_one_antenna(signal, self.config).snr_db

    def estimate_batch(self, data: np.ndarray) -> float:
        """Вернуть медиану snr_db для 2D [n_ant × n_samp]."""
        median_db, _, _ = estimate_snr_many_antennas(data, self.config)
        return median_db
