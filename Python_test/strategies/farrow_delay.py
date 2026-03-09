#!/usr/bin/env python3
"""
FarrowDelay -- numpy reference implementation of Lagrange fractional delay
===========================================================================

Port of LchFarrow (C++/OpenCL/ROCm) algorithm to pure numpy.
Uses the same Lagrange 48x5 coefficient matrix.

Algorithm:
  For each output sample n:
    read_pos = n - delay_samples
    if read_pos out of bounds: output[n] = 0
    else: output[n] = 5-point Lagrange interpolation at read_pos

  Lagrange matrix: 48 rows (fractional subdivisions), 5 columns (interpolation points)
    frac_idx = round(frac_part * 48) % 48
    coeffs = matrix[frac_idx]  -> 5 coefficients
    output[n] = sum(coeffs[k] * input[read_base - 2 + k], k=0..4)

Usage:
  farrow = FarrowDelay()  # loads built-in 48x5 matrix

  # Apply delay per antenna
  S_delayed = farrow.apply(S, delays_samples)

  # Or from physical delays (seconds)
  S_delayed = farrow.apply_seconds(S, delays_s, sample_rate)

Author: Kodo (AI Assistant)
Date: 2026-03-08
"""

import json
import os
import numpy as np
from typing import Optional


# ============================================================================
# Built-in Lagrange 48x5 matrix (from lch_farrow module)
# ============================================================================

# Path to JSON file in lch_farrow module
_DEFAULT_MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "modules", "lch_farrow", "lagrange_matrix_48x5.json"
)


def load_lagrange_matrix(json_path: Optional[str] = None) -> np.ndarray:
    """Load Lagrange coefficient matrix from JSON.

    Args:
        json_path: path to JSON file, None = built-in 48x5

    Returns:
        np.ndarray [rows, columns] float32  (matches GPU float precision)
    """
    path = json_path or _DEFAULT_MATRIX_PATH
    with open(path, 'r') as f:
        data = json.load(f)
    arr = np.array(data['data'], dtype=np.float32)
    return arr.reshape(data['rows'], data['columns'])


# ============================================================================
# FarrowDelay class
# ============================================================================

class FarrowDelay:
    """Numpy reference implementation of Lagrange 48x5 fractional delay.

    Applies per-antenna fractional delay to complex signal matrix.
    Compatible with C++ LchFarrow / LchFarrowROCm algorithm.

    Attributes:
        matrix: Lagrange coefficient matrix [48, 5]
        n_subdivisions: number of fractional subdivisions (48)
        n_points: interpolation order + 1 (5)
    """

    def __init__(self, matrix_path: Optional[str] = None):
        """Initialize with Lagrange coefficient matrix.

        Args:
            matrix_path: path to JSON, None = built-in modules/lch_farrow/lagrange_matrix_48x5.json
        """
        self.matrix = load_lagrange_matrix(matrix_path)
        self.n_subdivisions = self.matrix.shape[0]  # 48
        self.n_points = self.matrix.shape[1]         # 5
        self._half_points = self.n_points // 2       # 2 (center offset)

    def apply_single(self, signal_1d: np.ndarray,
                     delay_samples: float) -> np.ndarray:
        """Apply fractional delay to single channel.

        Mirrors GPU kernel (lch_farrow.cl) exactly:
          read_pos = n - delay_samples   (per-sample, float32)
          center   = floor(read_pos)
          frac     = read_pos - center   (fractional part of read_pos, NOT of delay!)
          row      = int(frac * 48) % 48
          output[n] = sum(L[row][k] * input[center - 1 + k], k=0..4)

        Args:
            signal_1d: [n_samples] complex signal
            delay_samples: delay in samples (float, can have fractional part)

        Returns:
            [n_samples] complex delayed signal
        """
        n = len(signal_1d)
        output = np.zeros(n, dtype=signal_1d.dtype)
        # Convert to float32 to match GPU arithmetic precision
        delay_f32 = np.float32(delay_samples)

        for i in range(n):
            read_pos = float(np.float32(i)) - float(delay_f32)
            if read_pos < 0.0:
                continue
            center = int(np.floor(read_pos))
            frac = read_pos - center                          # frac of read_pos (not of delay!)
            row = int(frac * self.n_subdivisions) % self.n_subdivisions  # int(), not round()
            coeffs = self.matrix[row]
            val = 0.0 + 0.0j
            for k in range(self.n_points):
                src_idx = center - 1 + k                     # centre-1 .. centre+3
                if 0 <= src_idx < n:
                    val += float(coeffs[k]) * complex(signal_1d[src_idx])
            output[i] = val

        return output.astype(signal_1d.dtype)

    def apply(self, signal_2d: np.ndarray,
              delays_samples: np.ndarray) -> np.ndarray:
        """Apply per-antenna fractional delay.

        Args:
            signal_2d: [n_ant, n_samples] complex signal
            delays_samples: [n_ant] delays in samples (float)

        Returns:
            [n_ant, n_samples] complex delayed signal
        """
        n_ant, n_samples = signal_2d.shape
        assert len(delays_samples) == n_ant, \
            f"delays_samples length {len(delays_samples)} != n_ant {n_ant}"

        output = np.zeros_like(signal_2d)
        for ant in range(n_ant):
            output[ant] = self.apply_single(signal_2d[ant], delays_samples[ant])

        return output

    def apply_seconds(self, signal_2d: np.ndarray,
                      delays_s: np.ndarray,
                      sample_rate: float) -> np.ndarray:
        """Apply delay specified in seconds.

        Args:
            signal_2d: [n_ant, n_samples] complex signal
            delays_s: [n_ant] delays in seconds
            sample_rate: sampling frequency (Hz)

        Returns:
            [n_ant, n_samples] complex delayed signal
        """
        delays_samples = delays_s * sample_rate
        return self.apply(signal_2d, delays_samples)

    def compensate(self, signal_2d: np.ndarray,
                   delays_samples: np.ndarray) -> np.ndarray:
        """Compensate (remove) delay -- apply NEGATIVE delay.

        For beamforming: compensate natural delays to align antennas.

        Args:
            signal_2d: [n_ant, n_samples] signal WITH delays
            delays_samples: [n_ant] delays to REMOVE (positive values)

        Returns:
            [n_ant, n_samples] aligned signal (delays removed)
        """
        return self.apply(signal_2d, -delays_samples)

    def compensate_seconds(self, signal_2d: np.ndarray,
                           delays_s: np.ndarray,
                           sample_rate: float) -> np.ndarray:
        """Compensate delay specified in seconds.

        Args:
            signal_2d: [n_ant, n_samples] signal WITH delays
            delays_s: [n_ant] delays to REMOVE (seconds, positive)
            sample_rate: Hz

        Returns:
            [n_ant, n_samples] aligned signal
        """
        delays_samples = delays_s * sample_rate
        return self.compensate(signal_2d, delays_samples)
