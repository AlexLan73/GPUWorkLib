"""
numpy_reference.py — CPU-эталон pipeline strategies
=====================================================

Information Expert (GRASP):
    NumpyReference хранит все эталонные данные и умеет
    вычислять статистику по той же формуле что и GPU-kernel.

Value Object (GoF):
    После инициализации данные не изменяются.
    Все поля доступны только для чтения.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class StatsRef:
    """Эталонная статистика одного луча (beam).

    Повторяет структуру StatisticsResult из C++.
    Все поля float32 / complex64 — как в GPU.
    """
    beam_id       : int
    mean          : complex  # np.complex64 — комплексное среднее
    variance      : float    # float32 — Var(|z|) от магнитуд
    std_dev       : float    # float32 — sqrt(Var(|z|))
    mean_magnitude: float    # float32 — E[|z|]


class NumpyReference:
    """CPU-эталон для всех шагов pipeline AntennaProcessor.

    Вычисляет W@S, Hamming, FFT, статистику одновременно при инициализации.
    Используется в PipelineStepValidator для сравнения с GPU.

    Usage:
        ref = NumpyReference(
            S=S_cpu,          # [n_ant, n_samples] complex64
            W=W_cpu,          # [n_ant, n_ant] complex64
            fs=12e6,
            f0=2e6,
            n_fft=8192,
        )
        print(ref.X_ref.shape)        # (n_ant, n_samples)
        print(ref.spec_ref.shape)     # (n_ant, n_fft)
        print(ref.input_stats[0])     # StatsRef(beam_id=0, mean=..., ...)
    """

    def __init__(self,
                 S: np.ndarray,
                 W: np.ndarray,
                 fs: float,
                 f0: float,
                 n_fft: int):
        """
        Args:
            S:     входной сигнал [n_ant, n_samples] complex64
            W:     весовая матрица [n_ant, n_ant] complex64
            fs:    частота дискретизации (Гц)
            f0:    ожидаемая частота сигнала (Гц) — для проверки пика
            n_fft: размер FFT (>= n_samples, обычно следующая степень 2)
        """
        # Сохранить параметры
        self._fs    = float(fs)
        self._f0    = float(f0)
        self._n_fft = int(n_fft)
        self._n_ant, self._n_samples = S.shape

        # --- STEP 0: Сохранить входные данные как complex64 ---
        self.S_ref = S.astype(np.complex64)   # [n_ant, n_samples]
        self.W_ref = W.astype(np.complex64)   # [n_ant, n_ant]

        # --- STEP 1: Статистика входного сигнала d_S ---
        self.input_stats: list = self._compute_stats(self.S_ref)

        # --- STEP 2: GEMM X = W @ S ---
        # Использовать float64 для вычисления (точность), потом cast в complex64
        self.X_ref = (self.W_ref.astype(np.complex128) @
                      self.S_ref.astype(np.complex128)).astype(np.complex64)
        # shape: [n_ant, n_samples]

        # --- STEP 3: Статистика после GEMM d_X ---
        self.gemm_stats: list = self._compute_stats(self.X_ref)

        # --- STEP 4: Hamming + zero-pad + FFT ---
        # Окно Хэмминга (float32, как в GPU)
        self.hamm = np.hamming(self._n_samples).astype(np.float32)  # [n_samples]

        # Применить окно + zero-pad
        padded = np.zeros((self._n_ant, self._n_fft), dtype=np.complex64)
        padded[:, :self._n_samples] = (
            self.X_ref * self.hamm[np.newaxis, :]
        ).astype(np.complex64)

        # FFT (NumPy FFT работает корректно с complex64)
        self.spec_ref = np.fft.fft(padded).astype(np.complex64)  # [n_ant, n_fft]

        # Магнитуды (float32)
        self.mag_ref = np.abs(self.spec_ref).astype(np.float32)  # [n_ant, n_fft]

        # Ожидаемый бин пика: round(f0 * n_fft / fs)
        self.expected_peak_bin: int = round(self._f0 * self._n_fft / self._fs)

        # --- STEP 5: Статистика |spectrum| ---
        self.fft_stats: list = self._compute_stats(self.mag_ref)

    # ── Публичные свойства ───────────────────────────────────────────────────

    @property
    def n_ant(self) -> int:
        return self._n_ant

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def fs(self) -> float:
        return self._fs

    @property
    def f0(self) -> float:
        return self._f0

    def compute_dynamic_range_db(self) -> np.ndarray:
        """Вычислить dynamic_range_dB per beam (для CHECK-6.3).

        Формула: 20 * log10(max / max(min, 1e-30))
        Повторяет MinMaxResult.dynamic_range_dB из C++.

        Returns:
            np.ndarray shape [n_ant] dtype float32
        """
        max_mag = self.mag_ref.max(axis=1)   # [n_ant]
        min_mag = np.maximum(self.mag_ref.min(axis=1), 1e-30)  # [n_ant]
        return (20 * np.log10(max_mag / min_mag)).astype(np.float32)

    # ── Приватные методы ─────────────────────────────────────────────────────

    def _compute_stats(self, arr: np.ndarray) -> list:
        """Вычислить статистику по каждому лучу (beam = строка arr).

        Повторяет формулы GPU-kernel Statistics (welford_fused):
            mean         = mean(arr, axis=1)           ← комплексное
            variance     = var(|arr|, axis=1, ddof=0)  ← от магнитуд!
            std_dev      = std(|arr|, axis=1, ddof=0)
            mean_magnitude = mean(|arr|, axis=1)

        Args:
            arr: np.ndarray shape [n_ant, N]
                 dtype: complex64 (для d_S, d_X) или float32 (для mag)

        Returns:
            List[StatsRef] длиной n_ant
        """
        n_ant = arr.shape[0]
        stats = []
        for b in range(n_ant):
            row = arr[b]
            # mean: комплексное среднее (или реальное если arr=float)
            mean_val = np.mean(row).astype(np.complex64)
            # magnitude для variance/std/mean_mag
            mag = np.abs(row).astype(np.float32)
            stats.append(StatsRef(
                beam_id        = b,
                mean           = complex(mean_val),
                variance       = float(np.var(mag, ddof=0)),
                std_dev        = float(np.std(mag, ddof=0)),
                mean_magnitude = float(np.mean(mag)),
            ))
        return stats
