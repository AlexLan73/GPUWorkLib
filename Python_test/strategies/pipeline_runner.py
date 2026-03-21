#!/usr/bin/env python3
"""
PipelineRunner -- полный beamforming pipeline с checkpoint'ами и статистикой
==============================================================================

Зеркалит C++ AntennaProcessor debug points + добавляет Farrow шаг.

Две программные ветки:
  Pipeline A (без Farrow):
    Step 0: S_raw → [stats] → [save]
    Step 1: GEMM(W_phase, S_raw) → X → [stats] → [save]
    Step 2: Window+FFT → spectrum → [stats] → [save]
    Step 3: Peak detection → results

  Pipeline B (с Farrow):
    Step 0: S_raw → [stats] → [save]
    Step 0.5: FarrowDelay → S_aligned → [stats] → [save]
    Step 1: GEMM(W_sum, S_aligned) → X → [stats] → [save]
    Step 2: Window+FFT → spectrum → [stats] → [save]
    Step 3: Peak detection → results

Checkpoint'ы (опциональные):
  - save_input:     S_raw.npy
  - save_aligned:   S_aligned.npy (только Pipeline B)
  - save_gemm:      X_gemm.npy
  - save_spectrum:  spectrum.npy
  - save_stats:     stats.json (всегда, если output_dir задан)
  - save_results:   results.json

Usage:
  from pipeline_runner import PipelineRunner, PipelineConfig
  from scenario_builder import make_single_target

  scenario = make_single_target(n_ant=8, theta_deg=30, fdev_hz=1e6)

  config = PipelineConfig(
      save_input=True,
      save_aligned=True,
      save_stats=True,
  )

  runner = PipelineRunner(output_dir="Results/strategies/test_01")
  result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6, config=config)
  result_b = runner.run_pipeline_b(scenario, steer_theta=30, config=config)

  # Сравнение
  runner.compare(result_a, result_b)

Author: Kodo (AI Assistant)
Date: 2026-03-08
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from scenario_builder import ULAGeometry, ScenarioBuilder, EmitterSignal
from farrow_delay import FarrowDelay


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Конфигурация checkpoint'ов и статистики.

    Что сохранять на диск (опционально):
      save_input:    S_raw [n_ant × n_samples]
      save_aligned:  S_aligned после Farrow (только Pipeline B)
      save_gemm:     X после GEMM [n_beams × n_samples]
      save_spectrum: spectrum после FFT [n_beams × nFFT]
      save_stats:    stats.json со статистикой на каждом этапе
      save_results:  results.json с пиками и метриками
    """
    save_input: bool = False
    save_aligned: bool = False
    save_gemm: bool = False
    save_spectrum: bool = False
    save_stats: bool = True
    save_results: bool = True


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class ChannelStats:
    """Статистика одного канала/луча.

    Аналог C++ StatisticsProcessor output.
    """
    channel_id: int = 0
    mean_re: float = 0.0
    mean_im: float = 0.0
    mean_abs: float = 0.0
    std_re: float = 0.0
    std_im: float = 0.0
    std_abs: float = 0.0
    max_abs: float = 0.0
    min_abs: float = 0.0
    power: float = 0.0        # mean(|x|^2)
    n_samples: int = 0
    n_nonzero: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_channel_stats(data_1d: np.ndarray, channel_id: int = 0) -> ChannelStats:
    """Вычислить статистику одного канала.

    Args:
        data_1d: [n_samples] complex или float
        channel_id: ID канала для маркировки
    """
    s = ChannelStats(channel_id=channel_id, n_samples=len(data_1d))

    if np.iscomplexobj(data_1d):
        s.mean_re = float(np.mean(data_1d.real))
        s.mean_im = float(np.mean(data_1d.imag))
        s.std_re = float(np.std(data_1d.real))
        s.std_im = float(np.std(data_1d.imag))
        abs_vals = np.abs(data_1d)
    else:
        s.mean_re = float(np.mean(data_1d))
        s.std_re = float(np.std(data_1d))
        abs_vals = np.abs(data_1d)

    s.mean_abs = float(np.mean(abs_vals))
    s.std_abs = float(np.std(abs_vals))
    s.max_abs = float(np.max(abs_vals))
    s.min_abs = float(np.min(abs_vals))
    s.power = float(np.mean(abs_vals ** 2))
    s.n_nonzero = int(np.count_nonzero(abs_vals > 1e-12))

    return s


def compute_matrix_stats(data_2d: np.ndarray, label: str = "") -> List[ChannelStats]:
    """Вычислить per-channel статистику для матрицы [n_channels × n_samples].

    Args:
        data_2d: [n_channels, n_samples]
        label: метка для логов
    """
    n_ch = data_2d.shape[0]
    return [compute_channel_stats(data_2d[ch], ch) for ch in range(n_ch)]


# ============================================================================
# Peak Detection
# ============================================================================

@dataclass
class PeakInfo:
    """Информация о пике в спектре."""
    beam_id: int = 0
    bin_index: int = 0
    freq_hz: float = 0.0
    magnitude: float = 0.0
    phase_rad: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def find_peaks_per_beam(magnitudes: np.ndarray, freq_axis: np.ndarray,
                        n_peaks: int = 5,
                        freq_range: tuple = None) -> List[List[PeakInfo]]:
    """Найти top-N пиков для каждого beam.

    Args:
        magnitudes: [n_beams, nFFT] float
        freq_axis: [nFFT] float (Hz)
        n_peaks: количество пиков на beam
        freq_range: (f_min, f_max) -- ограничить поиск

    Returns:
        List[List[PeakInfo]] — per beam, per peak
    """
    n_beams, nFFT = magnitudes.shape
    half = nFFT // 2
    all_peaks = []

    for beam in range(n_beams):
        mags = magnitudes[beam, :half].copy()
        mags[0] = 0  # skip DC
        freqs = freq_axis[:half]

        if freq_range is not None:
            mask = (freqs < freq_range[0]) | (freqs > freq_range[1])
            mags[mask] = 0

        beam_peaks = []
        for _ in range(min(n_peaks, half)):
            idx = int(np.argmax(mags))
            if mags[idx] <= 0:
                break
            beam_peaks.append(PeakInfo(
                beam_id=beam,
                bin_index=idx,
                freq_hz=float(freqs[idx]),
                magnitude=float(mags[idx]),
            ))
            # Zero out neighborhood to find next peak
            lo = max(0, idx - 5)
            hi = min(half, idx + 6)
            mags[lo:hi] = 0

        all_peaks.append(beam_peaks)

    return all_peaks


# ============================================================================
# Pipeline Result
# ============================================================================

@dataclass
class PipelineResult:
    """Результат pipeline -- все промежуточные данные + статистика + пики."""

    pipeline_name: str = ""       # "A" или "B"

    # Промежуточные данные (numpy arrays, доступны для Python тестов)
    S_raw: Optional[np.ndarray] = None           # [n_ant, n_samples] input
    S_aligned: Optional[np.ndarray] = None        # [n_ant, n_samples] после Farrow (B only)
    X_gemm: Optional[np.ndarray] = None           # [n_beams, n_samples] после GEMM
    spectrum: Optional[np.ndarray] = None          # [n_beams, nFFT] complex
    magnitudes: Optional[np.ndarray] = None        # [n_beams, nFFT] float
    W: Optional[np.ndarray] = None                 # [n_beams, n_ant] weight matrix

    # Статистика на каждом этапе
    stats_input: Optional[List[ChannelStats]] = None
    stats_aligned: Optional[List[ChannelStats]] = None   # B only
    stats_gemm: Optional[List[ChannelStats]] = None
    stats_spectrum: Optional[List[ChannelStats]] = None

    # Пики
    peaks: Optional[List[List[PeakInfo]]] = None

    # Метаинформация
    nFFT: int = 0
    freq_axis: Optional[np.ndarray] = None
    steer_theta_deg: float = 0.0
    steer_freq_hz: float = 0.0

    def peak_summary(self) -> str:
        """Краткая сводка по пикам."""
        if not self.peaks:
            return "No peaks"
        lines = [f"Pipeline {self.pipeline_name} peaks:"]
        for beam_peaks in self.peaks:
            if beam_peaks:
                p = beam_peaks[0]
                lines.append(
                    f"  Beam {p.beam_id}: f={p.freq_hz/1e6:.3f} MHz, "
                    f"mag={p.magnitude:.4f}"
                )
        return "\n".join(lines)

    def stats_summary(self) -> str:
        """Краткая сводка статистики."""
        lines = [f"Pipeline {self.pipeline_name} stats:"]
        for label, stats in [
            ("Input", self.stats_input),
            ("Aligned", self.stats_aligned),
            ("GEMM", self.stats_gemm),
            ("Spectrum", self.stats_spectrum),
        ]:
            if stats:
                total_power = sum(s.power for s in stats) / len(stats)
                max_abs = max(s.max_abs for s in stats)
                lines.append(
                    f"  {label:10s}: avg_power={total_power:.6f}, "
                    f"max_abs={max_abs:.4f}, channels={len(stats)}"
                )
        return "\n".join(lines)


# ============================================================================
# Pipeline Helpers
# ============================================================================

def hamming_window(n: int) -> np.ndarray:
    """Hamming window matching C++ kernel."""
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def next_pow2_x2(n: int) -> int:
    """nFFT = next_pow2(n) * 2."""
    p = 1
    while p < n:
        p <<= 1
    return p * 2


# ============================================================================
# PipelineRunner
# ============================================================================

class PipelineRunner:
    """Запускает Pipeline A и/или B с checkpoint'ами.

    Args:
        output_dir: директория для сохранения (None = без сохранения)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir

    def _ensure_dir(self, subdir: str) -> str:
        """Создать поддиректорию если нужно."""
        if self.output_dir is None:
            return ""
        path = os.path.join(self.output_dir, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_npy(self, path: str, filename: str, data: np.ndarray):
        """Сохранить numpy массив."""
        if path:
            np.save(os.path.join(path, filename), data)

    def _save_json(self, path: str, filename: str, data: Any):
        """Сохранить JSON."""
        if path:
            with open(os.path.join(path, filename), 'w') as f:
                json.dump(data, f, indent=2, default=str)

    # ---- Pipeline A ----

    def run_pipeline_a(self, scenario: dict,
                       steer_theta: float,
                       steer_freq: float,
                       config: Optional[PipelineConfig] = None) -> PipelineResult:
        """Pipeline A: S_raw → GEMM(W_phase) → Window+FFT → peaks.

        W_phase[b][a] = (1/√N) · exp(-j·2π·f·τ_a) — фазовая коррекция.

        Args:
            scenario: dict from ScenarioBuilder.build()
            steer_theta: угол наведения (градусы)
            steer_freq: частота для фазовой коррекции (Гц)
            config: что сохранять

        Returns:
            PipelineResult со всеми данными
        """
        cfg = config or PipelineConfig()
        save_dir = self._ensure_dir("pipeline_a")

        S = scenario['S']
        array: ULAGeometry = scenario['array']
        n_ant = array.n_ant
        fs = scenario['fs']
        n_samples = scenario['n_samples']

        result = PipelineResult(
            pipeline_name="A",
            S_raw=S,
            steer_theta_deg=steer_theta,
            steer_freq_hz=steer_freq,
        )

        # Step 0: Input stats
        result.stats_input = compute_matrix_stats(S, "input")
        if cfg.save_input:
            self._save_npy(save_dir, "S_raw.npy", S)

        # Step 1: W_phase matrix
        delays = array.compute_delays(steer_theta)
        inv_sqrt_n = 1.0 / np.sqrt(n_ant)
        W = np.zeros((n_ant, n_ant), dtype=np.complex64)
        for beam in range(n_ant):
            for ant in range(n_ant):
                W[beam, ant] = inv_sqrt_n * np.exp(
                    -1j * 2.0 * np.pi * steer_freq * delays[ant]
                )
        result.W = W

        # Step 2: GEMM
        X = W @ S
        result.X_gemm = X
        result.stats_gemm = compute_matrix_stats(X, "gemm")
        if cfg.save_gemm:
            self._save_npy(save_dir, "X_gemm.npy", X)

        # Step 3: Window + FFT
        nFFT = next_pow2_x2(n_samples)
        win = hamming_window(n_samples)
        spectrum = np.zeros((n_ant, nFFT), dtype=np.complex64)
        magnitudes = np.zeros((n_ant, nFFT), dtype=np.float32)

        for b in range(n_ant):
            padded = np.zeros(nFFT, dtype=np.complex64)
            padded[:n_samples] = X[b] * win
            spec = np.fft.fft(padded)
            spectrum[b] = spec
            magnitudes[b] = np.abs(spec).astype(np.float32)

        result.spectrum = spectrum
        result.magnitudes = magnitudes
        result.nFFT = nFFT
        result.freq_axis = np.fft.fftfreq(nFFT, 1.0 / fs)
        result.stats_spectrum = compute_matrix_stats(magnitudes, "spectrum")

        if cfg.save_spectrum:
            self._save_npy(save_dir, "spectrum.npy", spectrum)
            self._save_npy(save_dir, "magnitudes.npy", magnitudes)

        # Step 4: Peak detection
        result.peaks = find_peaks_per_beam(magnitudes, result.freq_axis)

        # Save stats & results
        if cfg.save_stats and save_dir:
            stats_data = {
                'pipeline': 'A',
                'steer_theta_deg': steer_theta,
                'steer_freq_hz': steer_freq,
                'n_ant': n_ant,
                'n_samples': n_samples,
                'nFFT': nFFT,
                'stats_input': [s.to_dict() for s in result.stats_input],
                'stats_gemm': [s.to_dict() for s in result.stats_gemm],
                'stats_spectrum': [s.to_dict() for s in result.stats_spectrum],
            }
            self._save_json(save_dir, "stats.json", stats_data)

        if cfg.save_results and save_dir:
            results_data = {
                'pipeline': 'A',
                'peaks': [[p.to_dict() for p in beam] for beam in result.peaks],
            }
            self._save_json(save_dir, "results.json", results_data)

        return result

    # ---- Pipeline B ----

    def run_pipeline_b(self, scenario: dict,
                       steer_theta: float,
                       config: Optional[PipelineConfig] = None) -> PipelineResult:
        """Pipeline B: S_raw → FarrowDelay → S_aligned → GEMM(W_sum) → FFT → peaks.

        W_sum[b][a] = 1/√N — когерентное суммирование.

        Args:
            scenario: dict from ScenarioBuilder.build()
            steer_theta: угол наведения для компенсации (градусы)
            config: что сохранять

        Returns:
            PipelineResult со всеми данными (включая S_aligned)
        """
        cfg = config or PipelineConfig()
        save_dir = self._ensure_dir("pipeline_b")

        S = scenario['S']
        array: ULAGeometry = scenario['array']
        n_ant = array.n_ant
        fs = scenario['fs']
        n_samples = scenario['n_samples']

        result = PipelineResult(
            pipeline_name="B",
            S_raw=S,
            steer_theta_deg=steer_theta,
        )

        # Step 0: Input stats
        result.stats_input = compute_matrix_stats(S, "input")
        if cfg.save_input:
            self._save_npy(save_dir, "S_raw.npy", S)

        # Step 0.5: Farrow compensation
        farrow = FarrowDelay()
        delays_s = array.compute_delays(steer_theta)
        S_aligned = farrow.compensate_seconds(S, delays_s, fs)

        result.S_aligned = S_aligned
        result.stats_aligned = compute_matrix_stats(S_aligned, "aligned")
        if cfg.save_aligned:
            self._save_npy(save_dir, "S_aligned.npy", S_aligned)

        # Step 1: W_sum matrix (когерентное суммирование)
        inv_sqrt_n = 1.0 / np.sqrt(n_ant)
        W = np.full((n_ant, n_ant), inv_sqrt_n, dtype=np.complex64)
        result.W = W

        # Step 2: GEMM
        X = W @ S_aligned
        result.X_gemm = X
        result.stats_gemm = compute_matrix_stats(X, "gemm")
        if cfg.save_gemm:
            self._save_npy(save_dir, "X_gemm.npy", X)

        # Step 3: Window + FFT
        nFFT = next_pow2_x2(n_samples)
        win = hamming_window(n_samples)
        spectrum = np.zeros((n_ant, nFFT), dtype=np.complex64)
        magnitudes = np.zeros((n_ant, nFFT), dtype=np.float32)

        for b in range(n_ant):
            padded = np.zeros(nFFT, dtype=np.complex64)
            padded[:n_samples] = X[b] * win
            spec = np.fft.fft(padded)
            spectrum[b] = spec
            magnitudes[b] = np.abs(spec).astype(np.float32)

        result.spectrum = spectrum
        result.magnitudes = magnitudes
        result.nFFT = nFFT
        result.freq_axis = np.fft.fftfreq(nFFT, 1.0 / fs)
        result.stats_spectrum = compute_matrix_stats(magnitudes, "spectrum")

        if cfg.save_spectrum:
            self._save_npy(save_dir, "spectrum.npy", spectrum)
            self._save_npy(save_dir, "magnitudes.npy", magnitudes)

        # Step 4: Peak detection
        result.peaks = find_peaks_per_beam(magnitudes, result.freq_axis)

        # Save stats & results
        if cfg.save_stats and save_dir:
            stats_data = {
                'pipeline': 'B',
                'steer_theta_deg': steer_theta,
                'n_ant': n_ant,
                'n_samples': n_samples,
                'nFFT': nFFT,
                'stats_input': [s.to_dict() for s in result.stats_input],
                'stats_aligned': [s.to_dict() for s in result.stats_aligned],
                'stats_gemm': [s.to_dict() for s in result.stats_gemm],
                'stats_spectrum': [s.to_dict() for s in result.stats_spectrum],
            }
            self._save_json(save_dir, "stats.json", stats_data)

        if cfg.save_results and save_dir:
            results_data = {
                'pipeline': 'B',
                'peaks': [[p.to_dict() for p in beam] for beam in result.peaks],
            }
            self._save_json(save_dir, "results.json", results_data)

        return result

    # ---- Comparison ----

    def compare(self, result_a: PipelineResult,
                result_b: PipelineResult) -> dict:
        """Сравнить результаты Pipeline A и B.

        Returns:
            dict с метриками сравнения
        """
        comparison = {
            'pipeline_a': result_a.pipeline_name,
            'pipeline_b': result_b.pipeline_name,
        }

        # Пиковые магнитуды (beam 0)
        if result_a.peaks and result_b.peaks:
            peak_a = result_a.peaks[0][0] if result_a.peaks[0] else None
            peak_b = result_b.peaks[0][0] if result_b.peaks[0] else None

            if peak_a and peak_b:
                comparison['beam0_peak_a'] = {
                    'freq_hz': peak_a.freq_hz,
                    'magnitude': peak_a.magnitude,
                }
                comparison['beam0_peak_b'] = {
                    'freq_hz': peak_b.freq_hz,
                    'magnitude': peak_b.magnitude,
                }
                comparison['magnitude_ratio_b_over_a'] = (
                    peak_b.magnitude / max(peak_a.magnitude, 1e-10)
                )
                comparison['freq_diff_hz'] = abs(peak_b.freq_hz - peak_a.freq_hz)

        # Средняя мощность после GEMM
        if result_a.stats_gemm and result_b.stats_gemm:
            power_a = sum(s.power for s in result_a.stats_gemm) / len(result_a.stats_gemm)
            power_b = sum(s.power for s in result_b.stats_gemm) / len(result_b.stats_gemm)
            comparison['avg_gemm_power_a'] = power_a
            comparison['avg_gemm_power_b'] = power_b

        # Сохраняем если output_dir задан
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self._save_json(self.output_dir, "comparison.json", comparison)

        return comparison

    def print_comparison(self, result_a: PipelineResult,
                         result_b: PipelineResult):
        """Вывести сравнение на консоль."""
        comp = self.compare(result_a, result_b)

        print("\n" + "=" * 60)
        print("Pipeline Comparison: A (phase) vs B (Farrow)")
        print("=" * 60)

        if 'beam0_peak_a' in comp:
            pa = comp['beam0_peak_a']
            pb = comp['beam0_peak_b']
            print(f"  Beam 0 peak A: f={pa['freq_hz']/1e6:.3f} MHz, "
                  f"mag={pa['magnitude']:.4f}")
            print(f"  Beam 0 peak B: f={pb['freq_hz']/1e6:.3f} MHz, "
                  f"mag={pb['magnitude']:.4f}")
            print(f"  Magnitude ratio B/A: {comp['magnitude_ratio_b_over_a']:.3f}")
            print(f"  Frequency diff: {comp['freq_diff_hz']:.1f} Hz")

        if 'avg_gemm_power_a' in comp:
            print(f"  Avg GEMM power A: {comp['avg_gemm_power_a']:.6f}")
            print(f"  Avg GEMM power B: {comp['avg_gemm_power_b']:.6f}")

        print()
        print(result_a.stats_summary())
        print()
        print(result_b.stats_summary())

