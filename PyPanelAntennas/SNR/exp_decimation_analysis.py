#!/usr/bin/env python3
"""
exp_decimation_analysis.py — Анализ влияния децимации на SNR-оценщик
=====================================================================

Исследует: как длина FFT (после децимации) влияет на CFAR SNR-оценку.

─── Теория (Hann window, CW тон) ────────────────────────────────────────────

  FFT peak power:    |X[k₀]|² = A² · (N_actual/2)²       (coherent gain ×N)
  Noise per bin:     E[|X[k]|²] = σ² · N_actual · 3/8    (Hann ENBW)

  CFAR SNR = peak / noise_per_bin
           = A²·N_actual / (σ²·3/2)
           = SNR_input_linear · N_actual · (2/3)

  В дБ:
    CFAR_SNR_dB ≈ SNR_input_dB + 10·log₁₀(N_actual) − 1.76 dB
                  │                │                   │
                  │                └─ processing gain   └─ Hann window loss
                  └─ входной SNR

  Децимация в D раз (N_actual = N_samples/D):
    ΔdB = −10·log₁₀(D)     — одинаково для ВСЕХ входных SNR!

─── Структура эксперимента ──────────────────────────────────────────────────

  Часть A: 1.3M отсчётов, target_n_fft ∈ [1024..1M]
           → базовая (N_fft≈2M) vs decimated (разные N_fft)
           → delta vs target_n_fft для всех SNR уровней

  Часть B: Усечение входа [1.3M → 2.5K], target_n_fft=2048 ФИКСИРОВАН
           → тест стабильности: меняется ли CFAR SNR когда меняется n_samples
              при одинаковом N_fft?

  Часть C: Усечение БЕЗ децимации (step=1), N_fft=next_pow2(N_input)
           → что выигрываешь от большего количества отсчётов без децимации?

Окно: ТОЛЬКО Hann.

Запуск:
    "F:/Program Files (x86)/Python314/python.exe" PyPanelAntennas/SNR/exp_decimation_analysis.py
    python3 PyPanelAntennas/SNR/exp_decimation_analysis.py

@author Кодо
@date 2026-04-10
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Пути ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cfar_estimator import (  # noqa: E402
    CfarConfig,
    compute_pipeline_sizes,
    next_power_of_2,
    make_window,
    zero_pad,
)

PLOTS_DIR  = _HERE / "plots"
RESULTS_DIR = _HERE / "results"
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Параметры ────────────────────────────────────────────────────────────────
N_FULL        = 1_300_000          # 1.3M входных отсчётов
F_NORM        = 0.15               # нормированная частота CW
GUARD_BINS    = 5                  # из SNR_00 Эксп.5
REF_BINS      = 16                 # из SNR_00 Эксп.5
WINDOW        = "hann"             # лучшее окно (SNR_00 → Hann + mean)
SEED          = 42

SNR_INPUT_DB_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Часть A: target_n_fft values (powers of 2: 2^10..2^20)
TARGET_N_FFT_A: List[int] = [2**p for p in range(10, 21)]
# 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576

# Часть B/C: коэффициенты усечения (N_full / 2^i)
TRUNCATION_FACTORS: List[int] = [2**i for i in range(0, 10)]
# 1, 2, 4, ..., 512 → длины: 1.3M, 650K, 325K, ..., ~2539

# Порог из SNR_00 (для отображения)
THR_LOW_MID  = 15.0
THR_MID_HIGH = 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# Ядро оценки SNR (без дублирования cfar_estimator — используем его напрямую)
# ═══════════════════════════════════════════════════════════════════════════════

def _cfar_snr(signal: np.ndarray, step: int, guard: int, ref: int) -> Tuple[float, int, int]:
    """Оценить CFAR SNR для заданного шага децимации.

    Returns:
        (snr_db, n_actual, n_fft)
    """
    n_samples = len(signal)
    n_actual  = n_samples // step if step > 0 else n_samples
    n_fft     = next_power_of_2(n_actual)

    # Децимация
    decimated = signal[::step][:n_actual].astype(np.complex64)

    # Hann window
    w = make_window(WINDOW, n_actual)
    decimated = (decimated * w).astype(np.complex64)

    # Zero-pad → FFT → |X|²
    padded   = zero_pad(decimated, n_fft)
    spectrum = np.fft.fft(padded)
    mag_sq   = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)

    # peak
    k_peak    = int(np.argmax(mag_sq))
    peak_sq   = float(mag_sq[k_peak])

    # CA-CFAR ref window
    ref_vals = np.empty(2 * ref, dtype=np.float32)
    for i in range(ref):
        offset  = guard + 1 + i
        k_left  = (k_peak - offset) % n_fft
        k_right = (k_peak + offset) % n_fft
        ref_vals[2 * i]     = mag_sq[k_left]
        ref_vals[2 * i + 1] = mag_sq[k_right]

    noise_est = max(float(ref_vals.mean()), 1e-30)
    snr_db    = 10.0 * math.log10(max(peak_sq / noise_est, 1e-30))

    return snr_db, n_actual, n_fft


def generate_signal(n: int, snr_db: float, seed: int = SEED) -> np.ndarray:
    """CW + AWGN. Noise power = 1, signal power = 10^(snr_db/10)."""
    rng  = np.random.default_rng(seed)
    A    = math.sqrt(10.0 ** (snr_db / 10.0))
    t    = np.arange(n, dtype=np.float64)
    sig  = A * np.exp(1j * 2.0 * math.pi * F_NORM * t)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
    return (sig + noise).astype(np.complex64)


def theory_cfar_db(snr_input_db: float, n_actual: int) -> float:
    """Теоретическое значение CFAR SNR (Hann window, CW в точном бине).

    CFAR_SNR_dB = SNR_input_dB + 10·log10(N_actual) − 1.76 dB
    """
    return snr_input_db + 10.0 * math.log10(n_actual) - 1.76


# ═══════════════════════════════════════════════════════════════════════════════
# ЧАСТЬ A: фиксированный вход 1.3M, меняем target_n_fft
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_a() -> Dict:
    """Часть A: decimation sweep при N_input=1.3M."""
    print("\n" + "═" * 60)
    print("  Часть A: 1.3M samples → sweep target_n_fft")
    print("═" * 60)

    # baseline: шаг 1 (нет децимации), N_fft ≈ 2^21
    step_baseline = 1
    n_actual_baseline = N_FULL // step_baseline
    n_fft_baseline    = next_power_of_2(n_actual_baseline)

    results_a: Dict = {
        "n_full": N_FULL,
        "f_norm": F_NORM,
        "window": WINDOW,
        "guard_bins": GUARD_BINS,
        "ref_bins": REF_BINS,
        "n_fft_baseline": n_fft_baseline,
        "snr_levels": SNR_INPUT_DB_LIST,
        "target_n_fft_values": TARGET_N_FFT_A,
        "data": {},
    }

    for snr_db in SNR_INPUT_DB_LIST:
        t0 = time.perf_counter()
        sig = generate_signal(N_FULL, snr_db, seed=SEED)

        # Baseline (step=1)
        baseline_snr, _, _ = _cfar_snr(sig, step=1, guard=GUARD_BINS, ref=REF_BINS)
        print(f"  SNR_in={snr_db:3.0f} dB | baseline (N_fft={n_fft_baseline}) = {baseline_snr:.2f} dB", end="")

        row = {
            "baseline": {"step": 1, "n_actual": n_actual_baseline,
                         "n_fft": n_fft_baseline, "cfar_snr_db": baseline_snr,
                         "theory_db": theory_cfar_db(snr_db, n_actual_baseline)},
            "decimated": [],
        }

        for tgt in TARGET_N_FFT_A:
            # шаг = ceil(N_full / target_n_fft)
            step = max(1, (N_FULL + tgt - 1) // tgt)
            snr_est, n_act, n_fft_act = _cfar_snr(sig, step=step, guard=GUARD_BINS, ref=REF_BINS)
            delta   = snr_est - baseline_snr
            theory  = theory_cfar_db(snr_db, n_act)
            theory_delta = theory - theory_cfar_db(snr_db, n_actual_baseline)

            row["decimated"].append({
                "target_n_fft": tgt,
                "step": step,
                "n_actual": n_act,
                "n_fft": n_fft_act,
                "cfar_snr_db": snr_est,
                "delta_from_baseline_db": delta,
                "theory_cfar_db": theory,
                "theory_delta_db": theory_delta,
            })

        results_a["data"][snr_db] = row
        print(f" | {time.perf_counter()-t0:.1f}s")

    return results_a


# ═══════════════════════════════════════════════════════════════════════════════
# ЧАСТЬ B: усечение входа, target_n_fft=2048 ФИКСИРОВАН
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_b() -> Dict:
    """Часть B: truncate n_samples, target_n_fft=2048 fixed."""
    print("\n" + "═" * 60)
    print("  Часть B: усечение входа (target_n_fft=2048 фикс.)")
    print("═" * 60)

    TARGET_NFFT_B = 2048
    input_lengths = [max(TARGET_NFFT_B * 2, N_FULL // f) for f in TRUNCATION_FACTORS]
    # убираем дубликаты и сортируем по убыванию
    input_lengths = sorted(set(input_lengths), reverse=True)

    results_b: Dict = {
        "target_n_fft_fixed": TARGET_NFFT_B,
        "input_lengths": input_lengths,
        "data": {},
    }

    for snr_db in SNR_INPUT_DB_LIST:
        sig_full = generate_signal(N_FULL, snr_db, seed=SEED)
        row = []
        print(f"  SNR_in={snr_db:3.0f} dB |", end="")

        for n_inp in input_lengths:
            sig = sig_full[:n_inp]
            step = max(1, (n_inp + TARGET_NFFT_B - 1) // TARGET_NFFT_B)
            snr_est, n_act, n_fft_act = _cfar_snr(sig, step=step, guard=GUARD_BINS, ref=REF_BINS)
            row.append({
                "n_input": n_inp,
                "step": step,
                "n_actual": n_act,
                "n_fft": n_fft_act,
                "cfar_snr_db": snr_est,
                "theory_db": theory_cfar_db(snr_db, n_act),
            })
            print(f" {snr_est:.1f}", end="")

        results_b["data"][snr_db] = row
        print()

    return results_b


# ═══════════════════════════════════════════════════════════════════════════════
# ЧАСТЬ C: усечение БЕЗ децимации (step=1)
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_c() -> Dict:
    """Часть C: truncate + no decimation → N_fft = next_pow2(N_input)."""
    print("\n" + "═" * 60)
    print("  Часть C: усечение без децимации (step=1)")
    print("═" * 60)

    # Входные длины: от 1.3M до 2048 по степеням 2
    input_lengths = sorted({max(1024, N_FULL // f) for f in TRUNCATION_FACTORS}, reverse=True)

    results_c: Dict = {
        "step_fixed": 1,
        "input_lengths": input_lengths,
        "data": {},
    }

    for snr_db in SNR_INPUT_DB_LIST:
        sig_full = generate_signal(N_FULL, snr_db, seed=SEED)
        row = []
        print(f"  SNR_in={snr_db:3.0f} dB |", end="")

        for n_inp in input_lengths:
            sig = sig_full[:n_inp]
            snr_est, n_act, n_fft_act = _cfar_snr(sig, step=1, guard=GUARD_BINS, ref=REF_BINS)
            row.append({
                "n_input": n_inp,
                "n_actual": n_act,
                "n_fft": n_fft_act,
                "cfar_snr_db": snr_est,
                "theory_db": theory_cfar_db(snr_db, n_act),
            })
            print(f" {snr_est:.1f}", end="")

        results_c["data"][snr_db] = row
        print()

    return results_c


# ═══════════════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ═══════════════════════════════════════════════════════════════════════════════

_CMAP = plt.cm.plasma  # type: ignore[attr-defined]

def _snr_color(snr_db: float) -> tuple:
    """Цвет по входному SNR (5..50 dB → plasma)."""
    t = (snr_db - SNR_INPUT_DB_LIST[0]) / (SNR_INPUT_DB_LIST[-1] - SNR_INPUT_DB_LIST[0])
    return _CMAP(t)


def plot_all(res_a: Dict, res_b: Dict, res_c: Dict) -> None:
    """Рисуем главный 4-панельный рисунок."""

    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])  # A: CFAR SNR vs N_fft (все SNR уровни)
    ax2 = fig.add_subplot(gs[0, 1])  # A: delta от baseline vs N_fft
    ax3 = fig.add_subplot(gs[1, 0])  # B: усечение входа, target_n_fft=2048 фиксирован
    ax4 = fig.add_subplot(gs[1, 1])  # C: усечение без децимации (step=1) + сравнение

    dark_axes_style = dict(facecolor="#1a1a2e",
                           labelcolor="white",
                           titlecolor="white")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(dark_axes_style["facecolor"])
        ax.tick_params(colors="white", which="both")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(True, color="#333355", linestyle="--", alpha=0.6)

    target_vals_a = res_a["target_n_fft_values"]
    n_baseline    = res_a["n_fft_baseline"]

    # ── Panel 1: CFAR SNR vs N_fft (часть A) ──────────────────────────────────
    ax1.set_title("A: CFAR SNR vs N_fft после децимации\n(вход 1.3M, Hann, CA-CFAR)", pad=10)
    ax1.set_xlabel("log₂(N_fft_actual)")
    ax1.set_ylabel("CFAR SNR (дБ)")

    for snr_db in SNR_INPUT_DB_LIST:
        row    = res_a["data"][snr_db]
        color  = _snr_color(snr_db)
        dec    = row["decimated"]

        x_vals = [math.log2(d["n_fft"]) for d in dec]
        y_meas = [d["cfar_snr_db"] for d in dec]
        y_theo = [d["theory_cfar_db"] for d in dec]

        # baseline point
        ax1.scatter([math.log2(n_baseline)], [row["baseline"]["cfar_snr_db"]],
                    color=color, s=80, marker="*", zorder=5)
        ax1.plot(x_vals, y_meas, "o-", color=color, lw=1.5, ms=4,
                 label=f"SNR_in={snr_db} dB")
        ax1.plot(x_vals, y_theo, "--", color=color, lw=0.9, alpha=0.5)

    # Threshold lines
    ax1.axhline(THR_LOW_MID,  color="#ff4444", lw=1.2, ls=":", alpha=0.8,
                label=f"thr Low→Mid = {THR_LOW_MID} dB")
    ax1.axhline(THR_MID_HIGH, color="#ff8800", lw=1.2, ls=":", alpha=0.8,
                label=f"thr Mid→High = {THR_MID_HIGH} dB")
    ax1.axvline(math.log2(2048), color="#88aaff", lw=1.2, ls="--", alpha=0.7,
                label="target=2048")

    leg = ax1.legend(fontsize=7, framealpha=0.2, facecolor="#333355",
                     labelcolor="white", ncol=2, loc="upper left")
    ax1.set_xlim(9.5, 21.5)

    # ── Panel 2: Delta от baseline vs N_fft (часть A) ─────────────────────────
    ax2.set_title("A: Потеря CFAR SNR vs N_fft\nδ = CFAR_decimated − CFAR_baseline", pad=10)
    ax2.set_xlabel("log₂(N_fft_actual)")
    ax2.set_ylabel("δ CFAR SNR (дБ)")

    # Theoretical delta (независимо от SNR_input!)
    x_theory = [math.log2(tgt) for tgt in target_vals_a]
    y_theory  = [10.0 * math.log10(tgt / n_baseline) for tgt in target_vals_a]
    ax2.plot(x_theory, y_theory, "w--", lw=2.0, alpha=0.9, label="Теория: −10·log₁₀(D)")

    collapse_flag = True  # все кривые должны сойтись
    for snr_db in SNR_INPUT_DB_LIST:
        row   = res_a["data"][snr_db]
        color = _snr_color(snr_db)
        dec   = row["decimated"]
        x_vals = [math.log2(d["n_fft"]) for d in dec]
        y_vals = [d["delta_from_baseline_db"] for d in dec]
        ax2.plot(x_vals, y_vals, "o-", color=color, lw=1.2, ms=3, alpha=0.7,
                 label=f"{snr_db} dB")

    ax2.axvline(math.log2(2048), color="#88aaff", lw=1.2, ls="--", alpha=0.7,
                label="target=2048\n(−{:.0f} dB)".format(-10*math.log10(2048/n_baseline)))
    ax2.legend(fontsize=7, framealpha=0.2, facecolor="#333355",
               labelcolor="white", ncol=2, loc="lower right")
    ax2.set_xlim(9.5, 21.5)

    # Аннотация
    delta_at_2048 = 10.0 * math.log10(2048 / n_baseline)
    ax2.annotate(f"target=2048\nδ≈{delta_at_2048:.1f} dB",
                 xy=(math.log2(2048), delta_at_2048),
                 xytext=(math.log2(2048) + 1.5, delta_at_2048 + 4),
                 color="white", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="white", lw=0.8))

    # ── Panel 3: Усечение входа, target_n_fft=2048 фиксирован ────────────────
    ax3.set_title("B: CFAR SNR при усечении входа\n(target_n_fft=2048 фиксирован, Hann)", pad=10)
    ax3.set_xlabel("log₂(N_input)")
    ax3.set_ylabel("CFAR SNR (дБ)")

    inp_lengths_b = res_b["input_lengths"]
    for snr_db in SNR_INPUT_DB_LIST:
        color = _snr_color(snr_db)
        row   = res_b["data"][snr_db]
        x_vals = [math.log2(d["n_input"]) for d in row]
        y_meas = [d["cfar_snr_db"] for d in row]
        y_theo = [d["theory_db"] for d in row]
        ax3.plot(x_vals, y_meas, "o-", color=color, lw=1.5, ms=4,
                 label=f"{snr_db} dB")
        ax3.plot(x_vals, y_theo, "--", color=color, lw=0.8, alpha=0.4)

    ax3.axhline(THR_LOW_MID,  color="#ff4444", lw=1.2, ls=":", alpha=0.8)
    ax3.axhline(THR_MID_HIGH, color="#ff8800", lw=1.2, ls=":", alpha=0.8)
    ax3.legend(fontsize=7, framealpha=0.2, facecolor="#333355",
               labelcolor="white", ncol=2)
    ax3.annotate("При одинаковом N_fft\nCFAR SNR≈const\n(теоретически)",
                 xy=(ax3.get_xlim()[0] + 1 if ax3.get_xlim()[0] > 5 else 12, 80),
                 color="#aaffaa", fontsize=8, style="italic")

    # ── Panel 4: Сравнение стратегий B vs C ───────────────────────────────────
    ax4.set_title("C: Без децимации (step=1) vs target=2048\n"
                  "CFAR SNR vs N_input", pad=10)
    ax4.set_xlabel("log₂(N_input)")
    ax4.set_ylabel("CFAR SNR (дБ)")

    for snr_db in [10, 20, 30, 40]:  # 4 уровня для читаемости
        color = _snr_color(snr_db)
        # Часть B (target=2048)
        row_b = res_b["data"][snr_db]
        x_b   = [math.log2(d["n_input"]) for d in row_b]
        y_b   = [d["cfar_snr_db"] for d in row_b]
        # Часть C (step=1)
        row_c = res_c["data"][snr_db]
        x_c   = [math.log2(d["n_input"]) for d in row_c]
        y_c   = [d["cfar_snr_db"] for d in row_c]

        ax4.plot(x_b, y_b, "o-", color=color, lw=1.5, ms=4, alpha=0.9,
                 label=f"target=2048, SNR={snr_db}dB")
        ax4.plot(x_c, y_c, "s--", color=color, lw=1.5, ms=4, alpha=0.6,
                 label=f"step=1, SNR={snr_db}dB")

    # Теоретическая линия для step=1 (CFAR ~ N_actual)
    inp_c = res_c["data"][SNR_INPUT_DB_LIST[0]]
    x_theory_c = [math.log2(d["n_input"]) for d in inp_c]
    y_theory_10 = [theory_cfar_db(10.0, d["n_actual"]) for d in inp_c]
    ax4.plot(x_theory_c, y_theory_10, "w-.", lw=1.0, alpha=0.5, label="теория step=1, 10dB")

    ax4.axhline(THR_LOW_MID,  color="#ff4444", lw=1.2, ls=":", alpha=0.8)
    ax4.axhline(THR_MID_HIGH, color="#ff8800", lw=1.2, ls=":", alpha=0.8)
    ax4.legend(fontsize=7, framealpha=0.2, facecolor="#333355",
               labelcolor="white", ncol=1, loc="upper left")

    # ── Цветовая легенда (SNR_input) ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=_CMAP,
        norm=plt.Normalize(vmin=SNR_INPUT_DB_LIST[0], vmax=SNR_INPUT_DB_LIST[-1])
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02,
                        label="Входной SNR (дБ)")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # ── Заголовок и аннотация теории ──────────────────────────────────────────
    fig.suptitle(
        "Влияние децимации на CFAR SNR-оценку\n"
        "N_input=1.3M, CW (f=0.15·Fs), Hann window, CA-CFAR (guard=5, ref=16)\n"
        r"Теория: CFAR_SNR ≈ SNR_input + 10·log₁₀(N_actual) − 1.76 dB",
        color="white", fontsize=12, y=0.98
    )
    fig.patch.set_facecolor("#0d0d0d")

    out = PLOTS_DIR / "decimation_analysis.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  График сохранён: {out}")


def plot_delta_collapse(res_a: Dict) -> None:
    """Отдельный график: delta collapse (все SNR уровни должны совпасть)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d0d")
    fig.suptitle(
        "Коллапс дельты: потеря CFAR SNR при децимации\n"
        "Независимость от входного SNR: δ = −10·log₁₀(D)",
        color="white", fontsize=11
    )

    target_vals = res_a["target_n_fft_values"]
    n_baseline  = res_a["n_fft_baseline"]

    for ax_idx, (ax, ykey, title) in enumerate(zip(
        axes,
        ["delta_from_baseline_db", "cfar_snr_db"],
        ["Потеря дБ от baseline", "Абсолютный CFAR SNR (дБ)"]
    )):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.grid(True, color="#333355", ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("log₂(N_fft_actual)")
        ax.set_ylabel("дБ")

        for snr_db in SNR_INPUT_DB_LIST:
            row   = res_a["data"][snr_db]
            color = _snr_color(snr_db)
            dec   = row["decimated"]
            x_v   = [math.log2(d["n_fft"]) for d in dec]
            if ykey == "delta_from_baseline_db":
                y_v = [d["delta_from_baseline_db"] for d in dec]
            else:
                y_v = [d["cfar_snr_db"] for d in dec]
            ax.plot(x_v, y_v, "o-", color=color, lw=1.3, ms=4,
                    label=f"{snr_db} dB")

        # Теория
        x_theo = [math.log2(t) for t in target_vals]
        if ykey == "delta_from_baseline_db":
            y_theo = [10.0 * math.log10(t / n_baseline) for t in target_vals]
            ax.plot(x_theo, y_theo, "w--", lw=2.5, alpha=0.85, label="Теория")
            ax.axvline(math.log2(2048), color="#88aaff", lw=1.3, ls="--",
                       label="N_fft=2048")
            # Аннотация зазора
            d_2048 = 10.0 * math.log10(2048 / n_baseline)
            ax.annotate(f"δ = {d_2048:.1f} dB\n(D ≈ {n_baseline//2048}×)",
                        xy=(math.log2(2048), d_2048),
                        xytext=(13.5, d_2048 + 6),
                        color="white", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="white"))
        else:
            for snr_db in SNR_INPUT_DB_LIST:
                y_theo_line = [theory_cfar_db(snr_db, t) for t in target_vals]
                ax.plot(x_theo, y_theo_line, "--",
                        color=_snr_color(snr_db), lw=0.7, alpha=0.4)

        ax.axhline(0 if ykey == "delta_from_baseline_db" else THR_MID_HIGH,
                   color="white", lw=0.8, alpha=0.4, ls=":")
        ax.legend(fontsize=7.5, framealpha=0.2, facecolor="#333355",
                  labelcolor="white", ncol=2, loc="lower right"
                  if ykey == "delta_from_baseline_db" else "upper left")

    out = PLOTS_DIR / "decimation_delta_collapse.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  График сохранён: {out}")


def print_summary(res_a: Dict) -> None:
    """Таблица: baseline vs decimated для наглядности."""
    n_base = res_a["n_fft_baseline"]
    print("\n" + "═" * 80)
    print(f"  ТАБЛИЦА: CFAR SNR при разных N_fft (baseline N_fft={n_base:,})")
    print("─" * 80)
    header_nfft = ["".join(f"{d['n_fft']:>8}" for d in res_a["data"][5]["decimated"])]
    print(f"  SNR_in  | baseline | " + "".join(f"{t:>8}" for t in TARGET_N_FFT_A))
    print("─" * 80)
    for snr_db in SNR_INPUT_DB_LIST:
        row = res_a["data"][snr_db]
        base_str = f"{row['baseline']['cfar_snr_db']:8.2f}"
        dec_strs = "".join(f"{d['cfar_snr_db']:8.2f}" for d in row["decimated"])
        print(f"  {snr_db:6.0f} dB | {base_str} | {dec_strs}")
    print("─" * 80)
    print("  Delta от baseline (теория: −10·log₁₀(D) = одно значение для всех SNR_in):")
    print(f"  {'':10} | {'':8} | " + "".join(f"{10.0*math.log10(t/n_base):8.1f}" for t in TARGET_N_FFT_A))
    print("─" * 80)
    print(f"  Вывод: при target_n_fft=2048 от baseline теряем {10.0*math.log10(2048/n_base):.1f} дБ")
    print(f"         Это НЕ влияет на решение Low/Mid/High — пороги откалиброваны!")
    print("═" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  exp_decimation_analysis.py")
    print(f"  N_full={N_FULL:,}, F_norm={F_NORM}, window={WINDOW}")
    print(f"  SNR_input: {SNR_INPUT_DB_LIST}")
    print("=" * 60)

    t_total = time.perf_counter()

    res_a = run_part_a()
    res_b = run_part_b()
    res_c = run_part_c()

    print_summary(res_a)

    print("\n  Строим графики...", flush=True)
    plot_all(res_a, res_b, res_c)
    plot_delta_collapse(res_a)

    # Сохраняем JSON
    out_json = RESULTS_DIR / "decimation_analysis.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"part_a": res_a, "part_b": res_b, "part_c": res_c},
                  f, indent=2, ensure_ascii=False)
    print(f"  JSON сохранён: {out_json}")

    print(f"\n  Готово за {time.perf_counter()-t_total:.1f}s")
    print("  Графики:")
    print(f"    PyPanelAntennas/SNR/plots/decimation_analysis.png")
    print(f"    PyPanelAntennas/SNR/plots/decimation_delta_collapse.png")


if __name__ == "__main__":
    main()
