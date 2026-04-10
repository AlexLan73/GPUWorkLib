"""
debug_peak_grows.py — НАРОДНАЯ ВЕРСИЯ: как растёт пик при увеличении SNR_in.

Серия панелей 2×4 = 8 спектров при SNR_in от -20 до +25 dB.
Каждая панель — увеличенный вид спектра вокруг пика с:
  - |X|² кривой (Hann + обработка)
  - Показанный ref-окно CFAR
  - Text box с параметрами и результатом

Визуально видно как пик "вырастает" из шума → порог CFAR работает надёжно.

Обработка: Hann window + CA-CFAR (mean), guard=5, ref=16.

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfar_estimator import (
    CfarConfig,
    estimate_snr_one_antenna,
    make_window,
)
from lfm_signal_generator import make_cw_with_snr, make_awgn


# Финальные параметры
FINAL_WINDOW = "hann"
FINAL_ESTIMATOR = "mean"
FINAL_GUARD = 5
FINAL_REF = 16


def compute_mag_sq_hann(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """|X|² для Hann-windowed + zero-padded сигнала."""
    n = len(signal)
    w = make_window("hann", n).astype(np.float32)
    w_sig = (signal * w).astype(np.complex64)
    padded = np.zeros(n_fft, dtype=np.complex64)
    padded[:n] = w_sig
    X = np.fft.fft(padded)
    return (np.abs(X) ** 2).astype(np.float32)


def branch_color(snr_db: float) -> str:
    """Цвет по бренчу (для визуализации)."""
    if snr_db < 15.0:
        return "#d62728"  # красный — Low
    elif snr_db < 30.0:
        return "#ff7f0e"  # оранжевый — Mid
    else:
        return "#2ca02c"  # зелёный — High


def branch_name(snr_db: float) -> str:
    if snr_db < 15.0:
        return "LOW"
    elif snr_db < 30.0:
        return "MID"
    else:
        return "HIGH"


def main() -> None:
    # Серия SNR_in от "ничего нет" до "сильный сигнал"
    snr_in_list = [
        ("Только шум", -999.0),    # специальный случай — чистый шум
        ("Очень слабый", -20.0),
        ("Слабый", -10.0),
        ("Порог", 0.0),
        ("Средний", 10.0),
        ("Хороший", 15.0),
        ("Сильный", 20.0),
        ("Очень сильный", 25.0),
    ]

    n_samples = 5000
    freq_norm = 0.15
    noise_power = 1.0

    # Пред-вычисляем все спектры и CFAR для каждого SNR_in
    results = []
    for label, snr_in_db in snr_in_list:
        rng = np.random.default_rng(42)
        if snr_in_db == -999.0:
            # Только шум, без сигнала
            signal = make_awgn(n_samples, noise_power=noise_power, rng=rng)
        else:
            signal, _ = make_cw_with_snr(
                n_samples, freq_norm=freq_norm, snr_in_db=snr_in_db,
                noise_power=noise_power, rng=rng,
            )

        # Pipeline: step=1, N_actual=5000, N_fft=8192
        n_fft = 8192
        mag_sq = compute_mag_sq_hann(signal, n_fft)

        # CFAR
        cfg = CfarConfig(
            step_samples=1,
            guard_bins=FINAL_GUARD,
            ref_bins=FINAL_REF,
            window=FINAL_WINDOW,
            cfar_estimator=FINAL_ESTIMATOR,
        )
        res = estimate_snr_one_antenna(signal, cfg)

        results.append({
            "label": label,
            "snr_in_db": snr_in_db,
            "snr_fft_db": res.snr_db,
            "k_peak": res.k_peak,
            "noise_mean": res.noise_mean,
            "mag_sq": mag_sq,
            "n_fft": n_fft,
        })

    # ========================================================================
    # График 2x4 панелей
    # ========================================================================
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)

    for i, res in enumerate(results):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        mag_sq = res["mag_sq"]
        n_fft = res["n_fft"]
        k_peak = res["k_peak"]

        # ZOOM вокруг пика
        zoom = 80
        lo = max(0, k_peak - zoom)
        hi = min(n_fft, k_peak + zoom)
        x_rel = np.arange(lo, hi) - k_peak

        mag_db = 10 * np.log10(mag_sq[lo:hi] + 1e-30)

        # Основная кривая
        col_color = branch_color(res["snr_fft_db"])
        ax.plot(x_rel, mag_db, color=col_color, linewidth=1.2)

        # Пик (маркер)
        ax.plot(0, 10 * np.log10(mag_sq[k_peak] + 1e-30),
                "v", color=col_color, markersize=10,
                markeredgecolor="black")

        # Ref-окно CFAR (красный фон)
        ax.axvspan(-(FINAL_GUARD + FINAL_REF) - 0.5, -(FINAL_GUARD + 1) + 0.5,
                   alpha=0.15, color="red")
        ax.axvspan((FINAL_GUARD + 1) - 0.5, (FINAL_GUARD + FINAL_REF) + 0.5,
                   alpha=0.15, color="red")

        # Guard-зона (серый фон)
        ax.axvspan(-FINAL_GUARD - 0.5, FINAL_GUARD + 0.5,
                   alpha=0.20, color="gray")

        # Линия noise_mean
        noise_db = 10 * math.log10(max(res["noise_mean"], 1e-30))
        ax.axhline(noise_db, color="blue", linestyle=":",
                   alpha=0.7, linewidth=1.3)

        # Заголовок каждой панели
        if res["snr_in_db"] == -999.0:
            title_snr_in = "SNR_in = −∞ (только шум)"
        else:
            title_snr_in = f"SNR_in = {res['snr_in_db']:+.0f} dB"

        ax.set_title(
            f"{res['label']}\n"
            f"{title_snr_in}  →  SNR_fft = {res['snr_fft_db']:.1f} dB  "
            f"[{branch_name(res['snr_fft_db'])}]",
            fontsize=10, fontweight="bold",
            color=col_color,
        )

        ax.set_xlabel("смещение от пика (бины)", fontsize=8)
        ax.set_ylabel("|X[k]|² (dB)", fontsize=8)
        ax.set_xlim(-zoom, zoom)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Text box с числами
        textstr = (
            f"peak: {10*math.log10(mag_sq[k_peak]+1e-30):.1f} dB\n"
            f"noise: {noise_db:.1f} dB\n"
            f"SNR:  {res['snr_fft_db']:.1f} dB"
        )
        ax.text(
            0.02, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=8, family="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=col_color,
                linewidth=1.5,
                alpha=0.85,
            ),
        )

        # Легенда только в первом графике
        if i == 0:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            handles = [
                Patch(facecolor="red", alpha=0.15,
                      label=f"ref-окно (guard={FINAL_GUARD}, ref={FINAL_REF})"),
                Patch(facecolor="gray", alpha=0.2, label="guard зона"),
                Line2D([0], [0], color="blue", linestyle=":",
                       label="noise_mean (CFAR)"),
            ]
            ax.legend(handles=handles, loc="lower right", fontsize=7)

    # Общий заголовок и пояснения
    plt.suptitle(
        "Как растёт пик с увеличением SNR_in\n"
        "Обработка: Hann window + CA-CFAR (mean), guard=5, ref=16, N_actual=5000, N_fft=8192",
        fontsize=13, fontweight="bold", y=1.00,
    )

    # Пояснительный text box
    explanation = (
        "ЦВЕТ панели = BranchSelector решение:\n"
        "  КРАСНЫЙ (LOW)  — SNR_fft < 15 dB  → накопление (слабый сигнал)\n"
        "  ОРАНЖ  (MID)   — 15 ≤ SNR_fft < 30 → стандартная обработка\n"
        "  ЗЕЛЁНЫЙ (HIGH) — SNR_fft ≥ 30 dB  → точная обработка (сильный сигнал)\n\n"
        "Треугольник  = позиция пика (argmax |X|²)\n"
        "Красный фон  = ref-окно CFAR (усреднение для оценки шума)\n"
        "Серый фон    = guard-зона (исключена из оценки шума)\n"
        "Синяя точка  = уровень noise_mean из CFAR"
    )
    fig.text(
        0.5, -0.02, explanation,
        ha="center", va="top",
        fontsize=9, family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fffbea",
            edgecolor="#333",
            linewidth=1,
        ),
    )

    out = Path(__file__).resolve().parent / "plots" / "debug_peak_grows.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")

    # Печать результатов
    print()
    print("=" * 70)
    print(f"{'Label':<18} {'SNR_in':<12} {'SNR_fft':<12} {'Branch':<8}")
    print("-" * 70)
    for res in results:
        snr_in = ("-inf (noise)" if res["snr_in_db"] == -999.0
                  else f"{res['snr_in_db']:+.0f} dB")
        print(f"{res['label']:<18} {snr_in:<12} "
              f"{res['snr_fft_db']:>6.1f} dB   "
              f"{branch_name(res['snr_fft_db']):<8}")


if __name__ == "__main__":
    main()
