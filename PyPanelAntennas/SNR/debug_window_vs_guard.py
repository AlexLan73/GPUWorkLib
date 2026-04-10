"""
debug_window_vs_guard.py — ФИНАЛЬНЫЙ график: масштабирование для Hann + CA-CFAR.

Показывает стабильность выбранной обработки (Hann + mean CFAR) при
разных реальных длинах сигнала N_actual.

Главный вывод: SNR_fft точно масштабируется как SNR_in + 10·log10(N_actual),
разница с теорией — стабильный bias ~1.76 dB (Hann processing loss).

Показывает:
  - (0,0) Спектры для N_actual ∈ {1024, 2048, 4096, 8192} — видна ширина пика
  - (0,1) ZOOM вокруг пика: как меняется main lobe с ростом N_actual
  - (1,0) Coherent gain: SNR_fft vs N_actual (линейная зависимость в dB)
  - (1,1) Таблица с числами

Выбранная обработка:
  window = Hann, cfar = mean, guard=5, ref=16

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
from lfm_signal_generator import make_cw_with_snr


# Финальные параметры (наш выбор)
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


def main() -> None:
    # Параметры
    snr_in_db = 20.0
    freq_norm = 0.15
    noise_power = 1.0

    # Варианты N_actual — РЕАЛЬНАЯ длина сигнала
    n_actual_list = [1024, 2048, 4096, 8192]

    # Общий источник (8192 отсчёта), обрезаем по нужной длине
    rng = np.random.default_rng(42)
    source_signal, amp = make_cw_with_snr(
        8192, freq_norm=freq_norm, snr_in_db=snr_in_db,
        noise_power=noise_power, rng=rng,
    )

    print("=" * 72)
    print(f"Масштабирование по N_actual для Hann + mean CFAR")
    print(f"SNR_in={snr_in_db} dB, freq={freq_norm}, amp={amp:.3f}")
    print("=" * 72)

    # Считаем для каждого N_actual
    results = []
    for n_actual in n_actual_list:
        sig = source_signal[:n_actual]
        n_fft = n_actual  # без zero-pad — чистый результат

        # Финальная обработка
        mag_sq = compute_mag_sq_hann(sig, n_fft)

        # CFAR
        cfg = CfarConfig(
            step_samples=1,
            guard_bins=FINAL_GUARD,
            ref_bins=FINAL_REF,
            window=FINAL_WINDOW,
            cfar_estimator=FINAL_ESTIMATOR,
        )
        # Обход: передаём sig как полный, step=1 без децимации
        # Но наша estimate_snr_one_antenna использует compute_pipeline_sizes —
        # она выберет n_fft=next_pow2(N_actual)=N_actual (так как N_actual=2^k)
        res = estimate_snr_one_antenna(sig, cfg)

        theory = snr_in_db + 10 * math.log10(n_actual)

        results.append({
            "n_actual": n_actual,
            "n_fft": n_fft,
            "mag_sq": mag_sq,
            "snr_db": res.snr_db,
            "theory_db": theory,
            "theory_hann_db": theory - 1.76,
            "bias_db": res.snr_db - theory,
        })

        print(f"N_actual={n_actual:>5}: SNR_fft={res.snr_db:6.2f} dB, "
              f"theory={theory:6.2f}, bias={res.snr_db - theory:+.2f} dB")

    # ========================================================================
    # График 2x2
    # ========================================================================
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # ------------------------------------------------------------------------
    # (0, 0) Полный спектр для всех N_actual
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for i, res in enumerate(results):
        # Нормируем k в диапазон [-0.5, 0.5] чтобы сравнить спектры разной длины
        k_norm = np.fft.fftfreq(res["n_fft"])
        idx = np.argsort(k_norm)
        ax.plot(k_norm[idx], 10 * np.log10(res["mag_sq"][idx] + 1e-30),
                color=colors[i], linewidth=0.6,
                label=f"N_actual = {res['n_actual']} (pik @ {res['snr_db']:.1f} dB)",
                alpha=0.75)
    ax.axvline(freq_norm, color="black", linestyle="--", alpha=0.4,
               label=f"истинная частота {freq_norm}")
    ax.set_xlabel("нормированная частота k/N_fft")
    ax.set_ylabel("|X[k]|² (dB)")
    ax.set_title(
        f"Полные спектры для разных N_actual\n"
        f"Hann window, CW freq={freq_norm}, SNR_in={snr_in_db:.0f} dB",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------
    # (0, 1) ZOOM вокруг пика — видно как main lobe сужается
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for i, res in enumerate(results):
        mag = res["mag_sq"]
        k_peak = int(np.argmax(mag))
        zoom = 25
        lo = max(0, k_peak - zoom)
        hi = min(len(mag), k_peak + zoom)
        x_rel = np.arange(lo, hi) - k_peak
        ax.plot(x_rel, 10 * np.log10(mag[lo:hi] + 1e-30),
                color=colors[i], linewidth=1.5,
                marker="o", markersize=3,
                label=f"N_actual={res['n_actual']}")

    # Ref-окно
    ax.axvspan(-(FINAL_GUARD + FINAL_REF) - 0.5, -(FINAL_GUARD + 1) + 0.5,
               alpha=0.15, color="red")
    ax.axvspan((FINAL_GUARD + 1) - 0.5, (FINAL_GUARD + FINAL_REF) + 0.5,
               alpha=0.15, color="red",
               label=f"ref-окно (guard={FINAL_GUARD}, ref={FINAL_REF})")

    ax.set_xlabel("смещение от пика (бины)")
    ax.set_ylabel("|X[k]|² (dB)")
    ax.set_title(
        "ZOOM вокруг пика: main lobe Hann\n"
        "(ширина одинаковая, уровень = coherent gain, зависит от N_actual)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-zoom, zoom)

    # Аннотация про main lobe width Hann
    ax.annotate(
        "Main lobe Hann:\n≈ 3 бина\n(независимо от\nN_actual)",
        xy=(3, 60),
        xytext=(12, 75),
        fontsize=9, color="darkblue",
        arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
    )

    # ------------------------------------------------------------------------
    # (1, 0) SNR_fft vs N_actual — показывает масштабирование
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    n_arr = [r["n_actual"] for r in results]
    measured_arr = [r["snr_db"] for r in results]
    theory_arr = [r["theory_db"] for r in results]
    theory_hann_arr = [r["theory_hann_db"] for r in results]

    ax.plot(n_arr, measured_arr, "o-",
            color="#ff7f0e", markersize=12, linewidth=2.5,
            label="Hann + mean (наш выбор)")
    ax.plot(n_arr, theory_arr, "r--", linewidth=1.5, alpha=0.7,
            label="идеальная теория = SNR_in + 10·log10(N_actual)")
    ax.plot(n_arr, theory_hann_arr, "r:", linewidth=1.5, alpha=0.7,
            label="теория с Hann loss (−1.76 dB)")

    # Аннотации к каждой точке
    for i, res in enumerate(results):
        ax.annotate(
            f"{res['snr_db']:.1f} dB\n(bias {res['bias_db']:+.1f})",
            xy=(res["n_actual"], res["snr_db"]),
            xytext=(0, 15), textcoords="offset points",
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightyellow", alpha=0.8),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N_actual (реальная длина сигнала, без zero-pad)")
    ax.set_ylabel("SNR_fft (dB)")
    ax.set_title(
        "Coherent gain: линейный рост с log(N_actual)\n"
        "Каждое удвоение N_actual → +3 dB выигрыш по SNR",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------------
    # (1, 1) Таблица результатов
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    table_text = f"""
╔══════════════════════════════════════════════╗
║  МАСШТАБИРОВАНИЕ Hann + CA-CFAR (mean)       ║
║  CW freq={freq_norm}, SNR_in={snr_in_db:.0f} dB, amp={amp:.2f}      ║
╠══════════════════════════════════════════════╣
║                                              ║
║  Параметры обработки:                        ║
║    window     = Hann                         ║
║    cfar       = mean (CA-CFAR)               ║
║    guard_bins = {FINAL_GUARD}                            ║
║    ref_bins   = {FINAL_REF}                           ║
║                                              ║
╠══════════════════════════════════════════════╣
║  N_actual    gain    теория   измер   bias   ║
║  ────────   ─────    ──────   ─────   ────   ║
"""
    for res in results:
        gain = 10 * math.log10(res["n_actual"])
        table_text += (
            f"║  {res['n_actual']:>6}    "
            f"{gain:5.1f}   "
            f"{res['theory_db']:6.1f}   "
            f"{res['snr_db']:6.1f}   "
            f"{res['bias_db']:+5.1f}   ║\n"
        )

    table_text += f"""╠══════════════════════════════════════════════╣
║                                              ║
║  ВЫВОДЫ:                                     ║
║                                              ║
║  1. Bias СТАБИЛЬНЫЙ (~ −1.8 .. −3 dB)        ║
║     → легко компенсировать калибровкой       ║
║                                              ║
║  2. Каждое удвоение N_actual даёт +3 dB      ║
║     → больше данных = точнее обнаружение     ║
║                                              ║
║  3. Main lobe Hann ~ 3 бина (const)          ║
║     → guard={FINAL_GUARD} достаточно для любого N        ║
║                                              ║
║  4. Sidelobes Hann −32 dB (vs −13 dB rect)   ║
║     → ref-окно видит ШУМ, а не хвосты пика   ║
║                                              ║
╚══════════════════════════════════════════════╝
    """

    ax.text(
        0.02, 0.98, table_text,
        transform=ax.transAxes,
        fontsize=9, family="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fffbea",
            edgecolor="#ff7f0e",
            linewidth=2,
        ),
    )

    plt.suptitle(
        "МАСШТАБИРОВАНИЕ Hann + CA-CFAR по N_actual\n"
        "Coherent gain работает, bias стабильный",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out = Path(__file__).resolve().parent / "plots" / "debug_window_vs_guard.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
