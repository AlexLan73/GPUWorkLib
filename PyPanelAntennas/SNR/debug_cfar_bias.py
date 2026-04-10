"""
debug_cfar_bias.py — ФИНАЛЬНЫЙ график выбранной обработки: Hann + CA-CFAR (mean).

Показывает что наша финальная связка решает проблему sinc sidelobes:
  - Слева сверху: полный спектр (rect vs Hann) — видна разница в шумовой полке
  - Справа сверху: ZOOM вокруг пика — видно как Hann убивает sinc sidelobes
  - Слева снизу: SNR vs guard_bins для Hann+mean — плато! guard не важен
  - Справа снизу: таблица с результатами и пояснения

Выбранная обработка (после ревью 2026-04-09):
  window       = Hann
  cfar         = CA-CFAR (mean)
  guard_bins   = 5
  ref_bins     = 16
  target_n_fft = 2048 (по умолчанию, гибкий)

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
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfar_estimator import (
    CfarConfig,
    estimate_snr_one_antenna,
    compute_pipeline_sizes,
    make_window,
)
from lfm_signal_generator import make_cw_with_snr


# ============================================================================
# Параметры финальной обработки (наш выбор после ревью)
# ============================================================================

FINAL_WINDOW = "hann"
FINAL_ESTIMATOR = "mean"
FINAL_GUARD = 5
FINAL_REF = 16


def compute_spectrum(
    signal: np.ndarray,
    n_fft: int,
    window: str = "rect",
) -> np.ndarray:
    """Вычислить |X[k]|² с опциональным окном."""
    n = len(signal)
    if window == "rect":
        w_sig = signal.astype(np.complex64)
    else:
        w = make_window(window, n).astype(np.float32)
        w_sig = (signal * w).astype(np.complex64)

    padded = np.zeros(n_fft, dtype=np.complex64)
    padded[:n] = w_sig
    X = np.fft.fft(padded)
    return (np.abs(X) ** 2).astype(np.float32)


def cfar_snr_at_guard(
    mag_sq: np.ndarray,
    guard: int,
    ref: int,
) -> tuple[float, int, float]:
    """CA-CFAR (mean). Возвращает (snr_db, k_peak, noise_mean)."""
    n = len(mag_sq)
    k_peak = int(np.argmax(mag_sq))
    peak = float(mag_sq[k_peak])

    values = np.empty(2 * ref, dtype=np.float32)
    for i in range(ref):
        off = guard + 1 + i
        values[2 * i] = mag_sq[(k_peak - off) % n]
        values[2 * i + 1] = mag_sq[(k_peak + off) % n]
    noise = max(float(values.mean()), 1e-30)
    ratio = max(peak / noise, 1e-30)
    return 10.0 * math.log10(ratio), k_peak, noise


def main() -> None:
    # ========================================================================
    # Параметры эксперимента
    # ========================================================================
    n_samples = 5000
    snr_in_db = 20.0
    freq_norm = 0.15
    noise_power = 1.0

    rng = np.random.default_rng(42)
    signal, amp = make_cw_with_snr(
        n_samples, freq_norm=freq_norm, snr_in_db=snr_in_db,
        noise_power=noise_power, rng=rng,
    )

    # Pipeline параметры (реплика SnrEstimatorOp)
    step, n_actual, n_fft = compute_pipeline_sizes(n_samples, 0, 1)  # step=1
    coherent_gain_db = 10.0 * math.log10(n_actual)
    theory_db = snr_in_db + coherent_gain_db

    print(f"n_samples={n_samples}, N_actual={n_actual}, N_fft={n_fft}")
    print(f"theory SNR_fft = {theory_db:.2f} dB")

    # Два спектра для сравнения
    spec_rect = compute_spectrum(signal, n_fft, window="rect")
    spec_hann = compute_spectrum(signal, n_fft, window="hann")

    k_peak_rect = int(np.argmax(spec_rect))
    k_peak_hann = int(np.argmax(spec_hann))

    # "Истинный" noise floor — далеко от пика
    def true_noise(spec, k_peak, exclude=300):
        n_fft_loc = len(spec)
        mask = np.ones(n_fft_loc, dtype=bool)
        lo = max(0, k_peak - exclude)
        hi = min(n_fft_loc, k_peak + exclude)
        mask[lo:hi] = False
        return float(spec[mask].mean())

    noise_rect = true_noise(spec_rect, k_peak_rect)
    noise_hann = true_noise(spec_hann, k_peak_hann)

    # CFAR результаты для разных guard (Hann + mean)
    guard_sweep = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    snr_rect_sweep = []
    snr_hann_sweep = []
    for g in guard_sweep:
        sr, _, _ = cfar_snr_at_guard(spec_rect, g, FINAL_REF)
        sh, _, _ = cfar_snr_at_guard(spec_hann, g, FINAL_REF)
        snr_rect_sweep.append(sr)
        snr_hann_sweep.append(sh)

    # Финальный результат выбранной обработки
    cfg_final = CfarConfig(
        step_samples=1,
        guard_bins=FINAL_GUARD,
        ref_bins=FINAL_REF,
        window=FINAL_WINDOW,
        cfar_estimator=FINAL_ESTIMATOR,
    )
    final_result = estimate_snr_one_antenna(signal, cfg_final)

    # ========================================================================
    # ГРАФИК 2x2
    # ========================================================================
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # ------------------------------------------------------------------------
    # (0, 0) Полный спектр: сравнение rect vs Hann
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    k_arr = np.arange(n_fft)
    ax.plot(k_arr, 10 * np.log10(spec_rect + 1e-30),
            color="#aec7e8", linewidth=0.5,
            label="БЕЗ окна (rectangular)")
    ax.plot(k_arr, 10 * np.log10(spec_hann + 1e-30),
            color="#ff7f0e", linewidth=0.6,
            label="С Hann window (наш выбор)")
    ax.axvline(k_peak_rect, color="red", linestyle="--", alpha=0.4,
               label=f"пик @ k={k_peak_rect}")
    ax.axhline(10 * math.log10(noise_hann), color="green",
               linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"истинный шум ≈ {10*math.log10(noise_hann):.0f} dB")
    ax.set_xlabel("частотный бин k")
    ax.set_ylabel("|X[k]|² (dB)")
    ax.set_title(
        f"Полный спектр FFT\n"
        f"N_actual={n_actual}, N_fft={n_fft}, SNR_in={snr_in_db:.0f} dB",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------------
    # (0, 1) ZOOM вокруг пика — главный показ!
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    zoom = 60
    lo = max(0, k_peak_hann - zoom)
    hi = min(n_fft, k_peak_hann + zoom)
    x_rel = np.arange(lo, hi) - k_peak_hann

    ax.plot(x_rel, 10 * np.log10(spec_rect[lo:hi] + 1e-30),
            color="#1f77b4", linewidth=1.3, marker="o", markersize=2.5,
            label="rect — sidelobes −13 dB")
    ax.plot(x_rel, 10 * np.log10(spec_hann[lo:hi] + 1e-30),
            color="#ff7f0e", linewidth=1.5, marker="s", markersize=3,
            label="Hann — sidelobes −32 dB")

    # Зона ref-window (guard=5, ref=16) — красным фоном
    ax.axvspan(-(FINAL_GUARD + FINAL_REF) - 0.5, -(FINAL_GUARD + 1) + 0.5,
               alpha=0.15, color="red")
    ax.axvspan((FINAL_GUARD + 1) - 0.5, (FINAL_GUARD + FINAL_REF) + 0.5,
               alpha=0.15, color="red",
               label=f"ref-окно CFAR (guard={FINAL_GUARD}, ref={FINAL_REF})")

    # Зона guard (пик + защита) — серым
    ax.axvspan(-FINAL_GUARD - 0.5, FINAL_GUARD + 0.5,
               alpha=0.20, color="gray",
               label=f"guard-зона (±{FINAL_GUARD} бинов)")

    # Истинный шум линия
    ax.axhline(10 * math.log10(noise_hann), color="green",
               linestyle=":", linewidth=1.5,
               label=f"истинный шум {10*math.log10(noise_hann):.0f} dB")

    ax.set_xlabel("смещение от пика (бины)")
    ax.set_ylabel("|X[k]|² (dB)")
    ax.set_title(
        "ZOOM вокруг пика: почему Hann работает\n"
        "(ref-окно Hann попадает в ШУМ, а не в хвосты sinc)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-zoom, zoom)

    # Аннотация стрелка на ref-окно
    ax.annotate(
        "Hann кривая\nсходится к шуму\nчерез ~10 бинов",
        xy=(15, 10 * math.log10(noise_hann) + 3),
        xytext=(35, 50),
        fontsize=9, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # ------------------------------------------------------------------------
    # (1, 0) SNR vs guard_bins — плато для Hann!
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(guard_sweep, snr_rect_sweep, "o-",
            color="#1f77b4", markersize=8, linewidth=1.5,
            label="rect (без окна)")
    ax.plot(guard_sweep, snr_hann_sweep, "s-",
            color="#ff7f0e", markersize=8, linewidth=2.0,
            label="Hann (наш выбор)")

    # Теоретические линии
    ax.axhline(theory_db, color="red", linestyle="--", alpha=0.6,
               label=f"теория {theory_db:.1f} dB")
    ax.axhline(theory_db - 1.76, color="red", linestyle=":", alpha=0.5,
               label=f"теория − 1.76 dB (Hann loss) ≈ {theory_db-1.76:.1f} dB")

    # Вертикальная линия — наш default guard
    ax.axvline(FINAL_GUARD, color="green", linestyle="-", alpha=0.4, linewidth=2)
    ax.text(FINAL_GUARD * 1.1, theory_db - 25,
            f"наш default\nguard={FINAL_GUARD}",
            fontsize=9, color="darkgreen", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("guard_bins")
    ax.set_ylabel("измеренный SNR_fft (dB)")
    ax.set_title(
        "Зависимость SNR от guard_bins\n"
        "(Hann — ПЛАТО уже с guard=5, rect требует guard≥200)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------------
    # (1, 1) Текстовая сводка — параметры + результаты
    # ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    summary_text = f"""
╔══════════════════════════════════════════════╗
║  ФИНАЛЬНАЯ ОБРАБОТКА (выбор 2026-04-09)      ║
╠══════════════════════════════════════════════╣
║                                              ║
║   Window function:   Hann                    ║
║                      w[n] = 0.5·(1 − cos θ)  ║
║                      θ = 2π·n/(N-1)          ║
║                                              ║
║   CFAR estimator:    CA-CFAR (mean)          ║
║                      noise_est = mean(ref)   ║
║                                              ║
║   guard_bins:        {FINAL_GUARD}                       ║
║   ref_bins:          {FINAL_REF}                      ║
║   target_n_fft:      2048 (auto, гибкий)     ║
║                                              ║
╠══════════════════════════════════════════════╣
║  РЕЗУЛЬТАТЫ ТЕКУЩЕГО ТЕСТА                   ║
╠══════════════════════════════════════════════╣
║                                              ║
║   Вход:              CW freq={freq_norm}, SNR_in={snr_in_db:.0f} dB ║
║   N_actual / N_fft:  {n_actual} / {n_fft}             ║
║                                              ║
║   Теория SNR_fft:    {theory_db:6.2f} dB              ║
║   (с Hann loss):     {theory_db - 1.76:6.2f} dB              ║
║                                              ║
║   Измерено (rect):   {cfar_snr_at_guard(spec_rect, FINAL_GUARD, FINAL_REF)[0]:6.2f} dB  [BAD]   ║
║   Измерено (Hann):   {final_result.snr_db:6.2f} dB  [OK]    ║
║                                              ║
║   Разница с теорией:                         ║
║     rect:  {cfar_snr_at_guard(spec_rect, FINAL_GUARD, FINAL_REF)[0] - theory_db:+6.2f} dB (катастрофа)      ║
║     Hann:  {final_result.snr_db - theory_db:+6.2f} dB (приемлемо)       ║
║                                              ║
╠══════════════════════════════════════════════╣
║  ПОЧЕМУ HANN РАБОТАЕТ                        ║
╠══════════════════════════════════════════════╣
║                                              ║
║  Hann window "скругляет" края сигнала →      ║
║  частотные sidelobes сигнала падают с        ║
║  −13 dB до −32 dB относительно пика.         ║
║                                              ║
║  Результат: ref-окно CFAR видит реальный     ║
║  шум, а не хвосты пика → правильный SNR.     ║
║                                              ║
╚══════════════════════════════════════════════╝
    """

    ax.text(
        0.02, 0.98, summary_text,
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
        "ФИНАЛЬНАЯ ОБРАБОТКА: Hann window + CA-CFAR (mean)\n"
        "SNR-estimator, выбранный набор параметров",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out = Path(__file__).resolve().parent / "plots" / "debug_cfar_bias.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
