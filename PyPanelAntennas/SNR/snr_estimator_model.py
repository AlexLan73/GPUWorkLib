"""
snr_estimator_model.py — главный скрипт Python модели SNR-estimator.

Запускает 5 экспериментов для КАЖДОЙ комбинации (window × cfar_estimator):
    - windows:    rect, hann, hamming, blackman
    - estimators: mean (CA-CFAR), median (OS-CFAR)
    = 8 комбинаций

Эксперименты:
    0. Быстрая сравнительная таблица — какая комбинация лучше?
    1. Базовая кривая SNR (проверка теории) + H0 артефакт
    2. Масштабирование n_samples (2K → 1.3M)
    3. Влияние step_samples
    4. Стабилизация по числу антенн (медиана по антеннам)
    5. Калибровка порогов branch (ROC)

Результаты:
    plots/exp{0..5}_*.png     — графики (каждый эксп. рисует сравнение по комбинациям)
    results/exp{1..5}_*.json  — численные результаты
    results/exp5_thresholds.json — ФИНАЛЬНЫЕ пороги для C++ (главный выход)

Запуск:
    "F:\\Program Files (x86)\\Python314\\python.exe" snr_estimator_model.py
    python3 snr_estimator_model.py

    --exp N     — запустить только эксперимент N (0..5)
    --quick     — быстрый прогон (меньше повторений)

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cfar_estimator import (  # noqa: E402
    CfarConfig,
    estimate_snr_one_antenna,
    estimate_snr_many_antennas,
    compute_pipeline_sizes,
    K_TARGET_N_FFT,
)
from lfm_signal_generator import make_cw_with_snr, make_awgn  # noqa: E402


# ============================================================================
# Конфигурация: комбинации (window × estimator)
# ============================================================================

WINDOWS = ["rect", "hann", "hamming", "blackman"]
ESTIMATORS = ["mean", "median"]

# Default для всех экспериментов (для JSON porогов из Эксп.5)
DEFAULT_WINDOW = "hann"
DEFAULT_ESTIMATOR = "mean"
DEFAULT_GUARD = 5
DEFAULT_REF = 16

# Цвета для графиков (одинаковые через все эксперименты)
COMBO_COLORS = {
    ("rect", "mean"):       "#1f77b4",  # синий
    ("rect", "median"):     "#aec7e8",  # св. синий
    ("hann", "mean"):       "#ff7f0e",  # оранжевый
    ("hann", "median"):     "#ffbb78",  # св. оранжевый
    ("hamming", "mean"):    "#2ca02c",  # зелёный
    ("hamming", "median"):  "#98df8a",  # св. зелёный
    ("blackman", "mean"):   "#d62728",  # красный
    ("blackman", "median"): "#ff9896",  # св. красный
}


def combo_label(window: str, estimator: str) -> str:
    return f"{window} + {estimator}"


def combo_color(window: str, estimator: str) -> str:
    return COMBO_COLORS.get((window, estimator), "black")


def combo_linestyle(estimator: str) -> str:
    return "-" if estimator == "mean" else "--"


# ============================================================================
# Infrastructure
# ============================================================================

PLOTS_DIR = _HERE / "plots"
RESULTS_DIR = _HERE / "results"
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=float)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def make_config(
    window: str,
    estimator: str,
    step_samples: int = 0,
    guard: int = DEFAULT_GUARD,
    ref: int = DEFAULT_REF,
) -> CfarConfig:
    return CfarConfig(
        target_n_fft=0,  # auto
        step_samples=step_samples,
        step_antennas=0,
        guard_bins=guard,
        ref_bins=ref,
        search_full_spectrum=True,
        window=window,
        cfar_estimator=estimator,
    )


# ============================================================================
# Эксперимент 0 — Быстрая сравнительная таблица всех комбинаций
# ============================================================================

def experiment_0_comparison(n_trials: int = 20) -> dict:
    """Эксп.0: быстро увидеть какая комбинация window×estimator лучше.

    Фиксированные параметры:
      - n_samples = 5000, SNR_in = 20 dB, freq_norm = 0.15
      - step_samples = 1 (без децимации) → N_actual=5000, N_fft=8192
      - guard=5, ref=16
      - Усредняем по n_trials trials с разным seed

    Метрики:
      - bias (dB) vs theory
      - std (dB) — повторяемость
    """
    log("=" * 70)
    log("EXP 0: Сравнение всех комбинаций window × estimator")
    log("=" * 70)

    n_samples = 5000
    snr_in_db = 20.0

    step, n_actual, n_fft = compute_pipeline_sizes(n_samples, 0, 1)
    theory_db = snr_in_db + 10.0 * math.log10(n_actual)
    log(f"  n_samples={n_samples}, N_actual={n_actual}, N_fft={n_fft}")
    log(f"  theory SNR_fft = {theory_db:.2f} dB")

    results = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            cfg = make_config(window, estimator, step_samples=1)
            trials_snr = np.zeros(n_trials)
            trials_noise_h0 = np.zeros(n_trials)  # H0: только шум
            for t in range(n_trials):
                rng = np.random.default_rng(42 + t)
                # Сигнал + шум
                sig, _ = make_cw_with_snr(
                    n_samples, 0.15, snr_in_db, 1.0, rng=rng
                )
                trials_snr[t] = estimate_snr_one_antenna(sig, cfg).snr_db
                # Только шум (H0 artifact)
                rng2 = np.random.default_rng(1000 + t)
                noise = make_awgn(n_samples, 1.0, rng=rng2)
                trials_noise_h0[t] = estimate_snr_one_antenna(noise, cfg).snr_db

            mean_snr = float(trials_snr.mean())
            std_snr = float(trials_snr.std())
            bias = mean_snr - theory_db
            h0_mean = float(trials_noise_h0.mean())
            h0_std = float(trials_noise_h0.std())

            results[combo_label(window, estimator)] = {
                "window": window,
                "estimator": estimator,
                "snr_mean_db": mean_snr,
                "snr_std_db": std_snr,
                "bias_vs_theory_db": bias,
                "h0_mean_db": h0_mean,
                "h0_std_db": h0_std,
            }

    # --- Печать таблицы ---
    log("")
    log(f"  {'Combo':<20} {'SNR_fft':<18} {'bias':<10} {'H0 noise':<18}")
    log(f"  {'-'*20} {'-'*18} {'-'*10} {'-'*18}")
    for label, r in results.items():
        log(f"  {label:<20} "
            f"{r['snr_mean_db']:>6.2f} ± {r['snr_std_db']:<5.2f} dB "
            f"{r['bias_vs_theory_db']:>+6.2f} dB  "
            f"{r['h0_mean_db']:>6.2f} ± {r['h0_std_db']:<5.2f} dB")

    # --- График: bar chart с богатыми аннотациями ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # (0): SNR measured vs theory
    ax = axes[0]
    labels = list(results.keys())
    snr_means = [results[l]["snr_mean_db"] for l in labels]
    snr_stds = [results[l]["snr_std_db"] for l in labels]
    colors = [combo_color(results[l]["window"], results[l]["estimator"]) for l in labels]
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, snr_means, yerr=snr_stds, color=colors, capsize=4)

    # Текст bias над каждым баром
    for i, (l, m) in enumerate(zip(labels, snr_means)):
        bias = results[l]["bias_vs_theory_db"]
        ax.text(i, m + 1.5, f"{bias:+.1f}", ha="center", fontsize=8,
                color="darkred" if bias < -10 else "black",
                fontweight="bold" if abs(bias) < 3 else "normal")

    ax.axhline(theory_db, color="red", linestyle="--", linewidth=2,
               label=f"Идеальная теория = SNR_in + 10·log10(N_actual) = {theory_db:.1f} dB")
    ax.axhline(theory_db - 1.76, color="orange", linestyle=":", linewidth=1.5,
               label=f"Теория − 1.76 dB (Hann processing loss) = {theory_db-1.76:.1f} dB")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("измеренный SNR_fft (dB)", fontsize=10)
    ax.set_title(
        f"ЧТО ПОКАЗАНО: измерение SNR_fft для 8 комбинаций обработки\n"
        f"(сигнал: CW SNR_in={snr_in_db} dB, N_actual={n_actual}, {n_trials} реализаций)\n"
        f"Чем ближе к красной линии — тем точнее. Числа над барами = bias в dB.",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=20)

    # (1): H0 artifact
    ax = axes[1]
    h0_means = [results[l]["h0_mean_db"] for l in labels]
    h0_stds = [results[l]["h0_std_db"] for l in labels]
    ax.bar(x_pos, h0_means, yerr=h0_stds, color=colors, capsize=4)

    for i, m in enumerate(h0_means):
        ax.text(i, m + 0.3, f"{m:.1f}", ha="center", fontsize=8)

    ax.axhline(10.0, color="red", linestyle="--", linewidth=2,
               label="Теоретический уровень ≈ 10 dB (ln(N_fft)+γ)")
    ax.axhline(15.0, color="green", linestyle=":", linewidth=1.5,
               label="Порог low_to_mid = 15 dB (с запасом от H0)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("H0 SNR (только шум, dB)", fontsize=10)
    ax.set_title(
        "H0 артефакт CFAR: что показывает детектор когда НЕТ сигнала\n"
        "(чистый AWGN, никакого CW) — должен быть НИЖЕ low_to_mid порога\n"
        "Стабильность ~10 dB → порог 15 dB надёжно разделяет шум и сигнал",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Эксп.0: Сравнение 8 комбинаций (window × CFAR estimator)\n"
        f"Параметры: N_actual={n_actual}, N_fft={n_fft}, guard={DEFAULT_GUARD}, ref={DEFAULT_REF}\n"
        f"ВЫБРАНА ФИНАЛЬНАЯ: {DEFAULT_WINDOW} + {DEFAULT_ESTIMATOR} (простота + P_correct=97.9%)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp0_comparison.png", dpi=110, bbox_inches="tight")
    plt.close()

    save_json(RESULTS_DIR / "exp0_comparison.json", {
        "n_samples": n_samples,
        "snr_in_db": snr_in_db,
        "n_actual": n_actual,
        "n_fft": n_fft,
        "theory_db": theory_db,
        "n_trials": n_trials,
        "results": results,
    })
    log(f"  Saved: {PLOTS_DIR / 'exp0_comparison.png'}")
    return results


# ============================================================================
# Эксперимент 1 — Базовая кривая SNR (для всех комбинаций)
# ============================================================================

def experiment_1_basic_curve(n_trials: int = 20) -> dict:
    """Эксп.1: кривые SNR_fft vs SNR_in для всех комбинаций + H0 артефакт."""
    log("=" * 70)
    log("EXP 1: Базовая кривая SNR_in → SNR_fft (все комбинации)")
    log("=" * 70)

    n_samples = 5000
    snr_in_list = [5, 10, 15, 20, 25, 30, 35, 40]

    step, n_actual, n_fft = compute_pipeline_sizes(n_samples, 0, 1)
    theory_line = np.array(snr_in_list) + 10.0 * math.log10(n_actual)

    combo_results = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            cfg = make_config(window, estimator, step_samples=1)
            measured = np.zeros(len(snr_in_list))
            stds = np.zeros(len(snr_in_list))

            for i, snr_in_db in enumerate(snr_in_list):
                trials = np.zeros(n_trials)
                for t in range(n_trials):
                    rng = np.random.default_rng(42 + t)
                    sig, _ = make_cw_with_snr(
                        n_samples, 0.15, float(snr_in_db), 1.0, rng
                    )
                    trials[t] = estimate_snr_one_antenna(sig, cfg).snr_db
                measured[i] = trials.mean()
                stds[i] = trials.std()

            # H0
            h0 = np.zeros(n_trials)
            for t in range(n_trials):
                rng = np.random.default_rng(1000 + t)
                noise = make_awgn(n_samples, 1.0, rng)
                h0[t] = estimate_snr_one_antenna(noise, cfg).snr_db

            combo_results[combo_label(window, estimator)] = {
                "window": window,
                "estimator": estimator,
                "snr_measured_db": measured.tolist(),
                "snr_std_db": stds.tolist(),
                "h0_mean_db": float(h0.mean()),
                "h0_std_db": float(h0.std()),
            }
            log(f"  {window:<10} {estimator:<8}: "
                f"SNR@20→{measured[3]:.1f} dB, H0={h0.mean():.1f} dB")

    # --- График 2×1 (sig и H0) с богатыми описаниями ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for label, r in combo_results.items():
        ax.errorbar(snr_in_list, r["snr_measured_db"], yerr=r["snr_std_db"],
                    fmt="o-" if r["estimator"] == "mean" else "s--",
                    color=combo_color(r["window"], r["estimator"]),
                    label=label, alpha=0.85, markersize=5, capsize=2)
    ax.plot(snr_in_list, theory_line, "k--", alpha=0.7, linewidth=2,
            label=f"Идеальная теория = SNR_in + {10*math.log10(n_actual):.1f} dB")

    # Аннотация: идеальная линия 1:1 наклон
    ax.annotate(
        "Хорошая обработка\nидёт ПАРАЛЛЕЛЬНО теории\n(наклон = 1)",
        xy=(25, 25 + 10*math.log10(n_actual)),
        xytext=(30, 45),
        fontsize=9, color="darkblue",
        arrowprops=dict(arrowstyle="->", color="darkblue"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
    )

    ax.set_xlabel("SNR_in (dB) — входное отношение сигнал/шум", fontsize=10)
    ax.set_ylabel("SNR_fft (dB) — измеренное CFAR", fontsize=10)
    ax.set_title(
        f"ЧТО ПОКАЗАНО: базовая кривая SNR_in → SNR_fft\n"
        f"(N_actual={n_actual}, N_fft={n_fft}, {n_trials} реализаций)\n"
        f"Чем ближе к чёрной пунктирной линии — тем меньше bias.",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = list(combo_results.keys())
    h0_means = [combo_results[l]["h0_mean_db"] for l in labels]
    h0_stds = [combo_results[l]["h0_std_db"] for l in labels]
    colors = [combo_color(combo_results[l]["window"],
                          combo_results[l]["estimator"]) for l in labels]
    x = np.arange(len(labels))
    ax.bar(x, h0_means, yerr=h0_stds, color=colors, capsize=4)
    for i, m in enumerate(h0_means):
        ax.text(i, m + 0.3, f"{m:.1f}", ha="center", fontsize=8)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.5,
               label="Теоретический артефакт 10·log10(ln(N_fft)+γ) ≈ 10 dB")
    ax.axhline(15.0, color="green", linestyle=":", linewidth=1.5,
               label="Порог low_to_mid = 15 dB (калибровано)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("H0 SNR (dB) — измерение на чистом шуме", fontsize=10)
    ax.set_title(
        "H0 артефакт: что показывает детектор БЕЗ сигнала\n"
        "(CW убран, остался только AWGN) — должно быть стабильно < порога",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Эксп.1: Базовая кривая SNR_in→SNR_fft + H0 артефакт\n"
        "Проверяем линейность связи и устойчивость оценки шума",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_basic_curve.png", dpi=110, bbox_inches="tight")
    plt.close()

    out = {
        "n_samples": n_samples,
        "snr_in_db": snr_in_list,
        "theory_line_db": theory_line.tolist(),
        "n_actual": n_actual,
        "n_fft": n_fft,
        "n_trials": n_trials,
        "combinations": combo_results,
    }
    save_json(RESULTS_DIR / "exp1_results.json", out)
    log(f"  Saved: {PLOTS_DIR / 'exp1_basic_curve.png'}")
    return out


# ============================================================================
# Эксперимент 2 — Масштабирование n_samples (только для best combos)
# ============================================================================

def experiment_2_scaling(n_trials: int = 10) -> dict:
    """Эксп.2: как SNR_fft зависит от n_samples для лучших комбинаций.

    Прогоняем все 8 комбинаций на SNR_in=10 dB, медиана по 5 антеннам.
    """
    log("=" * 70)
    log("EXP 2: Масштабирование n_samples (2K → 1.3M)")
    log("=" * 70)

    n_antennas = 5
    # Сокращённый список для скорости (убрали самые тяжёлые)
    n_samples_list = [2_000, 4_000, 8_000, 16_000, 32_000,
                      128_000, 512_000, 1_300_000]
    snr_in_db = 10.0

    combo_results = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            cfg = make_config(window, estimator)
            snr_measured = np.zeros(len(n_samples_list))
            snr_std = np.zeros(len(n_samples_list))
            step_auto = np.zeros(len(n_samples_list), dtype=int)
            n_act_auto = np.zeros(len(n_samples_list), dtype=int)
            n_fft_auto = np.zeros(len(n_samples_list), dtype=int)

            for j, n_samples in enumerate(n_samples_list):
                step, n_act, n_fft = compute_pipeline_sizes(n_samples, 0, 0)
                step_auto[j] = step
                n_act_auto[j] = n_act
                n_fft_auto[j] = n_fft

                trials = np.zeros(n_trials)
                for t in range(n_trials):
                    data = np.zeros((n_antennas, n_samples), dtype=np.complex64)
                    for a in range(n_antennas):
                        rng_a = np.random.default_rng(42 + t * 1000 + a + j * 5)
                        sig, _ = make_cw_with_snr(
                            n_samples, 0.1 + 0.03 * a, snr_in_db, 1.0, rng_a
                        )
                        data[a] = sig
                    median_db, _, _ = estimate_snr_many_antennas(data, cfg)
                    trials[t] = median_db
                snr_measured[j] = trials.mean()
                snr_std[j] = trials.std()

            combo_results[combo_label(window, estimator)] = {
                "window": window,
                "estimator": estimator,
                "snr_measured_db": snr_measured.tolist(),
                "snr_std_db": snr_std.tolist(),
                "step_auto": step_auto.tolist(),
                "n_actual_auto": n_act_auto.tolist(),
                "n_fft_auto": n_fft_auto.tolist(),
            }
            log(f"  {window:<10} {estimator:<8}: "
                f"min_SNR={snr_measured.min():.1f}, "
                f"max_SNR={snr_measured.max():.1f}, "
                f"max_std={snr_std.max():.2f}")

    # --- График 2×2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): SNR vs n_samples
    ax = axes[0, 0]
    for label, r in combo_results.items():
        ax.errorbar(n_samples_list, r["snr_measured_db"],
                    yerr=r["snr_std_db"],
                    fmt="o-" if r["estimator"] == "mean" else "s--",
                    color=combo_color(r["window"], r["estimator"]),
                    label=label, alpha=0.85, markersize=4, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("SNR_fft (dB)")
    ax.set_title(f"SNR_fft vs n_samples (SNR_in={snr_in_db} dB, медиана по 5 ант.)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # (0,1): std vs n_samples
    ax = axes[0, 1]
    for label, r in combo_results.items():
        ax.plot(n_samples_list, r["snr_std_db"],
                "o-" if r["estimator"] == "mean" else "s--",
                color=combo_color(r["window"], r["estimator"]),
                label=label, alpha=0.85, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("σ(SNR_fft) (dB)")
    ax.set_title("Стабильность оценки")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,0): auto step
    ax = axes[1, 0]
    any_combo = combo_results[combo_label("hann", "mean")]
    ax.plot(n_samples_list, any_combo["step_auto"], "s-", color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("step_samples (auto)")
    ax.set_title("Auto step_samples")
    ax.grid(True, alpha=0.3, which="both")

    # (1,1): auto N_actual / N_fft
    ax = axes[1, 1]
    ax.plot(n_samples_list, any_combo["n_actual_auto"], "o-",
            color="orange", label="N_actual")
    ax.plot(n_samples_list, any_combo["n_fft_auto"], "s-",
            color="green", label="N_fft (next pow2)")
    ax.set_xscale("log")
    ax.set_xlabel("n_samples")
    ax.set_ylabel("samples")
    ax.set_title("Auto N_actual / N_fft")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "Эксп.2: Масштабирование по n_samples (2K → 1.3M)\n"
        f"Как меняется SNR_fft при росте входных данных (SNR_in={snr_in_db} dB, медиана по {n_antennas} антеннам)\n"
        "Графики (0,0) — SNR измеренный, (0,1) — стабильность (σ), (1,0) — auto step, (1,1) — auto N_actual/N_fft",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_scaling.png", dpi=110, bbox_inches="tight")
    plt.close()

    out = {
        "n_antennas": n_antennas,
        "n_samples_list": n_samples_list,
        "snr_in_db": snr_in_db,
        "n_trials": n_trials,
        "combinations": combo_results,
    }
    save_json(RESULTS_DIR / "exp2_results.json", out)
    log(f"  Saved: {PLOTS_DIR / 'exp2_scaling.png'}")
    return out


# ============================================================================
# Эксперимент 3 — Влияние step_samples
# ============================================================================

def experiment_3_step_samples(n_trials: int = 15) -> dict:
    """Эксп.3: SNR vs step_samples при фиксированном n_samples."""
    log("=" * 70)
    log("EXP 3: Влияние step_samples")
    log("=" * 70)

    n_samples = 5000
    step_list = [1, 2, 4, 8, 16]
    snr_in_db = 20.0

    combo_results = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            snr_measured = np.zeros(len(step_list))
            snr_std = np.zeros(len(step_list))
            for j, step in enumerate(step_list):
                cfg = make_config(window, estimator, step_samples=step)
                trials = np.zeros(n_trials)
                for t in range(n_trials):
                    rng = np.random.default_rng(42 + t + j * 1000)
                    sig, _ = make_cw_with_snr(
                        n_samples, 0.12, snr_in_db, 1.0, rng
                    )
                    trials[t] = estimate_snr_one_antenna(sig, cfg).snr_db
                snr_measured[j] = trials.mean()
                snr_std[j] = trials.std()
            combo_results[combo_label(window, estimator)] = {
                "window": window, "estimator": estimator,
                "snr_measured_db": snr_measured.tolist(),
                "snr_std_db": snr_std.tolist(),
            }
            log(f"  {window:<10} {estimator:<8}: {snr_measured.tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for label, r in combo_results.items():
        ax.errorbar(step_list, r["snr_measured_db"], yerr=r["snr_std_db"],
                    fmt="o-" if r["estimator"] == "mean" else "s--",
                    color=combo_color(r["window"], r["estimator"]),
                    label=label, alpha=0.85, markersize=5, capsize=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("step_samples")
    ax.set_ylabel("SNR_fft (dB)")
    ax.set_title(f"SNR_fft vs step_samples (SNR_in={snr_in_db} dB)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for label, r in combo_results.items():
        ax.plot(step_list, r["snr_std_db"],
                "o-" if r["estimator"] == "mean" else "s--",
                color=combo_color(r["window"], r["estimator"]),
                label=label, alpha=0.85)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("step_samples")
    ax.set_ylabel("σ(SNR_fft) (dB)")
    ax.set_title("Разброс — ищем колено")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        f"Эксп.3: Влияние step_samples на точность (n_samples={n_samples}, SNR_in={snr_in_db} dB)\n"
        "step=1 (без децимации) vs step=16 — ищем 'колено' точность/скорость.\n"
        "Левый график: SNR (чем выше — тем точнее). Правый: σ (чем ниже — тем стабильнее).",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_step_samples.png", dpi=110, bbox_inches="tight")
    plt.close()

    out = {
        "n_samples": n_samples,
        "step_list": step_list,
        "snr_in_db": snr_in_db,
        "n_trials": n_trials,
        "combinations": combo_results,
    }
    save_json(RESULTS_DIR / "exp3_results.json", out)
    log(f"  Saved: {PLOTS_DIR / 'exp3_step_samples.png'}")
    return out


# ============================================================================
# Эксперимент 4 — Стабилизация по числу антенн (медиана по антеннам!)
# ============================================================================

def experiment_4_antennas(n_trials: int = 20) -> dict:
    """Эксп.4: Var(median(SNR_per_ant)) vs N_ants для всех комбинаций.

    ЭТО ТА МЕДИАНА ПО АНТЕННАМ которая у нас уже есть в estimate_snr_many_antennas.
    Проверяем: при каком N_ants дисперсия медианы < 1 dB?
    """
    log("=" * 70)
    log("EXP 4: Стабилизация медианы ПО АНТЕННАМ")
    log("=" * 70)

    n_samples = 5000
    snr_in_db = 10.0
    n_ants_list = [2, 5, 10, 20, 50, 100]

    combo_results = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            cfg = make_config(window, estimator)
            var_by_n = []
            for n_ants in n_ants_list:
                medians = np.zeros(n_trials)
                for t in range(n_trials):
                    data = np.zeros((n_ants, n_samples), dtype=np.complex64)
                    for a in range(n_ants):
                        rng = np.random.default_rng(42 + t * 10000 + a)
                        freq = 0.05 + 0.25 * ((a * 137) % 100) / 100  # псевдо-случайно
                        sig, _ = make_cw_with_snr(
                            n_samples, freq, snr_in_db, 1.0, rng
                        )
                        data[a] = sig
                    median_db, _, _ = estimate_snr_many_antennas(data, cfg)
                    medians[t] = median_db
                var_by_n.append({
                    "n_ants": n_ants,
                    "median_mean_db": float(medians.mean()),
                    "median_std_db": float(medians.std()),
                })

            combo_results[combo_label(window, estimator)] = {
                "window": window, "estimator": estimator,
                "variance_by_n_ants": var_by_n,
            }
            stds = [v["median_std_db"] for v in var_by_n]
            log(f"  {window:<10} {estimator:<8}: stds={['%.2f'%s for s in stds]}")

    fig, ax = plt.subplots(figsize=(12, 8))
    for label, r in combo_results.items():
        n_arr = [v["n_ants"] for v in r["variance_by_n_ants"]]
        std_arr = [v["median_std_db"] for v in r["variance_by_n_ants"]]
        ax.plot(n_arr, std_arr,
                "o-" if r["estimator"] == "mean" else "s--",
                color=combo_color(r["window"], r["estimator"]),
                label=label, alpha=0.85, markersize=7, linewidth=1.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=2,
               label="Целевая стабильность σ < 1 dB")
    ax.set_xscale("log")
    ax.set_xlabel("N антенн в медиане", fontsize=11)
    ax.set_ylabel("σ(median SNR_fft) (dB) — разброс оценки", fontsize=11)
    ax.set_title(
        f"Эксп.4: Стабилизация МЕДИАНЫ ПО АНТЕННАМ (SNR_in={snr_in_db} dB)\n"
        "Сколько нужно антенн чтобы медиана была стабильна?\n"
        "Тот самый 'MedianRadixSortOp::ExecuteFloat(1, n_ant_used)' в C++ коде.",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # Аннотация "default 50 антенн"
    ax.axvline(50, color="green", linestyle=":", alpha=0.6, linewidth=2)
    ax.text(50 * 1.05, ax.get_ylim()[1] * 0.9,
            "default:\n50 антенн",
            fontsize=9, color="darkgreen", fontweight="bold",
            verticalalignment="top")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_antennas.png", dpi=110, bbox_inches="tight")
    plt.close()

    out = {
        "n_samples": n_samples,
        "snr_in_db": snr_in_db,
        "n_ants_list": n_ants_list,
        "n_trials": n_trials,
        "combinations": combo_results,
    }
    save_json(RESULTS_DIR / "exp4_results.json", out)
    log(f"  Saved: {PLOTS_DIR / 'exp4_antennas.png'}")
    return out


# ============================================================================
# Эксперимент 5 — Калибровка порогов (для DEFAULT комбинации)
# ============================================================================

def experiment_5_thresholds(n_trials: int = 200) -> dict:
    """Эксп.5: калибровка порогов branch для default combo + сравнение всех.

    Истинные классы по SNR_in:
      - Low:  SNR_in < -15 dB
      - Mid:  -15 <= SNR_in < 0 dB
      - High: SNR_in >= 0 dB

    Для каждой комбинации считаем лучшие (low_to_mid, mid_to_high) по grid search.
    Сохраняем ГЛАВНЫЕ пороги — для default combo (Hann + mean).
    """
    log("=" * 70)
    log(f"EXP 5: Калибровка порогов (n_trials={n_trials})")
    log("=" * 70)

    n_samples = 5000
    n_antennas = 5
    snr_in_range = np.arange(-30.0, 21.0, 1.0)

    def true_branch(snr_in_db: float) -> int:
        if snr_in_db < -15.0:
            return 0
        elif snr_in_db < 0.0:
            return 1
        else:
            return 2

    # Собираем данные для всех комбинаций
    combo_thresholds = {}
    for window in WINDOWS:
        for estimator in ESTIMATORS:
            cfg = make_config(window, estimator)
            log(f"  {combo_label(window, estimator)} ...")

            all_samples = []  # (snr_fft_measured, true_branch)
            snr_fft_stats = {}  # snr_in -> mean/std
            for snr_in_db in snr_in_range:
                trials = np.zeros(n_trials)
                for t in range(n_trials):
                    data = np.zeros((n_antennas, n_samples), dtype=np.complex64)
                    for a in range(n_antennas):
                        rng = np.random.default_rng(42 + t * 100 + a)
                        sig, _ = make_cw_with_snr(
                            n_samples, 0.1 + 0.03 * a,
                            float(snr_in_db), 1.0, rng
                        )
                        data[a] = sig
                    median_db, _, _ = estimate_snr_many_antennas(data, cfg)
                    trials[t] = median_db

                tb = true_branch(float(snr_in_db))
                for v in trials:
                    all_samples.append((float(v), tb))
                snr_fft_stats[float(snr_in_db)] = {
                    "mean_db": float(trials.mean()),
                    "std_db": float(trials.std()),
                }

            # Grid search
            snr_arr = np.array([s[0] for s in all_samples])
            true_arr = np.array([s[1] for s in all_samples], dtype=int)

            threshold_candidates = np.arange(0.0, 40.0, 1.0)
            best_p = 0.0
            best_low = 0.0
            best_high = 0.0
            for tl in threshold_candidates:
                for th in threshold_candidates:
                    if th <= tl + 1.0:
                        continue
                    pred = np.zeros_like(snr_arr, dtype=int)
                    pred[snr_arr >= th] = 2
                    pred[(snr_arr >= tl) & (snr_arr < th)] = 1
                    p = float((pred == true_arr).mean())
                    if p > best_p:
                        best_p = p
                        best_low = float(tl)
                        best_high = float(th)

            combo_thresholds[combo_label(window, estimator)] = {
                "window": window,
                "estimator": estimator,
                "low_to_mid_db": best_low,
                "mid_to_high_db": best_high,
                "p_correct": best_p,
                "snr_fft_by_snr_in": snr_fft_stats,
            }
            log(f"    low={best_low:.1f}, high={best_high:.1f}, P={best_p:.3f}")

    # --- График: 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): SNR_fft vs SNR_in для всех комбинаций
    ax = axes[0, 0]
    for label, r in combo_thresholds.items():
        snr_in_sorted = sorted(r["snr_fft_by_snr_in"].keys())
        means = [r["snr_fft_by_snr_in"][k]["mean_db"] for k in snr_in_sorted]
        ax.plot(snr_in_sorted, means,
                "o-" if r["estimator"] == "mean" else "s--",
                color=combo_color(r["window"], r["estimator"]),
                label=label, alpha=0.85, markersize=3)
    ax.axvline(-15, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("SNR_in (dB)")
    ax.set_ylabel("SNR_fft measured (dB)")
    ax.set_title("SNR_fft vs SNR_in (истинные границы -15, 0 dB)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    # (0,1): P_correct для всех
    ax = axes[0, 1]
    labels = list(combo_thresholds.keys())
    ps = [combo_thresholds[l]["p_correct"] for l in labels]
    colors = [combo_color(combo_thresholds[l]["window"],
                          combo_thresholds[l]["estimator"]) for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, ps, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.9, color="red", linestyle=":", alpha=0.7, label="P_correct > 0.9")
    ax.set_ylabel("P_correct")
    ax.set_title("Точность классификации по комбинациям")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0): пороги low/high для всех
    ax = axes[1, 0]
    lows = [combo_thresholds[l]["low_to_mid_db"] for l in labels]
    highs = [combo_thresholds[l]["mid_to_high_db"] for l in labels]
    width = 0.35
    ax.bar(x - width / 2, lows, width, label="low_to_mid", color="orange")
    ax.bar(x + width / 2, highs, width, label="mid_to_high", color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Порог (dB)")
    ax.set_title("Калиброванные пороги по комбинациям")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1): default combo (Hann + mean) — SNR_fft кривая
    ax = axes[1, 1]
    default_key = combo_label(DEFAULT_WINDOW, DEFAULT_ESTIMATOR)
    r = combo_thresholds[default_key]
    snr_in_sorted = sorted(r["snr_fft_by_snr_in"].keys())
    means = np.array([r["snr_fft_by_snr_in"][k]["mean_db"] for k in snr_in_sorted])
    stds = np.array([r["snr_fft_by_snr_in"][k]["std_db"] for k in snr_in_sorted])
    ax.errorbar(snr_in_sorted, means, yerr=stds,
                fmt="o-", color="darkblue", capsize=2, markersize=4)
    ax.axhline(r["low_to_mid_db"], color="orange", linestyle="--",
               label=f"low_to_mid = {r['low_to_mid_db']:.1f}")
    ax.axhline(r["mid_to_high_db"], color="red", linestyle="--",
               label=f"mid_to_high = {r['mid_to_high_db']:.1f}")
    ax.axvline(-15, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("SNR_in (dB)")
    ax.set_ylabel("SNR_fft (dB)")
    ax.set_title(f"DEFAULT: {default_key} (P_correct = {r['p_correct']:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Эксп.5: КАЛИБРОВКА ПОРОГОВ branch selector (главный выход Python модели)\n"
        "Истинные классы: Low (SNR_in < −15), Mid (−15..0), High (≥ 0)\n"
        "Grid search по порогам → P_correct. ФИНАЛЬНЫЕ ЗНАЧЕНИЯ для C++ в правом нижнем.",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp5_roc_thresholds.png", dpi=110, bbox_inches="tight")
    plt.close()

    # --- Сохранение порогов (главный выход!) ---
    default_combo = combo_thresholds[combo_label(DEFAULT_WINDOW, DEFAULT_ESTIMATOR)]
    thresholds_output = {
        "low_to_mid_db": default_combo["low_to_mid_db"],
        "mid_to_high_db": default_combo["mid_to_high_db"],
        "hysteresis_db": 2.0,
        "p_correct": default_combo["p_correct"],
        "default_combo": {
            "window": DEFAULT_WINDOW,
            "estimator": DEFAULT_ESTIMATOR,
        },
        "all_combinations": {
            label: {
                "window": r["window"],
                "estimator": r["estimator"],
                "low_to_mid_db": r["low_to_mid_db"],
                "mid_to_high_db": r["mid_to_high_db"],
                "p_correct": r["p_correct"],
            }
            for label, r in combo_thresholds.items()
        },
        "calibration_parameters": {
            "n_samples": n_samples,
            "n_antennas": n_antennas,
            "target_n_fft": K_TARGET_N_FFT,
            "n_trials_per_snr": n_trials,
            "snr_in_range_db": [float(snr_in_range[0]), float(snr_in_range[-1])],
            "guard_bins": DEFAULT_GUARD,
            "ref_bins": DEFAULT_REF,
        },
        "true_branch_definition": {
            "Low":  "SNR_in < -15 dB",
            "Mid":  "-15 <= SNR_in < 0 dB",
            "High": "SNR_in >= 0 dB",
        },
    }
    save_json(RESULTS_DIR / "exp5_thresholds.json", thresholds_output)
    log(f"  Saved: {PLOTS_DIR / 'exp5_roc_thresholds.png'}")
    log(f"  Saved: {RESULTS_DIR / 'exp5_thresholds.json'}  ← ГЛАВНЫЙ ВЫХОД")
    log("")
    log("  Best combinations by P_correct:")
    sorted_by_p = sorted(
        combo_thresholds.items(),
        key=lambda kv: -kv[1]["p_correct"]
    )
    for i, (label, r) in enumerate(sorted_by_p[:5]):
        log(f"    {i+1}. {label:<20} P={r['p_correct']:.3f}  "
            f"low={r['low_to_mid_db']:.1f}  high={r['mid_to_high_db']:.1f}")
    return thresholds_output


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SNR-estimator Python model")
    parser.add_argument("--exp", type=int, default=0,
                        help="Запустить только эксперимент N (0..5), -1 = все")
    parser.add_argument("--quick", action="store_true",
                        help="Быстрый прогон (меньше повторений)")
    args = parser.parse_args()

    if args.quick:
        trials = {"0": 5, "1": 5, "2": 3, "3": 5, "4": 5, "5": 30}
    else:
        trials = {"0": 20, "1": 20, "2": 10, "3": 15, "4": 20, "5": 200}

    log("=" * 70)
    log("SNR-estimator Python model")
    log(f"Quick: {args.quick}, exp: {args.exp}")
    log(f"Plots: {PLOTS_DIR}")
    log("=" * 70)

    t_start = time.time()

    if args.exp in (0, -1):
        experiment_0_comparison(n_trials=trials["0"])
    if args.exp in (1, -1):
        experiment_1_basic_curve(n_trials=trials["1"])
    if args.exp in (2, -1):
        experiment_2_scaling(n_trials=trials["2"])
    if args.exp in (3, -1):
        experiment_3_step_samples(n_trials=trials["3"])
    if args.exp in (4, -1):
        experiment_4_antennas(n_trials=trials["4"])
    if args.exp in (5, -1):
        experiment_5_thresholds(n_trials=trials["5"])

    elapsed = time.time() - t_start
    log("=" * 70)
    log(f"DONE in {elapsed:.1f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
