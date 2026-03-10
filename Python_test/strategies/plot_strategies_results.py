#!/usr/bin/env python3
"""
plot_strategies_results.py — визуализация Pipeline A vs B для strategies модуля
==============================================================================

Генерирует 4 графика:
  1. Спектры Pipeline A vs Pipeline B (5 антенн, 8000 точек)
  2. Debug checkpoints 2.1 / 2.2 / 2.3 — сигнал на каждом этапе
  3. Сравнение найденных пиков по антеннам
  4. Farrow delay effect — разница между S_raw и S_aligned

Сохраняет в Results/Plots/strategies/

Usage:
  python3 plot_strategies_results.py

Author: Kodo (AI Assistant)
Date: 2026-03-10
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI
import matplotlib.pyplot as plt

# Добавляем путь к нашим модулям
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from scenario_builder import ScenarioBuilder, ULAGeometry
from farrow_delay import FarrowDelay
from pipeline_runner import PipelineRunner, PipelineConfig

OUT_DIR = os.path.normpath(os.path.join(script_dir, "..", "..", "Results", "Plots", "strategies"))
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Сценарий: 5 антенн, 1 источник на 30°, f0=2 МГц
# ============================================================================

def build_scenario():
    geom = ULAGeometry(n_ant=5, d_ant_m=0.075)
    builder = ScenarioBuilder(array=geom, fs=12e6, n_samples=8000)
    builder.add_target(theta_deg=30.0, f0_hz=2e6, amplitude=1.0)
    return builder.build()


def compute_pipelines(scenario):
    runner = PipelineRunner(output_dir=None)
    config = PipelineConfig(
        save_input=False, save_aligned=False,
        save_gemm=False, save_spectrum=False,
        save_stats=False, save_results=False,
    )
    result_a = runner.run_pipeline_a(
        scenario, steer_theta=30.0, steer_freq=2e6, config=config)
    result_b = runner.run_pipeline_b(
        scenario, steer_theta=30.0, config=config)
    return result_a, result_b


# ============================================================================
# График 1: Спектры Pipeline A vs B (луч 0)
# ============================================================================

def plot_spectra(result_a, result_b, fs):
    n_fft_a = result_a.nFFT
    n_fft_b = result_b.nFFT

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Strategies: Pipeline A vs B — Спектры (луч 0)", fontsize=14)

    for ax, result, label, n_fft in zip(
            axes,
            [result_a, result_b],
            ["Pipeline A (без Farrow)", "Pipeline B (с Farrow)"],
            [n_fft_a, n_fft_b]):

        if result.spectrum is None:
            ax.text(0.5, 0.5, f"{label}: нет данных спектра",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        freqs = np.fft.fftfreq(n_fft, d=1.0 / fs) / 1e6  # МГц
        spec = result.spectrum[0]  # луч 0
        mag_db = 20 * np.log10(np.abs(spec[:n_fft // 2]) + 1e-12)
        ax.plot(freqs[:n_fft // 2], mag_db, linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("Амплитуда, дБ")
        ax.set_ylim(bottom=mag_db.max() - 80)
        ax.grid(True, alpha=0.4)

        # Помечаем найденный пик
        if result.peaks:
            for beam_peaks in result.peaks:
                if beam_peaks:
                    freq_mhz = beam_peaks[0].freq_hz / 1e6
                    ax.axvline(freq_mhz, color='r', linestyle='--', alpha=0.7,
                               label=f"Пик: {freq_mhz:.3f} МГц")
                    ax.legend(fontsize=9)
                    break

    axes[-1].set_xlabel("Частота, МГц")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "spectra_pipeline_a_vs_b.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Сохранено: {out}")


# ============================================================================
# График 2: Checkpoints 2.1 / 2.2 / 2.3 (Pipeline A, антенна 0)
# ============================================================================

def plot_checkpoints(result_a):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Strategies: Debug checkpoints (Pipeline A, антенна 0)", fontsize=14)

    items = [
        ("2.1 Входной сигнал S_raw (Re)",  result_a.S_raw,   'Re'),
        ("2.2 После GEMM X (Re)",           result_a.X_gemm,  'Re'),
        ("2.3 Спектр |spectrum|",            result_a.spectrum, 'Mag'),
    ]

    for ax, (title, data, mode) in zip(axes, items):
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.4)

        if data is None:
            ax.text(0.5, 0.5, "нет данных (checkpoint не сохранялся)",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        row = data[0]  # первая антенна/луч
        if mode == 'Mag':
            ax.plot(np.abs(row), linewidth=0.7)
            ax.set_ylabel("|spectrum|")
        else:
            ax.plot(np.real(row), linewidth=0.7)
            ax.set_ylabel("Re")

    axes[-1].set_xlabel("Отсчёт")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "checkpoints_2_1_2_2_2_3.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Сохранено: {out}")


# ============================================================================
# График 3: Сравнение найденных пиков по антеннам
# ============================================================================

def plot_peak_comparison(result_a, result_b, n_ant=5):
    freqs_a = [bp[0].freq_hz / 1e6 if bp else 0.0 for bp in (result_a.peaks or [])]
    freqs_b = [bp[0].freq_hz / 1e6 if bp else 0.0 for bp in (result_b.peaks or [])]

    while len(freqs_a) < n_ant:
        freqs_a.append(0.0)
    while len(freqs_b) < n_ant:
        freqs_b.append(0.0)

    x = np.arange(n_ant)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, freqs_a[:n_ant], width, label='Pipeline A', alpha=0.8)
    ax.bar(x + width / 2, freqs_b[:n_ant], width, label='Pipeline B', alpha=0.8)
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.7, label='Целевая f₀=2 МГц')

    ax.set_xlabel("Луч (антенна)")
    ax.set_ylabel("Найденная частота, МГц")
    ax.set_title("Сравнение найденных пиков: Pipeline A vs B")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Beam {i}" for i in range(n_ant)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "peak_comparison_a_vs_b.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Сохранено: {out}")


# ============================================================================
# График 4: Farrow effect — S_raw vs S_aligned (антенна 1)
# ============================================================================

def plot_farrow_effect(scenario):
    S = scenario['S']          # [n_ant, n_samples]
    array = scenario['array']  # ULAGeometry
    fs = scenario['fs']

    delays_s = array.compute_delays(30.0)   # [n_ant] секунды
    delays_samples = delays_s * fs           # в отсчётах

    farrow = FarrowDelay()
    S_aligned = farrow.apply(S, delays_samples)

    n_show = 500
    t = np.arange(n_show)
    ant = 1  # антенна с ненулевой задержкой

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Farrow delay: S_raw vs S_aligned (Re, антенна {ant})", fontsize=14)

    axes[0].plot(t, np.real(S[ant, :n_show]), linewidth=0.8, label="S_raw")
    axes[0].set_title("До Farrow (S_raw, Re)")
    axes[0].set_ylabel("Амплитуда")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, np.real(S_aligned[ant, :n_show]), linewidth=0.8,
                 color='orange',
                 label=f"S_aligned (τ={delays_samples[ant]:.2f} отсч)")
    axes[1].set_title("После Farrow (S_aligned, Re)")
    axes[1].set_ylabel("Амплитуда")
    axes[1].set_xlabel("Отсчёт")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "farrow_raw_vs_aligned.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Сохранено: {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== plot_strategies_results.py ===")
    print(f"Выходная папка: {OUT_DIR}\n")

    print("Строим сценарий (5 антенн, CW f0=2 МГц, theta=30°)...")
    scenario = build_scenario()
    S = scenario['S']
    print(f"  S.shape = {S.shape}, fs = {scenario['fs']/1e6:.0f} МГц\n")

    print("Запускаем Pipeline A и B (NumPy)...")
    result_a, result_b = compute_pipelines(scenario)
    print(f"  Pipeline A: nFFT={result_a.nFFT}, peaks={len(result_a.peaks or [])}")
    print(f"  Pipeline B: nFFT={result_b.nFFT}, peaks={len(result_b.peaks or [])}\n")

    print("Строим графики:")

    print("  График 1: Спектры A vs B...")
    plot_spectra(result_a, result_b, scenario['fs'])

    print("  График 2: Debug checkpoints...")
    plot_checkpoints(result_a)

    print("  График 3: Peak comparison...")
    plot_peak_comparison(result_a, result_b, n_ant=5)

    print("  График 4: Farrow effect...")
    plot_farrow_effect(scenario)

    print(f"\nГотово! Все графики сохранены в: {OUT_DIR}")


if __name__ == '__main__':
    main()
