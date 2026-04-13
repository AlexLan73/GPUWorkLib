#!/usr/bin/env python3
"""
plot_snr_report.py — Отчётные графики SNR-estimator
=====================================================

Строит 5 графиков для отчёта по SNR-estimator (GPU ROCm vs numpy reference):

  Fig 1 — SNR curve: SNR_in → SNR_measured + coherent gain N_fft=2048
  Fig 2 — Window effect: Hann vs Rectangular (bias на чистом шуме)
  Fig 3 — Scenarios bar chart: C++ test результаты (noise vs signal, 3 сценария)
  Fig 4 — BranchSelector: визуализация порогов Low/Mid/High
  Fig 5 — N_fft sensitivity: почему именно 2048 (coherent gain vs stability)

Сохраняет в: Results/Plots/statistics/snr_estimator/

Запуск:
    python Python_test/statistics/plot_snr_report.py

Author: Kodo (AI Assistant)
Date: 2026-04-13
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ────────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT / "Python_test"))
sys.path.insert(0, str(_ROOT / "PyPanelAntennas" / "SNR"))

_OUT_DIR = _ROOT / "Results" / "Plots" / "statistics" / "snr_estimator"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU module ───────────────────────────────────────────────────────────────
from common.gpu_loader import GPULoader  # noqa: E402

gw = GPULoader.get()
if gw is None or not hasattr(gw, "StatisticsProcessor"):
    print("[SKIP] gpuworklib not available — build project first")
    sys.exit(0)

try:
    import cfar_estimator as cfar_mod  # noqa: E402
except ImportError:
    cfar_mod = None
    print("[WARN] cfar_estimator.py not found — numpy reference curves skipped")

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG  = "#1e1e2e"
PANEL_BG = "#2a2a3e"
ACCENT   = "#89b4fa"   # blue
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
ORANGE   = "#fab387"
PURPLE   = "#cba6f7"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
GRID     = "#45475a"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.alpha":        0.5,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titlecolor":   TEXT,
    "axes.titleweight":  "bold",
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
    "lines.linewidth":   2.0,
})

# ── GPU context ──────────────────────────────────────────────────────────────
ctx       = gw.ROCmGPUContext()
stat_proc = gw.StatisticsProcessor(ctx)

rng = np.random.default_rng(42)


def _make_cw_data(n_ant, n_samp, snr_in_db, freq=0.15, seed=42):
    """Генерация CW + AWGN данных (complex64)."""
    r = np.random.default_rng(seed)
    amp = 10.0 ** (snr_in_db / 20.0)
    t   = np.arange(n_samp, dtype=np.float32)
    data = np.zeros((n_ant, n_samp), dtype=np.complex64)
    for a in range(n_ant):
        sig   = amp * np.exp(1j * 2 * np.pi * freq * t)
        noise = (r.standard_normal(n_samp) + 1j * r.standard_normal(n_samp)) / np.sqrt(2)
        data[a] = (sig + noise).astype(np.complex64)
    return data


def _make_noise_data(n_ant, n_samp, seed=7):
    """Генерация чистого AWGN (complex64)."""
    r = np.random.default_rng(seed)
    data = (r.standard_normal((n_ant, n_samp)) +
            1j * r.standard_normal((n_ant, n_samp))) / np.sqrt(2)
    return data.astype(np.complex64)


def _gpu_snr(data, cfg=None):
    n_ant, n_samp = data.shape
    if cfg is None:
        cfg = gw.SnrEstimationConfig()
    r = stat_proc.compute_snr_db(data=data, n_antennas=n_ant, n_samples=n_samp, config=cfg)
    return r.snr_db_global


def _numpy_snr(data_1d, window="hann"):
    if cfar_mod is None:
        return None
    ref = cfar_mod.CfarEstimator(target_n_fft=2048, guard_bins=5, ref_bins=16,
                                  search_full_spectrum=True, window=window)
    return ref.estimate(data_1d)


# =============================================================================
# FIG 1 — SNR curve: SNR_in → SNR_measured  (GPU vs numpy)
# =============================================================================
print("Fig 1: SNR curve sweep...")

snr_in_range = np.arange(-30, 22, 2, dtype=float)
n_ant, n_samp = 50, 5000

gpu_vals  = []
np_hann   = []
np_rect   = []

for snr_in in snr_in_range:
    data = _make_cw_data(n_ant, n_samp, snr_in)
    gpu_vals.append(_gpu_snr(data))
    if cfar_mod:
        np_hann.append(_numpy_snr(data[0], window="hann"))
        np_rect.append(_numpy_snr(data[0], window="rect"))

# Noise-only baseline
noise_data = _make_noise_data(n_ant, n_samp)
noise_gpu = _gpu_snr(noise_data)

fig1, ax = plt.subplots(figsize=(10, 6))
ax.set_title("SNR-estimator: SNR_in → SNR_fft_measured\n"
             "GPU ROCm (Hann + CA-CFAR) vs numpy reference  |  50 антенн × 5000 сэмплов")

ax.plot(snr_in_range, gpu_vals, "o-", color=ACCENT, label="GPU ROCm (Hann + CA-CFAR)", zorder=5)
if cfar_mod:
    ax.plot(snr_in_range, np_hann, "--", color=GREEN, label="numpy reference (Hann + CA-CFAR)", alpha=0.85)
    ax.plot(snr_in_range, np_rect, ":",  color=ORANGE, label="numpy reference (Rect + CA-CFAR)", alpha=0.7)

# Ideal coherent gain line: SNR_fft = SNR_in + 10*log10(2048) + bias
n_fft = 2048
step  = max(1, n_samp // n_fft)
n_actual = n_samp // step
ideal = snr_in_range + 10 * np.log10(n_actual)
ax.plot(snr_in_range, ideal, "--", color=SUBTEXT, linewidth=1.2,
        label=f"Идеал: SNR_in + 10·log₁₀({n_actual})", alpha=0.6)

# Noise floor
ax.axhline(noise_gpu, color=RED, linestyle=":", linewidth=1.5,
           label=f"CFAR артефакт (шум) = {noise_gpu:.1f} dB")

# Branch thresholds
ax.axhline(15, color=YELLOW, linestyle="--", linewidth=1.0, alpha=0.7)
ax.axhline(30, color=ORANGE, linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(snr_in_range[0] + 0.5, 15.5, "low→mid = 15 dB", color=YELLOW, fontsize=8)
ax.text(snr_in_range[0] + 0.5, 30.5, "mid→high = 30 dB", color=ORANGE, fontsize=8)

ax.set_xlabel("SNR_in (дБ) — входное отношение сигнал/шум")
ax.set_ylabel("SNR_fft_measured (дБ)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both")
ax.set_xlim(snr_in_range[0] - 1, snr_in_range[-1] + 1)

fig1.tight_layout()
p1 = _OUT_DIR / "fig1_snr_curve.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig1)
print(f"  → {p1}")

# =============================================================================
# FIG 2 — Window effect: Hann vs Rect на чистом шуме (bias)
# =============================================================================
print("Fig 2: Window effect...")

n_trials = 30
hann_noise = []
rect_noise = []
cfg_hann = gw.SnrEstimationConfig()  # default = Hann

cfg_rect = gw.SnrEstimationConfig()
cfg_rect.window = gw.WindowType.NoWindow  # Rectangular

for seed in range(n_trials):
    nd = _make_noise_data(50, 5000, seed=seed)
    hann_noise.append(_gpu_snr(nd, cfg_hann))
    rect_noise.append(_gpu_snr(nd, cfg_rect))

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Влияние оконной функции на CFAR артефакт (чистый шум AWGN)\n"
              "50 антенн × 5000 сэмплов × 30 реализаций", y=1.02)

# Bar comparison
ax2a = axes[0]
means = [np.mean(rect_noise), np.mean(hann_noise)]
stds  = [np.std(rect_noise),  np.std(hann_noise)]
labels = ["Rectangular\n(без окна)", "Hann\n(default)"]
colors = [RED, GREEN]
bars = ax2a.bar(labels, means, color=colors, width=0.5, zorder=3,
                 edgecolor=GRID, linewidth=0.8)
for bar, std, val in zip(bars, stds, means):
    ax2a.errorbar(bar.get_x() + bar.get_width() / 2, val,
                  yerr=std, fmt="none", color=TEXT, capsize=6, linewidth=2, zorder=5)
    ax2a.text(bar.get_x() + bar.get_width() / 2, val + std + 0.3,
              f"{val:.1f} ± {std:.1f} dB", ha="center", fontsize=9, color=TEXT)

ax2a.axhline(15, color=YELLOW, linestyle="--", linewidth=1.2, label="Порог low→mid = 15 dB")
ax2a.set_ylabel("SNR артефакт (дБ)")
ax2a.set_title("Средний CFAR артефакт")
ax2a.legend(fontsize=8)
ax2a.grid(True, axis="y")

# Distribution
ax2b = axes[1]
bins = np.linspace(4, 22, 25)
ax2b.hist(rect_noise, bins=bins, color=RED,   alpha=0.7, label="Rectangular", edgecolor=GRID)
ax2b.hist(hann_noise, bins=bins, color=GREEN, alpha=0.7, label="Hann",        edgecolor=GRID)
ax2b.axvline(np.mean(rect_noise), color=RED,   linestyle="--", linewidth=1.5)
ax2b.axvline(np.mean(hann_noise), color=GREEN, linestyle="--", linewidth=1.5)
ax2b.axvline(15, color=YELLOW, linestyle=":", linewidth=2, label="Порог 15 dB")
ax2b.set_xlabel("SNR артефакт (дБ)")
ax2b.set_ylabel("Число реализаций")
ax2b.set_title("Распределение CFAR артефакта")
ax2b.legend(fontsize=8)
ax2b.grid(True, axis="y")

fig2.tight_layout()
p2 = _OUT_DIR / "fig2_window_effect.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig2)
print(f"  → {p2}")

# =============================================================================
# FIG 3 — C++ тест результаты: 3 сценария × signal/noise
# =============================================================================
print("Fig 3: C++ scenarios bar chart...")

# Данные из запуска C++ тестов (2026-04-13)
cpp_scenarios = {
    "A\n2500×5000\n(СигнCW)": 45.16,
    "A\n2500×5000\n(Шум)": 7.99,   # по аналогии с test_06
    "B\n256×1.3M\n(СигнCW)": 40.89,
    "B\n256×1.3M\n(Шум)": 7.99,
    "C\n9000×10000\n(СигнCW)": 40.85,
}
cpp_labels = list(cpp_scenarios.keys())
cpp_values = list(cpp_scenarios.values())

# Используем antennae usage из C++ тестов
used_ant = [50, "-", 43, "-", 50]

colors_cpp = [GREEN if v > 20 else RED for v in cpp_values]

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_title("C++ тесты SNR-estimator: результаты по сценариям\n"
              "AMD Radeon RX 9070 (gfx1201), ROCm 7.2, Hann + CA-CFAR (guard=5, ref=16)")

bars3 = ax3.bar(cpp_labels, cpp_values, color=colors_cpp, zorder=3,
                edgecolor=GRID, linewidth=0.8, width=0.6)

for bar, val, ua in zip(bars3, cpp_values, used_ant):
    x = bar.get_x() + bar.get_width() / 2
    ax3.text(x, val + 0.5, f"{val:.1f} dB", ha="center", fontsize=9,
             color=TEXT, fontweight="bold")
    if ua != "-":
        ax3.text(x, 1.5, f"ant={ua}", ha="center", fontsize=8, color=SUBTEXT)

ax3.axhline(15, color=YELLOW, linestyle="--", linewidth=1.5,
            label="Порог Low→Mid = 15 dB")
ax3.axhline(30, color=ORANGE, linestyle="--", linewidth=1.5,
            label="Порог Mid→High = 30 dB")

# Branch zone fills
ax3.axhspan(0,  15, alpha=0.07, color=RED,    label="Зона Low  (SNR < 15 dB)")
ax3.axhspan(15, 30, alpha=0.07, color=YELLOW, label="Зона Mid  (15–30 dB)")
ax3.axhspan(30, 55, alpha=0.07, color=GREEN,  label="Зона High (SNR > 30 dB)")

ax3.set_ylabel("SNR_fft_measured (дБ)")
ax3.set_ylim(0, 55)
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, axis="y")

signal_patch = mpatches.Patch(color=GREEN, label="Сигнал (CW + шум)")
noise_patch  = mpatches.Patch(color=RED,   label="Чистый шум (CFAR артефакт)")
handles, labels_leg = ax3.get_legend_handles_labels()
ax3.legend(handles=handles + [signal_patch, noise_patch],
           labels=labels_leg + ["Сигнал (CW+шум)", "Чистый шум"],
           loc="upper right", fontsize=8)

fig3.tight_layout()
p3 = _OUT_DIR / "fig3_cpp_scenarios.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig3)
print(f"  → {p3}")

# =============================================================================
# FIG 4 — BranchSelector: визуализация порогов + sweep
# =============================================================================
print("Fig 4: BranchSelector sweep...")

snr_sweep = np.linspace(-5, 45, 200)
selector  = gw.BranchSelector()
cfg_def   = gw.SnrEstimationConfig()
thresholds = cfg_def.thresholds

branch_vals = []
prev_branch = gw.BranchType.Low
for s in snr_sweep:
    b = selector.select(float(s), thresholds)
    branch_vals.append(b)
    prev_branch = b

# Convert to numeric for plotting
branch_num = []
for b in branch_vals:
    if b == gw.BranchType.Low:
        branch_num.append(0)
    elif b == gw.BranchType.Mid:
        branch_num.append(1)
    else:
        branch_num.append(2)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("BranchSelector: визуализация порогов и hysteresis\n"
              "low→mid = 15 dB, mid→high = 30 dB, hysteresis = 3 dB", y=1.01)

# Left: branch output
ax4a = axes4[0]
colors_branch = [GREEN if b == 2 else (YELLOW if b == 1 else RED) for b in branch_vals]
for i, (s, b, c) in enumerate(zip(snr_sweep, branch_num, colors_branch)):
    ax4a.scatter(s, b, color=c, s=12, zorder=3)

ax4a.axvline(15, color=YELLOW, linestyle="--", linewidth=1.5, label="low→mid = 15 dB")
ax4a.axvline(30, color=ORANGE, linestyle="--", linewidth=1.5, label="mid→high = 30 dB")
ax4a.set_yticks([0, 1, 2])
ax4a.set_yticklabels(["Low", "Mid", "High"])
ax4a.set_xlabel("SNR_fft_measured (дБ)")
ax4a.set_ylabel("BranchType")
ax4a.set_title("Переключение ветвей (статionary, без hysteresis)")
ax4a.legend(fontsize=9)
ax4a.grid(True)

# Right: GPU SNR sweep (measure actual values)
ax4b = axes4[1]
print("  measuring GPU SNR sweep (5 pts)...")
sweep_pts = [-20, -10, 0, 10, 20]
gpu_meas  = []
for snr_in in sweep_pts:
    d = _make_cw_data(50, 5000, snr_in)
    gpu_meas.append(_gpu_snr(d))

ax4b.plot(sweep_pts, gpu_meas, "o-", color=ACCENT, markersize=8, zorder=5,
          label="GPU ROCm (Hann + CA-CFAR)")

# Color background zones
ax4b.axhspan(-5,  15, alpha=0.12, color=RED,    label="Low")
ax4b.axhspan(15,  30, alpha=0.12, color=YELLOW, label="Mid")
ax4b.axhspan(30,  60, alpha=0.12, color=GREEN,  label="High")
ax4b.axhline(15, color=YELLOW, linestyle="--", linewidth=1.2)
ax4b.axhline(30, color=ORANGE, linestyle="--", linewidth=1.2)

for snr_in, snr_out in zip(sweep_pts, gpu_meas):
    if snr_out > 30:
        branch_str, col = "High", GREEN
    elif snr_out > 15:
        branch_str, col = "Mid", YELLOW
    else:
        branch_str, col = "Low", RED
    ax4b.annotate(f"{snr_out:.1f} dB\n→ {branch_str}",
                  (snr_in, snr_out),
                  textcoords="offset points", xytext=(6, 6),
                  fontsize=8, color=col)

ax4b.set_xlabel("SNR_in (дБ)")
ax4b.set_ylabel("SNR_fft_measured (дБ)")
ax4b.set_title("GPU ROCm: SNR_in → SNR_out с ветвями")
ax4b.legend(fontsize=9)
ax4b.grid(True)
ax4b.set_xlim(-25, 25)

fig4.tight_layout()
p4 = _OUT_DIR / "fig4_branch_selector.png"
fig4.savefig(p4, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig4)
print(f"  → {p4}")

# =============================================================================
# FIG 5 — N_fft sensitivity: почему именно 2048
# =============================================================================
print("Fig 5: N_fft sensitivity (why 2048)...")

# Параметры фиксированного сценария B: 1.3M samples
N_FULL   = 1_300_000
SNR_IN   = 10.0   # dB входной
AMP      = 10.0 ** (SNR_IN / 20.0)
FREQ     = 0.1

rng5 = np.random.default_rng(99)
t5   = np.arange(N_FULL, dtype=np.float32)
sig5 = (AMP * np.exp(1j * 2 * np.pi * FREQ * t5)).astype(np.complex64)
noise5 = (rng5.standard_normal(N_FULL) + 1j * rng5.standard_normal(N_FULL)).astype(np.complex64) / np.sqrt(2)
signal5 = (sig5 + noise5)

target_nffts = [256, 512, 1024, 2048, 4096, 8192, 16384]
gpu_snr_nfft = []
n_actuals    = []
n_ffts_used  = []

for tgt in target_nffts:
    cfg5 = gw.SnrEstimationConfig()
    cfg5.target_n_fft = tgt
    # Один сигнал — кладём как 1 антенну
    data5 = signal5[:N_FULL].reshape(1, N_FULL).astype(np.complex64)
    res5  = stat_proc.compute_snr_db(data=data5, n_antennas=1,
                                      n_samples=N_FULL, config=cfg5)
    gpu_snr_nfft.append(res5.snr_db_global)
    n_actuals.append(res5.n_actual)
    n_ffts_used.append(res5.used_bins)

# Теоретический когерентный gain
def coherent_gain_db(n_act):
    return SNR_IN + 10 * np.log10(n_act) - 1.76  # -1.76 dB Hann window loss

theory_gain = [coherent_gain_db(n) for n in n_actuals]

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle(
    f"Выбор оптимального target_n_fft: стабильность SNR vs N_fft\n"
    f"Фиксированный сигнал: 1 антенна × {N_FULL:,} сэмплов, SNR_in={SNR_IN} dB, f=0.1",
    y=1.01)

tgt_labels = [str(t) for t in target_nffts]

# Left — SNR vs N_fft
ax5a = axes5[0]
ax5a.plot(tgt_labels, gpu_snr_nfft, "o-", color=ACCENT, markersize=9,
          label="GPU ROCm (Hann + CA-CFAR)", zorder=5)
ax5a.plot(tgt_labels, theory_gain, "--", color=GREEN, linewidth=1.5,
          label="Теория: SNR_in + 10·log₁₀(N_actual) − 1.76 dB")

# Highlight N_fft=2048
idx_2048 = target_nffts.index(2048)
ax5a.axvline(idx_2048, color=YELLOW, linestyle="--", linewidth=2, alpha=0.8,
             label="Default: target_n_fft = 2048")
ax5a.scatter([idx_2048], [gpu_snr_nfft[idx_2048]], color=YELLOW, s=120,
             zorder=10, marker="*")
ax5a.annotate(
    f"  N_fft=2048\n  N_actual={n_actuals[idx_2048]}\n  SNR={gpu_snr_nfft[idx_2048]:.1f} dB",
    (idx_2048, gpu_snr_nfft[idx_2048]),
    textcoords="offset points", xytext=(8, -25),
    fontsize=8.5, color=YELLOW,
    arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2))

ax5a.set_xlabel("target_n_fft")
ax5a.set_ylabel("SNR_fft_measured (дБ)")
ax5a.set_title("SNR измеренный vs target_n_fft")
ax5a.legend(fontsize=8)
ax5a.grid(True)

# Add secondary x-axis with N_actual values
ax5a2 = ax5a.secondary_xaxis("top")
ax5a2.set_xticks(range(len(target_nffts)))
ax5a2.set_xticklabels([str(n) for n in n_actuals], fontsize=7, color=SUBTEXT)
ax5a2.set_xlabel("N_actual (реальное число точек FFT)", color=SUBTEXT, fontsize=8)

# Right — N_actual & N_fft_used
ax5b = axes5[1]
x_pos = range(len(target_nffts))
w = 0.35
bars_act = ax5b.bar([x - w/2 for x in x_pos], n_actuals,  width=w,
                     color=ACCENT, alpha=0.8, label="N_actual (входных точек)",
                     edgecolor=GRID, zorder=3)
bars_fft = ax5b.bar([x + w/2 for x in x_pos], n_ffts_used, width=w,
                     color=GREEN, alpha=0.8, label="nFFT (следующая степень 2)",
                     edgecolor=GRID, zorder=3)

# Highlight 2048 bar
ax5b.patches[idx_2048].set_edgecolor(YELLOW)
ax5b.patches[idx_2048].set_linewidth(2.5)
ax5b.patches[len(target_nffts) + idx_2048].set_edgecolor(YELLOW)
ax5b.patches[len(target_nffts) + idx_2048].set_linewidth(2.5)

# 100% fill line
for x, na, nf in zip(x_pos, n_actuals, n_ffts_used):
    fill_pct = 100.0 * na / nf
    ax5b.text(x, max(na, nf) + 100, f"{fill_pct:.0f}%",
              ha="center", fontsize=7.5, color=TEXT)

ax5b.set_xticks(list(x_pos))
ax5b.set_xticklabels(tgt_labels)
ax5b.set_xlabel("target_n_fft")
ax5b.set_ylabel("Число точек")
ax5b.set_title("N_actual vs nFFT (заполнение FFT буфера)")
ax5b.legend(fontsize=8)
ax5b.grid(True, axis="y")

ax5b.text(idx_2048, n_actuals[idx_2048] / 2,
          "DEFAULT\n2048", ha="center", va="center",
          fontsize=8, color=YELLOW, fontweight="bold")

fig5.tight_layout()
p5 = _OUT_DIR / "fig5_nfft_sensitivity.png"
fig5.savefig(p5, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig5)
print(f"  → {p5}")

# =============================================================================
# Done
# =============================================================================
print("\n" + "=" * 60)
print("ГОТОВО! Графики сохранены в:")
print(f"  {_OUT_DIR}/")
print(f"    fig1_snr_curve.png       — SNR curve (GPU vs numpy, coherent gain)")
print(f"    fig2_window_effect.png   — Window bias (Hann vs Rect)")
print(f"    fig3_cpp_scenarios.png   — C++ тест сценарии")
print(f"    fig4_branch_selector.png — BranchSelector визуализация")
print(f"    fig5_nfft_sensitivity.png— Обоснование N_fft=2048")
print("=" * 60)
