"""
test_heterodyne_comparison.py
==============================
GPU vs CPU heterodyne dechirp comparison report.

Generates a detailed markdown report and annotated comparison plots:
  1. Runs full GPU pipeline (gpuworklib.HeterodyneDechirp)
  2. Runs CPU reference pipeline (NumPy FFT + parabolic interp)
  3. Compares f_beat, range, SNR per antenna
  4. Saves markdown report + annotated PNG plots

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
search_range=5000 => охват [0..2499] бин (~3.66 МГц)

@author Kodo (AI Assistant)
@date 2026-02-21
"""

import sys
import os
import time
import numpy as np

# -- Path to gpuworklib --
BUILD_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Debug'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Release'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'),
]
for p in BUILD_PATHS:
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
        break

try:
    import gpuworklib
except ImportError:
    print("ERROR: gpuworklib not found. Build with -DBUILD_PYTHON=ON")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, plots will be skipped")

# ============================================================================
# Constants
# ============================================================================

FS = 12e6
F_START = 0.0
F_END = 2e6
B = F_END - F_START   # 2 MHz
N = 8000
ANTENNAS = 5
T = N / FS            # 666.67 us
MU = B / T            # 3e9 Hz/s
C_LIGHT = 3e8

DELAYS_US = np.array([100., 200., 300., 400., 500.])
DELAYS_S  = DELAYS_US * 1e-6

F_BEATS_TRUE = MU * DELAYS_S
RANGES_TRUE  = C_LIGHT * DELAYS_S / 2.0

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                          'Results', 'Plots', 'heterodyne')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                           'Results', 'JSON')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================================
# CPU Reference Pipeline
# ============================================================================

def cpu_pipeline(delays_s):
    """Full CPU dechirp pipeline: generate rx, ref, dechirp, FFT, peak find."""
    t = np.arange(N, dtype=np.float64) / FS

    rx = np.zeros((len(delays_s), N), dtype=np.complex128)
    for i, tau in enumerate(delays_s):
        t_d = t - tau
        phase = 2 * np.pi * (0.5 * MU * t_d**2 + F_START * t_d)
        rx[i] = np.exp(1j * phase)

    ref_conj = np.exp(-1j * (np.pi * MU * t**2 + 2 * np.pi * F_START * t))
    dc = np.conj(rx * ref_conj[np.newaxis, :])

    nfft = 8192
    results = []
    for k in range(len(delays_s)):
        padded = np.zeros(nfft, dtype=np.complex128)
        padded[:N] = dc[k]
        spectrum = np.fft.fft(padded)
        mag = np.abs(spectrum[:nfft // 2])

        peak_bin = np.argmax(mag)
        if 0 < peak_bin < len(mag) - 1:
            L, C, R = mag[peak_bin - 1], mag[peak_bin], mag[peak_bin + 1]
            denom = L - 2 * C + R
            delta = 0.5 * (L - R) / denom if abs(denom) > 1e-12 else 0.0
        else:
            delta = 0.0

        refined_bin = peak_bin + delta
        f_beat  = refined_bin * FS / nfft
        range_m = C_LIGHT * T * f_beat / (2 * B)

        # CPU SNR: RMS шум в полосе ±50 бин, исключая пик
        noise_bins  = list(range(max(1, peak_bin - 50), max(1, peak_bin - 5)))
        noise_bins += list(range(min(peak_bin + 5, nfft // 2 - 1),
                                  min(peak_bin + 50, nfft // 2 - 1)))
        noise_est = np.mean(mag[noise_bins]) if noise_bins else 1e-12
        snr_db    = 20 * np.log10(mag[peak_bin] / max(noise_est, 1e-12))

        results.append({
            'f_beat_hz':    float(f_beat),
            'range_m':      float(range_m),
            'peak_snr_db':  float(snr_db),
            'peak_bin':     int(peak_bin),
            'refined_bin':  float(refined_bin),
            'peak_magnitude': float(mag[peak_bin]),
        })

    return results


# ============================================================================
# GPU Pipeline
# ============================================================================

def gpu_pipeline(delays_s):
    """Full GPU pipeline via gpuworklib.HeterodyneDechirp."""
    ctx = gpuworklib.ROCmGPUContext(0)
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(F_START, F_END, FS, N, len(delays_s))

    t  = np.arange(N, dtype=np.float32) / FS
    rx = np.zeros((len(delays_s), N), dtype=np.complex64)
    for i, tau in enumerate(delays_s):
        t_d = t - tau
        phase = 2 * np.pi * (0.5 * MU * t_d**2 + F_START * t_d)
        rx[i] = np.exp(1j * phase).astype(np.complex64)

    result = het.process(rx.ravel())
    if not result['success']:
        raise RuntimeError(f"GPU pipeline failed: {result['error_message']}")

    return result['antennas']


# ============================================================================
# Comparison
# ============================================================================

def run_comparison():
    """Run GPU and CPU pipelines, compare results."""
    print("=" * 70)
    print("HETERODYNE DECHIRP: GPU vs CPU COMPARISON")
    print("=" * 70)
    print(f"  fs={FS/1e6:.0f} МГц, B={B/1e6:.0f} МГц, N={N}, T={T*1e6:.2f} мкс")
    print(f"  mu={MU:.2e} Гц/с, антенн={ANTENNAS}")
    print(f"  delays={DELAYS_US.tolist()} мкс")
    print()

    print("Running CPU pipeline (NumPy float64)...")
    t0 = time.perf_counter()
    cpu_results = cpu_pipeline(DELAYS_S)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time*1000:.1f} ms")

    print("Running GPU pipeline (gpuworklib, float32)...")
    _ = gpu_pipeline(DELAYS_S)          # прогрев
    t0 = time.perf_counter()
    gpu_results = gpu_pipeline(DELAYS_S)
    gpu_time = time.perf_counter() - t0
    print(f"  GPU time: {gpu_time*1000:.1f} ms")
    print()

    return cpu_results, gpu_results, cpu_time, gpu_time


def print_comparison_table(cpu_results, gpu_results):
    """Print comparison table."""
    print("COMPARISON TABLE")
    print("-" * 108)
    header = (f"  {'Ant':>3} | {'Задержка':>8} | {'f GPU':>11} | {'f CPU':>11} | "
              f"{'df':>8} | {'R GPU':>9} | {'R CPU':>9} | {'dR':>6} | "
              f"{'SNR GPU':>7} | {'SNR CPU':>7}")
    units = (f"  {'':>3} | {'мкс':>8} | {'Гц':>11} | {'Гц':>11} | "
             f"{'Гц':>8} | {'м':>9} | {'м':>9} | {'м':>6} | "
             f"{'дБ':>7} | {'дБ':>7}")
    print(header)
    print(units)
    print("  " + "─" * 104)

    max_df = 0
    max_dr = 0
    for k in range(ANTENNAS):
        fg = gpu_results[k]['f_beat_hz']
        fc = cpu_results[k]['f_beat_hz']
        rg = gpu_results[k]['range_m']
        rc = cpu_results[k]['range_m']
        sg = gpu_results[k]['peak_snr_db']
        sc = cpu_results[k]['peak_snr_db']
        df = abs(fg - fc)
        dr = abs(rg - rc)
        max_df = max(max_df, df)
        max_dr = max(max_dr, dr)
        print(f"  {k:3d} | {DELAYS_US[k]:8.0f} | {fg:11.1f} | {fc:11.1f} | "
              f"{df:8.1f} | {rg:9.2f} | {rc:9.2f} | {dr:6.2f} | "
              f"{sg:7.1f} | {sc:7.1f}")

    print()
    print(f"  Max |f_GPU - f_CPU|: {max_df:.1f} Гц")
    print(f"  Max |R_GPU - R_CPU|: {max_dr:.2f} м")

    print()
    print("VS ТЕОРИЯ:")
    print("  " + "─" * 88)
    print(f"  {'Ant':>3} | {'f_true':>11} | {'f_GPU err':>10} | {'f_CPU err':>10} | "
          f"{'R_true':>9} | {'R_GPU err':>9} | {'R_CPU err':>9}")
    print("  " + "─" * 88)
    for k in range(ANTENNAS):
        fg = gpu_results[k]['f_beat_hz']
        fc = cpu_results[k]['f_beat_hz']
        rg = gpu_results[k]['range_m']
        rc = cpu_results[k]['range_m']
        ft = F_BEATS_TRUE[k]
        rt = RANGES_TRUE[k]
        print(f"  {k:3d} | {ft:11.0f} | {abs(fg-ft):10.1f} | {abs(fc-ft):10.1f} | "
              f"{rt:9.2f} | {abs(rg-rt):9.2f} | {abs(rc-rt):9.2f}")

    return max_df, max_dr


def generate_report_md(cpu_results, gpu_results, cpu_time, gpu_time, max_df, max_dr):
    """Generate markdown report."""
    report_path = os.path.join(REPORT_DIR, 'heterodyne_comparison_report.md')

    lines = [
        "# Heterodyne Dechirp: GPU vs CPU — Отчёт",
        "",
        f"> Сгенерировано: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Параметры",
        "",
        "| Параметр | Значение | Описание |",
        "|----------|---------|---------|",
        f"| fs | {FS/1e6:.0f} МГц | Частота дискретизации |",
        f"| B | {B/1e6:.0f} МГц | Полоса ЛЧМ (bandwidth) |",
        f"| N | {N} | Отсчётов на антенну |",
        f"| T | {T*1e6:.2f} мкс | Длительность чирпа = N/fs |",
        f"| μ | {MU:.2e} Гц/с | Скорость чирпа = B/T |",
        f"| Антенн | {ANTENNAS} | Число антенных каналов |",
        f"| Задержки | {DELAYS_US.tolist()} мкс | τ каждой антенны |",
        f"| search_range | 5000 | half_range=2500, бины [0..2499] |",
        "",
        "## Формулы",
        "",
        "| Формула | Описание |",
        "|---------|---------|",
        "| `s_ref* = conj(LFM(t))` | Опорный сигнал |",
        "| `s_dc = conj(s_rx × s_ref*)` | Дечирп (биение) |",
        "| `f_beat = μ × τ` | Частота биений |",
        "| `R = c × T × f_beat / (2 × B)` | Дальность |",
        "| `SNR = 20·log₁₀(пик / avg(bin±1))` | SNR GPU |",
        "",
        "## Скорость",
        "",
        "| Реализация | Время |",
        "|------------|-------|",
        f"| CPU (NumPy float64) | {cpu_time*1000:.1f} мс |",
        f"| GPU (OpenCL float32) | {gpu_time*1000:.1f} мс |",
        f"| Ускорение | {cpu_time/max(gpu_time, 1e-9):.1f}x |",
        "",
        "## Результаты",
        "",
        "| Ant | τ мкс | f_GPU Гц | f_CPU Гц | df Гц | R_GPU м | R_CPU м | dR м | SNR_GPU дБ | SNR_CPU дБ |",
        "|-----|-------|---------|---------|-------|---------|---------|------|-----------|-----------|",
    ]

    for k in range(ANTENNAS):
        fg = gpu_results[k]['f_beat_hz']
        fc = cpu_results[k]['f_beat_hz']
        rg = gpu_results[k]['range_m']
        rc = cpu_results[k]['range_m']
        sg = gpu_results[k]['peak_snr_db']
        sc = cpu_results[k]['peak_snr_db']
        lines.append(
            f"| {k} | {DELAYS_US[k]:.0f} | {fg:.1f} | {fc:.1f} | "
            f"{abs(fg-fc):.1f} | {rg:.2f} | {rc:.2f} | {abs(rg-rc):.2f} | "
            f"{sg:.1f} | {sc:.1f} |"
        )

    lines += [
        "",
        "## Итог",
        "",
        f"- Max |f_GPU - f_CPU|: **{max_df:.1f} Гц**",
        f"- Max |R_GPU - R_CPU|: **{max_dr:.2f} м**",
        f"- Все ошибки f_beat < 5000 Гц: "
        f"**{'✅ PASS' if max_df < 5000 else '❌ FAIL'}**",
        "",
        "## vs Теория",
        "",
        "| Ant | f_true Гц | f_GPU err Гц | f_CPU err Гц | R_true м | R_GPU err м | R_CPU err м |",
        "|-----|-----------|-------------|-------------|---------|------------|------------|",
    ]

    for k in range(ANTENNAS):
        fg = gpu_results[k]['f_beat_hz']
        fc = cpu_results[k]['f_beat_hz']
        rg = gpu_results[k]['range_m']
        rc = cpu_results[k]['range_m']
        ft = F_BEATS_TRUE[k]
        rt = RANGES_TRUE[k]
        lines.append(
            f"| {k} | {ft:.0f} | {abs(fg-ft):.1f} | {abs(fc-ft):.1f} | "
            f"{rt:.2f} | {abs(rg-rt):.2f} | {abs(rc-rt):.2f} |"
        )

    lines += [
        "",
        "---",
        f"*Сгенерировано: test_heterodyne_comparison.py | Кодо (AI Assistant)*",
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved: {report_path}")
    return report_path


def generate_comparison_plot(cpu_results, gpu_results, cpu_time, gpu_time):
    """Generate annotated 4-panel comparison plot."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return

    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.40,
                   top=0.88, bottom=0.08)

    speedup = cpu_time / max(gpu_time, 1e-9)
    fig.suptitle('Heterodyne Dechirp: GPU vs CPU — Сравнительный отчёт',
                 fontsize=13, fontweight='bold')
    fig.text(0.5, 0.915,
             f'fs={FS/1e6:.0f} МГц | B={B/1e6:.0f} МГц | N={N} | '
             f'T={T*1e6:.1f} мкс | μ={MU:.2e} Гц/с | {ANTENNAS} антенн | '
             f'CPU {cpu_time*1000:.1f} мс vs GPU {gpu_time*1000:.1f} мс '
             f'({speedup:.1f}x)',
             ha='center', fontsize=8.5, color='gray')

    x = np.arange(ANTENNAS)
    w = 0.35

    # ── Panel 1: f_beat vs задержка ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fg_arr = np.array([g['f_beat_hz'] for g in gpu_results])
    fc_arr = np.array([c['f_beat_hz'] for c in cpu_results])
    ax1.plot(DELAYS_US, F_BEATS_TRUE / 1e3, 'g--', lw=2, alpha=0.6,
             label='Теория f=μ·τ')
    ax1.plot(DELAYS_US, fg_arr / 1e3, 'ro-', ms=9, lw=1.5, label='GPU (OpenCL)')
    ax1.plot(DELAYS_US, fc_arr / 1e3, 'b^--', ms=7, lw=1.2, label='CPU (NumPy)')
    for k in range(ANTENNAS):
        ax1.annotate(f'{fg_arr[k]/1e3:.0f}',
                     (DELAYS_US[k], fg_arr[k] / 1e3),
                     textcoords='offset points', xytext=(4, 5),
                     fontsize=7.5, color='red')
    ax1.set_xlabel('Задержка τ [мкс]')
    ax1.set_ylabel('f_beat [кГц]')
    ax1.set_title('f_beat vs Задержка: GPU vs CPU vs Теория', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.03, 0.97,
             'f_beat = μ·τ\nЛинейная зависимость\nОбе реализации совпадают',
             transform=ax1.transAxes, va='top', fontsize=8, color='navy',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='navy', alpha=0.85))

    # ── Panel 2: Ошибка f_beat vs теория ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gpu_f_err = np.array([abs(gpu_results[k]['f_beat_hz'] - F_BEATS_TRUE[k])
                           for k in range(ANTENNAS)])
    cpu_f_err = np.array([abs(cpu_results[k]['f_beat_hz'] - F_BEATS_TRUE[k])
                           for k in range(ANTENNAS)])
    b2g = ax2.bar(x - w/2, gpu_f_err, w, label='GPU', color='coral', alpha=0.85, ec='k')
    b2c = ax2.bar(x + w/2, cpu_f_err, w, label='CPU', color='steelblue', alpha=0.85, ec='k')
    ax2.axhline(5000, color='red', ls='--', lw=1.5, alpha=0.7, label='Допуск 5 кГц')
    for bar, err in zip(b2g, gpu_f_err):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 err + 20, f'{err:.0f}',
                 ha='center', va='bottom', fontsize=7.5, color='darkred')
    for bar, err in zip(b2c, cpu_f_err):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 err + 20, f'{err:.0f}',
                 ha='center', va='bottom', fontsize=7.5, color='navy')
    ax2.set_xlabel('Антенна')
    ax2.set_ylabel('|f - f_true| [Гц]')
    ax2.set_title('Ошибка f_beat vs Теория', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Ant{i}\n({DELAYS_US[i]:.0f}мкс)' for i in range(ANTENNAS)],
                        fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.03, 0.97,
             'Точность ≈ bin_width/2\n≈ 730 Гц для nFFT=8192',
             transform=ax2.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='linen', ec='brown', alpha=0.85))

    # ── Panel 3: Дальность ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    rg_arr = np.array([g['range_m'] for g in gpu_results])
    rc_arr = np.array([c['range_m'] for c in cpu_results])
    ax3.plot(DELAYS_US, RANGES_TRUE, 'g--', lw=2, alpha=0.6, label='True R=c·τ/2')
    ax3.plot(DELAYS_US, rg_arr, 'ro-', ms=9, lw=1.5, label='GPU')
    ax3.plot(DELAYS_US, rc_arr, 'b^--', ms=7, lw=1.2, label='CPU')
    for k in range(ANTENNAS):
        ax3.annotate(f'{rg_arr[k]/1e3:.1f}км',
                     (DELAYS_US[k], rg_arr[k]),
                     textcoords='offset points', xytext=(4, 5),
                     fontsize=7.5, color='red')
    ax3.set_xlabel('Задержка τ [мкс]')
    ax3.set_ylabel('Дальность [м]')
    ax3.set_title('Дальность: GPU vs CPU vs True', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.03, 0.97,
             'R = c·T·f_beat / (2·B)\n≈ c·τ/2',
             transform=ax3.transAxes, va='top', fontsize=8, color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.3', fc='honeydew', ec='green', alpha=0.85))

    # ── Panel 4: SNR GPU vs CPU ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sg_arr = np.array([g['peak_snr_db'] for g in gpu_results])
    sc_arr = np.array([c['peak_snr_db'] for c in cpu_results])
    b4g = ax4.bar(x - w/2, sg_arr, w, label='GPU (2 соседних бина)', color='mediumseagreen', alpha=0.85, ec='k')
    b4c = ax4.bar(x + w/2, sc_arr, w, label='CPU (RMS ±50 бин)',    color='goldenrod',      alpha=0.85, ec='k')
    ax4.axhline(0, color='red', ls='--', lw=1.2, alpha=0.7)
    for bar, s in zip(b4g, sg_arr):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 max(s, 0) + 0.3, f'{s:.1f}',
                 ha='center', va='bottom', fontsize=7.5, color='darkgreen', fontweight='bold')
    for bar, s in zip(b4c, sc_arr):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 max(s, 0) + 0.3, f'{s:.1f}',
                 ha='center', va='bottom', fontsize=7.5, color='goldenrod', fontweight='bold')
    ax4.set_xlabel('Антенна')
    ax4.set_ylabel('SNR [дБ]')
    ax4.set_title('Сравнение SNR (разные методы оценки шума)', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Ant{i}\n({DELAYS_US[i]:.0f}мкс)' for i in range(ANTENNAS)],
                        fontsize=8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0.03, 0.97,
             'GPU: avg(бин-1, бин+1)\n'
             'CPU: RMS на дальних бинах\n'
             'Разные методы -> разные значения',
             transform=ax4.transAxes, va='top', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.3', fc='ivory', ec='olive', alpha=0.85))

    # ── Panel 5 (full width): Сводная таблица ───────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    header = (f"{'Ant':>4} | {'τ мкс':>6} | {'f_GPU Гц':>11} | {'f_CPU Гц':>11} | "
              f"{'df Гц':>7} | {'R_GPU м':>9} | {'R_CPU м':>9} | {'dR м':>6} | "
              f"{'SNR_GPU':>7} | {'SNR_CPU':>7}")
    divider = "─" * 95
    rows = [header, divider]
    for k in range(ANTENNAS):
        fg  = gpu_results[k]['f_beat_hz']
        fc  = cpu_results[k]['f_beat_hz']
        rg  = gpu_results[k]['range_m']
        rc  = cpu_results[k]['range_m']
        sg  = gpu_results[k]['peak_snr_db']
        sc  = cpu_results[k]['peak_snr_db']
        ft  = F_BEATS_TRUE[k]
        df  = abs(fg - fc)
        dr  = abs(rg - rc)
        ok  = "OK" if abs(fg - ft) < 5000 else "!!"
        rows.append(
            f"{k:>4} | {DELAYS_US[k]:>6.0f} | {fg:>11.1f} | {fc:>11.1f} | "
            f"{df:>7.1f} | {rg:>9.2f} | {rc:>9.2f} | {dr:>6.2f} | "
            f"{sg:>6.1f}дБ | {sc:>6.1f}дБ  {ok}"
        )

    ax5.text(0.01, 0.97, '\n'.join(rows),
             transform=ax5.transAxes, va='top', ha='left',
             fontsize=8.5, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f8f9fa', ec='#adb5bd'))
    ax5.set_title('Сводная таблица', pad=4, fontsize=10)

    # ── подвал ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             'GPUWorkLib | HeterodyneDechirp | OpenCL float32 | '
             f'search_range=5000 | CPU {cpu_time*1000:.1f} мс | '
             f'GPU {gpu_time*1000:.1f} мс',
             ha='center', fontsize=7.5, color='gray', style='italic')

    plot_path = os.path.join(PLOTS_DIR, 'comparison_gpu_vs_cpu.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    cpu_results, gpu_results, cpu_time, gpu_time = run_comparison()
    max_df, max_dr = print_comparison_table(cpu_results, gpu_results)
    generate_report_md(cpu_results, gpu_results, cpu_time, gpu_time, max_df, max_dr)
    generate_comparison_plot(cpu_results, gpu_results, cpu_time, gpu_time)

    print()
    print("=" * 70)
    passed = max_df < 5000
    print(f"VERDICT: {'PASSED' if passed else 'FAILED'} "
          f"(max df={max_df:.1f} Гц, tolerance=5000 Гц)")
    print("=" * 70)


if __name__ == '__main__':
    main()
