"""
test_heterodyne_comparison.py
==============================
GPU vs CPU heterodyne dechirp comparison report.

Generates a detailed markdown report and 4-panel comparison plot:
  1. Runs full GPU pipeline (gpuworklib.HeterodyneDechirp)
  2. Runs CPU reference pipeline (NumPy FFT + parabolic interp)
  3. Compares f_beat, range, SNR per antenna
  4. Saves markdown report + PNG plot

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s

@author Kodo (AI Assistant)
@date 2026-02-21
"""

import sys
import os
import time
import numpy as np

# -- Path to gpuworklib --
BUILD_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Release'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Debug'),
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
DELAYS_S = DELAYS_US * 1e-6

F_BEATS_TRUE = MU * DELAYS_S
RANGES_TRUE = C_LIGHT * DELAYS_S / 2.0

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

    # Generate rx (delayed LFM)
    rx = np.zeros((len(delays_s), N), dtype=np.complex128)
    for i, tau in enumerate(delays_s):
        t_d = t - tau
        phase = 2 * np.pi * (0.5 * MU * t_d**2 + F_START * t_d)
        rx[i] = np.exp(1j * phase)

    # Generate conjugate ref (delay=0)
    ref_conj = np.exp(-1j * (np.pi * MU * t**2 + 2 * np.pi * F_START * t))

    # Dechirp: conj(rx * ref_conj)
    dc = np.conj(rx * ref_conj[np.newaxis, :])

    # FFT
    nfft = 8192
    results = []
    for k in range(len(delays_s)):
        padded = np.zeros(nfft, dtype=np.complex128)
        padded[:N] = dc[k]
        spectrum = np.fft.fft(padded)
        mag = np.abs(spectrum[:nfft // 2])

        # Peak with parabolic interpolation
        peak_bin = np.argmax(mag)
        if 0 < peak_bin < len(mag) - 1:
            L, C, R = mag[peak_bin - 1], mag[peak_bin], mag[peak_bin + 1]
            denom = L - 2 * C + R
            if abs(denom) > 1e-12:
                delta = 0.5 * (L - R) / denom
            else:
                delta = 0.0
        else:
            delta = 0.0

        refined_bin = peak_bin + delta
        f_beat = refined_bin * FS / nfft
        range_m = C_LIGHT * T * f_beat / (2 * B)

        # SNR estimate
        noise_bins = list(range(max(1, peak_bin - 50), max(1, peak_bin - 5)))
        noise_bins += list(range(min(peak_bin + 5, nfft // 2 - 1),
                                  min(peak_bin + 50, nfft // 2 - 1)))
        if noise_bins:
            noise_est = np.mean(mag[noise_bins])
        else:
            noise_est = 1e-12
        snr_db = 20 * np.log10(mag[peak_bin] / max(noise_est, 1e-12))

        results.append({
            'f_beat_hz': float(f_beat),
            'range_m': float(range_m),
            'peak_snr_db': float(snr_db),
            'peak_bin': int(peak_bin),
            'refined_bin': float(refined_bin),
            'peak_magnitude': float(mag[peak_bin]),
        })

    return results


# ============================================================================
# GPU Pipeline
# ============================================================================

def gpu_pipeline(delays_s):
    """Full GPU pipeline via gpuworklib.HeterodyneDechirp."""
    ctx = gpuworklib.GPUContext(0)
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(F_START, F_END, FS, N, len(delays_s))

    # Generate rx on CPU (same as CPU pipeline but float32)
    t = np.arange(N, dtype=np.float32) / FS
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
    print(f"  fs={FS/1e6:.0f} MHz, B={B/1e6:.0f} MHz, N={N}, T={T*1e6:.2f} us")
    print(f"  mu={MU:.2e} Hz/s, antennas={ANTENNAS}")
    print(f"  delays={DELAYS_US.tolist()} us")
    print()

    # --- CPU ---
    print("Running CPU pipeline (NumPy float64)...")
    t0 = time.perf_counter()
    cpu_results = cpu_pipeline(DELAYS_S)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time*1000:.1f} ms")

    # --- GPU ---
    print("Running GPU pipeline (gpuworklib, float32)...")
    # Warmup
    _ = gpu_pipeline(DELAYS_S)
    t0 = time.perf_counter()
    gpu_results = gpu_pipeline(DELAYS_S)
    gpu_time = time.perf_counter() - t0
    print(f"  GPU time: {gpu_time*1000:.1f} ms")
    print()

    return cpu_results, gpu_results, cpu_time, gpu_time


def print_comparison_table(cpu_results, gpu_results):
    """Print comparison table."""
    print("COMPARISON TABLE")
    print("-" * 100)
    header = (f"  {'Ant':>3} | {'Delay':>6} | {'f GPU':>11} | {'f CPU':>11} | "
              f"{'df':>8} | {'R GPU':>9} | {'R CPU':>9} | {'dR':>6} | "
              f"{'SNR GPU':>7} | {'SNR CPU':>7}")
    units = (f"  {'':>3} | {'us':>6} | {'Hz':>11} | {'Hz':>11} | "
             f"{'Hz':>8} | {'m':>9} | {'m':>9} | {'m':>6} | "
             f"{'dB':>7} | {'dB':>7}")
    print(header)
    print(units)
    print("  " + "-" * 96)

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

        print(f"  {k:3d} | {DELAYS_US[k]:6.0f} | {fg:11.1f} | {fc:11.1f} | "
              f"{df:8.1f} | {rg:9.2f} | {rc:9.2f} | {dr:6.2f} | "
              f"{sg:7.1f} | {sc:7.1f}")

    print()
    print(f"  Max |f_GPU - f_CPU|: {max_df:.1f} Hz")
    print(f"  Max |R_GPU - R_CPU|: {max_dr:.2f} m")

    # Check vs true values
    print()
    print("VS THEORY:")
    print("  " + "-" * 80)
    print(f"  {'Ant':>3} | {'f_true':>11} | {'f_GPU err':>10} | {'f_CPU err':>10} | "
          f"{'R_true':>9} | {'R_GPU err':>9} | {'R_CPU err':>9}")
    print("  " + "-" * 80)
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
        "# Heterodyne Dechirp: GPU vs CPU Comparison Report",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| fs | {FS/1e6:.0f} MHz |",
        f"| B (bandwidth) | {B/1e6:.0f} MHz |",
        f"| N (samples) | {N} |",
        f"| T (chirp duration) | {T*1e6:.2f} us |",
        f"| mu (chirp rate) | {MU:.2e} Hz/s |",
        f"| Antennas | {ANTENNAS} |",
        f"| Delays | {DELAYS_US.tolist()} us |",
        "",
        "## Timing",
        "",
        f"| Pipeline | Time |",
        f"|----------|------|",
        f"| CPU (NumPy float64) | {cpu_time*1000:.1f} ms |",
        f"| GPU (OpenCL float32) | {gpu_time*1000:.1f} ms |",
        f"| Speedup | {cpu_time/max(gpu_time, 1e-9):.1f}x |",
        "",
        "## Results",
        "",
        "| Ant | Delay us | f_GPU Hz | f_CPU Hz | df Hz | R_GPU m | R_CPU m | dR m | SNR_GPU dB | SNR_CPU dB |",
        "|-----|----------|----------|----------|-------|---------|---------|------|------------|------------|",
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
        "## Summary",
        "",
        f"- Max |f_GPU - f_CPU|: **{max_df:.1f} Hz**",
        f"- Max |R_GPU - R_CPU|: **{max_dr:.2f} m**",
        f"- All f_beat errors < 5000 Hz tolerance: "
        f"**{'PASS' if max_df < 5000 else 'FAIL'}**",
        "",
        "## vs Theory",
        "",
        "| Ant | f_true Hz | f_GPU err Hz | f_CPU err Hz | R_true m | R_GPU err m | R_CPU err m |",
        "|-----|-----------|-------------|-------------|----------|------------|------------|",
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
        f"*Generated by test_heterodyne_comparison.py | Kodo (AI Assistant)*",
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved: {report_path}")
    return report_path


def generate_comparison_plot(cpu_results, gpu_results):
    """Generate 4-panel comparison plot."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Heterodyne Dechirp: GPU vs CPU Comparison',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: f_beat comparison ---
    ax = axes[0, 0]
    fg = [g['f_beat_hz'] for g in gpu_results]
    fc = [c['f_beat_hz'] for c in cpu_results]
    ax.plot(DELAYS_US, np.array(F_BEATS_TRUE) / 1e3, 'g--',
            label='Theory', linewidth=2, alpha=0.6)
    ax.plot(DELAYS_US, np.array(fg) / 1e3, 'ro-',
            label='GPU (OpenCL)', markersize=8)
    ax.plot(DELAYS_US, np.array(fc) / 1e3, 'b^--',
            label='CPU (NumPy)', markersize=7)
    ax.set_xlabel('Delay [us]')
    ax.set_ylabel('f_beat [kHz]')
    ax.set_title('f_beat vs Delay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: f_beat error ---
    ax = axes[0, 1]
    gpu_f_err = [abs(gpu_results[k]['f_beat_hz'] - F_BEATS_TRUE[k])
                 for k in range(ANTENNAS)]
    cpu_f_err = [abs(cpu_results[k]['f_beat_hz'] - F_BEATS_TRUE[k])
                 for k in range(ANTENNAS)]
    x = np.arange(ANTENNAS)
    w = 0.35
    ax.bar(x - w/2, gpu_f_err, w, label='GPU', color='coral')
    ax.bar(x + w/2, cpu_f_err, w, label='CPU', color='steelblue')
    ax.axhline(y=5000, color='red', linestyle='--', alpha=0.5, label='Tolerance 5kHz')
    ax.set_xlabel('Antenna')
    ax.set_ylabel('|f_error| [Hz]')
    ax.set_title('f_beat Error vs Theory')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: Range comparison ---
    ax = axes[1, 0]
    rg = [g['range_m'] for g in gpu_results]
    rc = [c['range_m'] for c in cpu_results]
    ax.plot(DELAYS_US, RANGES_TRUE, 'g--', label='True R', linewidth=2, alpha=0.6)
    ax.plot(DELAYS_US, rg, 'ro-', label='GPU', markersize=8)
    ax.plot(DELAYS_US, rc, 'b^--', label='CPU', markersize=7)
    ax.set_xlabel('Delay [us]')
    ax.set_ylabel('Range [m]')
    ax.set_title('Range vs Delay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: SNR comparison ---
    ax = axes[1, 1]
    sg = [g['peak_snr_db'] for g in gpu_results]
    sc = [c['peak_snr_db'] for c in cpu_results]
    ax.bar(x - w/2, sg, w, label='GPU', color='green', alpha=0.7)
    ax.bar(x + w/2, sc, w, label='CPU', color='orange', alpha=0.7)
    ax.set_xlabel('Antenna')
    ax.set_ylabel('SNR [dB]')
    ax.set_title('Peak SNR')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
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
    generate_comparison_plot(cpu_results, gpu_results)

    print()
    print("=" * 70)
    passed = max_df < 5000
    print(f"VERDICT: {'PASSED' if passed else 'FAILED'} "
          f"(max df={max_df:.1f} Hz, tolerance=5000 Hz)")
    print("=" * 70)


if __name__ == '__main__':
    main()
