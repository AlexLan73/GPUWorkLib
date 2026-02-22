"""
test_heterodyne_step_by_step.py
=================================
Step-by-step dechirp pipeline test with intermediate values and plots.

Each step prints values and saves a plot to Results/Plots/heterodyne/step_XX_*.png.
Runs GPU (gpuworklib.HeterodyneDechirp) and CPU (NumPy) in parallel for comparison.

Steps:
  1. Generate s_rx (5 antennas, linear delays) — GPU + NumPy
  2. Generate s_ref* (conjugate LFM, delay=0) — NumPy reference
  3. Dechirp: s_dc = conj(s_rx * s_ref*) — NumPy
  4. FFT of dechirped signal — NumPy
  5. FindMaxima: f_beat, R — GPU pipeline + NumPy argmax
  6. Dechirp correct: compensate f_beat — NumPy
  7. Verify DC: final FFT — NumPy
  8. Summary table: GPU vs CPU errors

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s

@author Kodo (AI Assistant)
@date 2026-02-21
"""

import sys
import os
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
# Constants (match C++ test parameters)
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

DELAYS_LINEAR_US = np.array([100., 200., 300., 400., 500.])
DELAYS_LINEAR_S = DELAYS_LINEAR_US * 1e-6

F_BEATS_EXPECTED = MU * DELAYS_LINEAR_S
RANGES_TRUE = C_LIGHT * T * F_BEATS_EXPECTED / (2 * B)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                          'Results', 'Plots', 'heterodyne')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def generate_rx_numpy(delays_s):
    """Generate delayed LFM signals on CPU."""
    t = np.arange(N, dtype=np.float64) / FS
    rx = np.zeros((len(delays_s), N), dtype=np.complex64)
    for i, tau in enumerate(delays_s):
        t_delayed = t - tau
        phase = 2 * np.pi * (0.5 * MU * t_delayed**2 + F_START * t_delayed)
        rx[i, :] = np.exp(1j * phase).astype(np.complex64)
    return rx


def generate_ref_conjugate_numpy():
    """Generate conjugate LFM reference: s_ref* = exp(-j*[pi*mu*t^2 + 2pi*f0*t])."""
    t = np.arange(N, dtype=np.float64) / FS
    phase = -(np.pi * MU * t**2 + 2 * np.pi * F_START * t)
    return np.exp(1j * phase).astype(np.complex64)


def parabolic_interp(mag, idx):
    """Parabolic interpolation around peak bin."""
    if idx == 0 or idx >= len(mag) - 1:
        return float(idx), mag[idx]
    L, C, R = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = L - 2 * C + R
    if abs(denom) < 1e-12:
        return float(idx), C
    delta = 0.5 * (L - R) / denom
    refined_mag = C - 0.25 * (L - R) * delta
    return idx + delta, refined_mag


def save_plot(filename, fig):
    """Save figure to PLOTS_DIR."""
    if not HAS_MATPLOTLIB:
        return
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Plot saved: {path}")


# ============================================================================
# Steps
# ============================================================================

def step01_generate_rx():
    """Step 1: Generate received LFM signals (5 antennas, linear delays)."""
    print("\n" + "=" * 60)
    print("STEP 1: Generate s_rx (5 antennas, linear delays)")
    print("=" * 60)

    rx_cpu = generate_rx_numpy(DELAYS_LINEAR_S)

    for k in range(ANTENNAS):
        max_val = np.max(np.abs(rx_cpu[k]))
        print(f"  Ant {k}: delay={DELAYS_LINEAR_US[k]:.0f} us, max|s_rx|={max_val:.4f}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(ANTENNAS, 1, figsize=(12, 8), sharex=True)
        t_us = np.arange(N) / FS * 1e6
        for k in range(ANTENNAS):
            axes[k].plot(t_us[:500], np.real(rx_cpu[k, :500]), linewidth=0.5)
            axes[k].set_ylabel(f'Ant {k}')
            axes[k].set_title(f'delay={DELAYS_LINEAR_US[k]:.0f} us', fontsize=9)
        axes[-1].set_xlabel('Time [us]')
        fig.suptitle('Step 1: s_rx — received LFM (Re part, first 500 samples)')
        fig.tight_layout()
        save_plot('step_01_rx_signals.png', fig)

    return rx_cpu


def step02_generate_ref_conjugate():
    """Step 2: Generate conjugate LFM reference s_ref*."""
    print("\n" + "=" * 60)
    print("STEP 2: Generate s_ref* = conj(LFM), delay=0")
    print("=" * 60)

    ref_cpu = generate_ref_conjugate_numpy()
    print(f"  ref[0]   = {ref_cpu[0]:.6f}")
    print(f"  ref[100] = {ref_cpu[100]:.6f}")
    print(f"  max|ref| = {np.max(np.abs(ref_cpu)):.6f}")

    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
        t_us = np.arange(N) / FS * 1e6
        ax1.plot(t_us, np.real(ref_cpu), linewidth=0.3)
        ax1.set_ylabel('Re(s_ref*)')
        ax1.set_title('Step 2: Conjugate LFM reference')
        ax2.plot(t_us, np.unwrap(np.angle(ref_cpu)), linewidth=0.5, color='orange')
        ax2.set_ylabel('Phase [rad]')
        ax2.set_xlabel('Time [us]')
        fig.tight_layout()
        save_plot('step_02_ref_conjugate.png', fig)

    return ref_cpu


def step03_dechirp(rx_cpu, ref_cpu):
    """Step 3: Dechirp s_dc = conj(s_rx * s_ref*)."""
    print("\n" + "=" * 60)
    print("STEP 3: Dechirp s_dc = conj(s_rx * s_ref*)")
    print("=" * 60)

    dc_cpu = np.zeros_like(rx_cpu)
    for k in range(ANTENNAS):
        product = rx_cpu[k] * ref_cpu
        dc_cpu[k] = np.conj(product)

    for k in range(ANTENNAS):
        print(f"  Ant {k}: max|s_dc|={np.max(np.abs(dc_cpu[k])):.4f}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(ANTENNAS, 1, figsize=(12, 8), sharex=True)
        t_us = np.arange(N) / FS * 1e6
        for k in range(ANTENNAS):
            axes[k].plot(t_us[:1000], np.real(dc_cpu[k, :1000]), linewidth=0.5)
            axes[k].set_ylabel(f'Ant {k}')
        axes[-1].set_xlabel('Time [us]')
        fig.suptitle('Step 3: Dechirped signal Re(s_dc) — visible beat tones')
        fig.tight_layout()
        save_plot('step_03_dechirp.png', fig)

    return dc_cpu


def step04_fft(dc_cpu):
    """Step 4: FFT of dechirped signal."""
    print("\n" + "=" * 60)
    print("STEP 4: FFT of dechirped signal")
    print("=" * 60)

    nfft = 8192  # next power of 2 >= N
    spec_cpu = np.zeros((ANTENNAS, nfft), dtype=np.complex64)
    for k in range(ANTENNAS):
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[:N] = dc_cpu[k]
        spec_cpu[k] = np.fft.fft(padded)

    freqs = np.fft.fftfreq(nfft, d=1.0 / FS)

    print(f"  nFFT = {nfft}, bin_width = {FS / nfft:.1f} Hz")
    for k in range(ANTENNAS):
        mag = np.abs(spec_cpu[k, :nfft // 2])
        peak_bin = np.argmax(mag)
        peak_freq = freqs[peak_bin]
        expected_freq = F_BEATS_EXPECTED[k]
        print(f"  Ant {k}: peak bin={peak_bin}, f_peak={peak_freq:.0f} Hz, "
              f"expected={expected_freq:.0f} Hz, error={abs(peak_freq - expected_freq):.0f} Hz")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        f_khz = freqs[:nfft // 2] / 1e3
        for k in range(ANTENNAS):
            mag_db = 20 * np.log10(np.abs(spec_cpu[k, :nfft // 2]) + 1e-12)
            ax.plot(f_khz, mag_db, label=f'Ant {k} ({DELAYS_LINEAR_US[k]:.0f} us)')
        ax.set_xlabel('Frequency [kHz]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_title('Step 4: FFT spectrum of dechirped signal')
        ax.legend()
        ax.set_xlim([0, 2000])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_plot('step_04_fft_spectrum.png', fig)

    return spec_cpu


def step05_find_maxima(spec_cpu):
    """Step 5: Find peak -> f_beat -> range (CPU parabolic interp)."""
    print("\n" + "=" * 60)
    print("STEP 5: FindMaxima -> f_beat -> Range")
    print("=" * 60)

    nfft = spec_cpu.shape[1]
    results_cpu = []

    print(f"  {'Ant':>3} | {'Delay us':>8} | {'f_beat Hz':>11} | {'Expected Hz':>11} | "
          f"{'Error Hz':>8} | {'Range m':>9} | {'True R m':>9} | {'dR m':>6}")
    print(f"  {'-'*3} | {'-'*8} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*6}")

    for k in range(ANTENNAS):
        mag = np.abs(spec_cpu[k, :nfft // 2])
        peak_bin = np.argmax(mag)
        refined_bin, _ = parabolic_interp(mag, peak_bin)
        f_beat = refined_bin * FS / nfft
        range_m = C_LIGHT * T * f_beat / (2 * B)
        expected_f = F_BEATS_EXPECTED[k]
        f_err = abs(f_beat - expected_f)
        r_err = abs(range_m - RANGES_TRUE[k])

        results_cpu.append({
            'f_beat': f_beat, 'range_m': range_m,
            'f_error': f_err, 'r_error': r_err
        })

        print(f"  {k:3d} | {DELAYS_LINEAR_US[k]:8.0f} | {f_beat:11.0f} | {expected_f:11.0f} | "
              f"{f_err:8.0f} | {range_m:9.2f} | {RANGES_TRUE[k]:9.2f} | {r_err:6.2f}")

    return results_cpu


def step06_dechirp_correct(dc_cpu, results_cpu):
    """Step 6: Frequency correction — compensate f_beat."""
    print("\n" + "=" * 60)
    print("STEP 6: Dechirp correction (compensate f_beat)")
    print("=" * 60)

    corrected = np.zeros_like(dc_cpu)
    t = np.arange(N, dtype=np.float64) / FS

    for k in range(ANTENNAS):
        f_beat = results_cpu[k]['f_beat']
        phase = -2 * np.pi * f_beat * t
        correction = np.exp(1j * phase).astype(np.complex64)
        corrected[k] = dc_cpu[k] * correction

    for k in range(ANTENNAS):
        print(f"  Ant {k}: max|corrected|={np.max(np.abs(corrected[k])):.4f}")

    return corrected


def step07_verify_dc(corrected):
    """Step 7: Verify peak at DC after correction."""
    print("\n" + "=" * 60)
    print("STEP 7: Verify DC (peak at 0 Hz after correction)")
    print("=" * 60)

    nfft = 8192
    all_dc = True
    for k in range(ANTENNAS):
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[:N] = corrected[k]
        spectrum = np.fft.fft(padded)
        mag = np.abs(spectrum[:nfft // 2])
        peak_bin = np.argmax(mag)
        print(f"  Ant {k}: peak at bin {peak_bin} (expect ~0)")
        if peak_bin > 3:
            all_dc = False

    print(f"  {'PASSED: all peaks at DC' if all_dc else 'FAILED: some peaks NOT at DC'}")


def step08_gpu_pipeline():
    """Step 8: Full GPU pipeline and compare with CPU."""
    print("\n" + "=" * 60)
    print("STEP 8: GPU Pipeline (HeterodyneDechirp.process)")
    print("=" * 60)

    ctx = gpuworklib.GPUContext(0)
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(F_START, F_END, FS, N, ANTENNAS)

    rx_cpu = generate_rx_numpy(DELAYS_LINEAR_S)
    result = het.process(rx_cpu.ravel())

    if not result['success']:
        print(f"  FAIL: {result['error_message']}")
        return None

    print(f"  {'Ant':>3} | {'f_beat GPU':>11} | {'Expected Hz':>11} | {'Error Hz':>8} | "
          f"{'Range m':>9} | {'SNR dB':>7}")
    print(f"  {'-'*3} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9} | {'-'*7}")

    gpu_results = []
    for k, ant in enumerate(result['antennas']):
        expected_f = F_BEATS_EXPECTED[k]
        f_err = abs(ant['f_beat_hz'] - expected_f)
        gpu_results.append(ant)
        print(f"  {k:3d} | {ant['f_beat_hz']:11.0f} | {expected_f:11.0f} | {f_err:8.0f} | "
              f"{ant['range_m']:9.2f} | {ant['peak_snr_db']:7.1f}")

    return gpu_results


def print_summary(cpu_results, gpu_results):
    """Final summary table: GPU vs CPU comparison."""
    print("\n" + "=" * 60)
    print("SUMMARY: GPU vs CPU comparison")
    print("=" * 60)

    if gpu_results is None:
        print("  GPU results not available")
        return

    print(f"  {'Ant':>3} | {'f GPU Hz':>11} | {'f CPU Hz':>11} | {'df Hz':>8} | "
          f"{'R GPU m':>9} | {'R CPU m':>9} | {'dR m':>6} | {'SNR dB':>7}")
    print(f"  {'-'*3} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*6} | {'-'*7}")

    for k in range(ANTENNAS):
        f_gpu = gpu_results[k]['f_beat_hz']
        f_cpu = cpu_results[k]['f_beat']
        r_gpu = gpu_results[k]['range_m']
        r_cpu = cpu_results[k]['range_m']
        df = abs(f_gpu - f_cpu)
        dr = abs(r_gpu - r_cpu)
        snr = gpu_results[k]['peak_snr_db']
        print(f"  {k:3d} | {f_gpu:11.0f} | {f_cpu:11.0f} | {df:8.0f} | "
              f"{r_gpu:9.2f} | {r_cpu:9.2f} | {dr:6.2f} | {snr:7.1f}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: f_beat comparison
        f_gpus = [g['f_beat_hz'] for g in gpu_results]
        f_cpus = [c['f_beat'] for c in cpu_results]
        axes[0].plot(DELAYS_LINEAR_US, np.array(f_gpus) / 1e3, 'ro-', label='GPU', markersize=8)
        axes[0].plot(DELAYS_LINEAR_US, np.array(f_cpus) / 1e3, 'b^--', label='CPU', markersize=7)
        axes[0].plot(DELAYS_LINEAR_US, F_BEATS_EXPECTED / 1e3, 'g--', label='Theory', alpha=0.5)
        axes[0].set_xlabel('Delay [us]')
        axes[0].set_ylabel('f_beat [kHz]')
        axes[0].set_title('f_beat: GPU vs CPU vs Theory')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Error |GPU - CPU|
        df_vals = [abs(f_gpus[k] - f_cpus[k]) for k in range(ANTENNAS)]
        axes[1].bar(range(ANTENNAS), df_vals, color='coral')
        axes[1].set_xlabel('Antenna')
        axes[1].set_ylabel('|f_GPU - f_CPU| [Hz]')
        axes[1].set_title('f_beat error GPU vs CPU')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Plot 3: SNR
        snrs = [g['peak_snr_db'] for g in gpu_results]
        axes[2].bar(range(ANTENNAS), snrs, color='green', alpha=0.7)
        axes[2].set_xlabel('Antenna')
        axes[2].set_ylabel('SNR [dB]')
        axes[2].set_title('Peak SNR per antenna')
        axes[2].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Summary: GPU vs CPU heterodyne dechirp')
        fig.tight_layout()
        save_plot('step_08_summary.png', fig)


# ============================================================================
# Main
# ============================================================================

def run_full_test():
    """Run all steps sequentially."""
    rx_cpu = step01_generate_rx()
    ref_cpu = step02_generate_ref_conjugate()
    dc_cpu = step03_dechirp(rx_cpu, ref_cpu)
    spec_cpu = step04_fft(dc_cpu)
    cpu_results = step05_find_maxima(spec_cpu)
    corrected = step06_dechirp_correct(dc_cpu, cpu_results)
    step07_verify_dc(corrected)
    gpu_results = step08_gpu_pipeline()
    print_summary(cpu_results, gpu_results)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_full_test()
