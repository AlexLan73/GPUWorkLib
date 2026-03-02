"""
test_heterodyne.py — Tests for HeterodyneDechirp (LFM dechirp pipeline)

Tests:
  1. Basic dechirp — single antenna, known delay, verify f_beat
  2. Multiple antennas — 5 antennas, linear delays, verify range
  3. SNR verification — all SNR values > 0 dB
  4. Plot: f_beat vs delay (linear relationship) + SNR bars

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
search_range=5000 => half_range=2500 => left bins [0..2499] (~0..3.66 MHz)

@author Kodo (AI Assistant)
@date 2026-02-21
"""

import sys
import os
import numpy as np
import pytest

# -- Path to gpuworklib (Python_test/heterodyne/ -> 2 levels up) --
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

if not hasattr(gpuworklib, 'HeterodyneDechirp'):
    print("ERROR: gpuworklib built without HeterodyneDechirp.")
    print("  Rebuild: cmake -B build -DBUILD_PYTHON=ON && cmake --build build")
    sys.exit(1)


def _is_amd_gpu():
    """Detect AMD GPU (ROCm available)."""
    try:
        import subprocess
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=2)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


USE_ROCM = _is_amd_gpu()

# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6           # sample rate, Hz
F_START = 0.0       # LFM start frequency, Hz
F_END = 2e6         # LFM end frequency, Hz
N = 8000            # samples per antenna
ANTENNAS = 5
BANDWIDTH = F_END - F_START  # 2 MHz
DURATION = N / FS            # 666.67 us
MU = BANDWIDTH / DURATION    # 3e9 Hz/s  (chirp rate)
C_LIGHT = 3e8                # speed of light, m/s

DELAYS_US = [100, 200, 300, 400, 500]  # delays in microseconds
F_BEAT_TOL_HZ = 5000.0                 # tolerance +/- 5 kHz

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Plots', 'heterodyne')


# ============================================================================
# Helper: generate delayed LFM signal on CPU (reference)
# ============================================================================

def generate_lfm_rx(delays_us, f_start=F_START, f_end=F_END, fs=FS, n=N):
    """Generate delayed LFM signal (complex IQ) for each antenna."""
    t = np.arange(n, dtype=np.float32) / fs
    mu = (f_end - f_start) / (n / fs)
    rx = np.zeros((len(delays_us), n), dtype=np.complex64)
    for i, delay_us in enumerate(delays_us):
        tau = delay_us * 1e-6
        t_delayed = t - tau
        phase = 2 * np.pi * (0.5 * mu * t_delayed**2 + f_start * t_delayed)
        rx[i, :] = np.exp(1j * phase).astype(np.complex64)
    return rx


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def ctx():
    if USE_ROCM and hasattr(gpuworklib, 'ROCmGPUContext'):
        return gpuworklib.ROCmGPUContext(0)
    return gpuworklib.GPUContext(0)

@pytest.fixture(scope="module")
def het(ctx):
    h = gpuworklib.HeterodyneDechirp(ctx)
    h.set_params(float(F_START), float(F_END), float(FS), int(N), int(ANTENNAS))
    return h


# ============================================================================
# Test 1: Basic dechirp — single antenna
# ============================================================================

def test_basic_dechirp_single_antenna(ctx):
    """Single antenna with delay=100us -> f_beat=300kHz."""
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(float(F_START), float(F_END), float(FS), int(N), 1)

    rx = generate_lfm_rx([100.0])
    result = het.process(rx.ravel())

    assert result['success'], f"Process failed: {result['error_message']}"
    assert len(result['antennas']) == 1

    expected_f_beat = MU * 100e-6
    actual_f_beat = result['antennas'][0]['f_beat_hz']
    error = abs(actual_f_beat - expected_f_beat)

    print(f"\n  Expected: {expected_f_beat:.0f} Hz  (mu * tau = 3e9 * 100e-6)")
    print(f"  Actual:   {actual_f_beat:.0f} Hz")
    print(f"  Error:    {error:.0f} Hz  (tolerance: {F_BEAT_TOL_HZ:.0f} Hz)")

    assert error < F_BEAT_TOL_HZ, \
        f"f_beat error {error:.0f} Hz exceeds tolerance {F_BEAT_TOL_HZ:.0f} Hz"


# ============================================================================
# Test 2: Multiple antennas — verify range
# ============================================================================

def test_multiple_antennas_range(het):
    """5 antennas with delays [100..500] us, verify f_beat and range."""
    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())

    assert result['success'], f"Process failed: {result['error_message']}"
    assert len(result['antennas']) == ANTENNAS

    print(f"\n  Formula: f_beat = mu * tau,  R = c * T * f_beat / (2 * B)")
    print(f"  mu={MU:.2e} Hz/s, T={DURATION*1e6:.2f} us, B={BANDWIDTH/1e6:.0f} MHz")
    print(f"  {'Ant':>3} | {'Delay us':>8} | {'f_beat Hz':>11} | {'Expected':>11} | "
          f"{'Error':>8} | {'Range m':>9}")
    print(f"  {'-'*3} | {'-'*8} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9}")

    for i, ant in enumerate(result['antennas']):
        delay_us = DELAYS_US[i]
        expected_f = MU * delay_us * 1e-6
        actual_f = ant['f_beat_hz']
        error = abs(actual_f - expected_f)
        print(f"  {i:3d} | {delay_us:8.0f} | {actual_f:11.0f} | {expected_f:11.0f} | "
              f"{error:8.0f} | {ant['range_m']:9.2f}")
        assert error < F_BEAT_TOL_HZ, \
            f"Antenna {i}: f_beat error {error:.0f} Hz exceeds tolerance"


# ============================================================================
# Test 3: SNR verification
# ============================================================================

def test_snr_positive(het):
    """All SNR values should be > 0 dB for clean LFM signal.

    SNR = 20*log10(peak_magnitude / noise_estimate)
    noise_estimate = average of two neighboring FFT bins.
    For clean sinusoid peak >> neighbors => SNR >> 0 dB.
    """
    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())
    assert result['success']

    print()
    print(f"  SNR = 20*log10(peak_mag / avg(left_bin, right_bin))")
    for i, ant in enumerate(result['antennas']):
        snr = ant['peak_snr_db']
        tau_us = DELAYS_US[i]
        f_beat = ant['f_beat_hz']
        print(f"  Ant {i}: delay={tau_us} us, f_beat={f_beat/1e3:.1f} kHz, SNR={snr:.1f} dB")
        assert snr > 0.0, f"Antenna {i}: SNR {snr:.1f} dB <= 0"
    print("  All SNR > 0 dB — PASSED")


# ============================================================================
# Test 4: Plot f_beat vs delay + SNR + range + errors (полный отчёт)
# ============================================================================

def test_plot_f_beat_vs_delay(het):
    """Generate multi-panel report plot for the heterodyne dechirp pipeline."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        pytest.skip("matplotlib not available")

    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())
    assert result['success']

    delays_arr = np.array(DELAYS_US, dtype=float)
    delays_s   = delays_arr * 1e-6
    expected_f = MU * delays_s
    actual_f   = np.array([a['f_beat_hz'] for a in result['antennas']])
    snr_vals   = np.array([a['peak_snr_db'] for a in result['antennas']])
    ranges     = np.array([a['range_m'] for a in result['antennas']])
    errors_hz  = np.abs(actual_f - expected_f)
    true_r     = C_LIGHT * delays_s / 2.0

    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.40,
                   top=0.88, bottom=0.09)

    # ── заголовок ────────────────────────────────────────────────────────────
    fig.suptitle('LFM Dechirp — GPU результаты (HeterodyneDechirp)',
                 fontsize=13, fontweight='bold')
    fig.text(0.5, 0.915,
             f'fs={FS/1e6:.0f} MHz | B={BANDWIDTH/1e6:.0f} MHz | '
             f'N={N} | T={DURATION*1e6:.1f} мкс | '
             f'μ={MU:.2e} Гц/с | {ANTENNAS} антенн',
             ha='center', fontsize=9, color='gray')

    # ── Panel 1: f_beat vs задержка ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(delays_arr, expected_f / 1e3, 'b--', lw=2, alpha=0.7,
             label='Теория: f = μ·τ')
    ax1.plot(delays_arr, actual_f / 1e3, 'ro-', ms=9, lw=1.5,
             label='GPU результат')
    for i in range(ANTENNAS):
        ax1.annotate(f'{actual_f[i]/1e3:.0f} кГц',
                     (delays_arr[i], actual_f[i] / 1e3),
                     textcoords='offset points', xytext=(4, 6),
                     fontsize=7.5, color='red')
    ax1.set_xlabel('Задержка τ [мкс]')
    ax1.set_ylabel('f_beat [кГц]')
    ax1.set_title('Частота биений f_beat vs задержка', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.03, 0.96,
             'f_beat = μ · τ\nμ = B/T = 3·10⁹ Гц/с\nЛинейная зависимость',
             transform=ax1.transAxes, va='top', fontsize=8, color='navy',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                       ec='navy', alpha=0.85))

    # ── Panel 2: SNR per antenna ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bar_c = ['#2ecc71' if s > 10 else '#f39c12' if s > 0 else '#e74c3c'
              for s in snr_vals]
    bars2 = ax2.bar(range(ANTENNAS), snr_vals, color=bar_c, alpha=0.85,
                    ec='black', lw=0.5)
    ax2.axhline(0,  color='red',   ls='--', lw=1.3, alpha=0.8, label='0 дБ')
    ax2.axhline(10, color='green', ls=':',  lw=1.0, alpha=0.7, label='10 дБ порог')
    for bar, s in zip(bars2, snr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 max(s, 0) + 0.5, f'{s:.1f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_xlabel('Антенна')
    ax2.set_ylabel('SNR [дБ]')
    ax2.set_title('Отношение сигнал/шум по антеннам', fontsize=10)
    ax2.set_xticks(range(ANTENNAS))
    ax2.set_xticklabels([f'Ant{i}\n({DELAYS_US[i]}мкс)' for i in range(ANTENNAS)],
                        fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.03, 0.96,
             'SNR = 20·log₁₀(пик / шум)\nшум = среднее бинов peak±1',
             transform=ax2.transAxes, va='top', fontsize=8, color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.3', fc='honeydew',
                       ec='green', alpha=0.85))

    # ── Panel 3: Дальность vs задержка ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(delays_arr, true_r / 1e3, 'g--', lw=2, alpha=0.7,
             label='True R = c·τ/2')
    ax3.plot(delays_arr, ranges / 1e3, 'ms-', ms=9, lw=1.5,
             label='GPU R')
    for i in range(ANTENNAS):
        ax3.annotate(f'{ranges[i]/1e3:.1f} км',
                     (delays_arr[i], ranges[i] / 1e3),
                     textcoords='offset points', xytext=(4, 6),
                     fontsize=7.5, color='purple')
    ax3.set_xlabel('Задержка τ [мкс]')
    ax3.set_ylabel('Дальность R [км]')
    ax3.set_title('Дальность до цели vs задержка', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.03, 0.96,
             'R = c·T·f_beat / (2·B)\n≈ c·τ/2',
             transform=ax3.transAxes, va='top', fontsize=8, color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.3', fc='honeydew',
                       ec='green', alpha=0.85))

    # ── Panel 4: Ошибка f_beat ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ec = ['#27ae60' if e < 1000 else '#e67e22' if e < F_BEAT_TOL_HZ
          else '#e74c3c' for e in errors_hz]
    bars4 = ax4.bar(range(ANTENNAS), errors_hz, color=ec, alpha=0.85,
                    ec='black', lw=0.5)
    ax4.axhline(F_BEAT_TOL_HZ, color='red', ls='--', lw=1.5,
                label=f'Допуск ±{F_BEAT_TOL_HZ/1e3:.0f} кГц')
    for bar, err in zip(bars4, errors_hz):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 err + 30, f'{err:.0f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax4.set_xlabel('Антенна')
    ax4.set_ylabel('|f_GPU - f_ожид| [Гц]')
    ax4.set_title('Ошибка определения f_beat', fontsize=10)
    ax4.set_xticks(range(ANTENNAS))
    ax4.set_xticklabels([f'Ant{i}\n({DELAYS_US[i]}мкс)' for i in range(ANTENNAS)],
                        fontsize=8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0.03, 0.96,
             'Точность ≈ bin_width / 2\nbin_width = fs/N_FFT ≈ 1465 Гц',
             transform=ax4.transAxes, va='top', fontsize=8, color='saddlebrown',
             bbox=dict(boxstyle='round,pad=0.3', fc='linen',
                       ec='saddlebrown', alpha=0.85))

    # ── Panel 5: Сводная таблица ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    header = (f"{'Ant':>4} | {'Задержка':>9} | {'f_beat GPU':>12} | "
              f"{'f_ожидаемая':>12} | {'Ошибка':>10} | "
              f"{'Дальность':>12} | {'SNR':>7}")
    divider = "─" * 85
    rows = [header, divider]
    for i, ant in enumerate(result['antennas']):
        tau = DELAYS_US[i]
        fg  = ant['f_beat_hz']
        fe  = MU * tau * 1e-6
        err = abs(fg - fe)
        r   = ant['range_m']
        s   = ant['peak_snr_db']
        ok  = "OK" if err < F_BEAT_TOL_HZ else "!!"
        rows.append(
            f"{i:>4} | {tau:>7.0f} мкс | {fg:>10.0f} Гц | "
            f"{fe:>10.0f} Гц | {err:>7.0f} Гц {ok} | "
            f"{r:>10.1f} м | {s:>6.1f} дБ"
        )
    ax5.text(0.01, 0.95, '\n'.join(rows),
             transform=ax5.transAxes, va='top', ha='left',
             fontsize=8.5, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f8f9fa', ec='#adb5bd'))
    ax5.set_title('Сводная таблица', pad=4, fontsize=10)

    # ── подвал ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             'GPUWorkLib | HeterodyneDechirp | OpenCL float32 | '
             'search_range=5000 (half_range=2500, охват [0..2499] бин ≈ [0..3.66 МГц])',
             ha='center', fontsize=7.5, color='gray', style='italic')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'test_heterodyne_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


# ============================================================================
# Main (standalone run)
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
