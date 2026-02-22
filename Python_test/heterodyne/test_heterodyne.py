"""
test_heterodyne.py — Tests for HeterodyneDechirp (LFM dechirp pipeline)

Tests:
  1. Basic dechirp — single antenna, known delay, verify f_beat
  2. Multiple antennas — 5 antennas, linear delays, verify range
  3. SNR verification — all SNR values > 0 dB
  4. Plot: f_beat vs delay (linear relationship)

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s

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

# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6           # sample rate
F_START = 0.0       # LFM start frequency
F_END = 2e6         # LFM end frequency
N = 8000            # samples per antenna
ANTENNAS = 5
BANDWIDTH = F_END - F_START  # 2 MHz
DURATION = N / FS            # 666.67 us
MU = BANDWIDTH / DURATION    # 3e9 Hz/s

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
    return gpuworklib.GPUContext(0)

@pytest.fixture(scope="module")
def het(ctx):
    h = gpuworklib.HeterodyneDechirp(ctx)
    h.set_params(F_START, F_END, FS, N, ANTENNAS)
    return h


# ============================================================================
# Test 1: Basic dechirp - single antenna
# ============================================================================

def test_basic_dechirp_single_antenna(ctx):
    """Single antenna with delay=100us -> f_beat=300kHz."""
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(F_START, F_END, FS, N, 1)

    rx = generate_lfm_rx([100.0])
    result = het.process(rx.ravel())

    assert result['success'], f"Process failed: {result['error_message']}"
    assert len(result['antennas']) == 1

    expected_f_beat = MU * 100e-6
    actual_f_beat = result['antennas'][0]['f_beat_hz']
    error = abs(actual_f_beat - expected_f_beat)

    print(f"\n  Expected: {expected_f_beat:.0f} Hz")
    print(f"  Actual:   {actual_f_beat:.0f} Hz")
    print(f"  Error:    {error:.0f} Hz")

    assert error < F_BEAT_TOL_HZ, \
        f"f_beat error {error:.0f} Hz exceeds tolerance {F_BEAT_TOL_HZ:.0f} Hz"


# ============================================================================
# Test 2: Multiple antennas - verify range
# ============================================================================

def test_multiple_antennas_range(het):
    """5 antennas with delays [100..500] us, verify f_beat and range."""
    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())

    assert result['success'], f"Process failed: {result['error_message']}"
    assert len(result['antennas']) == ANTENNAS

    print(f"\n  {'Ant':>3} | {'Delay us':>8} | {'f_beat Hz':>11} | {'Expected':>11} | {'Error':>8} | {'Range m':>9}")
    print(f"  {'-'*3} | {'-'*8} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9}")

    for i, ant in enumerate(result['antennas']):
        delay_us = DELAYS_US[i]
        expected_f = MU * delay_us * 1e-6
        actual_f = ant['f_beat_hz']
        error = abs(actual_f - expected_f)

        # Expected range: R = c*T*f_beat / (2*B)
        T = N / FS
        expected_range = (3e8 * T * expected_f) / (2.0 * BANDWIDTH)

        print(f"  {i:3d} | {delay_us:8.0f} | {actual_f:11.0f} | {expected_f:11.0f} | {error:8.0f} | {ant['range_m']:9.2f}")

        assert error < F_BEAT_TOL_HZ, \
            f"Antenna {i}: f_beat error {error:.0f} Hz exceeds tolerance"


# ============================================================================
# Test 3: SNR verification
# ============================================================================

def test_snr_positive(het):
    """All SNR values should be > 0 dB for clean LFM signal."""
    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())

    assert result['success']

    print()
    for i, ant in enumerate(result['antennas']):
        snr = ant['peak_snr_db']
        print(f"  Ant {i}: SNR = {snr:.1f} dB")
        assert snr > 0.0, f"Antenna {i}: SNR {snr:.1f} dB <= 0"


# ============================================================================
# Test 4: Plot f_beat vs delay (saves to Results/Plots/heterodyne/)
# ============================================================================

def test_plot_f_beat_vs_delay(het):
    """Generate plot of f_beat vs delay to verify linearity."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not available")

    rx = generate_lfm_rx(DELAYS_US)
    result = het.process(rx.ravel())
    assert result['success']

    delays_s = np.array(DELAYS_US) * 1e-6
    expected_f = MU * delays_s
    actual_f = np.array([a['f_beat_hz'] for a in result['antennas']])
    snr_values = np.array([a['peak_snr_db'] for a in result['antennas']])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: f_beat vs delay
    ax1 = axes[0]
    ax1.plot(np.array(DELAYS_US), expected_f / 1e3, 'b--', label='Expected (mu*tau)', linewidth=2)
    ax1.plot(np.array(DELAYS_US), actual_f / 1e3, 'ro-', label='GPU result', markersize=8)
    ax1.set_xlabel('Delay [us]')
    ax1.set_ylabel('f_beat [kHz]')
    ax1.set_title('Dechirp: f_beat vs delay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: SNR per antenna
    ax2 = axes[1]
    ax2.bar(range(ANTENNAS), snr_values, color='green', alpha=0.7)
    ax2.set_xlabel('Antenna')
    ax2.set_ylabel('SNR [dB]')
    ax2.set_title('Peak SNR per antenna')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

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
