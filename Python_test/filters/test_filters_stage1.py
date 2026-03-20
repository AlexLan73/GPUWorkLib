"""
Stage 1: scipy -> GPU filter pipeline test
============================================

Tests:
  1. FIR: scipy.signal.firwin() -> gpuworklib.FirFilter -> validate vs scipy.lfilter
  2. IIR: scipy.signal.butter(sos) -> gpuworklib.IirFilter -> validate vs scipy.sosfilt

Usage:
  python Python_test/filters/test_filters_stage1.py

Author: Kodo (AI Assistant)
Date: 2026-02-18
"""

import numpy as np
import sys
import os

# Python_test/filters/ -> 2 levels up to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for subdir in ["build/python/Release", "build/python/Debug", "build/Release", "build/Debug"]:
    p = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

# ============================================================================
# Try to import gpuworklib
# ============================================================================
try:
    import gpuworklib as gw
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found. Only CPU reference tests will run.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Skipping scipy validation tests.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ============================================================================
# Test parameters
# ============================================================================
CHANNELS = 8
POINTS = 4096
SAMPLE_RATE = 50000.0
F_LOW = 100.0      # Hz - should pass through LP filter
F_HIGH = 5000.0    # Hz - should be attenuated by LP filter
FIR_TAPS = 64
FIR_CUTOFF = 0.1   # Normalized (0-1, Nyquist=1)
IIR_ORDER = 2
IIR_CUTOFF = 0.1   # Normalized


def generate_test_signal(channels, points, fs):
    """Generate multi-channel complex test signal: CW_low + CW_high"""
    t = np.arange(points) / fs
    signal = np.zeros((channels, points), dtype=np.complex64)
    for ch in range(channels):
        phase = ch * 0.1
        re = (np.cos(2 * np.pi * F_LOW * t + phase)
              + 0.5 * np.cos(2 * np.pi * F_HIGH * t))
        im = (np.sin(2 * np.pi * F_LOW * t + phase)
              + 0.5 * np.sin(2 * np.pi * F_HIGH * t))
        signal[ch] = (re + 1j * im).astype(np.complex64)
    return signal


# ============================================================================
# FIR Tests
# ============================================================================

def test_fir_gpu_vs_scipy():
    """FIR: GPU result matches scipy.lfilter reference"""
    if not HAS_GPU:
        print("SKIP: gpuworklib not available")
        return
    if not HAS_SCIPY:
        print("SKIP: scipy not available")
        return

    # Design FIR via scipy
    taps = sig.firwin(FIR_TAPS, FIR_CUTOFF).astype(np.float32)

    # GPU
    ctx = gw.GPUContext(0)
    fir = gw.FirFilter(ctx)
    fir.set_coefficients(taps.tolist())

    signal = generate_test_signal(CHANNELS, POINTS, SAMPLE_RATE)

    result_gpu = fir.process(signal)

    # Reference: scipy per channel
    result_ref = np.zeros_like(signal)
    for ch in range(CHANNELS):
        result_ref[ch] = sig.lfilter(taps, [1.0], signal[ch]).astype(np.complex64)

    # Compare
    max_err = np.max(np.abs(result_gpu - result_ref))
    print(f"FIR GPU vs scipy: max_error = {max_err:.2e}")
    assert max_err < 1e-2, f"FIR error too large: {max_err}"
    print("  PASSED")


def test_fir_basic_properties():
    """FIR: basic property checks"""
    if not HAS_GPU:
        print("SKIP: gpuworklib not available")
        return
    if not HAS_SCIPY:
        print("SKIP: scipy not available")
        return

    taps = sig.firwin(FIR_TAPS, FIR_CUTOFF).astype(np.float32)

    ctx = gw.GPUContext(0)
    fir = gw.FirFilter(ctx)
    fir.set_coefficients(taps.tolist())

    assert fir.num_taps == FIR_TAPS
    assert len(fir.coefficients) == FIR_TAPS
    print(f"FIR properties: num_taps={fir.num_taps}, repr={repr(fir)}")
    print("  PASSED")


def test_fir_single_channel():
    """FIR: single channel (1D input)"""
    if not HAS_GPU:
        print("SKIP: gpuworklib not available")
        return
    if not HAS_SCIPY:
        print("SKIP: scipy not available")
        return

    taps = sig.firwin(32, 0.2).astype(np.float32)

    ctx = gw.GPUContext(0)
    fir = gw.FirFilter(ctx)
    fir.set_coefficients(taps.tolist())

    # 1D input
    signal_1d = generate_test_signal(1, POINTS, SAMPLE_RATE)[0]
    result = fir.process(signal_1d)

    assert result.ndim == 1
    assert result.shape[0] == POINTS
    print(f"FIR 1D: output shape = {result.shape}")
    print("  PASSED")


# ============================================================================
# IIR Tests
# ============================================================================

def test_iir_gpu_vs_scipy():
    """IIR: GPU result matches scipy.sosfilt reference"""
    if not HAS_GPU:
        print("SKIP: gpuworklib not available")
        return
    if not HAS_SCIPY:
        print("SKIP: scipy not available")
        return

    # Design IIR via scipy
    sos = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)

    # Convert SOS to list of dicts for gpuworklib
    sections = []
    for row in sos:
        # SOS format: [b0, b1, b2, a0, a1, a2] (a0=1)
        sections.append({
            'b0': float(row[0]),
            'b1': float(row[1]),
            'b2': float(row[2]),
            'a1': float(row[4]),  # skip a0 (always 1)
            'a2': float(row[5]),
        })

    # GPU
    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    signal = generate_test_signal(CHANNELS, POINTS, SAMPLE_RATE)

    result_gpu = iir.process(signal)

    # Reference: scipy per channel
    result_ref = np.zeros_like(signal)
    for ch in range(CHANNELS):
        result_ref[ch] = sig.sosfilt(sos, signal[ch]).astype(np.complex64)

    # Compare
    max_err = np.max(np.abs(result_gpu - result_ref))
    print(f"IIR GPU vs scipy: max_error = {max_err:.2e}")
    assert max_err < 5e-2, f"IIR error too large: {max_err}"
    print("  PASSED")


def test_iir_basic_properties():
    """IIR: basic property checks"""
    if not HAS_GPU:
        print("SKIP: gpuworklib not available")
        return

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections([
        {'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.56, 'a2': 0.64}
    ])

    assert iir.num_sections == 1
    assert len(iir.sections) == 1
    print(f"IIR properties: num_sections={iir.num_sections}, repr={repr(iir)}")
    print("  PASSED")


# ============================================================================
# Visualization (standalone mode)
# ============================================================================

def plot_filter_results():
    """Generate 4-panel plot showing filter results"""
    if not (HAS_GPU and HAS_SCIPY and HAS_PLOT):
        print("SKIP: missing dependencies for plotting")
        return

    taps = sig.firwin(FIR_TAPS, FIR_CUTOFF).astype(np.float32)

    ctx = gw.GPUContext(0)
    fir = gw.FirFilter(ctx)
    fir.set_coefficients(taps.tolist())

    signal = generate_test_signal(1, POINTS, SAMPLE_RATE)[0]
    result = fir.process(signal)

    t = np.arange(POINTS) / SAMPLE_RATE * 1000  # ms

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('GPU FIR Filter Test (Stage 1: scipy -> GPU)', fontsize=14)

    # 1. Signal before filtering
    axes[0, 0].plot(t[:500], signal.real[:500], 'b-', alpha=0.7, label='Re')
    axes[0, 0].plot(t[:500], signal.imag[:500], 'r-', alpha=0.5, label='Im')
    axes[0, 0].set_title('Input Signal (first 500 samples)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Signal after filtering
    axes[0, 1].plot(t[:500], result.real[:500], 'b-', alpha=0.7, label='Re')
    axes[0, 1].plot(t[:500], result.imag[:500], 'r-', alpha=0.5, label='Im')
    axes[0, 1].set_title('FIR Filtered Signal (first 500 samples)')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Frequency response of FIR filter
    w, h = sig.freqz(taps, worN=2048)
    freq_hz = w / np.pi * (SAMPLE_RATE / 2)
    axes[1, 0].plot(freq_hz, 20 * np.log10(np.abs(h) + 1e-12), 'g-', linewidth=2)
    axes[1, 0].axvline(FIR_CUTOFF * SAMPLE_RATE / 2, color='r', linestyle='--',
                        label=f'Cutoff = {FIR_CUTOFF * SAMPLE_RATE / 2:.0f} Hz')
    axes[1, 0].set_title('FIR Frequency Response')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].set_ylim([-80, 5])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Filter coefficients (impulse response)
    axes[1, 1].stem(range(len(taps)), taps, linefmt='b-', markerfmt='bo',
                     basefmt='k-')
    axes[1, 1].set_title(f'FIR Coefficients ({len(taps)} taps)')
    axes[1, 1].set_xlabel('Tap index')
    axes[1, 1].set_ylabel('h[k]')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    out_dir = os.path.join(PROJECT_ROOT, 'Results', 'Plots', 'filters')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'test_filters_stage1.png'), dpi=150)
    print(f"Plot saved to {os.path.join(out_dir, 'test_filters_stage1.png')}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" Filters Stage 1 Test: scipy -> GPU")
    print("=" * 60)

    test_fir_basic_properties()
    test_fir_single_channel()
    test_fir_gpu_vs_scipy()
    test_iir_basic_properties()
    test_iir_gpu_vs_scipy()

    print("\n--- Generating FIR plot ---")
    plot_filter_results()

    print("\nAll tests passed!")
