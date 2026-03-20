"""
IIR (Butterworth) Filter — Beautiful 4-Panel GPU Plot
=====================================================

Panels:
  1. Input signal (CW 100Hz + CW 5000Hz) — Re/Im
  2. IIR filtered signal (GPU Biquad Cascade, order=8) — Re/Im
  3. Frequency response COMPARISON: order 2, 4, 8 (magnitude dB)
  4. Pole-Zero diagram (order=8) with unit circle

Usage:
  python Python_test/filters/test_iir_plot.py

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
# Imports
# ============================================================================
try:
    import gpuworklib as gw
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: matplotlib not found.")


# ============================================================================
# Parameters
# ============================================================================
CHANNELS = 8
POINTS = 4096
SAMPLE_RATE = 50000.0
F_LOW = 100.0       # Hz - passes through LP
F_HIGH = 5000.0     # Hz - attenuated by LP
IIR_ORDER = 8       # Main filter order (was 2, now 8 for proper filtering)
IIR_CUTOFF = 0.1    # Normalized (0-1, Nyquist=1) => 2500 Hz


def sos_to_sections(sos):
    """Convert scipy SOS matrix to list of dicts for gpuworklib"""
    sections = []
    for row in sos:
        sections.append({
            'b0': float(row[0]), 'b1': float(row[1]), 'b2': float(row[2]),
            'a1': float(row[4]), 'a2': float(row[5]),
        })
    return sections


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
# IIR Tests
# ============================================================================

def test_iir_gpu_vs_scipy():
    """IIR: GPU result matches scipy.sosfilt reference (order=8)"""
    if not HAS_GPU or not HAS_SCIPY:
        print("SKIP: missing gpuworklib or scipy")
        return

    sos = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    signal = generate_test_signal(CHANNELS, POINTS, SAMPLE_RATE)
    result_gpu = iir.process(signal)

    result_ref = np.zeros_like(signal)
    for ch in range(CHANNELS):
        result_ref[ch] = sig.sosfilt(sos, signal[ch]).astype(np.complex64)

    max_err = np.max(np.abs(result_gpu - result_ref))
    print(f"IIR (order={IIR_ORDER}) GPU vs scipy: max_error = {max_err:.2e}")
    assert max_err < 5e-2, f"IIR error too large: {max_err}"
    print("  PASSED")


def test_iir_basic_properties():
    """IIR: basic property checks (order=8 => 4 biquad sections)"""
    if not HAS_GPU or not HAS_SCIPY:
        print("SKIP: gpuworklib not available")
        return

    sos = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos)

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    expected_sections = IIR_ORDER // 2  # Each biquad = 2nd order
    assert iir.num_sections == expected_sections
    print(f"IIR properties: order={IIR_ORDER}, "
          f"biquad_sections={iir.num_sections}, repr={repr(iir)}")
    print("  PASSED")


# ============================================================================
# 4-Panel IIR Plot
# ============================================================================

def plot_iir_results():
    """Generate beautiful 4-panel plot for IIR (Butterworth) filter

    Panel 1: Input signal
    Panel 2: Filtered signal (GPU, order=8)
    Panel 3: Frequency response COMPARISON: order 2, 4, 8
    Panel 4: Pole-Zero diagram (order=8)
    """
    if not (HAS_GPU and HAS_SCIPY and HAS_PLOT):
        print("SKIP: missing dependencies for IIR plotting")
        return

    cutoff_hz = IIR_CUTOFF * SAMPLE_RATE / 2
    t = np.arange(POINTS) / SAMPLE_RATE * 1000  # ms

    # ── Design main filter (order=8) ──
    sos_main = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)
    sections = sos_to_sections(sos_main)

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    signal = generate_test_signal(1, POINTS, SAMPLE_RATE)[0]
    result = iir.process(signal)

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f'GPU IIR Butterworth Low-Pass (fc={cutoff_hz:.0f} Hz, '
        f'fs={SAMPLE_RATE/1000:.0f} kHz)',
        fontsize=15, fontweight='bold', y=0.98)

    # ── 1. Input signal ──
    ax1 = axes[0, 0]
    ax1.plot(t[:500], signal.real[:500], '#2196F3', alpha=0.8, linewidth=0.8, label='Re')
    ax1.plot(t[:500], signal.imag[:500], '#F44336', alpha=0.6, linewidth=0.8, label='Im')
    ax1.set_title('Input: CW 100 Hz + CW 5000 Hz', fontsize=11)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t[499]])

    # ── 2. Filtered signal (GPU, order=8) ──
    ax2 = axes[0, 1]
    ax2.plot(t[:500], result.real[:500], '#2196F3', alpha=0.8, linewidth=0.8, label='Re')
    ax2.plot(t[:500], result.imag[:500], '#F44336', alpha=0.6, linewidth=0.8, label='Im')
    ax2.set_title(f'IIR Filtered — GPU (order={IIR_ORDER}, {IIR_ORDER//2} biquads)',
                  fontsize=11)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t[499]])

    # ── 3. АЧХ comparison: order 2, 4, 8 ──
    ax3 = axes[1, 0]
    colors = ['#FF9800', '#4CAF50', '#2196F3']
    orders = [2, 4, 8]

    for order, color in zip(orders, colors):
        sos_cmp = sig.butter(order, IIR_CUTOFF, output='sos')
        w, h_freq = sig.sosfreqz(sos_cmp, worN=2048)
        freq_hz = w / np.pi * (SAMPLE_RATE / 2)
        mag_db = 20 * np.log10(np.abs(h_freq) + 1e-12)
        lw = 2.5 if order == IIR_ORDER else 1.5
        alpha = 1.0 if order == IIR_ORDER else 0.7
        slope = order * 6  # dB/octave: N * 6
        ax3.plot(freq_hz, mag_db, color=color, linewidth=lw, alpha=alpha,
                 label=f'order {order}  ({slope} dB/oct)')

    ax3.axvline(cutoff_hz, color='r', linestyle='--', linewidth=1.2, alpha=0.7,
                label=f'fc = {cutoff_hz:.0f} Hz')
    ax3.axhline(-3, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax3.text(cutoff_hz + 200, -3 + 1.5, '-3 dB', fontsize=8, color='gray')

    ax3.set_title('Frequency Response: Order Comparison', fontsize=11)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_ylim([-80, 5])
    ax3.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # ── 4. Pole-Zero diagram (order=8) ──
    ax4 = axes[1, 1]

    # Extract zeros and poles
    z_all = np.array([], dtype=complex)
    p_all = np.array([], dtype=complex)
    for row in sos_main:
        z_sec = np.roots(row[:3])
        p_sec = np.roots(row[3:])
        z_all = np.concatenate([z_all, z_sec])
        p_all = np.concatenate([p_all, p_sec])

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 256)
    ax4.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.2, alpha=0.25)
    ax4.fill(np.cos(theta), np.sin(theta), color='#E3F2FD', alpha=0.3)

    # Zeros (o) — blue circles
    ax4.plot(z_all.real, z_all.imag, 'o', color='#1565C0', markersize=9,
             markerfacecolor='none', markeredgewidth=2,
             label=f'Zeros ({len(z_all)})', zorder=5)

    # Poles (x) — red crosses
    ax4.plot(p_all.real, p_all.imag, 'x', color='#C62828', markersize=11,
             markeredgewidth=2.5, label=f'Poles ({len(p_all)})', zorder=5)

    # Axes
    ax4.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax4.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax4.set_title(f'Pole-Zero Plot (Butterworth order {IIR_ORDER})', fontsize=11)
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Imaginary')
    ax4.set_aspect('equal')
    ax4.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    lim = max(1.3, np.max(np.abs(p_all)) * 1.3, np.max(np.abs(z_all)) * 1.3)
    ax4.set_xlim([-lim, lim])
    ax4.set_ylim([-lim, lim])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    out_dir = os.path.join(PROJECT_ROOT, 'Results', 'Plots', 'filters')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'test_iir_stage1.png')
    plt.savefig(out_path, dpi=150)
    print(f"IIR plot saved to {out_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(f" IIR Butterworth Filter Test (order={IIR_ORDER})")
    print("=" * 60)

    test_iir_basic_properties()
    test_iir_gpu_vs_scipy()

    print("\n--- Generating IIR 4-panel plot ---")
    plot_iir_results()

    print("\nDone!")
