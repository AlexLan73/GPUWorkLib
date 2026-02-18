"""
IIR (Butterworth) Filter — Beautiful 4-Panel GPU Plot
=====================================================

Panels:
  1. Input signal (CW 100Hz + CW 5000Hz) — Re/Im
  2. IIR filtered signal (GPU Biquad Cascade) — Re/Im
  3. Frequency response (magnitude dB + phase deg)
  4. Pole-Zero diagram with unit circle

Usage:
  python Python_test/filters/test_iir_plot.py          (standalone)
  pytest Python_test/filters/test_iir_plot.py -v        (pytest)

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
IIR_ORDER = 2
IIR_CUTOFF = 0.1    # Normalized (0-1, Nyquist=1)


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
    """IIR: GPU result matches scipy.sosfilt reference"""
    if not HAS_GPU or not HAS_SCIPY:
        print("SKIP: missing gpuworklib or scipy")
        return

    sos = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)

    sections = []
    for row in sos:
        sections.append({
            'b0': float(row[0]), 'b1': float(row[1]), 'b2': float(row[2]),
            'a1': float(row[4]), 'a2': float(row[5]),
        })

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    signal = generate_test_signal(CHANNELS, POINTS, SAMPLE_RATE)
    result_gpu = iir.process(signal)

    result_ref = np.zeros_like(signal)
    for ch in range(CHANNELS):
        result_ref[ch] = sig.sosfilt(sos, signal[ch]).astype(np.complex64)

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
# 4-Panel IIR Plot
# ============================================================================

def plot_iir_results():
    """Generate beautiful 4-panel plot for IIR (Butterworth) filter"""
    if not (HAS_GPU and HAS_SCIPY and HAS_PLOT):
        print("SKIP: missing dependencies for IIR plotting")
        return

    # ── Design IIR Butterworth LP ──
    sos = sig.butter(IIR_ORDER, IIR_CUTOFF, output='sos').astype(np.float64)

    sections = []
    for row in sos:
        sections.append({
            'b0': float(row[0]), 'b1': float(row[1]), 'b2': float(row[2]),
            'a1': float(row[4]), 'a2': float(row[5]),
        })

    ctx = gw.GPUContext(0)
    iir = gw.IirFilter(ctx)
    iir.set_sections(sections)

    signal = generate_test_signal(1, POINTS, SAMPLE_RATE)[0]
    result = iir.process(signal)

    t = np.arange(POINTS) / SAMPLE_RATE * 1000  # ms
    cutoff_hz = IIR_CUTOFF * SAMPLE_RATE / 2

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f'GPU IIR Butterworth Filter (order={IIR_ORDER}, '
        f'fc={cutoff_hz:.0f} Hz, fs={SAMPLE_RATE/1000:.0f} kHz)',
        fontsize=14, fontweight='bold')

    # ── 1. Input signal ──
    axes[0, 0].plot(t[:500], signal.real[:500], 'b-', alpha=0.7, linewidth=0.8, label='Re')
    axes[0, 0].plot(t[:500], signal.imag[:500], 'r-', alpha=0.5, linewidth=0.8, label='Im')
    axes[0, 0].set_title('Input Signal (first 500 samples)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # ── 2. Filtered signal (GPU) ──
    axes[0, 1].plot(t[:500], result.real[:500], 'b-', alpha=0.7, linewidth=0.8, label='Re')
    axes[0, 1].plot(t[:500], result.imag[:500], 'r-', alpha=0.5, linewidth=0.8, label='Im')
    axes[0, 1].set_title('IIR Filtered Signal — GPU (first 500 samples)')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    # ── 3. Frequency response (magnitude + phase) ──
    w, h_freq = sig.sosfreqz(sos, worN=2048)
    freq_hz = w / np.pi * (SAMPLE_RATE / 2)
    mag_db = 20 * np.log10(np.abs(h_freq) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(h_freq)))

    ax_mag = axes[1, 0]
    color_mag = '#2196F3'
    ax_mag.plot(freq_hz, mag_db, color=color_mag, linewidth=2, label='|H(f)| dB')
    ax_mag.axvline(cutoff_hz, color='r', linestyle='--', linewidth=1.5,
                   label=f'Cutoff = {cutoff_hz:.0f} Hz')
    ax_mag.axhline(-3, color='gray', linestyle=':', alpha=0.5, label='-3 dB')
    ax_mag.set_title('IIR Frequency Response')
    ax_mag.set_xlabel('Frequency (Hz)')
    ax_mag.set_ylabel('Magnitude (dB)', color=color_mag)
    ax_mag.tick_params(axis='y', labelcolor=color_mag)
    ax_mag.set_ylim([-80, 5])
    ax_mag.legend(loc='lower left', fontsize=8)
    ax_mag.grid(True, alpha=0.3)

    # Phase on secondary axis
    ax_phase = ax_mag.twinx()
    color_phase = '#FF9800'
    ax_phase.plot(freq_hz, phase_deg, color=color_phase, linewidth=1, alpha=0.7)
    ax_phase.set_ylabel('Phase (deg)', color=color_phase)
    ax_phase.tick_params(axis='y', labelcolor=color_phase)

    # ── 4. Pole-Zero diagram ──
    ax_pz = axes[1, 1]

    # Extract zeros and poles from SOS
    z_all = np.array([], dtype=complex)
    p_all = np.array([], dtype=complex)
    for row in sos:
        b_sec = row[:3]
        a_sec = row[3:]
        z_sec = np.roots(b_sec)
        p_sec = np.roots(a_sec)
        z_all = np.concatenate([z_all, z_sec])
        p_all = np.concatenate([p_all, p_sec])

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 256)
    ax_pz.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1, alpha=0.3)

    # Zeros (o) and Poles (x)
    ax_pz.plot(z_all.real, z_all.imag, 'bo', markersize=10, markerfacecolor='none',
               markeredgewidth=2, label=f'Zeros ({len(z_all)})', zorder=5)
    ax_pz.plot(p_all.real, p_all.imag, 'rx', markersize=12,
               markeredgewidth=2.5, label=f'Poles ({len(p_all)})', zorder=5)

    # Styling
    ax_pz.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax_pz.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax_pz.set_title(f'Pole-Zero Plot (Butterworth order {IIR_ORDER})')
    ax_pz.set_xlabel('Real')
    ax_pz.set_ylabel('Imaginary')
    ax_pz.set_aspect('equal')
    ax_pz.legend(loc='upper left', fontsize=9)
    ax_pz.grid(True, alpha=0.3)

    # Zoom to show details around unit circle
    lim = max(1.3, np.max(np.abs(p_all)) * 1.3, np.max(np.abs(z_all)) * 1.3)
    ax_pz.set_xlim([-lim, lim])
    ax_pz.set_ylim([-lim, lim])

    plt.tight_layout()

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
    print(" IIR (Butterworth) Filter Test + Plot")
    print("=" * 60)

    test_iir_basic_properties()
    test_iir_gpu_vs_scipy()

    print("\n--- Generating IIR 4-panel plot ---")
    plot_iir_results()

    print("\nDone!")
