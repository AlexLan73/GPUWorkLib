"""
GPUWorkLib: SpectrumMaximaFinder.find_all_maxima() Test Suite
==============================================================
Tests for GPU-accelerated finding of ALL local maxima in FFT spectrum.
Generates matplotlib plots for visual verification.

Tests:
  1. Single tone -> 1 main peak at expected frequency + plot
  2. Three tones -> 3 main peaks, compare with SciPy find_peaks + plot
  3. Multi-beam -> 5 beams with different frequencies + plot
  4. GPU vs SciPy: exact match of peak positions
  5. Performance: GPU time for large batch (256 beams)

Output plots saved to: Results/Plots/

Author: Kodo (AI Assistant)
Date: 2026-02-14
"""

import sys
import os
import time
import numpy as np

# Add gpuworklib path (Python_test/fft_maxima/ -> 2 levels up to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for subdir in ["build/python/Release", "build/python/Debug", "build/Release", "build/Debug"]:
    path = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(path):
        sys.path.insert(0, path)
        break
import gpuworklib

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional: SciPy for comparison
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  [WARNING] SciPy not found, skipping comparison tests")

# Output directory for plots: Results/Plots/fft_maxima/
PLOT_DIR = os.path.join(PROJECT_ROOT, "Results", "Plots", "fft_maxima")
os.makedirs(PLOT_DIR, exist_ok=True)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")


# ============================================================================
# Test 1: Single tone -> single main peak + plot
# ============================================================================
def test_single_tone():
    print_header("Test 1: Single Tone -> Find All Maxima + Plot")

    ctx = gpuworklib.GPUContext(0)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    print(f"  GPU: {ctx.device_name}")

    fs = 1000.0
    nFFT = 1024
    freq = 100.0

    # Generate signal
    t = np.arange(nFFT, dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t / fs).astype(np.complex64)

    # FFT on GPU
    spectrum = fft.process_complex(signal, sample_rate=fs)

    # Find all maxima
    result = finder.find_all_maxima(spectrum, sample_rate=fs)

    num = result["num_maxima"]
    positions = result["positions"]
    frequencies = result["frequencies"]
    magnitudes = result["magnitudes"]

    print(f"  Found {num} maxima")
    for i in range(min(num, 5)):
        print(f"    bin={positions[i]}, freq={frequencies[i]:.2f} Hz, mag={magnitudes[i]:.2f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Test 1: Single Tone ({freq:.0f} Hz) - FindAllMaxima", fontsize=14, fontweight='bold')

    # Time domain
    ax1.plot(t[:200], signal.real[:200], 'b-', linewidth=0.8)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Time Domain Signal (f={freq:.0f} Hz)")
    ax1.grid(True, alpha=0.3)

    # Frequency domain with peaks
    freq_axis = np.arange(nFFT // 2) * fs / nFFT
    mag_half = np.abs(spectrum[:nFFT // 2].astype(np.complex128))

    ax2.plot(freq_axis, mag_half, 'b-', linewidth=0.8, label='Spectrum')
    # Mark GPU peaks
    peak_mask = positions < nFFT // 2
    ax2.plot(frequencies[peak_mask], magnitudes[peak_mask], 'rv', markersize=10,
             label=f'GPU peaks ({num})', markeredgecolor='darkred')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("FFT Spectrum + GPU-detected Maxima")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test1_single_tone.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/test1_single_tone.png")

    # Check: expected bin for 100 Hz
    expected_bin = int(freq * nFFT / fs + 0.5)
    found_main = any(abs(int(p) - expected_bin) <= 1 for p in positions)

    passed = found_main and num > 0
    print_result("Single tone peak found", passed)
    return passed


# ============================================================================
# Test 2: Three tones -> three main peaks + plot
# ============================================================================
def test_three_tones():
    print_header("Test 2: Three Tones -> Find All Maxima + Plot")

    ctx = gpuworklib.GPUContext(0)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    fs = 1000.0
    nFFT = 1024
    freqs = [50.0, 120.0, 200.0]

    # Generate signal with 3 frequencies
    t = np.arange(nFFT, dtype=np.float32)
    signal = np.zeros(nFFT, dtype=np.complex64)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * t / fs).astype(np.complex64)

    # FFT on GPU
    spectrum = fft.process_complex(signal, sample_rate=fs)

    # Find all maxima
    result = finder.find_all_maxima(spectrum, sample_rate=fs)

    num = result["num_maxima"]
    positions = result["positions"]
    frequencies_found = result["frequencies"]
    magnitudes = result["magnitudes"]

    print(f"  Found {num} maxima total")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle("Test 2: Three Tones (50, 120, 200 Hz) - FindAllMaxima", fontsize=14, fontweight='bold')

    freq_axis = np.arange(nFFT // 2) * fs / nFFT
    mag_half = np.abs(spectrum[:nFFT // 2].astype(np.complex128))

    ax.plot(freq_axis, mag_half, 'b-', linewidth=0.8, label='Spectrum')

    # Mark GPU peaks
    peak_mask = positions < nFFT // 2
    ax.plot(frequencies_found[peak_mask], magnitudes[peak_mask], 'rv', markersize=10,
            label=f'GPU peaks ({num})', markeredgecolor='darkred')

    # Mark expected frequencies
    for f in freqs:
        ax.axvline(x=f, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=freqs[0], color='green', linestyle='--', alpha=0.5, linewidth=1, label='Expected freq')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, fs / 2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test2_three_tones.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/test2_three_tones.png")

    # Check each expected frequency
    all_found = True
    for ef in freqs:
        expected_bin = int(ef * nFFT / fs + 0.5)
        found = any(abs(int(p) - expected_bin) <= 1 for p in positions)
        print(f"    {ef:.0f} Hz (bin ~{expected_bin}): {'OK' if found else 'MISSING'}")
        if not found:
            all_found = False

    passed = all_found and num >= 3
    print_result("All three peaks found", passed)
    return passed


# ============================================================================
# Test 3: Multi-beam (5 beams with different frequencies) + plot
# ============================================================================
def test_multi_beam():
    print_header("Test 3: Multi-beam (5 beams) + Plot")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    fs = 1000.0
    length = 512
    f0 = 50.0
    freq_step = 50.0
    beam_count = 5
    beam_freqs = [f0 + i * freq_step for i in range(beam_count)]

    # Generate multi-beam signal using GPU
    signals = sig.generate_cw(freq=f0, fs=fs, length=length,
                               beam_count=beam_count, freq_step=freq_step)

    # FFT on GPU
    spectra = fft.process_complex(signals, sample_rate=fs)

    # Find all maxima (2D input)
    result = finder.find_all_maxima(spectra, sample_rate=fs)

    print(f"  {beam_count} beams processed")

    # --- Plot ---
    fig, axes = plt.subplots(beam_count, 1, figsize=(12, 3 * beam_count))
    fig.suptitle("Test 3: Multi-beam FindAllMaxima (5 beams, 50-250 Hz)",
                 fontsize=14, fontweight='bold')

    nFFT = spectra.shape[1] if spectra.ndim == 2 else len(spectra)
    freq_axis = np.arange(nFFT // 2) * fs / nFFT

    all_ok = True
    for b in range(beam_count):
        beam = result[b]
        expected_bin = int(beam_freqs[b] * length / fs + 0.5)
        found = any(abs(int(p) - expected_bin) <= 1 for p in beam["positions"])

        # Plot
        ax = axes[b]
        spec = spectra[b] if spectra.ndim == 2 else spectra
        mag = np.abs(spec[:nFFT // 2].astype(np.complex128))
        ax.plot(freq_axis, mag, 'b-', linewidth=0.8)

        peak_mask = beam["positions"] < nFFT // 2
        ax.plot(beam["frequencies"][peak_mask], beam["magnitudes"][peak_mask],
                'rv', markersize=8)
        ax.axvline(x=beam_freqs[b], color='green', linestyle='--', alpha=0.7)
        ax.set_ylabel("Mag")
        ax.set_title(f"Beam {b}: f={beam_freqs[b]:.0f} Hz, {beam['num_maxima']} peaks "
                     f"{'OK' if found else 'MISSING'}", fontsize=10)
        ax.grid(True, alpha=0.3)

        print(f"    Beam {b} ({beam_freqs[b]:.0f} Hz): {beam['num_maxima']} maxima, "
              f"expected_bin={expected_bin} -> {'OK' if found else 'MISSING'}")
        if not found:
            all_ok = False

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test3_multi_beam.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/test3_multi_beam.png")

    passed = all_ok
    print_result("All beams correct", passed)
    return passed


# ============================================================================
# Test 4: Compare GPU with SciPy find_peaks() + plot
# ============================================================================
def test_vs_scipy():
    if not HAS_SCIPY:
        print_header("Test 4: GPU vs SciPy (SKIPPED - no SciPy)")
        print("  [SKIP] SciPy not installed")
        return True

    print_header("Test 4: GPU vs SciPy find_peaks() + Plot")

    ctx = gpuworklib.GPUContext(0)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    fs = 1000.0
    nFFT = 1024

    # Signal with 3 frequencies
    t = np.arange(nFFT, dtype=np.float32)
    signal = np.zeros(nFFT, dtype=np.complex64)
    for f in [50.0, 120.0, 200.0]:
        signal += np.sin(2 * np.pi * f * t / fs).astype(np.complex64)

    # FFT on GPU
    spectrum = fft.process_complex(signal, sample_rate=fs)

    # GPU: find all maxima
    gpu_result = finder.find_all_maxima(spectrum, sample_rate=fs)
    gpu_positions = set(int(p) for p in gpu_result["positions"])

    # SciPy: find_peaks on magnitude spectrum (first half only)
    magnitude = np.abs(spectrum[:nFFT // 2].astype(np.complex128))
    scipy_peaks, _ = find_peaks(magnitude)
    scipy_positions = set(int(p) for p in scipy_peaks)

    # Compare
    common = gpu_positions & scipy_positions
    gpu_only = gpu_positions - scipy_positions
    scipy_only = scipy_positions - gpu_positions

    print(f"  GPU found: {len(gpu_positions)} maxima")
    print(f"  SciPy found: {len(scipy_positions)} maxima")
    print(f"  Common: {len(common)}")
    if gpu_only:
        print(f"  GPU only: {sorted(gpu_only)}")
    if scipy_only:
        print(f"  SciPy only: {sorted(scipy_only)}")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle("Test 4: GPU FindAllMaxima vs SciPy find_peaks()", fontsize=14, fontweight='bold')

    freq_axis = np.arange(nFFT // 2) * fs / nFFT
    ax.plot(freq_axis, magnitude, 'b-', linewidth=0.8, label='Spectrum')

    # GPU peaks
    gpu_freqs = gpu_result["frequencies"]
    gpu_mags = gpu_result["magnitudes"]
    peak_mask = gpu_result["positions"] < nFFT // 2
    ax.plot(gpu_freqs[peak_mask], gpu_mags[peak_mask], 'rv', markersize=12,
            label=f'GPU ({len(gpu_positions)})', markeredgecolor='darkred', zorder=5)

    # SciPy peaks
    scipy_freqs = scipy_peaks * fs / nFFT
    ax.plot(scipy_freqs, magnitude[scipy_peaks], 'g^', markersize=10,
            label=f'SciPy ({len(scipy_positions)})', markeredgecolor='darkgreen',
            markerfacecolor='none', linewidth=2, zorder=4)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, fs / 2)

    # Annotation
    match_text = f"Match: {len(common)}/{len(scipy_positions)} (100%)" if len(common) == len(scipy_positions) \
        else f"Match: {len(common)}/{len(scipy_positions)}"
    ax.text(0.98, 0.95, match_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if len(common) == len(scipy_positions) else 'lightyellow'))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test4_gpu_vs_scipy.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/test4_gpu_vs_scipy.png")

    # They should find the same peaks (within search range)
    passed = len(common) == len(scipy_positions) and len(scipy_only) == 0
    print_result(f"GPU matches SciPy ({len(common)}/{len(scipy_positions)} peaks)", passed)
    return passed


# ============================================================================
# Test 5: Performance benchmark (256 beams) + timing plot
# ============================================================================
def test_performance():
    print_header("Test 5: Performance (256 beams x 4096 FFT)")

    ctx = gpuworklib.GPUContext(0)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    fs = 10000.0
    nFFT = 4096
    beam_count = 256

    # Generate random multi-beam signal
    np.random.seed(42)
    signal = (np.random.randn(beam_count, nFFT) +
              1j * np.random.randn(beam_count, nFFT)).astype(np.complex64)

    # FFT
    spectra = fft.process_complex(signal, sample_rate=fs)

    # Warm-up
    _ = finder.find_all_maxima(spectra, sample_rate=fs)

    # Benchmark
    N_RUNS = 10
    times = []
    total_maxima = 0
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = finder.find_all_maxima(spectra, sample_rate=fs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
        total_maxima = sum(b["num_maxima"] for b in result)

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    std_ms = np.std(times)

    print(f"  {beam_count} beams x {nFFT} FFT")
    print(f"  Total maxima found: {total_maxima}")
    print(f"  Time: avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms, std={std_ms:.2f}ms")
    print(f"  Per beam: {avg_ms / beam_count:.3f} ms")

    # --- Plot: timing histogram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Test 5: Performance ({beam_count} beams x {nFFT} FFT)", fontsize=14, fontweight='bold')

    ax1.bar(range(N_RUNS), times, color='steelblue', edgecolor='navy', alpha=0.8)
    ax1.axhline(y=avg_ms, color='red', linestyle='--', linewidth=1.5, label=f'avg={avg_ms:.1f}ms')
    ax1.set_xlabel("Run #")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Execution Time per Run")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Maxima per beam distribution
    maxima_per_beam = [b["num_maxima"] for b in result]
    ax2.hist(maxima_per_beam, bins=30, color='coral', edgecolor='darkred', alpha=0.8)
    ax2.set_xlabel("Maxima count per beam")
    ax2.set_ylabel("Beam count")
    ax2.set_title(f"Maxima Distribution (total={total_maxima})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test5_performance.png"), dpi=150)
    plt.close()
    print(f"  Plot saved: {PLOT_DIR}/test5_performance.png")

    passed = avg_ms < 500  # should be well under 500ms
    print_result(f"Performance OK ({avg_ms:.1f}ms < 500ms)", passed)
    return passed


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SpectrumMaximaFinder: find_all_maxima() Tests")
    print("  Plots will be saved to: " + PLOT_DIR)
    print("=" * 60)

    tests = [
        ("Single tone", test_single_tone),
        ("Three tones", test_three_tones),
        ("Multi-beam", test_multi_beam),
        ("GPU vs SciPy", test_vs_scipy),
        ("Performance", test_performance),
    ]

    passed = 0
    total = len(tests)

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"  Plots saved to: {PLOT_DIR}")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)
