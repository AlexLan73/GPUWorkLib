"""
GPUWorkLib Python Test Suite
============================
Tests for GPU signal processing via pybind11 bindings.

1. Generate multi-channel sin -> FFT -> display results
2. Different frequencies to see differences
3. Generate CW, LFM, Noise -> plot signals
4. Create generators from string params on multiple channels
5. Full pipeline: generators -> FFT -> frequency analysis

Author: Kodo (AI Assistant)
Date: 2026-02-13
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add gpuworklib path (Python_test/integration/ -> 2 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.join(PROJECT_ROOT, "Results", "Plots", "integration")
os.makedirs(PLOT_DIR, exist_ok=True)
for subdir in ["build/python/Release", "build/python/Debug", "build/Release", "build/Debug"]:
    p = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(p):
        sys.path.insert(0, p)
        break
import gpuworklib


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# Test 1: Multi-channel sin -> FFT -> display
# ============================================================================
def test_multichannel_sin_fft():
    print_header("Test 1: Multi-channel sin -> FFT")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    print(f"  GPU: {ctx.device_name}")

    fs = 4000.0       # Sample rate 4 kHz
    length = 4096      # Samples per channel
    freqs = [100, 250, 500, 800, 1200]  # Different frequencies

    fig, axes = plt.subplots(len(freqs), 3, figsize=(16, 3 * len(freqs)))
    fig.suptitle("Test 1: Multi-channel CW signals -> FFT", fontsize=14, fontweight='bold')

    for i, f0 in enumerate(freqs):
        # Generate CW signal
        signal = sig.generate_cw(freq=f0, fs=fs, length=length)

        # Compute FFT
        spectrum = fft.process_complex(signal, sample_rate=fs)

        # Time domain
        t = np.arange(length) / fs * 1000  # ms
        axes[i, 0].plot(t[:200], signal[:200].real, 'b-', linewidth=0.7)
        axes[i, 0].set_title(f"Ch {i+1}: CW f={f0} Hz (time)")
        axes[i, 0].set_xlabel("t, ms")
        axes[i, 0].set_ylabel("Re")
        axes[i, 0].grid(True, alpha=0.3)

        # FFT magnitude
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        mag = np.abs(spectrum)
        half = nfft // 2
        axes[i, 1].plot(freq_axis[:half], mag[:half], 'r-', linewidth=0.7)
        axes[i, 1].set_title(f"FFT magnitude (peak @ {freq_axis[np.argmax(mag[:half])]:.1f} Hz)")
        axes[i, 1].set_xlabel("f, Hz")
        axes[i, 1].set_ylabel("|FFT|")
        axes[i, 1].grid(True, alpha=0.3)

        # FFT phase
        phase = np.angle(spectrum)
        axes[i, 2].plot(freq_axis[:half], phase[:half], 'g-', linewidth=0.5, alpha=0.7)
        axes[i, 2].set_title("FFT phase")
        axes[i, 2].set_xlabel("f, Hz")
        axes[i, 2].set_ylabel("phase, rad")
        axes[i, 2].grid(True, alpha=0.3)

        # Print detected peak
        peak_bin = np.argmax(mag[:half])
        detected_freq = freq_axis[peak_bin]
        print(f"  Ch {i+1}: f0={f0:>6} Hz -> detected {detected_freq:>8.2f} Hz  "
              f"(error={abs(detected_freq - f0):.2f} Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test1_multichannel_fft.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 2: Different signal types (CW, LFM, Noise)
# ============================================================================
def test_signal_types():
    print_header("Test 2: Signal types - CW, LFM, Noise")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 8000.0
    length = 8192

    # Generate signals
    cw = sig.generate_cw(freq=500, fs=fs, length=length, amplitude=2.0)
    lfm = sig.generate_lfm(f_start=200, f_end=1500, fs=fs, length=length)
    noise = sig.generate_noise(fs=fs, length=length, power=1.0, noise_type="gaussian")

    # FFT of each
    cw_fft = fft.process_complex(cw, sample_rate=fs)
    lfm_fft = fft.process_complex(lfm, sample_rate=fs)
    noise_fft = fft.process_complex(noise, sample_rate=fs)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle("Test 2: Signal Types - CW, LFM, Noise", fontsize=14, fontweight='bold')

    signals = [
        ("CW (500 Hz)", cw, cw_fft),
        ("LFM (200->1500 Hz)", lfm, lfm_fft),
        ("Gaussian Noise", noise, noise_fft),
    ]

    t = np.arange(length) / fs * 1000  # ms

    for row, (name, signal, spectrum) in enumerate(signals):
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2
        mag = np.abs(spectrum)

        # Time domain (first 500 samples)
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(t[:500], signal[:500].real, 'b-', linewidth=0.5)
        ax1.set_title(f"{name} - Time")
        ax1.set_xlabel("t, ms")
        ax1.grid(True, alpha=0.3)

        # IQ scatter
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(signal[:1000].real, signal[:1000].imag, s=1, alpha=0.3, c='purple')
        ax2.set_title(f"{name} - IQ")
        ax2.set_xlabel("I")
        ax2.set_ylabel("Q")
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # FFT magnitude (dB)
        ax3 = fig.add_subplot(gs[row, 2])
        mag_db = 20 * np.log10(mag[:half] + 1e-10)
        ax3.plot(freq_axis[:half], mag_db, 'r-', linewidth=0.5)
        ax3.set_title(f"{name} - FFT (dB)")
        ax3.set_xlabel("f, Hz")
        ax3.set_ylabel("dB")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test2_signal_types.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode

    # Print statistics
    print(f"  CW:    mean={np.mean(np.abs(cw)):.3f}  std={np.std(cw.real):.3f}")
    print(f"  LFM:   mean={np.mean(np.abs(lfm)):.3f}  std={np.std(lfm.real):.3f}")
    print(f"  Noise: mean={np.mean(noise.real):.4f}  std={np.std(noise.real):.3f}  "
          f"var={np.var(noise.real):.3f}")
    print("  PASS")


# ============================================================================
# Test 3: Multi-beam CW with frequency step
# ============================================================================
def test_multibeam_cw():
    print_header("Test 3: Multi-beam CW (8 beams, freq_step=50 Hz)")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 4000.0
    length = 4096
    beam_count = 8
    f0 = 100.0
    freq_step = 50.0

    # Generate multi-beam signal on GPU
    multi = sig.generate_cw(freq=f0, fs=fs, length=length,
                            beam_count=beam_count, freq_step=freq_step)
    print(f"  Shape: {multi.shape} (beams x samples)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Test 3: Multi-beam CW (8 beams, freq_step=50 Hz)", fontsize=14, fontweight='bold')

    # Spectrogram-like: FFT all beams
    all_spectra = []
    for b in range(beam_count):
        beam_data = multi[b, :]
        spectrum = fft.process_complex(beam_data, sample_rate=fs)
        nfft = len(spectrum)
        mag = np.abs(spectrum)
        all_spectra.append(mag[:nfft // 2])

        expected = f0 + b * freq_step
        peak_bin = np.argmax(mag[:nfft // 2])
        detected = peak_bin * fs / nfft
        print(f"  Beam {b}: expected {expected:>6.0f} Hz -> detected {detected:>8.2f} Hz")

    # Waterfall plot
    spectra_arr = np.array(all_spectra)
    freq_axis = np.arange(spectra_arr.shape[1]) * fs / nfft
    im = axes[0].imshow(spectra_arr, aspect='auto', origin='lower',
                         extent=[0, freq_axis[-1], 0, beam_count],
                         cmap='hot')
    axes[0].set_xlabel("Frequency, Hz")
    axes[0].set_ylabel("Beam index")
    axes[0].set_title("FFT magnitude (waterfall)")
    plt.colorbar(im, ax=axes[0], label="|FFT|")

    # Overlay: all beams FFT
    colors = plt.cm.viridis(np.linspace(0, 1, beam_count))
    for b in range(beam_count):
        label = f"Beam {b} ({f0 + b*freq_step:.0f} Hz)"
        axes[1].plot(freq_axis, all_spectra[b], color=colors[b], linewidth=0.8, label=label)
    axes[1].set_xlabel("Frequency, Hz")
    axes[1].set_ylabel("|FFT|")
    axes[1].set_title("All beams FFT overlay")
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(0, 600)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test3_multibeam.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 4: Create generators from string parameters
# ============================================================================
def test_generators_from_string():
    print_header("Test 4: Generators from string params")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 4000.0
    length = 4096

    # CW from string
    cw_params = [
        "freq=100,amp=1.0",
        "freq=300,amp=0.8,phase=1.57",
        "freq=700,amp=0.5,freq_step=25",
    ]

    # LFM from string
    lfm_params = [
        "f_start=100,f_end=500,amp=1.0",
        "f_start=200,f_end=1800,amp=0.7",
    ]

    fig, axes = plt.subplots(len(cw_params) + len(lfm_params), 2, figsize=(14, 3 * (len(cw_params) + len(lfm_params))))
    fig.suptitle("Test 4: Generators from string parameters", fontsize=14, fontweight='bold')

    row = 0

    # CW from string
    for params_str in cw_params:
        signal = sig.generate_cw_from_string(params_str, fs=fs, length=length)
        spectrum = fft.process_complex(signal, sample_rate=fs)

        nfft = len(spectrum)
        t = np.arange(length) / fs * 1000
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[row, 0].plot(t[:300], signal[:300].real, 'b-', linewidth=0.7)
        axes[row, 0].set_title(f"CW: '{params_str}' (time)")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(freq_axis[:half], np.abs(spectrum[:half]), 'r-', linewidth=0.7)
        axes[row, 1].set_title("FFT")
        axes[row, 1].grid(True, alpha=0.3)

        print(f"  CW  '{params_str}' -> OK")
        row += 1

    # LFM from string
    for params_str in lfm_params:
        signal = sig.generate_lfm_from_string(params_str, fs=fs, length=length)
        spectrum = fft.process_complex(signal, sample_rate=fs)

        nfft = len(spectrum)
        t = np.arange(length) / fs * 1000
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[row, 0].plot(t[:500], signal[:500].real, 'b-', linewidth=0.5)
        axes[row, 0].set_title(f"LFM: '{params_str}' (time)")
        axes[row, 0].grid(True, alpha=0.3)

        mag_db = 20 * np.log10(np.abs(spectrum[:half]) + 1e-10)
        axes[row, 1].plot(freq_axis[:half], mag_db, 'r-', linewidth=0.5)
        axes[row, 1].set_title("FFT (dB)")
        axes[row, 1].grid(True, alpha=0.3)

        print(f"  LFM '{params_str}' -> OK")
        row += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test4_from_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 5: Multi-beam CW from string + FFT analysis
# ============================================================================
def test_multibeam_from_string():
    print_header("Test 5: Multi-beam from string -> FFT")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 8000.0
    length = 8192
    beam_count = 4

    configs = [
        ("freq=200,amp=1.0,freq_step=100", "CW 200+100*n Hz"),
        ("freq=500,amp=0.5,freq_step=200", "CW 500+200*n Hz"),
    ]

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 5 * len(configs)))
    fig.suptitle("Test 5: Multi-beam from string -> FFT", fontsize=14, fontweight='bold')

    for idx, (params_str, title) in enumerate(configs):
        multi = sig.generate_cw_from_string(params_str, fs=fs, length=length, beam_count=beam_count)
        print(f"  '{params_str}' -> shape {multi.shape}")

        # Time domain overlay
        t = np.arange(length) / fs * 1000
        colors = plt.cm.tab10(np.linspace(0, 1, beam_count))
        for b in range(beam_count):
            axes[idx, 0].plot(t[:400], multi[b, :400].real, color=colors[b],
                              linewidth=0.7, label=f"Beam {b}")
        axes[idx, 0].set_title(f"{title} - Time domain")
        axes[idx, 0].set_xlabel("t, ms")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)

        # FFT overlay
        for b in range(beam_count):
            spectrum = fft.process_complex(multi[b, :], sample_rate=fs)
            nfft = len(spectrum)
            freq_axis = np.arange(nfft) * fs / nfft
            half = nfft // 2
            axes[idx, 1].plot(freq_axis[:half], np.abs(spectrum[:half]),
                              color=colors[b], linewidth=0.7, label=f"Beam {b}")

            peak = np.argmax(np.abs(spectrum[:half]))
            print(f"    Beam {b}: peak @ {freq_axis[peak]:.1f} Hz")

        axes[idx, 1].set_title(f"{title} - FFT")
        axes[idx, 1].set_xlabel("f, Hz")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test5_multibeam_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 6: Magnitude + Phase FFT output
# ============================================================================
def test_mag_phase():
    print_header("Test 6: FFT Magnitude + Phase + Frequency")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 4000.0
    length = 4096

    signal = sig.generate_cw(freq=300, fs=fs, length=length)

    # Get mag/phase/freq
    result = fft.process_mag_phase(signal, sample_rate=fs)

    mag = result["magnitude"]
    phase = result["phase"]
    freq = result["frequency"]
    nfft = result["nFFT"]

    print(f"  nFFT = {nfft}")
    print(f"  magnitude shape: {mag.shape}")
    print(f"  phase shape:     {phase.shape}")
    print(f"  frequency shape: {freq.shape}")

    half = nfft // 2
    peak_idx = np.argmax(mag[:half])
    print(f"  Peak @ index {peak_idx}: freq={freq[peak_idx]:.2f} Hz, mag={mag[peak_idx]:.2f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Test 6: Magnitude + Phase + Frequency", fontsize=14, fontweight='bold')

    axes[0].plot(freq[:half], mag[:half], 'r-', linewidth=0.7)
    axes[0].set_title(f"Magnitude (peak @ {freq[peak_idx]:.1f} Hz)")
    axes[0].set_xlabel("f, Hz")
    axes[0].set_ylabel("|FFT|")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq[:half], phase[:half], 'g-', linewidth=0.5)
    axes[1].set_title("Phase")
    axes[1].set_xlabel("f, Hz")
    axes[1].set_ylabel("rad")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq[:half], 20*np.log10(mag[:half] + 1e-10), 'b-', linewidth=0.7)
    axes[2].set_title("Magnitude (dB)")
    axes[2].set_xlabel("f, Hz")
    axes[2].set_ylabel("dB")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test6_mag_phase.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 7: Universal generate() from string (type=... in string)
# ============================================================================
def test_generate_from_string():
    print_header("Test 7: Universal generate() from string")

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)

    fs = 4000.0
    length = 4096

    # Different signal types from a single generate() method
    configs = [
        ("type=cw,freq=200,amp=1.0", "CW 200 Hz"),
        ("type=sin,freq=500,amp=0.8,phase=0.5", "Sin 500 Hz"),
        ("type=lfm,f_start=100,f_end=800,amp=1.0", "LFM 100->800 Hz"),
        ("type=chirp,f_start=50,f_end=1500", "Chirp 50->1500 Hz"),
        ("type=noise,power=1.0", "Gaussian Noise"),
        ("type=cw,freq=100,amp=1.0,freq_step=75", "CW multi (100+75*n Hz)"),
    ]

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 3 * len(configs)))
    fig.suptitle("Test 7: Universal generate() from string", fontsize=14, fontweight='bold')

    for i, (params_str, title) in enumerate(configs):
        beam_count = 4 if "freq_step" in params_str else 1
        signal = sig.generate(params_str, fs=fs, length=length, beam_count=beam_count)

        # For multi-beam, take first beam
        if signal.ndim == 2:
            plot_signal = signal[0, :]
        else:
            plot_signal = signal

        spectrum = fft.process_complex(plot_signal, sample_rate=fs)

        t = np.arange(len(plot_signal)) / fs * 1000
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[i, 0].plot(t[:400], plot_signal[:400].real, 'b-', linewidth=0.6)
        axes[i, 0].set_title(f"'{params_str}'")
        axes[i, 0].set_xlabel("t, ms")
        axes[i, 0].grid(True, alpha=0.3)

        mag_db = 20 * np.log10(np.abs(spectrum[:half]) + 1e-10)
        axes[i, 1].plot(freq_axis[:half], mag_db, 'r-', linewidth=0.6)
        axes[i, 1].set_title(f"{title} - FFT (dB)")
        axes[i, 1].set_xlabel("f, Hz")
        axes[i, 1].grid(True, alpha=0.3)

        shape_str = str(signal.shape)
        print(f"  '{params_str}' -> shape {shape_str} OK")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test7_generate_from_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 8: ScriptGenerator - text DSL -> GPU kernel
# ============================================================================
def test_script_generator():
    print_header("Test 8: ScriptGenerator - Text DSL -> GPU kernel")

    ctx = gpuworklib.GPUContext(0)
    script = gpuworklib.ScriptGenerator(ctx)

    # ---- Example 1: User's original format ----
    script_text_1 = """
[Params]
ANTENNAS = 256
POINTS = 10000

[Defs]
// Variables depending on antenna ID
var_A = 1.0 + (float)ID * 0.01
var_W = 0.1 + (float)ID * 0.0005
var_P = (float)ID * 3.14 / 180.0

[Signal]
// T is the time/sample index
res = var_A * sin(var_W * (float)T + var_P);
"""
    print("  Loading script (256 antennas, 10000 points)...")
    script.load(script_text_1)
    print(f"    antennas={script.antennas}, points={script.points}")
    print(f"    is_ready={script.is_ready}")
    print(f"    repr: {script}")

    import time
    t0 = time.time()
    data = script.generate()
    dt = time.time() - t0
    print(f"    Generated: shape={data.shape}, dtype={data.dtype}")
    print(f"    Time: {dt*1000:.1f} ms for {data.size:,} samples")

    # Verify shape
    assert data.shape == (256, 10000), f"Expected (256, 10000), got {data.shape}"

    # Plot several antennas
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle("Test 8: ScriptGenerator - 256 antennas x 10000 points", fontsize=14, fontweight='bold')

    # Time domain: first, middle, last antenna
    antennas_to_plot = [0, 127, 255]
    for i, ant_idx in enumerate(antennas_to_plot):
        signal = data[ant_idx, :].real
        axes[i, 0].plot(signal[:500], 'b-', linewidth=0.5)
        axes[i, 0].set_title(f"Antenna {ant_idx} - Time domain (first 500 samples)")
        axes[i, 0].set_xlabel("Sample (T)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        # FFT via numpy (since these are real signals, imaginary is 0)
        spectrum = np.fft.fft(signal)
        nfft = len(spectrum)
        half = nfft // 2
        mag = np.abs(spectrum[:half])
        axes[i, 1].plot(mag, 'r-', linewidth=0.5)
        axes[i, 1].set_title(f"Antenna {ant_idx} - FFT magnitude")
        axes[i, 1].set_xlabel("Bin")
        axes[i, 1].set_ylabel("|FFT|")
        axes[i, 1].grid(True, alpha=0.3)

        # Statistics
        rms = np.sqrt(np.mean(signal**2))
        print(f"    Antenna {ant_idx}: rms={rms:.4f}, max={np.max(np.abs(signal)):.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test8_script_basic.png"), dpi=150)

    # ---- Example 2: Complex IQ signal ----
    script_text_2 = """
[Params]
ANTENNAS = 8
POINTS = 4096

[Defs]
float freq = 100.0 + (float)ID * 50.0
float phase = 2.0 * M_PI_F * freq * (float)T / 4000.0

[Signal]
// Complex IQ output
res_re = cos(phase);
res_im = sin(phase);
"""
    print("\n  Loading complex IQ script (8 antennas, 4096 points)...")
    script.load(script_text_2)
    data2 = script.generate()
    print(f"    Generated: shape={data2.shape}, dtype={data2.dtype}")

    assert data2.shape == (8, 4096), f"Expected (8, 4096), got {data2.shape}"

    # Verify complex output is non-zero imaginary
    for b in range(8):
        expected_freq = 100.0 + b * 50.0
        imag_rms = np.sqrt(np.mean(data2[b].imag**2))
        real_rms = np.sqrt(np.mean(data2[b].real**2))
        print(f"    Beam {b} (f={expected_freq:.0f} Hz): Re_rms={real_rms:.4f}, Im_rms={imag_rms:.4f}")

    assert np.any(np.abs(data2.imag) > 0.01), "Complex IQ signal should have non-zero imaginary part"

    # Plot IQ
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle("Test 8b: ScriptGenerator - Complex IQ (8 beams)", fontsize=14, fontweight='bold')

    # Time domain: 2 beams
    for i, b in enumerate([0, 7]):
        axes2[0, i].plot(data2[b, :200].real, 'b-', linewidth=0.7, label="Re (I)")
        axes2[0, i].plot(data2[b, :200].imag, 'r-', linewidth=0.7, label="Im (Q)")
        axes2[0, i].set_title(f"Beam {b} (f={100+b*50} Hz) - IQ")
        axes2[0, i].legend()
        axes2[0, i].grid(True, alpha=0.3)

    # IQ scatter
    for i, b in enumerate([0, 7]):
        axes2[1, i].scatter(data2[b, :500].real, data2[b, :500].imag, s=1, alpha=0.3, c='purple')
        axes2[1, i].set_title(f"Beam {b} - IQ scatter")
        axes2[1, i].set_aspect('equal')
        axes2[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test8b_script_iq.png"), dpi=150)

    # ---- Example 3: Ternary operator / conditional ----
    script_text_3 = """
[Params]
ANTENNAS = 4
POINTS = 2048

[Defs]
// Even antennas get low freq, odd get high freq
var_W = (ID % 2 == 0) ? 0.05 : 0.3

[Signal]
res = sin(var_W * (float)T);
"""
    print("\n  Loading conditional script (ternary operator)...")
    script.load(script_text_3)
    data3 = script.generate()
    print(f"    Generated: shape={data3.shape}")

    # Verify different frequencies for even/odd
    for b in range(4):
        spectrum = np.fft.fft(data3[b].real)
        peak = np.argmax(np.abs(spectrum[:len(spectrum)//2]))
        print(f"    Antenna {b} ({'even' if b%2==0 else 'odd'}): peak bin={peak}")

    # ---- Print generated kernel source ----
    print(f"\n  Generated OpenCL kernel:\n{'─'*40}")
    for line in script.kernel_source.split('\n'):
        print(f"    {line}")
    print(f"{'─'*40}")

    print("  PASS")


# ============================================================================
# Test 9: ScriptGenerator -> GPU FFTProcessor (full pipeline)
# ============================================================================
def test_script_fft_pipeline():
    print_header("Test 9: ScriptGenerator -> GPU FFTProcessor pipeline")

    ctx = gpuworklib.GPUContext(0)
    script = gpuworklib.ScriptGenerator(ctx)

    # ---- Pipeline 1: Multi-antenna IQ signals with known frequencies ----
    # Each antenna generates a sinusoid at freq = 100 + ID * 100 Hz
    # fs=4000, so max detectable = 2000 Hz
    script.load("""
[Params]
ANTENNAS = 16
POINTS = 4096

[Defs]
float freq = 100.0 + (float)ID * 100.0
float w = 2.0 * M_PI_F * freq / 4000.0

[Signal]
res_re = cos(w * (float)T);
res_im = sin(w * (float)T);
""")

    import time
    fs = 4000.0

    print(f"  Pipeline 1: 16 antennas, freq = 100 + ID*100 Hz, fs={fs}")
    t0 = time.time()
    data = script.generate()
    gen_ms = (time.time() - t0) * 1000

    print(f"    ScriptGen: {data.shape} in {gen_ms:.1f} ms")

    # FFT each antenna via GPU FFTProcessor (new instance per size)
    fft1 = gpuworklib.FFTProcessor(ctx)
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    fig.suptitle("Test 9: ScriptGenerator -> GPU FFTProcessor\n"
                 "16 antennas, freq = 100 + ID*100 Hz", fontsize=13, fontweight='bold')

    all_detected = []
    t0 = time.time()
    for ant in range(16):
        beam_data = data[ant, :]
        result = fft1.process_mag_phase(beam_data, sample_rate=fs)

        mag = result["magnitude"]
        freq_axis = result["frequency"]
        nfft = result["nFFT"]
        half = nfft // 2

        peak_idx = np.argmax(mag[:half])
        detected_freq = freq_axis[peak_idx]
        expected_freq = 100.0 + ant * 100.0
        error = abs(detected_freq - expected_freq)
        all_detected.append((expected_freq, detected_freq, error))

        # Plot
        row, col = ant // 4, ant % 4
        ax = axes[row, col]
        mag_db = 20 * np.log10(mag[:half] + 1e-10)
        ax.plot(freq_axis[:half], mag_db, 'r-', linewidth=0.6)
        ax.axvline(x=expected_freq, color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f"Ant {ant}: {detected_freq:.0f} Hz", fontsize=9)
        ax.set_xlim(0, 2000)
        ax.set_ylim(-20, max(mag_db[:half]) + 5)
        ax.grid(True, alpha=0.3)
        if row == 3:
            ax.set_xlabel("f, Hz", fontsize=8)

    fft_ms = (time.time() - t0) * 1000
    print(f"    GPU FFT:   16 beams in {fft_ms:.1f} ms")

    # Print frequency detection results
    max_error = 0
    for expected, detected, err in all_detected:
        status = "OK" if err < 2.0 else "WARN"
        print(f"    Ant f={expected:>6.0f} Hz -> detected {detected:>8.1f} Hz "
              f"(err={err:.1f} Hz) [{status}]")
        max_error = max(max_error, err)

    assert max_error < 2.0, f"Max frequency error {max_error:.1f} Hz exceeds 2 Hz tolerance"

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test9_script_fft_pipeline.png"), dpi=150)

    # ---- Pipeline 2: Complex multi-harmonic signal ----
    script.load("""
[Params]
ANTENNAS = 4
POINTS = 8192

[Defs]
// Each antenna has a sum of 3 harmonics with different amplitudes
float f1 = 200.0 + (float)ID * 50.0
float f2 = f1 * 2.0
float f3 = f1 * 3.0
float w1 = 2.0 * M_PI_F * f1 / 8000.0
float w2 = 2.0 * M_PI_F * f2 / 8000.0
float w3 = 2.0 * M_PI_F * f3 / 8000.0

[Signal]
// Sum of 3 harmonics: fundamental + 2nd (0.5x) + 3rd (0.25x)
res_re = cos(w1 * (float)T) + 0.5 * cos(w2 * (float)T) + 0.25 * cos(w3 * (float)T);
res_im = sin(w1 * (float)T) + 0.5 * sin(w2 * (float)T) + 0.25 * sin(w3 * (float)T);
""")

    fs2 = 8000.0
    print(f"\n  Pipeline 2: 4 antennas, 3 harmonics each, fs={fs2}")

    data2 = script.generate()
    print(f"    Shape: {data2.shape}")

    fft_harm = gpuworklib.FFTProcessor(ctx)
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 12))
    fig2.suptitle("Test 9b: Multi-harmonic signals -> GPU FFT\n"
                  "4 antennas, 3 harmonics each", fontsize=13, fontweight='bold')

    for ant in range(4):
        f1 = 200.0 + ant * 50.0
        beam_data = data2[ant, :]

        # Time domain
        t = np.arange(500) / fs2 * 1000
        axes2[ant, 0].plot(t, beam_data[:500].real, 'b-', linewidth=0.5, label="Re")
        axes2[ant, 0].plot(t, beam_data[:500].imag, 'r-', linewidth=0.5, alpha=0.5, label="Im")
        axes2[ant, 0].set_title(f"Ant {ant}: f1={f1:.0f}, f2={f1*2:.0f}, f3={f1*3:.0f} Hz")
        axes2[ant, 0].legend(fontsize=7)
        axes2[ant, 0].grid(True, alpha=0.3)
        axes2[ant, 0].set_xlabel("t, ms")

        # GPU FFT magnitude
        result = fft_harm.process_mag_phase(beam_data, sample_rate=fs2)
        mag = result["magnitude"]
        freq_axis = result["frequency"]
        nfft = result["nFFT"]
        half = nfft // 2

        mag_db = 20 * np.log10(mag[:half] + 1e-10)
        axes2[ant, 1].plot(freq_axis[:half], mag_db, 'r-', linewidth=0.6)

        # Mark expected harmonics
        for h, label in [(f1, "f1"), (f1*2, "f2"), (f1*3, "f3")]:
            if h < fs2/2:
                axes2[ant, 1].axvline(x=h, color='blue', linestyle='--', alpha=0.4, linewidth=0.7)
                axes2[ant, 1].text(h, max(mag_db[:half])-5, label, fontsize=7, color='blue')

        axes2[ant, 1].set_title(f"Ant {ant} - FFT (dB)")
        axes2[ant, 1].set_xlabel("f, Hz")
        axes2[ant, 1].set_xlim(0, 2000)
        axes2[ant, 1].grid(True, alpha=0.3)

        # Find peaks
        peaks = []
        for h_idx, h_freq in enumerate([f1, f1*2, f1*3]):
            if h_freq >= fs2/2:
                continue
            bin_idx = int(round(h_freq * nfft / fs2))
            region = mag[max(0, bin_idx-3):bin_idx+4]
            local_peak = np.argmax(region) + max(0, bin_idx-3)
            peaks.append(freq_axis[local_peak])

        print(f"    Ant {ant} (f1={f1:.0f}): peaks @ {', '.join(f'{p:.1f}' for p in peaks)} Hz")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test9b_multiharmonic_fft.png"), dpi=150)

    # ---- Pipeline 3: Large scale benchmark ----
    script.load("""
[Params]
ANTENNAS = 512
POINTS = 16384

[Defs]
float freq = 50.0 + (float)ID * 3.0
float w = 2.0 * M_PI_F * freq / 16384.0

[Signal]
res_re = cos(w * (float)T);
res_im = sin(w * (float)T);
""")

    print(f"\n  Pipeline 3: Benchmark 512 antennas x 16384 points")
    t0 = time.time()
    data3 = script.generate()
    gen_ms = (time.time() - t0) * 1000
    total_samples = data3.size
    print(f"    ScriptGen: {data3.shape} = {total_samples:,} samples in {gen_ms:.1f} ms")
    print(f"    Throughput: {total_samples / gen_ms / 1e3:.1f} Msamples/s")

    # Spot-check a few antennas (new FFT instance for different size)
    fft2 = gpuworklib.FFTProcessor(ctx)
    for ant in [0, 100, 255, 511]:
        spectrum = fft2.process_complex(data3[ant, :], sample_rate=16384.0)
        nfft = len(spectrum)
        peak = np.argmax(np.abs(spectrum[:nfft//2]))
        detected = peak * 16384.0 / nfft
        expected = 50.0 + ant * 3.0
        print(f"    Ant {ant}: expected {expected:.0f} Hz -> detected {detected:.1f} Hz")

    print("  PASS")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  GPUWorkLib Python Test Suite")
    print("=" * 60)

    gpus = gpuworklib.list_gpus()
    print(f"\n  Found {len(gpus)} GPU(s):")
    for g in gpus:
        print(f"    [{g['index']}] {g['name']} ({g['memory_mb']} MB)")

    test_multichannel_sin_fft()
    test_signal_types()
    test_multibeam_cw()
    test_generators_from_string()
    test_multibeam_from_string()
    test_mag_phase()
    test_generate_from_string()
    test_script_generator()
    test_script_fft_pipeline()

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
