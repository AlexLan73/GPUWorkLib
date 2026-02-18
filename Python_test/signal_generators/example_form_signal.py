#!/usr/bin/env python3
"""
FormSignalGenerator + FormScriptGenerator — Demo & Presentation Plots

Примеры использования:
  python example_form_signal.py              # все демки + графики
  python example_form_signal.py --no-plot    # только текстовый вывод

Графики сохраняются в Results/Plots/signal_generators/FormSignal/

@author Кодо (AI Assistant)
@date 2026-02-17
"""

import sys
import os
import argparse
import numpy as np

# Auto-detect build path (Python_test/signal_generators/ -> 2 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for subdir in ["build/python/Release", "build/python/Debug", "build/python", "build/Release", "build/Debug"]:
    full = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(full):
        sys.path.insert(0, os.path.abspath(full))
        break

import gpuworklib


# ════════════════════════════════════════════════════════════════════════════
# NumPy Reference (getX formula)
# ════════════════════════════════════════════════════════════════════════════

def getX_numpy(fs, points, f0, amplitude, noise_amplitude, phase=0,
               fdev=0, norm=1/np.sqrt(2), tau=0, noise_seed=None):
    """CPU reference: getX formula in NumPy."""
    dt = 1 / fs
    ti = points * dt
    t = np.arange(points, dtype=np.float64) * dt + tau

    # Window
    mask = (t >= 0) & (t <= ti - dt)
    t_c = t - ti / 2

    # Signal
    ph = 2 * np.pi * f0 * t + np.pi * fdev / ti * (t_c ** 2) + phase
    X = amplitude * norm * np.exp(1j * ph)

    # Noise
    if noise_amplitude > 0 and noise_seed is not None:
        rng = np.random.RandomState(noise_seed)
        X += noise_amplitude * norm * (rng.randn(points) + 1j * rng.randn(points))

    X[~mask] = 0
    return X.astype(np.complex64), t


# ════════════════════════════════════════════════════════════════════════════
# Demo 1: FormSignalGenerator — Basic Usage
# ════════════════════════════════════════════════════════════════════════════

def demo_form_signal_generator(ctx):
    """Basic FormSignalGenerator usage."""
    print("\n" + "=" * 60)
    print("  Demo 1: FormSignalGenerator — Basic CW")
    print("=" * 60)

    gen = gpuworklib.FormSignalGenerator(ctx)

    # CW signal: 1 MHz, 8 channels
    gen.set_params(
        fs=12e6,
        f0=1e6,
        antennas=8,
        points=4096,
        amplitude=1.0,
        noise_amplitude=0.0,
        tau_step=1e-5   # 10 us step between channels
    )

    data = gen.generate()
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Max amplitude: {np.abs(data).max():.4f}")
    print(f"  Channels: {gen.antennas}, Points: {gen.points}, Fs: {gen.fs/1e6:.1f} MHz")

    return data


# ════════════════════════════════════════════════════════════════════════════
# Demo 2: FormScriptGenerator — DSL + Kernel Cache
# ════════════════════════════════════════════════════════════════════════════

def demo_form_script_generator(ctx):
    """FormScriptGenerator: DSL, compile, save/load kernel cache."""
    print("\n" + "=" * 60)
    print("  Demo 2: FormScriptGenerator — DSL + Kernel Cache")
    print("=" * 60)

    gen = gpuworklib.FormScriptGenerator(ctx)

    # Set params from string
    gen.set_params_from_string("f0=500000,a=1.0,an=0.05,antennas=4,points=8192,fs=10e6")
    print(f"  Params: f0=500kHz, 4ch, 8192pts, Fs=10MHz")

    # Show DSL script
    script = gen.generate_script()
    print(f"\n  DSL Script ({len(script)} chars):")
    for line in script.split('\n')[:8]:
        print(f"    {line}")
    print(f"    ...")

    # Compile and generate
    gen.compile()
    data = gen.generate()
    print(f"\n  Generated: {data.shape} complex64")

    # Save kernel to disk
    gen.save_kernel("demo_cw_500k", "Demo CW 500kHz, 4ch")
    print(f"  Saved kernel: 'demo_cw_500k'")

    # Load back and verify
    gen2 = gpuworklib.FormScriptGenerator(ctx)
    gen2.set_params_from_string("f0=500000,a=1.0,an=0.05,antennas=4,points=8192,fs=10e6")
    gen2.load_kernel("demo_cw_500k")
    data2 = gen2.generate()

    err = np.max(np.abs(data - data2))
    print(f"  Loaded kernel: err={err:.2e} (should be 0)")

    # List all kernels
    names = gen.list_kernels()
    print(f"  Available kernels: {names}")

    return data


# ════════════════════════════════════════════════════════════════════════════
# Demo 3: Chirp + Noise + FFT Pipeline
# ════════════════════════════════════════════════════════════════════════════

def demo_chirp_fft_pipeline(ctx):
    """Chirp signal → GPU FFT → spectrum analysis."""
    print("\n" + "=" * 60)
    print("  Demo 3: Chirp + Noise → FFT Pipeline")
    print("=" * 60)

    gen = gpuworklib.FormSignalGenerator(ctx)
    gen.set_params(
        fs=100000,
        f0=5000,
        fdev=20000,     # 20 kHz chirp
        amplitude=1.0,
        noise_amplitude=0.1,
        noise_seed=42,
        antennas=1,
        points=8192
    )

    signal = gen.generate()
    print(f"  Signal: {signal.shape}, chirp 5±10 kHz")

    # FFT on GPU
    fft = gpuworklib.FFTProcessor(ctx)
    spectrum = fft.process_complex(signal, sample_rate=100000)
    print(f"  Spectrum: {spectrum.shape}")

    mag = np.abs(spectrum[0] if spectrum.ndim == 2 else spectrum)
    freq = np.fft.fftfreq(len(mag), d=1/100000)
    peak_freq = freq[np.argmax(mag[:len(mag)//2])]
    print(f"  Peak frequency: {peak_freq:.0f} Hz")

    return signal, spectrum


# ════════════════════════════════════════════════════════════════════════════
# Demo 4: Multi-channel with Random Delay
# ════════════════════════════════════════════════════════════════════════════

def demo_random_delay(ctx):
    """Multi-channel generation with random per-channel delay."""
    print("\n" + "=" * 60)
    print("  Demo 4: Multi-channel + Random Delay (TAU_RANDOM)")
    print("=" * 60)

    gen = gpuworklib.FormSignalGenerator(ctx)
    gen.set_params(
        fs=50000,
        f0=2000,
        amplitude=1.0,
        noise_amplitude=0.0,
        antennas=16,
        points=2048,
        tau_min=0.0,
        tau_max=0.005,   # 0-5 ms random delay
        tau_seed=777
    )

    data = gen.generate()
    print(f"  Shape: {data.shape} (16 channels, 2048 pts)")

    # Show phase offset per channel (first nonzero sample)
    for ch in [0, 4, 8, 15]:
        first_nonzero = np.argmax(np.abs(data[ch]) > 0.01)
        print(f"  Channel {ch:2d}: first signal at sample {first_nonzero}")

    return data


# ════════════════════════════════════════════════════════════════════════════
# Demo 5: GPU vs NumPy Comparison
# ════════════════════════════════════════════════════════════════════════════

def demo_gpu_vs_numpy(ctx):
    """Compare GPU result with NumPy reference."""
    print("\n" + "=" * 60)
    print("  Demo 5: GPU vs NumPy Reference")
    print("=" * 60)

    params = dict(fs=12e6, points=4096, f0=1e6, amplitude=1.0,
                  noise_amplitude=0.0, phase=0.3, fdev=2000)

    # GPU
    gen = gpuworklib.FormSignalGenerator(ctx)
    gen.set_params(**params, antennas=1)
    gpu_data = gen.generate()

    # NumPy reference
    ref, t = getX_numpy(**params, norm=1/np.sqrt(2))

    err = np.max(np.abs(gpu_data.ravel() - ref))
    print(f"  Max error: {err:.2e}")
    print(f"  GPU matches NumPy: {'YES' if err < 1e-3 else 'NO'}")

    return gpu_data, ref, t


# ════════════════════════════════════════════════════════════════════════════
# Presentation Plots
# ════════════════════════════════════════════════════════════════════════════

def make_plots(ctx):
    """Generate presentation-quality plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'Results', 'Plots', 'signal_generators', 'FormSignal')
    os.makedirs(outdir, exist_ok=True)

    # Style
    plt.rcParams.update({
        'figure.facecolor': '#0a0e27',
        'axes.facecolor': '#121638',
        'axes.edgecolor': '#3a3f6b',
        'axes.labelcolor': '#e0e0e0',
        'text.color': '#e0e0e0',
        'xtick.color': '#aaaaaa',
        'ytick.color': '#aaaaaa',
        'grid.color': '#2a2f5b',
        'grid.alpha': 0.5,
        'font.size': 11,
    })

    # ── Plot 1: CW Time Domain + Spectrum ──
    print("\n  Generating Plot 1: CW Time + Spectrum...")
    gen = gpuworklib.FormSignalGenerator(ctx)
    gen.set_params(fs=100000, f0=5000, antennas=1, points=4096)
    cw = gen.generate().ravel()
    t = np.arange(len(cw)) / 100000 * 1000  # ms

    fft_cw = np.fft.fft(cw)
    freq = np.fft.fftfreq(len(cw), d=1/100000) / 1000  # kHz
    mag_db = 20 * np.log10(np.abs(fft_cw[:len(cw)//2]) + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    ax1.plot(t[:500], cw[:500].real, color='#00d4ff', lw=0.8, label='Re')
    ax1.plot(t[:500], cw[:500].imag, color='#ff6b9d', lw=0.8, label='Im')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('CW Signal: f0 = 5 kHz, Fs = 100 kHz', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(freq[:len(cw)//2], mag_db, color='#ffd700', lw=0.8)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('FFT Spectrum', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 50)
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(outdir, 'example_01_cw_spectrum.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

    # ── Plot 2: Chirp Spectrogram ──
    print("  Generating Plot 2: Chirp Spectrogram...")
    gen.set_params(fs=100000, f0=10000, fdev=30000, antennas=1, points=16384)
    chirp = gen.generate().ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    t_chirp = np.arange(len(chirp)) / 100000 * 1000

    ax1.plot(t_chirp[:2000], chirp[:2000].real, color='#00d4ff', lw=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Chirp Signal: f0=10kHz, fdev=30kHz', fontsize=14, fontweight='bold')
    ax1.grid(True)

    ax2.specgram(chirp, NFFT=256, Fs=100000, noverlap=240,
                 cmap='inferno', scale='dB')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(outdir, 'example_02_chirp_spectrogram.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

    # ── Plot 3: Multi-channel Waterfall ──
    print("  Generating Plot 3: Multi-channel Waterfall...")
    gen.set_params(fs=50000, f0=3000, antennas=16, points=2048,
                   tau_step=0.001, noise_amplitude=0.0)
    multi = gen.generate()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    im = ax1.imshow(np.abs(multi[:, :512]), aspect='auto', cmap='viridis',
                    extent=[0, 512/50000*1000, 15.5, -0.5])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Channel')
    ax1.set_title('16-Channel Magnitude (tau_step=1ms)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='|X|')

    # Overlay
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, 16))
    t_ms = np.arange(512) / 50000 * 1000
    for ch in range(16):
        ax2.plot(t_ms, multi[ch, :512].real + ch * 0.15,
                 color=colors[ch], lw=0.6)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Channel (offset)')
    ax2.set_title('Waveform Overlay (Re)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'example_03_multichannel.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

    # ── Plot 4: GPU vs NumPy ──
    print("  Generating Plot 4: GPU vs NumPy Comparison...")
    gen.set_params(fs=12e6, f0=1e6, antennas=1, points=4096, phase=0.3, fdev=2000)
    gpu = gen.generate().ravel()
    ref, t_ref = getX_numpy(fs=12e6, points=4096, f0=1e6, amplitude=1.0,
                            noise_amplitude=0.0, phase=0.3, fdev=2000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    n_show = 300
    t_us = t_ref[:n_show] * 1e6

    ax1.plot(t_us, gpu[:n_show].real, color='#00d4ff', lw=1.2, label='GPU')
    ax1.plot(t_us, ref[:n_show].real, '--', color='#ff6b9d', lw=1.2, label='NumPy')
    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('Re(X)')
    ax1.set_title('GPU vs NumPy Reference (Chirp, fdev=2kHz)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True)

    error = np.abs(gpu - ref)
    ax2.semilogy(np.arange(len(error)), error, color='#ffd700', lw=0.5)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('|Error|')
    ax2.set_title(f'Absolute Error (max = {error.max():.2e})', fontsize=13, fontweight='bold')
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(outdir, 'example_04_gpu_vs_numpy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

    # ── Plot 5: DSL Script + Kernel Source ──
    print("  Generating Plot 5: DSL Script Demo...")
    sgen = gpuworklib.FormScriptGenerator(ctx)
    sgen.set_params(fs=10e6, f0=500000, antennas=4, points=4096,
                    noise_amplitude=0.05, tau_step=0.0001)
    sgen.compile()
    script_data = sgen.generate()

    script_text = sgen.generate_script()
    kernel_text = sgen.generate_kernel_source()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # DSL script text
    ax1.text(0.02, 0.98, script_text[:600], transform=ax1.transAxes,
             fontsize=7, fontfamily='monospace', color='#00ff88',
             verticalalignment='top', horizontalalignment='left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('DSL Script (GenerateScript())', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Generated signal
    t_ms = np.arange(script_data.shape[1]) / 10e6 * 1000
    colors = ['#00d4ff', '#ff6b9d', '#ffd700', '#00ff88']
    for ch in range(4):
        ax2.plot(t_ms[:200], script_data[ch, :200].real,
                 color=colors[ch], lw=0.8, label=f'Ch {ch}')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Re(X)')
    ax2.set_title('FormScriptGenerator Output (4 channels)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(outdir, 'example_05_dsl_script.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n  All plots saved to {outdir}/")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FormSignalGenerator Demo')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    print("=" * 60)
    print("  FormSignalGenerator + FormScriptGenerator — Demo")
    print("=" * 60)

    ctx = gpuworklib.GPUContext(0)
    print(f"  GPU: {ctx.device_name}")

    # Run demos
    demo_form_signal_generator(ctx)
    demo_form_script_generator(ctx)
    demo_chirp_fft_pipeline(ctx)
    demo_random_delay(ctx)
    demo_gpu_vs_numpy(ctx)

    # Plots
    if not args.no_plot:
        make_plots(ctx)

    # Cleanup demo kernel
    import glob
    kdir = gpuworklib.FormScriptGenerator.get_kernels_dir()
    for f in glob.glob(os.path.join(kdir, 'demo_*')):
        os.remove(f)
    bdir = gpuworklib.FormScriptGenerator.get_kernels_bin_dir()
    for f in glob.glob(os.path.join(bdir, 'demo_*')):
        os.remove(f)

    print("\n" + "=" * 60)
    print("  ALL DEMOS COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
