"""
test_form_signal.py — Тесты FormSignalGenerator (GPU) vs NumPy (CPU) + графики

Формула getX:
  X = a * norm * exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
    + an * norm * (randn + j*randn)
  X = 0  при t < 0 или t > ti - dt

Тесты:
  1. CW без шума — GPU vs NumPy
  2. Chirp (fdev != 0) — GPU vs NumPy
  3. Окно (tau < 0) — нули в начале
  4. Multi-channel с TAU_STEP
  5. Шум — статистическая проверка
  6. Парсинг из строки (set_params_from_string)
  7. Signal + Noise combined

Графики (создаются по умолчанию, --no-plot чтобы отключить):
  1. CW: GPU vs NumPy overlay + error
  2. Chirp: spectrogram + instantaneous freq
  3. Window: signal с окном и задержкой
  4. Multi-channel: 8 антенн waterfall
  5. Noise: histogram + QQ plot
  6. Signal+Noise: спектр SNR

@author Кодо (AI Assistant)
@date 2026-02-17
"""

import sys
import os
import glob
import numpy as np

# ── Путь к gpuworklib (Python_test/signal_generators/ -> 2 levels up) ──
_root = os.path.join(os.path.dirname(__file__), '..', '..')
BUILD_PATHS = (
    glob.glob(os.path.join(_root, 'build', 'debian-*', 'python')) +
    [os.path.join(_root, 'build', 'python', 'Debug'),
     os.path.join(_root, 'build', 'python', 'Release'),
     os.path.join(_root, 'build', 'python')]
)
for p in BUILD_PATHS:
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
        break

try:
    import gpuworklib
except ImportError:
    print("ERROR: gpuworklib not found. Build with -DBUILD_PYTHON=ON")
    print(f"Searched: {BUILD_PATHS}")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: getX formula
# ════════════════════════════════════════════════════════════════════════════

def getX_numpy(fs, points, f0, amplitude, phase, fdev, norm_val, tau=0.0):
    """
    CPU reference (NumPy) — формула getX без шума.
    """
    dt = 1.0 / fs
    ti = points * dt
    t = np.arange(points, dtype=np.float64) * dt + tau

    in_window = (t >= 0.0) & (t <= ti - dt)

    t_centered = t - ti / 2.0
    ph = 2.0 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase

    X = amplitude * norm_val * np.exp(1j * ph)
    X[~in_window] = 0.0

    return X.astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════
# Тесты
# ════════════════════════════════════════════════════════════════════════════

def test_cw_no_noise():
    """FormSignal CW (f0=1 MHz, no noise) — GPU vs NumPy"""
    print("\n[Test 1] CW no noise: GPU vs NumPy...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    fs = 12e6
    points = 4096
    f0 = 1e6
    amplitude = 1.0
    phase = 0.3
    norm_val = 1.0 / np.sqrt(2.0)

    gen.set_params(fs=fs, points=points, f0=f0,
                   amplitude=amplitude, phase=phase,
                   norm=norm_val, noise_amplitude=0.0)

    gpu_data = gen.generate()
    cpu_ref = getX_numpy(fs, points, f0, amplitude, phase, 0.0, norm_val)

    max_err = np.max(np.abs(gpu_data - cpu_ref))
    passed = max_err < 1e-3

    print(f"  fs={fs:.0e} f0={f0:.0e} points={points}")
    print(f"  Max error: {max_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_chirp():
    """FormSignal Chirp (fdev=5000) — GPU vs NumPy"""
    print("\n[Test 2] Chirp (fdev=5000): GPU vs NumPy...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    fs = 100000.0
    points = 4096
    f0 = 1000.0
    fdev = 5000.0
    norm_val = 1.0 / np.sqrt(2.0)

    gen.set_params(fs=fs, points=points, f0=f0,
                   fdev=fdev, norm=norm_val, noise_amplitude=0.0)

    gpu_data = gen.generate()
    cpu_ref = getX_numpy(fs, points, f0, 1.0, 0.0, fdev, norm_val)

    max_err = np.max(np.abs(gpu_data - cpu_ref))
    passed = max_err < 1e-3

    print(f"  f0={f0} fdev={fdev}")
    print(f"  Max error: {max_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_window():
    """FormSignal Window: tau=-0.1 -> first 100 samples zero"""
    print("\n[Test 3] Window (tau=-0.1s)...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    gen.set_params(fs=1000.0, points=1000, f0=100.0,
                   amplitude=1.0, noise_amplitude=0.0,
                   tau_base=-0.1)

    gpu_data = gen.generate()

    zeros_first_100 = np.sum(np.abs(gpu_data[:100]) < 1e-6)
    nonzeros_mid = np.sum(np.abs(gpu_data[110:500]) > 0.01)

    passed = (zeros_first_100 >= 99) and (nonzeros_mid > 350)

    print(f"  tau=-0.1s, dt=0.001s")
    print(f"  Zeros in first 100: {zeros_first_100}/100")
    print(f"  Non-zeros in [110..500]: {nonzeros_mid}/390")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_multi_channel():
    """FormSignal 8 antennas with TAU_STEP=0.01"""
    print("\n[Test 4] Multi-channel (8 antennas, TAU_STEP=0.01)...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    fs = 10000.0
    points = 2048
    f0 = 500.0
    antennas = 8
    tau_step = 0.01
    norm_val = 1.0 / np.sqrt(2.0)

    gen.set_params(fs=fs, antennas=antennas, points=points, f0=f0,
                   noise_amplitude=0.0, tau_step=tau_step, norm=norm_val)

    gpu_data = gen.generate()
    assert gpu_data.shape == (antennas, points), \
        f"Shape mismatch: {gpu_data.shape} != ({antennas}, {points})"

    all_pass = True
    for a in range(antennas):
        tau = a * tau_step
        cpu_ref = getX_numpy(fs, points, f0, 1.0, 0.0, 0.0, norm_val, tau)
        err = np.max(np.abs(gpu_data[a] - cpu_ref))
        ok = err < 1e-3
        if not ok:
            all_pass = False
        print(f"  Antenna {a} (tau={tau:.4f}s): err={err:.2e} {'OK' if ok else 'FAIL'}")

    print(f"  {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_noise_statistics():
    """FormSignal noise only -- mean~0, variance~1"""
    print("\n[Test 5] Noise statistics (an=1.0, a=0.0)...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    gen.set_params(fs=10000.0, points=100000, f0=0.0,
                   amplitude=0.0, noise_amplitude=1.0,
                   norm=1.0, noise_seed=42)

    gpu_data = gen.generate()

    mean_re = np.mean(gpu_data.real)
    mean_im = np.mean(gpu_data.imag)
    var_re = np.var(gpu_data.real)
    var_im = np.var(gpu_data.imag)

    mean_ok = (abs(mean_re) < 0.05) and (abs(mean_im) < 0.05)
    var_ok = (abs(var_re - 1.0) < 0.1) and (abs(var_im - 1.0) < 0.1)
    passed = mean_ok and var_ok

    print(f"  N=100000")
    print(f"  Mean: re={mean_re:.4f} im={mean_im:.4f} (expected ~0)")
    print(f"  Variance: re={var_re:.4f} im={var_im:.4f} (expected ~1.0)")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_string_params():
    """FormSignal set_params_from_string -- same as set_params"""
    print("\n[Test 6] set_params_from_string vs set_params...")

    ctx = gpuworklib.ROCmGPUContext(0)

    gen1 = gpuworklib.FormSignalGeneratorROCm(ctx)
    gen1.set_params(fs=12e6, points=1024, f0=1e6,
                    amplitude=1.5, phase=0.5, noise_amplitude=0.0,
                    norm=1.0 / np.sqrt(2.0))
    data1 = gen1.generate()

    gen2 = gpuworklib.FormSignalGeneratorROCm(ctx)
    gen2.set_params_from_string("fs=12e6,points=1024,f0=1e6,a=1.5,phi=0.5")
    data2 = gen2.generate()

    max_err = np.max(np.abs(data1 - data2))
    passed = max_err < 1e-6

    print(f"  Max diff between set_params and from_string: {max_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_signal_plus_noise():
    """FormSignal signal+noise: amplitude envelope check"""
    print("\n[Test 7] Signal + Noise combined...")

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.FormSignalGeneratorROCm(ctx)

    fs = 100000.0
    points = 10000
    f0 = 5000.0
    amplitude = 2.0
    noise_amp = 0.5
    norm_val = 1.0 / np.sqrt(2.0)

    gen.set_params(fs=fs, points=points, f0=f0,
                   amplitude=amplitude, noise_amplitude=noise_amp,
                   norm=norm_val, noise_seed=777)

    gpu_data = gen.generate()
    cpu_pure = getX_numpy(fs, points, f0, amplitude, 0.0, 0.0, norm_val)
    noise_component = gpu_data - cpu_pure

    expected_std = noise_amp * norm_val
    actual_std_re = np.std(noise_component.real)
    actual_std_im = np.std(noise_component.imag)

    std_ok = (abs(actual_std_re - expected_std) < 0.05) and \
             (abs(actual_std_im - expected_std) < 0.05)

    signal_power = np.mean(np.abs(cpu_pure) ** 2)
    expected_signal_power = (amplitude * norm_val) ** 2
    power_ok = abs(signal_power - expected_signal_power) < 0.01

    passed = std_ok and power_ok

    print(f"  Signal: a={amplitude}, Noise: an={noise_amp}, norm={norm_val:.4f}")
    print(f"  Noise std (re): {actual_std_re:.4f} (expected {expected_std:.4f})")
    print(f"  Noise std (im): {actual_std_im:.4f} (expected {expected_std:.4f})")
    print(f"  Signal power: {signal_power:.4f} (expected {expected_signal_power:.4f})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ════════════════════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════════════════════

def make_plots(save_dir):
    """Генерация красивых графиков для FormSignalGenerator"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(save_dir, exist_ok=True)

    ctx = gpuworklib.ROCmGPUContext(0)
    gpu_name = ctx.device_name

    # ── Общий стиль ──
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.edgecolor': '#e94560',
        'axes.labelcolor': '#eaeaea',
        'text.color': '#eaeaea',
        'xtick.color': '#aaaaaa',
        'ytick.color': '#aaaaaa',
        'grid.color': '#2a2a4a',
        'grid.alpha': 0.5,
        'font.size': 10,
        'axes.titlesize': 13,
        'figure.titlesize': 15,
    })

    COLORS = {
        'gpu_re': '#00d2ff',     # cyan
        'gpu_im': '#ff6b6b',     # coral
        'cpu_re': '#ffd93d',     # yellow
        'cpu_im': '#6bff6b',     # lime
        'error':  '#e94560',     # red-pink
        'noise':  '#a855f7',     # purple
        'signal': '#00d2ff',     # cyan
        'window': '#ff9f43',     # orange
        'accent': '#e94560',     # red
    }

    print(f"\n  Generating plots to {save_dir}/...")

    # ══════════════════════════════════════════════════════════════════════
    # Plot 1: CW — GPU vs NumPy + Error
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 1] CW: GPU vs NumPy...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fs, points, f0 = 100000.0, 1024, 5000.0
    norm_val = 1.0 / np.sqrt(2.0)

    gen.set_params(fs=fs, points=points, f0=f0, norm=norm_val, noise_amplitude=0.0)
    gpu = gen.generate()
    cpu = getX_numpy(fs, points, f0, 1.0, 0.0, 0.0, norm_val)
    t_ms = np.arange(points) / fs * 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), height_ratios=[2, 2, 1])
    fig.suptitle(f'FormSignalGenerator: CW f0={f0/1e3:.0f} kHz  |  GPU: {gpu_name}',
                 fontweight='bold')

    # Re
    ax = axes[0]
    ax.plot(t_ms[:200], gpu.real[:200], color=COLORS['gpu_re'], lw=1.5,
            label='GPU Re', alpha=0.9)
    ax.plot(t_ms[:200], cpu.real[:200], color=COLORS['cpu_re'], lw=1.0,
            label='NumPy Re', ls='--', alpha=0.8)
    ax.set_ylabel('Re(X)')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.set_title('Real part')
    ax.grid(True)

    # Im
    ax = axes[1]
    ax.plot(t_ms[:200], gpu.imag[:200], color=COLORS['gpu_im'], lw=1.5,
            label='GPU Im', alpha=0.9)
    ax.plot(t_ms[:200], cpu.imag[:200], color=COLORS['cpu_im'], lw=1.0,
            label='NumPy Im', ls='--', alpha=0.8)
    ax.set_ylabel('Im(X)')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.set_title('Imaginary part')
    ax.grid(True)

    # Error
    ax = axes[2]
    err = np.abs(gpu - cpu)
    ax.semilogy(t_ms, err, color=COLORS['error'], lw=0.8, alpha=0.8)
    ax.axhline(np.max(err), color=COLORS['accent'], ls=':', lw=1,
               label=f'max = {np.max(err):.2e}')
    ax.set_ylabel('|Error|')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.set_title('Absolute Error (GPU - NumPy)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot1_cw_gpu_vs_numpy.png'), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Plot 2: Chirp — Spectrogram + waveform
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 2] Chirp: spectrogram + waveform...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fs_chirp, pts_chirp = 100000.0, 8192
    f0_chirp, fdev_chirp = 5000.0, 20000.0

    gen.set_params(fs=fs_chirp, points=pts_chirp, f0=f0_chirp,
                   fdev=fdev_chirp, norm=norm_val, noise_amplitude=0.0)
    chirp_data = gen.generate()
    t_chirp = np.arange(pts_chirp) / fs_chirp * 1000.0

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])
    fig.suptitle(f'FormSignalGenerator: Chirp  f0={f0_chirp/1e3:.0f} kHz, '
                 f'fdev={fdev_chirp/1e3:.0f} kHz  |  GPU: {gpu_name}',
                 fontweight='bold')

    # Waveform Re
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_chirp, chirp_data.real, color=COLORS['gpu_re'], lw=0.5, alpha=0.8)
    ax1.set_ylabel('Re(X)')
    ax1.set_title('Waveform (Real)')
    ax1.set_xlabel('Time (ms)')
    ax1.grid(True)

    # Amplitude envelope
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_chirp, np.abs(chirp_data), color=COLORS['signal'], lw=0.5, alpha=0.8)
    ax2.set_ylabel('|X|')
    ax2.set_title('Amplitude envelope')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True)

    # Spectrogram
    ax3 = fig.add_subplot(gs[1, :])
    nfft_spec = 256
    ax3.specgram(chirp_data.real, NFFT=nfft_spec, Fs=fs_chirp/1000.0,
                 noverlap=nfft_spec//2,
                 cmap='magma', scale='dB')
    ax3.set_ylabel('Frequency (kHz)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title(f'Spectrogram (NFFT={nfft_spec})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot2_chirp_spectrogram.png'), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Plot 3: Window + Delay — сигнал с окном
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 3] Window + Delay...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fs_win, pts_win = 10000.0, 2000

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'FormSignalGenerator: Window Effect (tau delay)  |  GPU: {gpu_name}',
                 fontweight='bold')

    tau_values = [0.0, -0.05, -0.1]
    labels = ['tau=0 (no delay)', 'tau=-50ms', 'tau=-100ms']
    colors = [COLORS['gpu_re'], COLORS['window'], COLORS['accent']]

    for ax, tau, label, color in zip(axes, tau_values, labels, colors):
        gen.set_params(fs=fs_win, points=pts_win, f0=500.0,
                       amplitude=1.0, noise_amplitude=0.0,
                       tau_base=tau, norm=norm_val)
        data = gen.generate()
        t_win = np.arange(pts_win) / fs_win * 1000.0

        ax.plot(t_win, data.real, color=color, lw=0.8, alpha=0.9, label='Re(X)')
        ax.fill_between(t_win, data.real, alpha=0.15, color=color)

        # Пометить зону нулей
        zero_mask = np.abs(data) < 1e-6
        if np.any(zero_mask):
            zero_end = np.where(~zero_mask)[0][0] if np.any(~zero_mask) else pts_win
            ax.axvspan(0, t_win[min(zero_end, pts_win-1)],
                       alpha=0.1, color='white', label=f'Zeros: {zero_end} samples')

        ax.set_ylabel('Re(X)')
        ax.set_title(label, fontsize=11)
        ax.legend(loc='upper right', framealpha=0.3)
        ax.grid(True)

    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot3_window_delay.png'), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Plot 4: Multi-channel waterfall
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 4] Multi-channel waterfall...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fs_mc, pts_mc, n_ant = 10000.0, 2048, 8
    tau_step_mc = 0.005

    gen.set_params(fs=fs_mc, antennas=n_ant, points=pts_mc, f0=500.0,
                   noise_amplitude=0.0, tau_step=tau_step_mc, norm=norm_val)
    mc_data = gen.generate()
    t_mc = np.arange(pts_mc) / fs_mc * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             height_ratios=[2, 1.2])
    fig.suptitle(f'FormSignalGenerator: {n_ant} Antennas, tau_step={tau_step_mc*1e3:.1f} ms'
                 f'  |  GPU: {gpu_name}', fontweight='bold')

    # Waterfall (heatmap)
    ax = axes[0]
    extent = [t_mc[0], t_mc[-1], n_ant - 0.5, -0.5]
    im = ax.imshow(mc_data.real, aspect='auto', extent=extent,
                   cmap='RdBu_r', interpolation='nearest',
                   vmin=-norm_val, vmax=norm_val)
    ax.set_ylabel('Antenna ID')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Waterfall: Re(X) per antenna')
    plt.colorbar(im, ax=ax, label='Re(X)', pad=0.01)

    # Overlay all channels
    ax2 = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_ant))
    for a in range(n_ant):
        tau = a * tau_step_mc
        ax2.plot(t_mc[:300], mc_data[a].real[:300],
                 color=cmap[a], lw=0.8, alpha=0.8,
                 label=f'Ant {a} (tau={tau*1e3:.1f}ms)')
    ax2.set_ylabel('Re(X)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('First 300 samples — delay visible')
    ax2.legend(loc='upper right', fontsize=7, ncol=4, framealpha=0.3)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot4_multichannel_waterfall.png'), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Plot 5: Noise — histogram + distribution
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 5] Noise: histogram + distribution...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    gen.set_params(fs=10000.0, points=100000, f0=0.0,
                   amplitude=0.0, noise_amplitude=1.0,
                   norm=1.0, noise_seed=42)
    noise_data = gen.generate()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'FormSignalGenerator: Noise Analysis (Philox + Box-Muller)  '
                 f'|  GPU: {gpu_name}', fontweight='bold')

    # Histogram Re
    ax = axes[0]
    ax.hist(noise_data.real, bins=200, density=True, color=COLORS['gpu_re'],
            alpha=0.7, edgecolor='none', label='GPU Re')
    x_gauss = np.linspace(-4, 4, 200)
    ax.plot(x_gauss, np.exp(-x_gauss**2 / 2) / np.sqrt(2 * np.pi),
            color=COLORS['cpu_re'], lw=2, label='N(0,1) theory')
    ax.set_title(f'Re histogram  (var={np.var(noise_data.real):.4f})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(framealpha=0.3)
    ax.grid(True)

    # Histogram Im
    ax = axes[1]
    ax.hist(noise_data.imag, bins=200, density=True, color=COLORS['gpu_im'],
            alpha=0.7, edgecolor='none', label='GPU Im')
    ax.plot(x_gauss, np.exp(-x_gauss**2 / 2) / np.sqrt(2 * np.pi),
            color=COLORS['cpu_im'], lw=2, label='N(0,1) theory')
    ax.set_title(f'Im histogram  (var={np.var(noise_data.imag):.4f})')
    ax.set_xlabel('Value')
    ax.legend(framealpha=0.3)
    ax.grid(True)

    # IQ scatter
    ax = axes[2]
    n_scatter = min(5000, len(noise_data))
    ax.scatter(noise_data.real[:n_scatter], noise_data.imag[:n_scatter],
               s=0.5, alpha=0.3, color=COLORS['noise'])
    circle = plt.Circle((0, 0), 1, fill=False, color=COLORS['accent'],
                         lw=1.5, ls='--', label='r=1 (1-sigma)')
    circle2 = plt.Circle((0, 0), 2, fill=False, color=COLORS['window'],
                          lw=1, ls=':', label='r=2 (2-sigma)')
    ax.add_patch(circle)
    ax.add_patch(circle2)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title('IQ Constellation')
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.legend(framealpha=0.3, loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot5_noise_analysis.png'), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Plot 6: Signal + Noise — spectrum & SNR
    # ══════════════════════════════════════════════════════════════════════
    print("  [Plot 6] Signal + Noise: spectrum...")

    gen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fs_sn, pts_sn, f0_sn = 100000.0, 8192, 10000.0
    amp_sn, an_sn = 1.0, 0.3

    gen.set_params(fs=fs_sn, points=pts_sn, f0=f0_sn,
                   amplitude=amp_sn, noise_amplitude=an_sn,
                   norm=norm_val, noise_seed=123)
    sn_data = gen.generate()

    # Чистый сигнал
    gen_pure = gpuworklib.FormSignalGeneratorROCm(ctx)
    gen_pure.set_params(fs=fs_sn, points=pts_sn, f0=f0_sn,
                        amplitude=amp_sn, noise_amplitude=0.0, norm=norm_val)
    pure_data = gen_pure.generate()

    # FFT
    freq_axis = np.fft.fftfreq(pts_sn, 1.0 / fs_sn)[:pts_sn // 2] / 1000.0  # kHz
    fft_sn = np.fft.fft(sn_data)[:pts_sn // 2]
    fft_pure = np.fft.fft(pure_data)[:pts_sn // 2]

    mag_sn_db = 20.0 * np.log10(np.abs(fft_sn) / pts_sn + 1e-12)
    mag_pure_db = 20.0 * np.log10(np.abs(fft_pure) / pts_sn + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'FormSignalGenerator: Signal + Noise Spectrum  '
                 f'f0={f0_sn/1e3:.0f} kHz, SNR ~ '
                 f'{20*np.log10(amp_sn/an_sn):.1f} dB  '
                 f'|  GPU: {gpu_name}', fontweight='bold')

    # Spectrum
    ax = axes[0]
    ax.plot(freq_axis, mag_pure_db, color=COLORS['signal'], lw=1.0,
            alpha=0.8, label=f'Pure signal (a={amp_sn})')
    ax.plot(freq_axis, mag_sn_db, color=COLORS['accent'], lw=0.6,
            alpha=0.7, label=f'Signal + Noise (an={an_sn})')
    ax.axvline(f0_sn / 1000.0, color=COLORS['cpu_re'], ls=':', lw=1,
               alpha=0.5, label=f'f0={f0_sn/1e3:.0f} kHz')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('FFT Spectrum')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.grid(True)
    ax.set_ylim(-100, 10)

    # Waveform comparison
    ax2 = axes[1]
    t_sn = np.arange(pts_sn) / fs_sn * 1000.0
    n_show = 500
    ax2.plot(t_sn[:n_show], pure_data.real[:n_show], color=COLORS['signal'],
             lw=1.5, alpha=0.9, label='Pure signal')
    ax2.plot(t_sn[:n_show], sn_data.real[:n_show], color=COLORS['accent'],
             lw=0.8, alpha=0.6, label='Signal + Noise')
    ax2.set_ylabel('Re(X)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Waveform (first 500 samples)')
    ax2.legend(loc='upper right', framealpha=0.3)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot6_signal_noise_spectrum.png'), dpi=150)
    plt.close()

    print(f"\n  All plots saved to {save_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FormSignalGenerator -- Python Tests (GPU vs NumPy)")
    print("=" * 60)

    gpus = gpuworklib.list_gpus()
    if not gpus:
        print("ERROR: No GPU found")
        return 1
    print(f"  GPU: {gpus[0]['name']} ({gpus[0]['memory_mb']} MB)")

    tests = [
        test_cw_no_noise,
        test_chirp,
        test_window,
        test_multi_channel,
        test_noise_statistics,
        test_string_params,
        test_signal_plus_noise,
    ]

    passed = 0
    total = len(tests)

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  FormSig Python Results: {passed}/{total} tests passed")
    print("=" * 60)

    # ── Графики (создаются по умолчанию) ──
    do_plot = '--no-plot' not in sys.argv
    if do_plot:
        plot_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Results',
                                'Plots', 'signal_generators', 'FormSignal')
        make_plots(os.path.abspath(plot_dir))

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
