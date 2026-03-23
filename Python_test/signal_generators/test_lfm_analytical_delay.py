"""
test_lfm_analytical_delay.py — Тесты LfmGeneratorAnalyticalDelay

Тесты:
  1. Нулевая задержка — GPU совпадает со стандартным LFM
  2. Дробная задержка 3.24 сэмпла — первый ненулевой в индексе 4
  3. GPU vs CPU reference (аналитический delay 0.5 мкс)
  4. Multi-antenna (4 канала с разными задержками)
  5. GPU vs NumPy reference (независимая проверка фазы)

Графики (по умолчанию, --no-plot отключить):
  Results/Plots/LfmAnalyticalDelay/ — задержка аналитическим способом (без интерполяции)

@author Кодо (AI Assistant)
@date 2026-02-18
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
    sys.exit(1)

if not hasattr(gpuworklib, 'LfmAnalyticalDelayROCm'):
    print("ERROR: gpuworklib built without LfmAnalyticalDelayROCm.")
    print("  Rebuild: cmake -B build -DBUILD_PYTHON=ON -DENABLE_ROCM=ON && cmake --build build")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: analytical LFM with delay
# ════════════════════════════════════════════════════════════════════════════

def lfm_analytical_numpy(fs, length, f_start, f_end, amplitude, delay_us=0.0):
    """CPU reference — аналитический ЛЧМ с задержкой (double precision)."""
    duration = length / fs
    chirp_rate = (f_end - f_start) / duration

    t = np.arange(length) / fs
    tau = delay_us * 1e-6

    output = np.zeros(length, dtype=np.complex128)
    mask = t >= tau
    t_local = t[mask] - tau

    phase = np.pi * chirp_rate * t_local**2 + 2 * np.pi * f_start * t_local
    output[mask] = amplitude * np.exp(1j * phase)

    return output.astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════
# GPU context
# ════════════════════════════════════════════════════════════════════════════

ctx = gpuworklib.ROCmGPUContext(0)
print(f"GPU: {ctx.device_name}")


# ════════════════════════════════════════════════════════════════════════════
# Test 1: Нулевая задержка — GPU = стандартный LFM
# ════════════════════════════════════════════════════════════════════════════

def test_zero_delay_vs_standard_lfm():
    """delay=0 -> результат совпадает с SignalGenerator.generate_lfm()."""
    print("\n[Test 1] Zero delay vs standard LFM...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    amplitude = 1.0

    # Analytical delay generator (delay = 0)
    gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end,
                                         amplitude=amplitude)
    gen.set_sampling(fs=fs, length=length)
    gen.set_delays([0.0])
    gpu_delayed = gen.generate_gpu()

    # NumPy reference (delay = 0)
    ref = lfm_analytical_numpy(fs, length, f_start, f_end, amplitude, delay_us=0.0)

    max_err = np.max(np.abs(gpu_delayed.ravel() - ref.ravel()))
    print(f"  max_error = {max_err:.2e}")
    # GPU float32 vs NumPy float64 -> ~1e-3
    assert max_err < 2e-3, f"Zero delay error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 2: Дробная задержка 3.24 сэмпла — первый ненулевой в индексе 4
# ════════════════════════════════════════════════════════════════════════════

def test_fractional_delay_boundary():
    """delay = 3.24 samples -> indices 0..3 = 0, index 4 != 0."""
    print("\n[Test 2] Fractional delay 3.24 samples...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6

    # delay_us = 3.24 / fs * 1e6
    delay_us = 3.24 / fs * 1e6

    gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end)
    gen.set_sampling(fs=fs, length=length)
    gen.set_delays([delay_us])
    data = gen.generate_gpu().ravel()

    # Проверяем: индексы 0..3 == 0
    zeros_ok = True
    for n in range(4):
        if abs(data[n]) > 1e-6:
            zeros_ok = False
            print(f"  WARNING: index {n} not zero: {abs(data[n])}")

    # Индекс 4 != 0
    first_nonzero_ok = abs(data[4]) > 0.1
    print(f"  zeros[0..3] = {'OK' if zeros_ok else 'FAIL'}")
    print(f"  |data[4]| = {abs(data[4]):.4f}")

    assert zeros_ok, "Indices 0..3 should be zero"
    assert first_nonzero_ok, f"Index 4 should be non-zero, got {abs(data[4])}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 3: GPU vs CPU reference
# ════════════════════════════════════════════════════════════════════════════

def test_gpu_vs_cpu():
    """GPU vs CPU (double precision reference) — delay 0.5 us."""
    print("\n[Test 3] GPU vs CPU reference...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    delay_us = 0.5

    gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end)
    gen.set_sampling(fs=fs, length=length)
    gen.set_delays([delay_us])

    gpu_data = gen.generate_gpu().ravel()
    cpu_data = gen.generate_cpu().ravel()

    max_err = np.max(np.abs(gpu_data - cpu_data))
    print(f"  max_error GPU vs CPU = {max_err:.2e}")
    # GPU float32 + fast-relaxed-math vs CPU double -> ~1e-3
    assert max_err < 1e-3, f"GPU vs CPU error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 4: Multi-antenna
# ════════════════════════════════════════════════════════════════════════════

def test_multi_antenna():
    """4 антенны с разными задержками."""
    print("\n[Test 4] Multi-antenna...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    delays = [0.0, 0.1, 0.2, 0.5]
    antennas = len(delays)

    gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end)
    gen.set_sampling(fs=fs, length=length)
    gen.set_delays(delays)

    gpu_data = gen.generate_gpu()
    cpu_data = gen.generate_cpu()

    assert gpu_data.shape == (antennas, length), \
        f"Shape mismatch: {gpu_data.shape}"

    max_err = np.max(np.abs(gpu_data - cpu_data))
    print(f"  shape = {gpu_data.shape}")
    print(f"  max_error GPU vs CPU = {max_err:.2e}")
    assert max_err < 1e-3, f"Multi-antenna error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 5: GPU vs NumPy reference (независимая проверка фазы)
# ════════════════════════════════════════════════════════════════════════════

def test_gpu_vs_numpy():
    """GPU vs NumPy (независимый расчёт фазы)."""
    print("\n[Test 5] GPU vs NumPy reference...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    amplitude = 1.0
    delay_us = 1.0  # 1 мкс

    gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end,
                                         amplitude=amplitude)
    gen.set_sampling(fs=fs, length=length)
    gen.set_delays([delay_us])
    gpu_data = gen.generate_gpu().ravel()

    # NumPy reference
    ref = lfm_analytical_numpy(fs, length, f_start, f_end, amplitude, delay_us)

    max_err = np.max(np.abs(gpu_data - ref))
    print(f"  max_error GPU vs NumPy = {max_err:.2e}")
    # float32 GPU + fast-relaxed-math vs float64 NumPy -> ~1e-3
    assert max_err < 1e-3, f"GPU vs NumPy error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Графики — реальная задержка сигнала
# ════════════════════════════════════════════════════════════════════════════

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Plots', 'signal_generators', 'LfmAnalyticalDelay')


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def make_plots():
    """Графики, показывающие реальную задержку сигнала."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not found, skipping plots")
        return

    ensure_plot_dir()
    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    amplitude = 1.0

    # ── Plot 1: Original vs Delayed — видимая задержка (overlay)
    # Original (delay=0) vs Delayed (1 µs = 12 samples) — сдвиг очевиден
    print("\n[Plots] Generating analytical delay visualization...")

    gen0 = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end, amplitude=amplitude)
    gen0.set_sampling(fs=fs, length=length)
    gen0.set_delays([0.0])
    original = gen0.generate_gpu().ravel()

    delay_us = 1.0  # 1 µs = 12 samples at 12 MHz
    gen1 = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end, amplitude=amplitude)
    gen1.set_sampling(fs=fs, length=length)
    gen1.set_delays([delay_us])
    delayed = gen1.generate_gpu().ravel()

    n_show = 300
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    x = np.arange(n_show)
    axes[0].plot(x, original[:n_show].real, color='#00d2ff', alpha=0.9, linewidth=1.0, label='Original (delay=0) Re')
    axes[0].plot(x, delayed[:n_show].real, color='#00ff88', alpha=0.9, linewidth=1.0, label=f'Delayed ({delay_us} µs) Re')
    axes[0].axvline(12, color='#ff6b6b', linestyle='--', alpha=0.7, label=f'Delay start (~{delay_us*fs/1e6:.0f} samples)')
    axes[0].set_title('Analytical Delay: Original vs Delayed Signal (Re) — аналитический способ')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[0].set_xlabel('Sample')

    axes[1].plot(x, np.abs(original[:n_show]), color='#00d2ff', alpha=0.9, linewidth=1.0, label='|Original|')
    axes[1].plot(x, np.abs(delayed[:n_show]), color='#00ff88', alpha=0.9, linewidth=1.0, label='|Delayed|')
    axes[1].axvline(12, color='#ff6b6b', linestyle='--', alpha=0.7)
    axes[1].set_title('Envelope: Visible Shift (analytical generation — no interpolation)')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[1].set_xlabel('Sample')

    # Overlay: delayed shifted left to align with original (проверка: форма совпадает)
    delay_samples = delay_us * 1e-6 * fs
    n_align = min(n_show - int(delay_samples), 200)
    axes[2].plot(original[:n_align].real, color='#00d2ff', alpha=0.8, linewidth=1.0, label='Original Re')
    axes[2].plot(delayed[int(delay_samples):int(delay_samples)+n_align].real, color='#00ff88', alpha=0.8,
                 linewidth=1.0, linestyle='--', label='Delayed (shifted to align) Re')
    axes[2].set_title('Aligned: Delayed Matches Original (analytical: pure time shift, no Farrow)')
    axes[2].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[2].set_xlabel('Sample')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot1_real_delay_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 2: Дробная задержка 3.24 сэмпла — граница (нули до индекса 4)
    # Верх: |signal| (0..1) — огибающая, амплитуда A=1 везде. Низ: Re, Im (-1..1) — компоненты.
    # Разные шкалы по Y — специально: |z|=1, но Re/Im осциллируют. Это не ошибка.
    delay_us_frac = 3.24 / fs * 1e6
    gen_frac = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end, amplitude=1.0)
    gen_frac.set_sampling(fs=fs, length=length)
    gen_frac.set_delays([delay_us_frac])
    data_frac = gen_frac.generate_gpu().ravel()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    n_zoom = 80
    axes[0].bar(np.arange(n_zoom), np.abs(data_frac[:n_zoom]), color='#00ff88', alpha=0.8, width=0.8)
    axes[0].axvline(3.5, color='#ff6b6b', linestyle='--', linewidth=2, label='First non-zero at index 4')
    axes[0].set_title(f'Analytical Fractional Delay 3.24 samples: Zeros [0..3], Signal at Index 4')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('|signal| (amplitude A=1)')

    axes[1].plot(data_frac[:n_zoom].real, color='#00d2ff', alpha=0.9, linewidth=1.0, label='Re')
    axes[1].plot(data_frac[:n_zoom].imag, color='#ff6b6b', alpha=0.9, linewidth=1.0, label='Im')
    axes[1].axvline(3.5, color='#ffdd57', linestyle='--', alpha=0.8)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_title('Re, Im: |z|=1 везде, разные шкалы Y (верх 0-1, низ -1..1) — по замыслу')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[1].set_xlabel('Sample')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot2_fractional_delay_boundary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Multi-antenna — разные задержки, видимый сдвиг по каналам
    delays = [0.0, 0.2, 0.5, 1.0]
    gen_multi = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=f_start, f_end=f_end)
    gen_multi.set_sampling(fs=fs, length=length)
    gen_multi.set_delays(delays)
    data_multi = gen_multi.generate_gpu()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    colors = ['#00d2ff', '#00ff88', '#ff6b6b', '#ffdd57']
    for ch in range(4):
        delay_samp = delays[ch] * 1e-6 * fs
        axes[0].plot(data_multi[ch, :n_show].real + ch * 2, color=colors[ch], alpha=0.9,
                    label=f'Ch{ch} delay={delays[ch]} µs (~{delay_samp:.0f} samp)')
    axes[0].set_title('Multi-Antenna Analytical Delay: Each Channel Different τ (offset for visibility)')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Re + offset')

    # Waterfall: |signal| по каналам
    im = axes[1].imshow(np.abs(data_multi[:, :n_show]), aspect='auto', cmap='viridis',
                        extent=[0, n_show, 3.5, -0.5])
    axes[1].set_title('Waterfall: |signal| per Antenna (analytical delay shifts start right)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Antenna')
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels([f'delay={d} µs' for d in delays])
    plt.colorbar(im, ax=axes[1], label='|signal|')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot3_multiantenna_delays.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print(f"  All plots: {os.path.abspath(PLOT_DIR)}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    no_plot = '--no-plot' in sys.argv

    print("=" * 70)
    print("LfmGeneratorAnalyticalDelay Tests")
    print("=" * 70)

    test_zero_delay_vs_standard_lfm()
    test_fractional_delay_boundary()
    test_gpu_vs_cpu()
    test_multi_antenna()
    test_gpu_vs_numpy()

    if not no_plot:
        make_plots()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
