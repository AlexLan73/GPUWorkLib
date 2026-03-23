"""
test_delayed_form_signal.py — Тесты DelayedFormSignalGenerator (Farrow 48×5)

Алгоритм:
  1. Генерация чистого сигнала getX (без шума)
  2. Применение дробной задержки: целый сдвиг + Lagrange 5-точечная интерполяция
  3. Добавление шума (опционально)

Тесты:
  1. Целая задержка — GPU vs NumPy (простой сдвиг)
  2. Дробная задержка — GPU vs NumPy (Lagrange интерполяция)
  3. Multi-channel с разными задержками
  4. Нулевая задержка — результат совпадает с FormSignalGenerator
  5. Задержка + шум — проверка SNR

Графики:
  1. Сравнение GPU vs NumPy для целой задержки
  2. Дробная задержка: исходный + задержанный сигнал overlay
  3. Multi-channel waterfall: 8 антенн с нарастающей задержкой
  4. Delay error: ошибка задержки vs задержка в сэмплах

@author Кодо (AI Assistant)
@date 2026-02-17
"""

import sys
import os
import json
import warnings
import numpy as np

# Подавить предупреждение Axes3D (тест не использует 3D; часто при двух установках matplotlib)
warnings.filterwarnings('ignore', message='Unable to import Axes3D', category=UserWarning)

# ── Путь к gpuworklib (Python_test/signal_generators/ -> 2 levels up) ──
BUILD_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'debian-radeon9070', 'python'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Debug'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Release'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'),
]
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

if not hasattr(gpuworklib, 'DelayedFormSignalGeneratorROCm'):
    print("ERROR: gpuworklib built without DelayedFormSignalGeneratorROCm.")
    print("  - Rebuild: cmake -B build -DBUILD_PYTHON=ON -DENABLE_ROCM=ON && cmake --build build")
    print("  - Loaded from:", getattr(gpuworklib, '__file__', '?'))
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Загрузка матрицы Lagrange 48×5
# ════════════════════════════════════════════════════════════════════════════

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'modules', 'lch_farrow',
    'lagrange_matrix_48x5.json')


def load_lagrange_matrix():
    """Загрузить матрицу 48×5 из JSON."""
    with open(MATRIX_PATH, 'r') as f:
        data = json.load(f)
    return np.array(data['data'], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: getX formula (без шума)
# ════════════════════════════════════════════════════════════════════════════

def getX_numpy(fs, points, f0, amplitude, phase, fdev, norm_val, tau=0.0):
    """CPU reference — формула getX без шума."""
    dt = 1.0 / fs
    ti = points / fs
    t = np.arange(points) * dt + tau

    # Окно
    in_window = (t >= 0) & (t <= ti - dt)

    t_centered = t - ti / 2
    ph = 2 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase

    sig = amplitude * norm_val * np.exp(1j * ph)
    sig[~in_window] = 0.0 + 0.0j
    return sig.astype(np.complex64)


def apply_delay_numpy(signal, delay_samples, lagrange_matrix):
    """
    CPU reference — применение дробной задержки через Lagrange 48×5.
    delay_samples — задержка в сэмплах (float).

    Зеркалит GPU-ядро (lch_farrow.cpp) точь-в-точь:
      read_pos = n - delay_samples   ← вычисляется PER-SAMPLE (не глобально!)
      center   = floor(read_pos)
      frac     = read_pos - center   ← дробная часть PER-SAMPLE
      row      = int(frac * 48) % 48 ← строка матрицы PER-SAMPLE

    ИСПРАВЛЕНО: старая версия использовала глобальный mu = delay - floor(delay),
    что давало неверные row/center для дробных задержек (ошибка ~3.5 вместо <0.01).
    """
    N = len(signal)
    output = np.zeros(N, dtype=np.complex64)
    delay_f32 = np.float32(delay_samples)

    for n in range(N):
        read_pos = float(n) - float(delay_f32)
        if read_pos < 0.0:
            output[n] = 0.0 + 0.0j
            continue
        center = int(np.floor(read_pos))
        frac = read_pos - center        # дробная часть: per-sample!
        row = int(frac * 48) % 48
        L = lagrange_matrix[row]
        val = 0.0 + 0.0j
        for k in range(5):
            idx = center - 1 + k
            if 0 <= idx < N:
                val += float(L[k]) * complex(signal[idx])
        output[n] = val

    return output


# ════════════════════════════════════════════════════════════════════════════
# Test 1: Целая задержка (integer delay)
# ════════════════════════════════════════════════════════════════════════════

def test_integer_delay():
    """Целая задержка (5 сэмплов) — GPU vs NumPy shift."""
    print("\n[Test 1] Integer delay...")

    fs = 1e6  # 1 MHz
    points = 4096
    f0 = 50000.0  # 50 kHz
    amplitude = 1.0
    norm_val = 1.0 / np.sqrt(2)

    # Задержка = 5 мкс при fs=1MHz → 5 сэмплов (целая)
    delay_us = 5.0
    delay_samples = delay_us * 1e-6 * fs  # = 5.0

    # GPU
    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    gen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                   amplitude=amplitude, norm=norm_val)
    gen.set_delays([delay_us])
    gpu_data = gen.generate()

    # NumPy reference
    clean = getX_numpy(fs, points, f0, amplitude, 0.0, 0.0, norm_val, tau=0.0)
    matrix = load_lagrange_matrix()
    ref = apply_delay_numpy(clean, delay_samples, matrix)

    # Compare
    max_err = np.max(np.abs(gpu_data.ravel() - ref))
    print(f"  delay_samples = {delay_samples}")
    print(f"  max_error = {max_err:.6e}")
    assert max_err < 1e-2, f"Integer delay error too large: {max_err}"
    print("  PASSED!")
    return gpu_data, ref, clean


# ════════════════════════════════════════════════════════════════════════════
# Test 2: Дробная задержка (fractional delay)
# ════════════════════════════════════════════════════════════════════════════

def test_fractional_delay():
    """Дробная задержка (2.7 сэмпла) — GPU vs NumPy Lagrange."""
    print("\n[Test 2] Fractional delay...")

    fs = 1e6
    points = 4096
    f0 = 50000.0
    amplitude = 1.0
    norm_val = 1.0 / np.sqrt(2)

    # Задержка = 2.7 мкс при fs=1MHz → 2.7 сэмплов
    delay_us = 2.7
    delay_samples = delay_us * 1e-6 * fs

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    gen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                   amplitude=amplitude, norm=norm_val)
    gen.set_delays([delay_us])
    gpu_data = gen.generate()

    clean = getX_numpy(fs, points, f0, amplitude, 0.0, 0.0, norm_val, tau=0.0)
    matrix = load_lagrange_matrix()
    ref = apply_delay_numpy(clean, delay_samples, matrix)

    max_err = np.max(np.abs(gpu_data.ravel() - ref))
    print(f"  delay_samples = {delay_samples}")
    print(f"  max_error = {max_err:.6e}")
    assert max_err < 1e-2, f"Fractional delay error too large: {max_err}"
    print("  PASSED!")
    return gpu_data, ref, clean


# ════════════════════════════════════════════════════════════════════════════
# Test 3: Multi-channel с нарастающей задержкой
# ════════════════════════════════════════════════════════════════════════════

def test_multichannel_delay():
    """8 антенн с задержкой 0, 1.5, 3.0, ..., 10.5 мкс."""
    print("\n[Test 3] Multi-channel delay...")

    fs = 1e6
    points = 4096
    f0 = 50000.0
    antennas = 8
    amplitude = 1.0
    norm_val = 1.0 / np.sqrt(2)
    delays = [i * 1.5 for i in range(antennas)]  # 0, 1.5, 3.0, ..., 10.5 мкс

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    gen.set_params(fs=fs, antennas=antennas, points=points, f0=f0,
                   amplitude=amplitude, norm=norm_val)
    gen.set_delays(delays)
    gpu_data = gen.generate()

    assert gpu_data.shape == (antennas, points), \
        f"Shape mismatch: {gpu_data.shape} != ({antennas}, {points})"

    # Check each channel vs reference
    clean = getX_numpy(fs, points, f0, amplitude, 0.0, 0.0, norm_val, tau=0.0)
    matrix = load_lagrange_matrix()
    max_errors = []

    for ch in range(antennas):
        delay_samples = delays[ch] * 1e-6 * fs
        ref = apply_delay_numpy(clean, delay_samples, matrix)
        err = np.max(np.abs(gpu_data[ch] - ref))
        max_errors.append(err)
        print(f"  ch{ch}: delay={delays[ch]:.1f}us ({delay_samples:.1f} samp) err={err:.6e}")

    max_err = max(max_errors)
    worst_ch = max_errors.index(max_err)
    # Допуск: float32 GPU vs float64 NumPy reference (fast-relaxed-math ~1e-3..1e-2)
    assert max_err < 1e-2, (
        f"Multi-channel error too large: {max_err} (worst channel: {worst_ch}, "
        f"delay={delays[worst_ch]:.1f} us)")
    print("  PASSED!")
    return gpu_data, delays


# ════════════════════════════════════════════════════════════════════════════
# Test 4: Нулевая задержка → совпадает с FormSignalGenerator
# ════════════════════════════════════════════════════════════════════════════

def test_zero_delay():
    """delay=0 → результат = FormSignalGenerator (без шума)."""
    print("\n[Test 4] Zero delay...")

    fs = 1e6
    points = 4096
    f0 = 100000.0
    amplitude = 1.0
    norm_val = 1.0 / np.sqrt(2)

    ctx = gpuworklib.ROCmGPUContext(0)
    # DelayedFormSignalGenerator с delay=0
    dgen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    dgen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                    amplitude=amplitude, norm=norm_val)
    dgen.set_delays([0.0])
    delayed = dgen.generate()

    # FormSignalGenerator (reference — без шума и задержки)
    fgen = gpuworklib.FormSignalGeneratorROCm(ctx)
    fgen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                    amplitude=amplitude, norm=norm_val)
    original = fgen.generate()

    max_err = np.max(np.abs(delayed.ravel() - original.ravel()))
    print(f"  max_error vs FormSignalGenerator = {max_err:.6e}")
    assert max_err < 1e-4, f"Zero delay error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 5: Задержка + шум — проверка SNR
# ════════════════════════════════════════════════════════════════════════════

def test_delay_with_noise():
    """Задержка + шум, проверяем что шум добавлен."""
    print("\n[Test 5] Delay + noise...")

    fs = 1e6
    points = 8192
    f0 = 50000.0
    amplitude = 1.0
    noise_amp = 0.2
    norm_val = 1.0 / np.sqrt(2)
    delay_us = 3.5

    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    gen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                   amplitude=amplitude, noise_amplitude=noise_amp,
                   norm=norm_val, noise_seed=42)
    gen.set_delays([delay_us])
    noisy = gen.generate()

    # Без шума
    gen2 = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
    gen2.set_params(fs=fs, antennas=1, points=points, f0=f0,
                    amplitude=amplitude, norm=norm_val)
    gen2.set_delays([delay_us])
    clean = gen2.generate()

    diff = noisy.ravel() - clean.ravel()
    noise_power = np.mean(np.abs(diff) ** 2)
    signal_power = np.mean(np.abs(clean.ravel()) ** 2)

    # Ожидаемая мощность шума: (noise_amp * norm)^2 * 2 (Re + Im)
    expected_noise = (noise_amp * norm_val) ** 2 * 2
    print(f"  noise_power = {noise_power:.6f}")
    print(f"  expected     = {expected_noise:.6f}")
    print(f"  signal_power = {signal_power:.6f}")

    # Проверяем порядок величины
    ratio = noise_power / expected_noise
    print(f"  ratio = {ratio:.3f} (should be ~1.0)")
    assert 0.5 < ratio < 2.0, f"Noise power ratio out of range: {ratio}"
    print("  PASSED!")
    return noisy, clean


# ════════════════════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════════════════════

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Plots', 'signal_generators', 'DelayedFormSignal')


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot1_integer_delay(gpu_data, ref, clean):
    """Plot 1: Целая задержка GPU vs NumPy."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not found, skipping plot")
        return

    ensure_plot_dir()

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

    x = np.arange(len(clean))

    # Original signal
    axes[0].plot(x[:200], clean[:200].real, color='#00d2ff', alpha=0.8, linewidth=0.8, label='Original Re')
    axes[0].plot(x[:200], clean[:200].imag, color='#ff6b6b', alpha=0.8, linewidth=0.8, label='Original Im')
    axes[0].set_title('Original Signal (no delay)')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    # GPU delayed vs reference
    gpu_flat = gpu_data.ravel()
    axes[1].plot(x[:200], gpu_flat[:200].real, color='#00ff88', alpha=0.8, linewidth=0.8, label='GPU Re')
    axes[1].plot(x[:200], ref[:200].real, color='#ff6b6b', alpha=0.6, linewidth=0.8, linestyle='--', label='NumPy Re')
    axes[1].set_title('Delayed Signal: GPU vs NumPy (Integer delay = 5 samples)')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    # Error
    error = np.abs(gpu_flat - ref)
    axes[2].semilogy(error[:500], color='#ffdd57', alpha=0.8, linewidth=0.5)
    axes[2].set_title(f'Absolute Error (max = {np.max(error):.2e})')
    axes[2].set_xlabel('Sample')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot1_integer_delay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def plot2_fractional_delay(gpu_data, ref, clean):
    """Plot 2: Дробная задержка overlay."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_plot_dir()

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

    gpu_flat = gpu_data.ravel()
    n_show = 200

    # Overlay: original vs delayed
    axes[0].plot(clean[:n_show].real, color='#00d2ff', alpha=0.8, linewidth=1.0, label='Original Re')
    axes[0].plot(gpu_flat[:n_show].real, color='#00ff88', alpha=0.8, linewidth=1.0, label='Delayed Re (GPU)')
    axes[0].plot(ref[:n_show].real, color='#ff6b6b', alpha=0.5, linewidth=1.0, linestyle='--', label='Delayed Re (NumPy)')
    axes[0].set_title('Fractional Delay = 2.7 samples')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    # Error
    error = np.abs(gpu_flat - ref)
    axes[1].semilogy(error[:1000], color='#ffdd57', alpha=0.8, linewidth=0.5)
    axes[1].set_title(f'Absolute Error (max = {np.max(error):.2e})')
    axes[1].set_xlabel('Sample')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot2_fractional_delay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def plot3_multichannel_waterfall(gpu_data, delays):
    """Plot 3: Multi-channel waterfall heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_plot_dir()

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

    n_show = 300

    # Heatmap
    heatmap = np.abs(gpu_data[:, :n_show])
    im = axes[0].imshow(heatmap, aspect='auto', cmap='inferno',
                        extent=[0, n_show, gpu_data.shape[0] - 0.5, -0.5])
    axes[0].set_title('Multi-Channel Waterfall (|signal|)')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Antenna')
    cb = plt.colorbar(im, ax=axes[0])
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('#444')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # Overlay: all channels Real part
    colors = plt.cm.viridis(np.linspace(0, 1, gpu_data.shape[0]))
    for ch in range(gpu_data.shape[0]):
        axes[1].plot(gpu_data[ch, :n_show].real, color=colors[ch],
                     alpha=0.7, linewidth=0.8,
                     label=f'ch{ch} ({delays[ch]:.1f}μs)')
    axes[1].set_title('Re(signal) per channel — visible delay shift')
    axes[1].set_xlabel('Sample')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
                   fontsize=7, ncol=4)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot3_multichannel_waterfall.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def plot4_delay_sweep():
    """Plot 4: Delay sweep — error vs fractional delay."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ensure_plot_dir()

    fs = 1e6
    points = 4096
    f0 = 50000.0
    amplitude = 1.0
    norm_val = 1.0 / np.sqrt(2)
    matrix = load_lagrange_matrix()

    delays_us = np.linspace(0.1, 10.0, 20)
    errors = []

    clean = getX_numpy(fs, points, f0, amplitude, 0.0, 0.0, norm_val)
    ctx = gpuworklib.ROCmGPUContext(0)

    for d_us in delays_us:
        gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
        gen.set_params(fs=fs, antennas=1, points=points, f0=f0,
                       amplitude=amplitude, norm=norm_val)
        gen.set_delays([float(d_us)])
        gpu = gen.generate().ravel()

        delay_samp = d_us * 1e-6 * fs
        ref = apply_delay_numpy(clean, delay_samp, matrix)
        err = np.max(np.abs(gpu - ref))
        errors.append(err)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#444')

    ax.semilogy(delays_us, errors, 'o-', color='#00ff88', markersize=5)
    ax.set_xlabel('Delay (μs)')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('GPU vs NumPy Error vs Delay (f0=50kHz, fs=1MHz)')
    ax.grid(True, alpha=0.2, color='white')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'plot4_delay_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Графики: по умолчанию ВКЛ. Выключить: --no-plot или env GPUWORKLIB_PLOT=0
    # В PyCharm: Run → Edit Configurations → Environment variables → GPUWORKLIB_PLOT=0 (выключить)
    no_plot = '--no-plot' in sys.argv
    if os.environ.get('GPUWORKLIB_PLOT', '').strip().lower() in ('0', 'false', 'no', 'off'):
        no_plot = True
    if os.environ.get('GPUWORKLIB_PLOT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        no_plot = False

    print("=" * 70)
    print("DelayedFormSignalGenerator Tests (Farrow 48×5)")
    print("=" * 70)

    # Tests
    gpu1, ref1, clean1 = test_integer_delay()
    gpu2, ref2, clean2 = test_fractional_delay()
    gpu3, delays3 = test_multichannel_delay()
    test_zero_delay()
    test_delay_with_noise()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

    # Plots
    if not no_plot:
        print("\n🎨 Generating plots...")
        plot1_integer_delay(gpu1, ref1, clean1)
        plot2_fractional_delay(gpu2, ref2, clean2)
        plot3_multichannel_waterfall(gpu3, delays3)
        plot4_delay_sweep()
        print("\nAll plots saved to:", PLOT_DIR)
