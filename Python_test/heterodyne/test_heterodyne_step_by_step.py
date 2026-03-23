"""
test_heterodyne_step_by_step.py
=================================
Step-by-step dechirp pipeline test with intermediate values and annotated plots.

Each step prints values and saves a plot to Results/Plots/heterodyne/step_XX_*.png.
Runs GPU (gpuworklib.HeterodyneDechirp) and CPU (NumPy) in parallel for comparison.

Steps:
  1. Generate s_rx (5 antennas, linear delays) - GPU + NumPy
  2. Generate s_ref* (conjugate LFM, delay=0) - NumPy reference
  3. Dechirp: s_dc = conj(s_rx * s_ref*) - NumPy
  4. FFT of dechirped signal - NumPy
  5. FindMaxima: f_beat, R - GPU pipeline + NumPy argmax
  6. Dechirp correct: compensate f_beat - NumPy
  7. Verify DC: final FFT - NumPy
  8. GPU Pipeline: full pass, GPU vs CPU summary

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s

@author Kodo (AI Assistant)
@date 2026-02-21
"""

import sys
import os
import numpy as np

# -- Path to gpuworklib --
BUILD_PATHS = [
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
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, plots will be skipped")

# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6
F_START = 0.0
F_END = 2e6
B = F_END - F_START   # 2 MHz
N = 8000
ANTENNAS = 5
T = N / FS            # 666.67 us
MU = B / T            # 3e9 Hz/s
C_LIGHT = 3e8

DELAYS_LINEAR_US = np.array([100., 200., 300., 400., 500.])
DELAYS_LINEAR_S  = DELAYS_LINEAR_US * 1e-6

F_BEATS_EXPECTED = MU * DELAYS_LINEAR_S
RANGES_TRUE      = C_LIGHT * DELAYS_LINEAR_S / 2.0

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                          'Results', 'Plots', 'heterodyne')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def generate_rx_numpy(delays_s):
    """Generate delayed LFM signals on CPU."""
    t = np.arange(N, dtype=np.float64) / FS
    rx = np.zeros((len(delays_s), N), dtype=np.complex64)
    for i, tau in enumerate(delays_s):
        t_delayed = t - tau
        phase = 2 * np.pi * (0.5 * MU * t_delayed**2 + F_START * t_delayed)
        rx[i, :] = np.exp(1j * phase).astype(np.complex64)
    return rx


def generate_ref_conjugate_numpy():
    """Generate conjugate LFM reference: s_ref* = exp(-j*[pi*mu*t^2 + 2pi*f0*t])."""
    t = np.arange(N, dtype=np.float64) / FS
    phase = -(np.pi * MU * t**2 + 2 * np.pi * F_START * t)
    return np.exp(1j * phase).astype(np.complex64)


def parabolic_interp(mag, idx):
    """Parabolic interpolation around peak bin."""
    if idx == 0 or idx >= len(mag) - 1:
        return float(idx), mag[idx]
    L, C, R = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = L - 2 * C + R
    if abs(denom) < 1e-12:
        return float(idx), C
    delta = 0.5 * (L - R) / denom
    refined_mag = C - 0.25 * (L - R) * delta
    return idx + delta, refined_mag


def save_plot(filename, fig):
    """Save figure to PLOTS_DIR."""
    if not HAS_MATPLOTLIB:
        return
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Plot saved: {path}")


def add_params_banner(fig, step_num, step_title, description):
    """Add a parameter/description banner at bottom of figure."""
    fig.text(0.5, 0.01,
             f'Шаг {step_num}: {step_title} | '
             f'fs={FS/1e6:.0f} МГц  B={B/1e6:.0f} МГц  N={N}  '
             f'T={T*1e6:.1f} мкс  μ={MU:.2e} Гц/с | '
             f'{description}',
             ha='center', fontsize=8, color='gray', style='italic',
             wrap=True)


# ============================================================================
# Steps
# ============================================================================

def step01_generate_rx():
    """Step 1: Generate received LFM signals (5 antennas, linear delays).

    s_rx[k, n] = exp( j * 2pi * ( 0.5*mu*(t[n]-tau_k)^2 + f_start*(t[n]-tau_k) ) )

    Каждая антенна принимает задержанную копию переданного чирпа.
    Задержка tau_k соответствует дальности R_k = c*tau_k/2.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Generate s_rx (5 antennas, linear delays)")
    print("=" * 60)
    print(f"  s_rx[k,n] = exp(j*2pi*(0.5*mu*(t-tau)^2 + f0*(t-tau)))")
    print(f"  tau in {DELAYS_LINEAR_US.tolist()} us")

    rx_cpu = generate_rx_numpy(DELAYS_LINEAR_S)

    for k in range(ANTENNAS):
        max_val = np.max(np.abs(rx_cpu[k]))
        f_true  = F_BEATS_EXPECTED[k]
        print(f"  Ant {k}: tau={DELAYS_LINEAR_US[k]:.0f} мкс, "
              f"max|s_rx|={max_val:.4f}, "
              f"f_beat_exp={f_true/1e3:.0f} кГц")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(ANTENNAS, 1, figsize=(13, 9), sharex=True)
        t_us = np.arange(N) / FS * 1e6
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, ANTENNAS))

        for k in range(ANTENNAS):
            axes[k].plot(t_us[:600], np.real(rx_cpu[k, :600]),
                         color=colors[k], lw=0.7)
            axes[k].set_ylabel(f'Re(s_rx[{k}])', fontsize=8)
            axes[k].set_ylim(-1.3, 1.5)
            axes[k].text(0.01, 0.85,
                         f'tau={DELAYS_LINEAR_US[k]:.0f} мкс  '
                         f'f_exp={F_BEATS_EXPECTED[k]/1e3:.0f} кГц',
                         transform=axes[k].transAxes, fontsize=8,
                         color=colors[k],
                         bbox=dict(fc='white', ec='gray', alpha=0.7, pad=2))
            axes[k].grid(True, alpha=0.25)

        axes[-1].set_xlabel('Время [мкс]')
        fig.suptitle('Шаг 1: Принятые LFM сигналы s_rx (5 антенн)\n'
                     'Реальная часть, первые 600 отсчётов',
                     fontsize=11, fontweight='bold')
        add_params_banner(fig, 1, 'Генерация s_rx',
                          'Каждая антенна = задержанная копия LFM-чирпа')
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_plot('step_01_rx_signals.png', fig)

    return rx_cpu


def step02_generate_ref_conjugate():
    """Step 2: Generate conjugate LFM reference s_ref*.

    s_ref*(n) = conj(LFM(n)) = exp(-j * (pi*mu*t^2 + 2*pi*f0*t))

    Опорный сигнал — задержка=0, фаза нарастает квадратично.
    После умножения rx * ref* и взятия conj() получаем чистое биение.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Generate s_ref* = conj(LFM), delay=0")
    print("=" * 60)
    print("  s_ref*(n) = exp(-j*(pi*mu*t^2 + 2*pi*f0*t))")

    ref_cpu = generate_ref_conjugate_numpy()
    print(f"  ref[0]   = {ref_cpu[0]:.6f}")
    print(f"  ref[100] = {ref_cpu[100]:.6f}")
    print(f"  max|ref| = {np.max(np.abs(ref_cpu)):.6f}  (всегда = 1.0)")

    if HAS_MATPLOTLIB:
        t_us = np.arange(N) / FS * 1e6
        fig, axes = plt.subplots(3, 1, figsize=(13, 8))

        axes[0].plot(t_us, np.real(ref_cpu), color='steelblue', lw=0.5)
        axes[0].set_ylabel('Re(s_ref*)', fontsize=9)
        axes[0].set_title('Вещественная часть — частота нарастает (LFM)', fontsize=9)
        axes[0].grid(True, alpha=0.25)
        axes[0].text(0.02, 0.85,
                     'Вид "chirp": частота мгновенно растёт от 0 до B=2 МГц',
                     transform=axes[0].transAxes, fontsize=8,
                     bbox=dict(fc='lightyellow', ec='navy', alpha=0.8, pad=3))

        axes[1].plot(t_us, np.imag(ref_cpu), color='coral', lw=0.5)
        axes[1].set_ylabel('Im(s_ref*)', fontsize=9)
        axes[1].set_title('Мнимая часть', fontsize=9)
        axes[1].grid(True, alpha=0.25)

        phase_unwrap = np.unwrap(np.angle(ref_cpu))
        axes[2].plot(t_us, phase_unwrap, color='purple', lw=0.8)
        axes[2].set_ylabel('Фаза [рад]', fontsize=9)
        axes[2].set_xlabel('Время [мкс]', fontsize=9)
        axes[2].set_title('Развёрнутая фаза — квадратичный рост (μt²)', fontsize=9)
        axes[2].grid(True, alpha=0.25)
        axes[2].text(0.02, 0.85,
                     f'phi(t) = -(pi*mu*t^2 + 2*pi*f0*t)\n'
                     f'mu = {MU:.2e} Гц/с',
                     transform=axes[2].transAxes, fontsize=8,
                     bbox=dict(fc='lavender', ec='purple', alpha=0.8, pad=3))

        fig.suptitle('Шаг 2: Опорный сигнал s_ref* = conj(LFM)\n'
                     'Нулевая задержка, нормированная амплитуда |s_ref*|=1',
                     fontsize=11, fontweight='bold')
        add_params_banner(fig, 2, 'Опорный сигнал',
                          'conj(LFM): квадратичная фаза, постоянная огибающая')
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_plot('step_02_ref_conjugate.png', fig)

    return ref_cpu


def step03_dechirp(rx_cpu, ref_cpu):
    """Step 3: Dechirp s_dc = conj(s_rx * s_ref*).

    Математика:
      s_rx * s_ref* = exp(j*2pi*(-mu*tau*t + 0.5*mu*tau^2))
      conj(...) = exp(j*2pi*(+mu*tau*t - 0.5*mu*tau^2))
                = exp(j*2pi*f_beat*t) * const_phase

    Результат — чистая комплексная синусоида с частотой f_beat = mu*tau.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Dechirp s_dc = conj(s_rx * s_ref*)")
    print("=" * 60)
    print("  Результат: s_dc[k] = exp(j*2pi*f_beat_k*t) * const")
    print("  f_beat_k = mu * tau_k")

    dc_cpu = np.zeros_like(rx_cpu)
    for k in range(ANTENNAS):
        product = rx_cpu[k] * ref_cpu
        dc_cpu[k] = np.conj(product)

    for k in range(ANTENNAS):
        f_exp = F_BEATS_EXPECTED[k]
        print(f"  Ant {k}: max|s_dc|={np.max(np.abs(dc_cpu[k])):.4f}, "
              f"f_beat_exp={f_exp/1e3:.0f} кГц")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(ANTENNAS, 1, figsize=(13, 9), sharex=True)
        t_us = np.arange(N) / FS * 1e6
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, ANTENNAS))

        for k in range(ANTENNAS):
            n_show = min(int(3e6 / F_BEATS_EXPECTED[k] * FS), 2000)
            axes[k].plot(t_us[:n_show], np.real(dc_cpu[k, :n_show]),
                         color=colors[k], lw=0.8)
            axes[k].set_ylabel(f'Re(s_dc[{k}])', fontsize=8)
            axes[k].set_ylim(-1.3, 1.5)
            f_k = F_BEATS_EXPECTED[k]
            axes[k].text(0.01, 0.82,
                         f'f_beat = {f_k/1e3:.0f} кГц  '
                         f'(tau={DELAYS_LINEAR_US[k]:.0f} мкс)',
                         transform=axes[k].transAxes, fontsize=8,
                         color=colors[k],
                         bbox=dict(fc='white', ec='gray', alpha=0.7, pad=2))
            axes[k].grid(True, alpha=0.25)

        axes[-1].set_xlabel('Время [мкс]')
        fig.suptitle('Шаг 3: Децимпированный сигнал Re(s_dc)\n'
                     's_dc = conj(s_rx × s_ref*)  —  чистые синусоидальные биения',
                     fontsize=11, fontweight='bold')
        add_params_banner(fig, 3, 'Дечирп',
                          'Видны биения: чем больше задержка -> выше частота')
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_plot('step_03_dechirp.png', fig)

    return dc_cpu


def step04_fft(dc_cpu):
    """Step 4: FFT of dechirped signal.

    N=8000 -> zero-pad to nFFT=8192 (next power of 2).
    Каждая антенна даёт узкий пик на частоте f_beat = mu*tau.
    Ширина бина: bin_width = fs / nFFT = 12e6 / 8192 ≈ 1465 Гц.
    """
    print("\n" + "=" * 60)
    print("STEP 4: FFT of dechirped signal")
    print("=" * 60)

    nfft = 8192
    spec_cpu = np.zeros((ANTENNAS, nfft), dtype=np.complex64)
    for k in range(ANTENNAS):
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[:N] = dc_cpu[k]
        spec_cpu[k] = np.fft.fft(padded)

    freqs = np.fft.fftfreq(nfft, d=1.0 / FS)
    bin_w = FS / nfft

    print(f"  nFFT = {nfft},  bin_width = {bin_w:.1f} Гц")
    for k in range(ANTENNAS):
        mag = np.abs(spec_cpu[k, :nfft // 2])
        peak_bin  = np.argmax(mag)
        peak_freq = freqs[peak_bin]
        exp_freq  = F_BEATS_EXPECTED[k]
        err       = abs(peak_freq - exp_freq)
        print(f"  Ant {k}: пик bin={peak_bin}, f_peak={peak_freq:.0f} Гц, "
              f"f_exp={exp_freq:.0f} Гц, ошибка={err:.0f} Гц "
              f"(~{err/bin_w:.2f} бина)")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))

        # Верхний: амплитудный спектр dB
        ax_db = axes[0]
        f_khz = freqs[:nfft // 2] / 1e3
        colors = plt.cm.tab10(np.linspace(0, 0.5, ANTENNAS))
        for k in range(ANTENNAS):
            mag_db = 20 * np.log10(np.abs(spec_cpu[k, :nfft // 2]) + 1e-6)
            ax_db.plot(f_khz, mag_db, color=colors[k], alpha=0.85, lw=1.0,
                       label=f'Ant{k} τ={DELAYS_LINEAR_US[k]:.0f}мкс')
            # Аннотация пика
            mag_lin = np.abs(spec_cpu[k, :nfft // 2])
            pk_bin = np.argmax(mag_lin)
            pk_f   = freqs[pk_bin] / 1e3
            pk_db  = mag_db[pk_bin]
            ax_db.annotate(f'{F_BEATS_EXPECTED[k]/1e3:.0f}кГц',
                           (pk_f, pk_db),
                           textcoords='offset points', xytext=(2, -14),
                           fontsize=7.5, color=colors[k],
                           arrowprops=dict(arrowstyle='->', color=colors[k],
                                           lw=0.8))
        ax_db.set_xlabel('Частота [кГц]', fontsize=9)
        ax_db.set_ylabel('Амплитуда [дБ]', fontsize=9)
        ax_db.set_title('FFT спектр дечирпированного сигнала — каждая антенна '
                        'даёт пик на f_beat', fontsize=10)
        ax_db.set_xlim([0, 1800])
        ax_db.legend(fontsize=8, loc='upper right')
        ax_db.grid(True, alpha=0.3)
        ax_db.text(0.02, 0.97,
                   f'N={N} -> zero-pad -> nFFT={nfft}\n'
                   f'bin_width = fs/nFFT = {bin_w:.0f} Гц\n'
                   f'Пик: f_beat = mu × tau',
                   transform=ax_db.transAxes, va='top', fontsize=8,
                   bbox=dict(fc='lightyellow', ec='navy', alpha=0.85, pad=3))

        # Нижний: линейная шкала (крупнее)
        ax_lin = axes[1]
        for k in range(ANTENNAS):
            mag_lin = np.abs(spec_cpu[k, :nfft // 2]) / N
            ax_lin.plot(f_khz, mag_lin, color=colors[k], alpha=0.85, lw=1.0,
                        label=f'Ant{k}')
        ax_lin.set_xlabel('Частота [кГц]', fontsize=9)
        ax_lin.set_ylabel('Нормированная амплитуда', fontsize=9)
        ax_lin.set_title('Линейная шкала — высота пиков ≈ 1.0 (идеальный сигнал)',
                         fontsize=10)
        ax_lin.set_xlim([0, 1800])
        ax_lin.legend(fontsize=8, loc='upper right')
        ax_lin.grid(True, alpha=0.3)

        fig.suptitle('Шаг 4: FFT дечирпированного сигнала\n'
                     'Каждый пик соответствует дальности одной антенны',
                     fontsize=11, fontweight='bold')
        add_params_banner(fig, 4, 'FFT спектр',
                          'Пики: f_beat = mu*tau. Ширина бина ≈ 1465 Гц')
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_plot('step_04_fft_spectrum.png', fig)

    return spec_cpu


def step05_find_maxima(spec_cpu):
    """Step 5: Find peak -> f_beat -> range (CPU parabolic interpolation)."""
    print("\n" + "=" * 60)
    print("STEP 5: FindMaxima -> f_beat -> Range")
    print("=" * 60)
    print("  Параболическая интерполяция: delta = 0.5*(L-R)/(L-2C+R)")

    nfft = spec_cpu.shape[1]
    results_cpu = []

    hdr = (f"  {'Ant':>3} | {'tau мкс':>7} | {'f_beat Гц':>10} | "
           f"{'f_exp Гц':>10} | {'df Гц':>8} | "
           f"{'R м':>8} | {'R_true м':>8} | {'dR м':>6}")
    print(hdr)
    print("  " + "─" * 77)

    for k in range(ANTENNAS):
        mag = np.abs(spec_cpu[k, :nfft // 2])
        peak_bin = np.argmax(mag)
        refined_bin, _ = parabolic_interp(mag, peak_bin)
        f_beat  = refined_bin * FS / nfft
        range_m = C_LIGHT * T * f_beat / (2 * B)
        exp_f   = F_BEATS_EXPECTED[k]
        f_err   = abs(f_beat - exp_f)
        r_err   = abs(range_m - RANGES_TRUE[k])

        results_cpu.append({
            'f_beat': f_beat, 'range_m': range_m,
            'f_error': f_err, 'r_error': r_err
        })
        print(f"  {k:>3} | {DELAYS_LINEAR_US[k]:>7.0f} | {f_beat:>10.0f} | "
              f"{exp_f:>10.0f} | {f_err:>8.0f} | "
              f"{range_m:>8.2f} | {RANGES_TRUE[k]:>8.2f} | {r_err:>6.2f}")

    return results_cpu


def step06_dechirp_correct(dc_cpu, results_cpu):
    """Step 6: Frequency correction — compensate f_beat.

    s_corr[k, n] = s_dc[k, n] * exp(-j*2pi*f_beat_k*t[n])

    После компенсации сигнал должен стать DC (нулевая частота).
    """
    print("\n" + "=" * 60)
    print("STEP 6: Dechirp correction (compensate f_beat)")
    print("=" * 60)
    print("  s_corr[k] = s_dc[k] * exp(-j*2pi*f_beat*t)")

    corrected = np.zeros_like(dc_cpu)
    t = np.arange(N, dtype=np.float64) / FS

    for k in range(ANTENNAS):
        f_beat = results_cpu[k]['f_beat']
        correction = np.exp(-1j * 2 * np.pi * f_beat * t).astype(np.complex64)
        corrected[k] = dc_cpu[k] * correction
        print(f"  Ant {k}: f_beat={f_beat/1e3:.1f} кГц, "
              f"max|corrected|={np.max(np.abs(corrected[k])):.4f}")

    return corrected


def step07_verify_dc(corrected):
    """Step 7: Verify peak at DC (bin=0) after correction.

    После частотной компенсации s_corr должен быть на нулевой частоте.
    FFT даёт пик в бине 0 (DC). Это подтверждает правильность f_beat.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Verify DC (peak at 0 Hz after correction)")
    print("=" * 60)
    print("  После компенсации FFT пик должен быть в бине 0")

    nfft = 8192
    all_dc = True
    dc_bins = []

    for k in range(ANTENNAS):
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[:N] = corrected[k]
        spectrum = np.fft.fft(padded)
        mag = np.abs(spectrum[:nfft // 2])
        peak_bin = np.argmax(mag)
        dc_bins.append(peak_bin)
        print(f"  Ant {k}: пик в бине {peak_bin} (ожидаем ~0)")
        if peak_bin > 3:
            all_dc = False

    print(f"  {'PASSED: все пики на DC' if all_dc else 'FAILED: некоторые пики не на DC'}")


def step08_gpu_pipeline():
    """Step 8: Full GPU pipeline, compare GPU vs CPU.

    GPU: gpuworklib.HeterodyneDechirp.process()
    CPU: NumPy FFT + parabolic interpolation

    Сравниваем f_beat, дальность и SNR между реализациями.
    """
    print("\n" + "=" * 60)
    print("STEP 8: GPU Pipeline (HeterodyneDechirp.process)")
    print("=" * 60)

    ctx = gpuworklib.ROCmGPUContext(0)
    het = gpuworklib.HeterodyneDechirp(ctx)
    het.set_params(F_START, F_END, FS, N, ANTENNAS)

    rx_cpu = generate_rx_numpy(DELAYS_LINEAR_S)
    result = het.process(rx_cpu.ravel())

    if not result['success']:
        print(f"  FAIL: {result['error_message']}")
        return None

    # CPU reference
    t = np.arange(N, dtype=np.float64) / FS
    ref_c = generate_ref_conjugate_numpy()
    cpu_f, cpu_r = [], []
    nfft = 8192
    for k in range(ANTENNAS):
        dc_k = np.conj(rx_cpu[k] * ref_c)
        padded = np.zeros(nfft, dtype=np.complex64)
        padded[:N] = dc_k
        spec = np.abs(np.fft.fft(padded)[:nfft // 2])
        pb = np.argmax(spec)
        rb, _ = parabolic_interp(spec, pb)
        f_c = rb * FS / nfft
        r_c = C_LIGHT * T * f_c / (2 * B)
        cpu_f.append(f_c)
        cpu_r.append(r_c)

    hdr = (f"  {'Ant':>3} | {'f_beat GPU':>11} | {'f_exp Гц':>11} | "
           f"{'df Гц':>8} | {'R GPU м':>9} | {'SNR дБ':>7}")
    print(hdr)
    print("  " + "─" * 66)

    gpu_results = []
    for k, ant in enumerate(result['antennas']):
        exp_f = F_BEATS_EXPECTED[k]
        f_err = abs(ant['f_beat_hz'] - exp_f)
        gpu_results.append(ant)
        print(f"  {k:3d} | {ant['f_beat_hz']:11.0f} | {exp_f:11.0f} | "
              f"{f_err:8.0f} | {ant['range_m']:9.2f} | {ant['peak_snr_db']:7.1f}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: f_beat GPU vs CPU vs Theory
        ax1 = axes[0]
        ax1.plot(DELAYS_LINEAR_US, F_BEATS_EXPECTED / 1e3, 'g--',
                 lw=2, alpha=0.6, label='Теория μ·τ')
        ax1.plot(DELAYS_LINEAR_US,
                 [a['f_beat_hz'] for a in gpu_results], 'ro-',
                 ms=9, lw=1.5, label='GPU')
        ax1.plot(DELAYS_LINEAR_US,
                 np.array(cpu_f) / 1e3, 'b^--',
                 ms=7, lw=1.2, label='CPU')
        for k in range(ANTENNAS):
            ax1.annotate(f'{gpu_results[k]["f_beat_hz"]/1e3:.0f}',
                         (DELAYS_LINEAR_US[k], gpu_results[k]['f_beat_hz']/1e3),
                         textcoords='offset points', xytext=(3, 5),
                         fontsize=7.5, color='red')
        ax1.set_xlabel('Задержка [мкс]')
        ax1.set_ylabel('f_beat [кГц]')
        ax1.set_title('f_beat: GPU vs CPU vs Теория', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.03, 0.97, 'f = μ·τ = 3e9·τ',
                 transform=ax1.transAxes, va='top', fontsize=8,
                 bbox=dict(fc='lightyellow', ec='navy', alpha=0.85, pad=3))

        # Panel 2: ошибки
        ax2 = axes[1]
        df_gpu = [abs(gpu_results[k]['f_beat_hz'] - F_BEATS_EXPECTED[k])
                  for k in range(ANTENNAS)]
        df_cpu = [abs(cpu_f[k] - F_BEATS_EXPECTED[k])
                  for k in range(ANTENNAS)]
        xb = np.arange(ANTENNAS)
        w  = 0.35
        ax2.bar(xb - w/2, df_gpu, w, label='GPU', color='coral', alpha=0.85)
        ax2.bar(xb + w/2, df_cpu, w, label='CPU', color='steelblue', alpha=0.85)
        ax2.axhline(5000, color='red', ls='--', lw=1.3, label='Допуск 5кГц')
        ax2.set_xlabel('Антенна')
        ax2.set_ylabel('|df| [Гц]')
        ax2.set_title('Ошибка f_beat (GPU vs CPU)', fontsize=10)
        ax2.set_xticks(xb)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.text(0.03, 0.97,
                 'Обе реализации:\nточность ≈ bin_width/2',
                 transform=ax2.transAxes, va='top', fontsize=8,
                 bbox=dict(fc='linen', ec='brown', alpha=0.85, pad=3))

        # Panel 3: SNR
        ax3 = axes[2]
        snrs = [g['peak_snr_db'] for g in gpu_results]
        bar_c = ['#2ecc71' if s > 10 else '#e74c3c' for s in snrs]
        bars = ax3.bar(range(ANTENNAS), snrs, color=bar_c, alpha=0.85, ec='black')
        ax3.axhline(0,  color='red',   ls='--', lw=1.2)
        ax3.axhline(10, color='green', ls=':',  lw=1.0)
        for bar, s in zip(bars, snrs):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     max(s, 0) + 0.3, f'{s:.1f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax3.set_xlabel('Антенна')
        ax3.set_ylabel('SNR [дБ]')
        ax3.set_title('SNR GPU (peak / соседние бины)', fontsize=10)
        ax3.set_xticks(range(ANTENNAS))
        ax3.set_xticklabels([f'A{i}\n{DELAYS_LINEAR_US[i]:.0f}мкс'
                              for i in range(ANTENNAS)], fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.text(0.03, 0.97,
                 'SNR = 20·log₁₀\n(пик / avg(left,right))',
                 transform=ax3.transAxes, va='top', fontsize=8,
                 bbox=dict(fc='honeydew', ec='green', alpha=0.85, pad=3))

        fig.suptitle('Шаг 8: GPU Pipeline — сравнение с CPU\n'
                     'HeterodyneDechirp.process() vs NumPy FFT',
                     fontsize=11, fontweight='bold')
        add_params_banner(fig, 8, 'GPU vs CPU',
                          'search_range=5000: охват [0..2499] бин (~3.66 МГц)')
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        save_plot('step_08_summary.png', fig)

    return gpu_results


def print_summary(cpu_results, gpu_results):
    """Final summary table: GPU vs CPU comparison."""
    print("\n" + "=" * 60)
    print("SUMMARY: GPU vs CPU comparison")
    print("=" * 60)

    if gpu_results is None:
        print("  GPU results not available")
        return

    hdr = (f"  {'Ant':>3} | {'f GPU Гц':>11} | {'f CPU Гц':>11} | "
           f"{'df Гц':>8} | {'R GPU м':>9} | {'R CPU м':>9} | "
           f"{'dR м':>6} | {'SNR дБ':>7}")
    print(hdr)
    print("  " + "─" * 76)

    for k in range(ANTENNAS):
        f_gpu = gpu_results[k]['f_beat_hz']
        f_cpu = cpu_results[k]['f_beat']
        r_gpu = gpu_results[k]['range_m']
        r_cpu = cpu_results[k]['range_m']
        df = abs(f_gpu - f_cpu)
        dr = abs(r_gpu - r_cpu)
        snr = gpu_results[k]['peak_snr_db']
        print(f"  {k:3d} | {f_gpu:11.0f} | {f_cpu:11.0f} | {df:8.0f} | "
              f"{r_gpu:9.2f} | {r_cpu:9.2f} | {dr:6.2f} | {snr:7.1f}")


# ============================================================================
# Main
# ============================================================================

def run_full_test():
    """Run all steps sequentially."""
    rx_cpu      = step01_generate_rx()
    ref_cpu     = step02_generate_ref_conjugate()
    dc_cpu      = step03_dechirp(rx_cpu, ref_cpu)
    spec_cpu    = step04_fft(dc_cpu)
    cpu_results = step05_find_maxima(spec_cpu)
    corrected   = step06_dechirp_correct(dc_cpu, cpu_results)
    step07_verify_dc(corrected)
    gpu_results = step08_gpu_pipeline()
    print_summary(cpu_results, gpu_results)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    run_full_test()
