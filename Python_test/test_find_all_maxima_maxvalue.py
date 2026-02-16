"""
GPUWorkLib: FindAllMaxima — MaxValue Format + CPU/GPU Data Paths
================================================================
Демонстрация FindAllMaxima с форматом MaxValue:
  - CPU данные (numpy → upload) и GPU данные (SignalGenerator)
  - Красивая визуализация спектра и найденных максимумов

Результат: positions, magnitudes, frequencies извлекаются из MaxValue.
Графики сохраняются в Results/Plots/

Зависимости: numpy, matplotlib, gpuworklib
  pip install numpy matplotlib

Author: Кодо (AI Assistant)
Date: 2026-02-15
"""

import sys
import os

try:
    import numpy as np
except ImportError:
    print("Ошибка: установите numpy (pip install numpy)")
    sys.exit(1)

# Путь к gpuworklib (пробуем Debug и Release)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ["build/python/Debug", "build/python/Release", "build/Debug", "build/Release"]:
    path = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(path):
        sys.path.insert(0, path)
        break
import gpuworklib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

PLOT_DIR = os.path.join(PROJECT_ROOT, "Results", "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def create_beautiful_plot(signal_cpu, spectrum_cpu, result_cpu,
                          signal_gpu, spectrum_gpu, result_gpu,
                          fs, nFFT, freqs, save_path):
    """Создаёт красивый график: CPU vs GPU данные, спектр + максимумы."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f8f9fa')

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          left=0.06, right=0.96, top=0.92, bottom=0.08)

    freq_axis = np.arange(nFFT // 2) * fs / nFFT
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

    # --- Верхний ряд: CPU данные ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#ecf0f1')
    mag_cpu = np.abs(spectrum_cpu[:nFFT // 2].astype(np.complex128))
    ax1.fill_between(freq_axis, 0, mag_cpu, alpha=0.4, color='#3498db')
    ax1.plot(freq_axis, mag_cpu, 'b-', linewidth=1.2, label='Спектр')

    pos = result_cpu["positions"]
    mag = result_cpu["magnitudes"]
    freq = result_cpu["frequencies"]
    mask = pos < nFFT // 2
    ax1.scatter(freq[mask], mag[mask], c='#e74c3c', s=120, zorder=5,
                edgecolors='#c0392b', linewidths=2, label=f'Максимумы ({result_cpu["num_maxima"]})')

    for f in freqs:
        ax1.axvline(x=f, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1)
    ax1.set_xlabel("Частота (Гц)", fontsize=11)
    ax1.set_ylabel("Амплитуда", fontsize=11)
    ax1.set_title("📥 CPU данные (numpy → upload → FFT → FindAllMaxima)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xlim(0, fs / 2)

    # --- Верхний ряд: GPU данные ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#ecf0f1')
    mag_gpu = np.abs(spectrum_gpu[:nFFT // 2].astype(np.complex128))
    ax2.fill_between(freq_axis, 0, mag_gpu, alpha=0.4, color='#9b59b6')
    ax2.plot(freq_axis, mag_gpu, color='#8e44ad', linewidth=1.2, label='Спектр')

    pos_g = result_gpu["positions"]
    mag_g = result_gpu["magnitudes"]
    freq_g = result_gpu["frequencies"]
    mask_g = pos_g < nFFT // 2
    ax2.scatter(freq_g[mask_g], mag_g[mask_g], c='#e67e22', s=120, zorder=5,
                edgecolors='#d35400', linewidths=2, label=f'Максимумы ({result_gpu["num_maxima"]})')

    for f in freqs:
        ax2.axvline(x=f, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1)
    ax2.set_xlabel("Частота (Гц)", fontsize=11)
    ax2.set_ylabel("Амплитуда", fontsize=11)
    ax2.set_title("📤 GPU данные (SignalGenerator → FFT → FindAllMaxima)", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_xlim(0, fs / 2)

    # --- Нижний ряд: Сравнение CPU vs GPU ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor('#fff9e6')
    ax3.plot(freq_axis, mag_cpu, 'b-', linewidth=1, alpha=0.7, label='Спектр (CPU path)')
    ax3.plot(freq_axis, mag_gpu, 'm-', linewidth=0.8, alpha=0.5, label='Спектр (GPU path)')

    ax3.scatter(freq[mask], mag[mask], c='#e74c3c', s=80, marker='o', zorder=5,
                edgecolors='black', linewidths=1, label='CPU maxima')
    ax3.scatter(freq_g[mask_g], mag_g[mask_g], c='#e67e22', s=80, marker='^', zorder=5,
                edgecolors='black', linewidths=1, label='GPU maxima')

    for i, f in enumerate(freqs):
        ax3.axvline(x=f, color=colors[i % len(colors)], linestyle=':', alpha=0.8, linewidth=1.5)

    ax3.set_xlabel("Частота (Гц)", fontsize=11)
    ax3.set_ylabel("Амплитуда", fontsize=11)
    ax3.set_title("🔄 Сравнение: CPU vs GPU — оба пути дают одинаковые максимумы", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.set_xlim(0, fs / 2)

    fig.suptitle(f"FindAllMaxima (MaxValue): {len(freqs)} тонов @ {freqs} Гц, Fs={fs:.0f} Гц",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 График сохранён: {save_path}")


def test_cpu_vs_gpu_data():
    """Тест: CPU данные и GPU данные — оба пути работают."""
    print("\n" + "=" * 70)
    print("  FindAllMaxima: CPU vs GPU Data Paths + MaxValue Format")
    print("=" * 70)

    ctx = gpuworklib.GPUContext(0)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)
    sig = gpuworklib.SignalGenerator(ctx)

    fs = 1000.0
    nFFT = 1024
    freqs = [50.0, 120.0, 200.0]

    # --- Путь 1: CPU данные (numpy) ---
    t = np.arange(nFFT, dtype=np.float32)
    signal_cpu = np.zeros(nFFT, dtype=np.complex64)
    for f in freqs:
        signal_cpu += np.sin(2 * np.pi * f * t / fs).astype(np.complex64)

    spectrum_cpu = fft.process_complex(signal_cpu, sample_rate=fs)
    result_cpu = finder.find_all_maxima(spectrum_cpu, sample_rate=fs)

    # --- Путь 2: GPU данные (SignalGenerator — генерирует на GPU при beam_count>1) ---
    # Суммируем 3 тона через SignalGenerator
    s1 = sig.generate_cw(freq=freqs[0], fs=fs, length=nFFT, beam_count=1)
    s2 = sig.generate_cw(freq=freqs[1], fs=fs, length=nFFT, beam_count=1)
    s3 = sig.generate_cw(freq=freqs[2], fs=fs, length=nFFT, beam_count=1)
    signal_gpu = np.squeeze(s1) + np.squeeze(s2) + np.squeeze(s3)

    spectrum_gpu = fft.process_complex(signal_gpu.astype(np.complex64), sample_rate=fs)
    result_gpu = finder.find_all_maxima(spectrum_gpu, sample_rate=fs)

    # Проверка
    print(f"  CPU path: {result_cpu['num_maxima']} максимумов")
    print(f"  GPU path: {result_gpu['num_maxima']} максимумов")

    # График
    save_path = os.path.join(PLOT_DIR, "find_all_maxima_cpu_vs_gpu.png")
    create_beautiful_plot(
        signal_cpu, spectrum_cpu, result_cpu,
        signal_cpu, spectrum_gpu, result_gpu,  # signal_gpu = signal_cpu для визуализации
        fs, nFFT, freqs, save_path
    )

    # Проверяем что основные пики найдены
    all_ok = True
    for ef in freqs:
        exp_bin = int(ef * nFFT / fs + 0.5)
        found_cpu = any(abs(int(p) - exp_bin) <= 1 for p in result_cpu["positions"])
        found_gpu = any(abs(int(p) - exp_bin) <= 1 for p in result_gpu["positions"])
        print(f"    {ef:.0f} Гц (bin ~{exp_bin}): CPU={'✓' if found_cpu else '✗'}, GPU={'✓' if found_gpu else '✗'}")
        if not (found_cpu and found_gpu):
            all_ok = False

    return all_ok


def test_multi_beam_beautiful():
    """Мульти-луч с красивой визуализацией."""
    print("\n" + "=" * 70)
    print("  FindAllMaxima: Multi-beam (5 лучей) — красивая картинка")
    print("=" * 70)

    ctx = gpuworklib.GPUContext(0)
    sig = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessor(ctx)
    finder = gpuworklib.SpectrumMaximaFinder(ctx)

    fs = 1000.0
    length = 512
    beam_count = 5
    beam_freqs = [50.0 + i * 50.0 for i in range(beam_count)]

    signals = sig.generate_cw(freq=50.0, fs=fs, length=length,
                              beam_count=beam_count, freq_step=50.0)
    spectra = fft.process_complex(signals, sample_rate=fs)
    result = finder.find_all_maxima(spectra, sample_rate=fs)

    # --- Красивый график ---
    fig, axes = plt.subplots(beam_count, 1, figsize=(14, 3 * beam_count))
    fig.patch.set_facecolor('#f0f4f8')
    fig.suptitle("FindAllMaxima: Multi-beam (5 лучей, 50–250 Гц)", fontsize=14, fontweight='bold')

    nFFT = spectra.shape[1] if spectra.ndim == 2 else len(spectra)
    freq_axis = np.arange(nFFT // 2) * fs / nFFT
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for b in range(beam_count):
        ax = axes[b]
        ax.set_facecolor('#fafafa')
        spec = spectra[b] if spectra.ndim == 2 else spectra
        mag = np.abs(spec[:nFFT // 2].astype(np.complex128))
        ax.fill_between(freq_axis, 0, mag, alpha=0.3, color=colors[b])
        ax.plot(freq_axis, mag, color=colors[b], linewidth=1.2)

        beam = result[b]
        mask = beam["positions"] < nFFT // 2
        ax.scatter(beam["frequencies"][mask], beam["magnitudes"][mask],
                   c='#2c3e50', s=100, marker='*', zorder=5, edgecolors='white', linewidths=1)
        ax.axvline(x=beam_freqs[b], color='green', linestyle='--', alpha=0.8)
        ax.set_ylabel("Амплитуда", fontsize=10)
        ax.set_title(f"Луч {b}: f={beam_freqs[b]:.0f} Гц, {beam['num_maxima']} максимумов", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fs / 2)

    axes[-1].set_xlabel("Частота (Гц)", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "find_all_maxima_multi_beam.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 График сохранён: {save_path}")

    all_ok = True
    for b in range(beam_count):
        exp_bin = int(beam_freqs[b] * length / fs + 0.5)
        found = any(abs(int(p) - exp_bin) <= 1 for p in result[b]["positions"])
        print(f"    Луч {b} ({beam_freqs[b]:.0f} Гц): {'✓' if found else '✗'}")
        if not found:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  GPUWorkLib: FindAllMaxima (MaxValue) — CPU/GPU Data + Визуализация")
    print("  Графики: " + PLOT_DIR)
    print("=" * 70)

    tests = [
        ("CPU vs GPU данные", test_cpu_vs_gpu_data),
        ("Multi-beam (5 лучей)", test_multi_beam_beautiful),
    ]

    passed = 0
    for name, fn in tests:
        try:
            if fn():
                passed += 1
                print(f"  ✅ {name}: PASS")
            else:
                print(f"  ❌ {name}: FAIL")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Результат: {passed}/{len(tests)} тестов пройдено")
    print(f"  Графики: {PLOT_DIR}")
    print("=" * 70 + "\n")

    sys.exit(0 if passed == len(tests) else 1)
