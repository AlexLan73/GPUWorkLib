"""
quick_test.py — smoke test для PyCharm отладки.

Быстрая проверка что все модули работают корректно:
  1. Генерируем CW + шум с известным SNR_in
  2. Прогоняем через estimate_snr_one_antenna
  3. Печатаем теорию vs реальность
  4. Показываем один график (спектр |X|²)

Запуск в PyCharm:
  - открыть этот файл
  - поставить breakpoint в интересующих местах
  - Run → Debug quick_test

@author Кодо
@date 2026-04-09
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# matplotlib — интерактивно для PyCharm (можно сменить на Agg если нет X)
import matplotlib
try:
    # Если запуск в PyCharm с UI — используем интерактивный backend
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfar_estimator import (
    CfarConfig,
    estimate_snr_one_antenna,
    compute_pipeline_sizes,
)
from lfm_signal_generator import make_cw_with_snr, make_awgn


def quick_test():
    print("=" * 60)
    print("QUICK TEST — SNR estimator smoke test")
    print("=" * 60)

    # --- Параметры ---
    n_samples = 5000
    snr_in_db = 20.0
    freq_norm = 0.15

    # --- Генерация ---
    rng = np.random.default_rng(42)
    signal, amplitude = make_cw_with_snr(
        n_samples, freq_norm=freq_norm, snr_in_db=snr_in_db,
        noise_power=1.0, rng=rng,
    )
    print(f"\nСгенерирован CW:")
    print(f"  n_samples  = {n_samples}")
    print(f"  freq_norm  = {freq_norm}")
    print(f"  SNR_in     = {snr_in_db} dB")
    print(f"  amplitude  = {amplitude:.3f}")
    print(f"  signal[0]  = {signal[0]}")

    # --- Pipeline sizes ---
    cfg = CfarConfig(target_n_fft=0)  # auto → 2048
    step, n_actual, n_fft = compute_pipeline_sizes(n_samples, 0, 0)
    coherent_gain = 10.0 * math.log10(n_actual)
    print(f"\nPipeline:")
    print(f"  step_samples = {step}")
    print(f"  N_actual     = {n_actual}")
    print(f"  N_fft        = {n_fft}")
    print(f"  coherent_gain = {coherent_gain:.2f} dB")

    # --- CFAR estimate ---
    result = estimate_snr_one_antenna(signal, cfg)
    print(f"\nCFAR результат:")
    print(f"  k_peak    = {result.k_peak}")
    print(f"  peak_sq   = {result.peak_sq:.3e}")
    print(f"  noise_mean = {result.noise_mean:.3e}")
    print(f"  SNR_fft   = {result.snr_db:.2f} dB")

    # --- Теория ---
    theory_db = snr_in_db + coherent_gain
    diff = result.snr_db - theory_db
    print(f"\nСравнение с теорией:")
    print(f"  Теория:    {theory_db:.2f} dB  (SNR_in + 10·log10(N_actual))")
    print(f"  Измерено:  {result.snr_db:.2f} dB")
    print(f"  Разница:   {diff:+.2f} dB  ({'OK' if abs(diff) < 3 else 'CHECK!'})")

    # --- H0 тест (только шум) ---
    noise = make_awgn(n_samples, noise_power=1.0, rng=np.random.default_rng(7))
    h0_result = estimate_snr_one_antenna(noise, cfg)
    h0_theory = 10.0 * math.log10(math.log(n_fft) + 0.5772)
    print(f"\nH0 (только шум) артефакт:")
    print(f"  Теория:    {h0_theory:.2f} dB  (10·log10(ln(N_fft) + γ))")
    print(f"  Измерено:  {h0_result.snr_db:.2f} dB")

    # --- Визуализация (спектр |X|²) ---
    decimated = signal[::step][:n_actual]
    padded = np.zeros(n_fft, dtype=np.complex64)
    padded[: len(decimated)] = decimated
    spectrum = np.fft.fft(padded)
    mag_sq = np.abs(spectrum) ** 2

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # (0): Спектр |X|² (dB)
    ax = axes[0]
    freqs = np.fft.fftfreq(n_fft)
    idx = np.argsort(freqs)
    ax.plot(freqs[idx], 10.0 * np.log10(mag_sq[idx] + 1e-30))
    ax.axvline(freq_norm, color="red", linestyle="--", alpha=0.5,
               label=f"True freq = {freq_norm}")
    # Пик в индексе с учётом центрирования
    peak_freq_norm = freqs[result.k_peak]
    ax.axvline(peak_freq_norm, color="green", linestyle=":",
               label=f"Detected peak @ {peak_freq_norm:.3f}")
    ax.set_xlabel("Нормированная частота f/fs")
    ax.set_ylabel("|X[k]|² (dB)")
    ax.set_title(f"Спектр |X|² (SNR_in={snr_in_db} dB → SNR_fft={result.snr_db:.1f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1): Вход сигнал (real часть первые 200 точек)
    ax = axes[1]
    ax.plot(signal[:200].real, label="Re", alpha=0.7)
    ax.plot(signal[:200].imag, label="Im", alpha=0.7)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Вход: CW freq={freq_norm} + AWGN, SNR_in={snr_in_db} dB (первые 200 точек)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Для PyCharm — показать интерактивно
    try:
        plt.show(block=False)
        print("\n[Window shown — close to exit]")
        plt.show()
    except Exception:
        # Fallback — save to file
        out = Path(__file__).resolve().parent / "plots" / "quick_test.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"\n[Plot saved: {out}]")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()
