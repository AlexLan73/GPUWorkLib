#!/usr/bin/env python3
"""
debug_pipeline_steps.py — Пошаговая отладка strategies pipeline в PyCharm
==========================================================================

КАК ИСПОЛЬЗОВАТЬ:
  1. Открыть в PyCharm
  2. Поставить breakpoint на каждую строку:  ← BREAKPOINT HERE
  3. Run → Debug (Shift+F9)
  4. Нажимать F9 (Resume) — переходить к следующему шагу
  5. Смотреть переменные в панели Variables

СТРУКТУРА PIPELINE (NumPy — без GPU):
  ┌─ STEP 0: S_raw    ─ входной сигнал  [n_ant × n_samples]  complex64
  ├─ STEP 1: W        ─ матрица весов   [n_ant × n_ant]       complex64
  ├─ STEP 2: X_gemm   ─ после GEMM     [n_ant × n_samples]   complex64  ← X = W @ S
  ├─ STEP 3: X_win    ─ после окна     [n_ant × n_samples]   complex64  ← * Hamming
  ├─ STEP 4: spectrum ─ после FFT      [n_ant × nFFT]         complex64
  │          magnitudes               [n_ant × nFFT]         float32
  ├─ STEP 5_1: one_max   ─ 1 пик + парабола (на луч)
  ├─ STEP 5_2: all_maxima ─ все локальные пики (на луч)
  └─ STEP 5_3: minmax    ─ глобальный MIN+MAX + dynamic_range_dB

ПАРАМЕТРЫ (совместимы с C++ тестом):
  n_ant=5, n_samples=8000, fs=12 МГц, f0=2 МГц, tau_step=100 мкс

Author: Kodo (AI Assistant)
Date: 2026-03-12
"""

import os
import sys
import numpy as np
from scipy import signal as sp_signal

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _stats(arr: np.ndarray, label: str):
    """Распечатать краткую статистику массива."""
    abs_vals = np.abs(arr)
    print(f"  [{label}]  shape={arr.shape}  dtype={arr.dtype}")
    print(f"    mean_abs={abs_vals.mean():.4f}  max={abs_vals.max():.4f}"
          f"  min={abs_vals.min():.4f}  std={abs_vals.std():.4f}")


def _next_pow2_x2(n: int) -> int:
    """nFFT = next_power_of_2(n) * 2."""
    p = 1
    while p < n:
        p <<= 1
    return p * 2


# ============================================================================
# ПАРАМЕТРЫ ТЕСТА
# ============================================================================

N_ANT      = 5          # количество антенн
N_SAMPLES  = 8000       # отсчётов на антенну
FS         = 12.0e6     # частота дискретизации [Гц]
F0         = 2.0e6      # несущая частота [Гц]
TAU_STEP   = 100e-6     # шаг задержки [с] (delay-and-sum)
AMPLITUDE  = 1.0

print("=" * 60)
print("strategies — Пошаговая отладка pipeline (NumPy)")
print("=" * 60)
print(f"  N_ANT={N_ANT}  N_SAMPLES={N_SAMPLES}  FS={FS/1e6:.0f} МГц"
      f"  F0={F0/1e6:.0f} МГц  TAU_STEP={TAU_STEP*1e6:.0f} мкс")


# ============================================================================
# STEP 0: Входной сигнал S_raw
# ============================================================================
#                        ← BREAKPOINT HERE (поставь сюда)
print("\n── STEP 0: Генерация входного сигнала S_raw ──")

dt = 1.0 / FS
t  = np.arange(N_SAMPLES) * dt         # [N_SAMPLES] временная ось

# Генерируем CW сигнал на каждую антенну с задержкой (delay-and-sum model)
S_raw = np.zeros((N_ANT, N_SAMPLES), dtype=np.complex64)
delays_s = np.arange(N_ANT) * TAU_STEP  # τ_i = i * tau_step [с]

for ant in range(N_ANT):
    tau = delays_s[ant]
    t_shifted = t - tau
    valid = t_shifted >= 0
    S_raw[ant, valid] = (AMPLITUDE * np.exp(1j * 2 * np.pi * F0 * t_shifted[valid])
                         ).astype(np.complex64)

# ── Что смотреть в Variables:
#   S_raw.shape       → (5, 8000)
#   S_raw.dtype       → complex64
#   np.abs(S_raw)     → почти везде 1.0, края нулевые из-за задержки
#   S_raw[0, :10]     → первые 10 отсчётов антенны 0 (нет задержки)
#   S_raw[4, :10]     → первые 10 отсчётов антенны 4 (задержана)
_stats(S_raw, "S_raw")
print(f"  delays_s (мкс): {delays_s * 1e6}")
print(f"  S_raw[0,:3]  = {S_raw[0, :3]}")
print(f"  S_raw[4,:3]  = {S_raw[4, :3]}  (задержана на {delays_s[4]*1e6:.0f} мкс)")


# ============================================================================
# STEP 1: Матрица весов W (delay-and-sum)
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 1: Матрица весов W [delay-and-sum] ──")

inv_sqrt_n = 1.0 / np.sqrt(N_ANT)
W = np.zeros((N_ANT, N_ANT), dtype=np.complex64)

for beam in range(N_ANT):
    for ant in range(N_ANT):
        tau = delays_s[ant]
        W[beam, ant] = inv_sqrt_n * np.exp(-1j * 2.0 * np.pi * F0 * tau)

# ── Что смотреть в Variables:
#   W.shape               → (5, 5)
#   np.linalg.norm(W, axis=1) → [1.0, 1.0, 1.0, 1.0, 1.0]  (unit-norm rows)
#   np.abs(W)             → все элементы = 1/sqrt(5) ≈ 0.4472
#   W.real, W.imag        → фазовые сдвиги для каждой антенны
row_norms = np.linalg.norm(W, axis=1)
_stats(W, "W")
print(f"  Нормы строк: {row_norms}")
print(f"  |W[0,:]| = {np.abs(W[0,:])}")  # все = 1/sqrt(5)
print(f"  W[0,0]   = {W[0,0]:.4f}   (антенна 0, луч 0)")
print(f"  W[0,4]   = {W[0,4]:.4f}   (антенна 4, луч 0)")


# ============================================================================
# STEP 2: GEMM — X = W @ S_raw  (формирование луча)
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 2: GEMM  X = W @ S_raw ──")

X_gemm = (W @ S_raw).astype(np.complex64)  # [N_ANT, N_SAMPLES]

# ── Что смотреть в Variables:
#   X_gemm.shape          → (5, 8000)
#   np.abs(X_gemm).mean() → > np.abs(S_raw).mean()  (когерентное усиление √N)
#   np.abs(X_gemm[0])     → почти равномерно ~1.0 (delay-and-sum выровнял фазы)
#   разница мощностей:  GEMM усиливает сигнал в sqrt(N) раз по амплитуде
coherent_gain = np.abs(X_gemm).mean() / (np.abs(S_raw).mean() + 1e-12)
_stats(X_gemm, "X_gemm")
print(f"  Когерентное усиление: {coherent_gain:.2f}x  (ожидаем ~{np.sqrt(N_ANT):.2f} = sqrt({N_ANT}))")
print(f"  X_gemm[0, 2000:2003] = {X_gemm[0, 2000:2003]}")


# ============================================================================
# STEP 3: Окно Хэмминга (перед FFT)
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 3: Оконное взвешивание (Hamming) ──")

nFFT = _next_pow2_x2(N_SAMPLES)  # 16384
hamming = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_SAMPLES) / (N_SAMPLES - 1))
           ).astype(np.float32)

# Применяем окно к каждому лучу
X_windowed = np.zeros((N_ANT, N_SAMPLES), dtype=np.complex64)
for beam in range(N_ANT):
    X_windowed[beam] = X_gemm[beam] * hamming

# ── Что смотреть в Variables:
#   hamming.shape          → (8000,)
#   hamming[0], hamming[-1]  → ~0.08  (края Хэмминга ≈ 0.08)
#   hamming[N_SAMPLES//2]  → ~1.0    (центр ≈ 1.0)
#   X_windowed vs X_gemm:  края обрезаны, центр почти не изменился
#   nFFT                   → 16384   (next_pow2(8000)*2)
_stats(X_windowed, "X_windowed")
print(f"  hamming[0]={hamming[0]:.4f}  hamming[{N_SAMPLES//2}]={hamming[N_SAMPLES//2]:.4f}"
      f"  hamming[-1]={hamming[-1]:.4f}")
print(f"  nFFT = {nFFT}  (next_pow2({N_SAMPLES}) * 2)")


# ============================================================================
# STEP 4: FFT → spectrum и magnitudes
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 4: Batch FFT → spectrum ──")

spectrum   = np.zeros((N_ANT, nFFT), dtype=np.complex64)
magnitudes = np.zeros((N_ANT, nFFT), dtype=np.float32)
freq_axis  = np.fft.fftfreq(nFFT, d=1.0 / FS)  # [nFFT] Гц

for beam in range(N_ANT):
    padded = np.zeros(nFFT, dtype=np.complex64)
    padded[:N_SAMPLES] = X_windowed[beam]          # zero-padding
    spec = np.fft.fft(padded)
    spectrum[beam]   = spec
    magnitudes[beam] = np.abs(spec).astype(np.float32)

# ── Что смотреть в Variables:
#   spectrum.shape    → (5, 16384)
#   magnitudes.shape  → (5, 16384)
#   freq_resolution   → FS / nFFT = 12e6 / 16384 ≈ 732 Гц
#   peak_bin          → bin где magnitudes[0] максимальна (должна быть ~2 МГц)
#   peak_freq_hz      → peak_bin * FS / nFFT (должна быть близко к F0=2 МГц)
freq_resolution = FS / nFFT
half            = nFFT // 2
peak_bin_beam0  = int(np.argmax(magnitudes[0, 1:half])) + 1
peak_freq_hz    = peak_bin_beam0 * FS / nFFT

_stats(magnitudes, "magnitudes (|FFT|)")
print(f"  freq_resolution = {freq_resolution:.1f} Гц")
print(f"  peak_bin[beam0] = {peak_bin_beam0}  →  {peak_freq_hz/1e6:.4f} МГц"
      f"  (ожидаем {F0/1e6:.1f} МГц, ∆={abs(peak_freq_hz-F0):.0f} Гц)")
print(f"  magnitudes[0, peak_bin] = {magnitudes[0, peak_bin_beam0]:.4f}")


# ============================================================================
# STEP 5_1: OneMax + Parabola — один пик + уточнение
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 5_1: OneMax + паrabolic interpolation ──")

def find_one_max_parabola(mags_1d, fs, nfft):
    """Один максимум + уточнение параболой (зеркало C++ one_max_no_phase kernel)."""
    half = nfft // 2
    bin_idx = int(np.argmax(mags_1d[1:half])) + 1
    magnitude = float(mags_1d[bin_idx])

    # Парабольная интерполяция по 3 точкам
    y0 = float(mags_1d[bin_idx - 1]) if bin_idx > 0 else 0.0
    y1 = float(mags_1d[bin_idx])
    y2 = float(mags_1d[bin_idx + 1]) if bin_idx < nfft - 1 else 0.0
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    freq_offset = (y2 - y0) / denom if abs(denom) > 1e-10 else 0.0
    refined_freq_hz = (bin_idx + freq_offset) * fs / nfft

    return {
        'bin_index':       bin_idx,
        'magnitude':       magnitude,
        'freq_offset':     freq_offset,    # [-0.5 .. +0.5] суббиновое уточнение
        'refined_freq_hz': refined_freq_hz,
    }

one_max = [find_one_max_parabola(magnitudes[b], FS, nFFT) for b in range(N_ANT)]

# ── Что смотреть в Variables:
#   one_max           → list[dict], один dict на луч
#   one_max[0]        → {'bin_index': N, 'magnitude': M, 'freq_offset': δ, 'refined_freq_hz': F}
#   refined_freq_hz   → должна быть ТОЧНЕЕ peak_freq_hz (паrabola subbin)
#   freq_offset       → [-0.5..+0.5], насколько пик сдвинут внутри бина
print(f"  Луч 0: bin={one_max[0]['bin_index']}  "
      f"mag={one_max[0]['magnitude']:.4f}  "
      f"offset={one_max[0]['freq_offset']:+.4f}  "
      f"freq={one_max[0]['refined_freq_hz']/1e6:.5f} МГц")
print(f"  ∆ от F0: {abs(one_max[0]['refined_freq_hz'] - F0):.1f} Гц")
for b in range(N_ANT):
    print(f"    Луч {b}: {one_max[b]['refined_freq_hz']/1e6:.5f} МГц")


# ============================================================================
# STEP 5_2: AllMaxima — все локальные пики
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 5_2: AllMaxima — все локальные максимумы (луч 0) ──")

def find_all_maxima(mags_1d, fs, nfft, search_start=1, search_end=None, limit=50):
    """Все локальные максимумы, отсортированные по убыванию (mirror C++ AllMaximaPipelineROCm).

    Сначала находим ВСЕ локальные максимумы, затем сортируем по амплитуде и обрезаем до limit.
    (C++ тоже возвращает отсортированный результат, limit=1000.)
    """
    if search_end is None:
        search_end = nfft // 2
    maxima = []
    for i in range(search_start + 1, search_end - 1):
        if mags_1d[i] > mags_1d[i - 1] and mags_1d[i] > mags_1d[i + 1]:
            maxima.append({
                'bin_index': i,
                'freq_hz':   i * fs / nfft,
                'magnitude': float(mags_1d[i]),
            })
    # Сортируем по убыванию амплитуды и обрезаем до limit
    maxima.sort(key=lambda x: -x['magnitude'])
    return maxima[:limit]

all_maxima = [find_all_maxima(magnitudes[b], FS, nFFT) for b in range(N_ANT)]

# ── Что смотреть в Variables:
#   all_maxima                → list[list[dict]]
#   all_maxima[0]             → пики луча 0, отсортированные по убыванию
#   len(all_maxima[0])        → сколько пиков (обычно 1-5 для CW сигнала)
#   all_maxima[0][0]['freq_hz'] → главный пик ≈ F0
print(f"  Луч 0: найдено {len(all_maxima[0])} пиков")
for i, pk in enumerate(all_maxima[0][:5]):
    print(f"    #{i+1}: bin={pk['bin_index']}  "
          f"freq={pk['freq_hz']/1e6:.4f} МГц  "
          f"mag={pk['magnitude']:.4f}")


# ============================================================================
# STEP 5_3: GlobalMinMax — глобальный MIN+MAX + dynamic range
# ============================================================================
#                        ← BREAKPOINT HERE
print("\n── STEP 5_3: GlobalMinMax + dynamic_range_dB ──")

def find_global_minmax(mags_1d, fs, nfft):
    """Глобальный MIN+MAX + dynamic range (mirror C++ global_minmax kernel)."""
    half = nfft // 2
    mags_half = mags_1d[1:half]   # пропускаем DC (bin 0)
    freqs = np.arange(1, half) * fs / nfft

    max_idx  = int(np.argmax(mags_half))
    min_idx  = int(np.argmin(mags_half))

    max_val  = float(mags_half[max_idx])
    min_val  = float(mags_half[min_idx])
    safe_min = max(min_val, 1e-30)
    dyn_db   = 20.0 * np.log10(max_val / safe_min) if max_val > 0 else 0.0

    return {
        'max_magnitude':    max_val,
        'max_bin':          max_idx + 1,
        'max_frequency_hz': float(freqs[max_idx]),
        'min_magnitude':    min_val,
        'min_bin':          min_idx + 1,
        'min_frequency_hz': float(freqs[min_idx]),
        'dynamic_range_dB': dyn_db,
    }

minmax = [find_global_minmax(magnitudes[b], FS, nFFT) for b in range(N_ANT)]

# ── Что смотреть в Variables:
#   minmax            → list[dict], один dict на луч
#   minmax[0]         → полная статистика луча 0
#   dynamic_range_dB  → должен быть > 40 дБ (хороший сигнал без шума)
#   max_frequency_hz  → должна быть ≈ F0
print(f"  Луч 0: MAX={minmax[0]['max_magnitude']:.4f}"
      f"  @ {minmax[0]['max_frequency_hz']/1e6:.4f} МГц")
print(f"         MIN={minmax[0]['min_magnitude']:.6f}"
      f"  @ {minmax[0]['min_frequency_hz']/1e6:.4f} МГц")
print(f"         dynamic_range = {minmax[0]['dynamic_range_dB']:.1f} дБ")
for b in range(N_ANT):
    print(f"    Луч {b}: DR={minmax[b]['dynamic_range_dB']:.1f} дБ")


# ============================================================================
# ИТОГ — Сводная таблица
# ============================================================================
#                        ← BREAKPOINT HERE (финальные данные)
print("\n" + "=" * 60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"{'Луч':>5}  {'Пик, МГц':>12}  {'∆F0, Гц':>9}  {'DR, дБ':>8}  {'N пиков':>8}")
print("-" * 55)
for b in range(N_ANT):
    freq     = one_max[b]['refined_freq_hz']
    delta    = abs(freq - F0)
    dr_db    = minmax[b]['dynamic_range_dB']
    n_peaks  = len(all_maxima[b])
    print(f"{b:>5}  {freq/1e6:>12.5f}  {delta:>9.1f}  {dr_db:>8.1f}  {n_peaks:>8}")

print()
print("Переменные для инспекции в PyCharm Variables:")
print("  S_raw      — входной сигнал  [N_ANT, N_SAMPLES]  complex64")
print("  W          — матрица весов   [N_ANT, N_ANT]       complex64")
print("  X_gemm     — после GEMM      [N_ANT, N_SAMPLES]  complex64")
print("  X_windowed — после Hamming   [N_ANT, N_SAMPLES]  complex64")
print("  spectrum   — после FFT       [N_ANT, nFFT]        complex64")
print("  magnitudes — |FFT|           [N_ANT, nFFT]        float32")
print("  one_max    — 1 пик+парабола  list[dict]  (на луч)")
print("  all_maxima — все пики        list[list[dict]]")
print("  minmax     — мин/макс/DR     list[dict]  (на луч)")
print()
print("Done. ✓")


# ============================================================================
# Опциональная визуализация (раскомментировать если нужно)
# ============================================================================

PLOT = False  # ← поменяй на True чтобы построить графики

if PLOT:
    import matplotlib
    matplotlib.use('Agg')  # без GUI
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"strategies pipeline: {N_ANT} антенн, f₀={F0/1e6:.0f} МГц", fontsize=13)

    # 1. S_raw — Re компонента (луч 0)
    ax = axes[0]
    ax.plot(np.real(S_raw[0]), linewidth=0.6, label="S_raw[0] Re")
    ax.plot(np.real(S_raw[4]), linewidth=0.6, label="S_raw[4] Re", alpha=0.7)
    ax.set_title("STEP 0: S_raw — входной сигнал (Re, антенны 0 и 4)")
    ax.set_ylabel("Амплитуда")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. X_gemm vs S_raw — после GEMM (луч 0)
    ax = axes[1]
    ax.plot(np.real(S_raw[0]), linewidth=0.5, label="S_raw[0] Re", alpha=0.5)
    ax.plot(np.real(X_gemm[0]), linewidth=0.8, label="X_gemm[0] Re (после GEMM)")
    ax.set_title("STEP 2: X_gemm — после GEMM (луч 0, Re)")
    ax.set_ylabel("Амплитуда")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Спектр (magnitudes, луч 0)
    ax = axes[2]
    freqs_mhz = freq_axis[:half] / 1e6
    mags_db   = 20 * np.log10(magnitudes[0, :half] + 1e-12)
    ax.plot(freqs_mhz, mags_db, linewidth=0.8)
    ax.axvline(F0 / 1e6, color='r', linestyle='--', alpha=0.7, label=f"f₀={F0/1e6:.0f} МГц")
    ax.axvline(one_max[0]['refined_freq_hz'] / 1e6, color='g', linestyle='--',
               alpha=0.7, label=f"найдено {one_max[0]['refined_freq_hz']/1e6:.4f} МГц")
    ax.set_title("STEP 4: Спектр (дБ, луч 0)")
    ax.set_ylabel("дБ")
    ax.set_ylim(bottom=mags_db.max() - 80)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. OneMax частоты по всем лучам
    ax = axes[3]
    beams = list(range(N_ANT))
    freqs_found = [one_max[b]['refined_freq_hz'] / 1e6 for b in beams]
    ax.bar(beams, freqs_found, alpha=0.8, label="Найдено (параbola)")
    ax.axhline(F0 / 1e6, color='r', linestyle='--', alpha=0.8, label=f"f₀={F0/1e6:.0f} МГц")
    ax.set_title("STEP 5_1: OneMax — найденная частота по лучам")
    ax.set_xlabel("Луч")
    ax.set_ylabel("Частота, МГц")
    ax.set_xticks(beams)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Results", "Plots", "strategies",
                     "debug_pipeline_steps.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nГрафик сохранён: {out}")
    plt.close()
