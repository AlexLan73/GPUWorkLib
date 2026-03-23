"""
test_lch_farrow.py — Тесты LchFarrow (standalone Lagrange 48x5 fractional delay)

Тесты:
  1. Нулевая задержка — output ≈ input
  2. Целая задержка (5 сэмплов) — output = shift(input)
  3. Дробная задержка (2.7 сэмпла) — GPU vs NumPy Lagrange
  4. Multi-antenna — 4 канала с разными задержками
  5. Сравнение с LfmAnalyticalDelay (standalone LchFarrow + LFM vs analytical)

@author Кодо (AI Assistant)
@date 2026-02-18
"""

import sys
import os
import json
import glob
import numpy as np

# ── Путь к gpuworklib (Python_test/lch_farrow/ -> 2 levels up) ──
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

if not hasattr(gpuworklib, 'LchFarrowROCm'):
    print("ERROR: gpuworklib built without LchFarrow.")
    print("  Rebuild: cmake -B build -DBUILD_PYTHON=ON && cmake --build build")
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
    arr = np.array(data['data'], dtype=np.float32)
    return arr.reshape(data['rows'], data['columns'])


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: Lagrange fractional delay
# ════════════════════════════════════════════════════════════════════════════

def apply_delay_numpy(signal, delay_samples, lagrange_matrix):
    """CPU reference — дробная задержка через Lagrange 48×5.

    Зеркалит GPU-ядро (lch_farrow.cpp) точь-в-точь:
      read_pos = n - delay_samples   ← вычисляется PER-SAMPLE (не глобально!)
      center   = floor(read_pos)
      frac     = read_pos - center   ← дробная часть PER-SAMPLE
      row      = int(frac * 48) % 48 ← строка матрицы PER-SAMPLE
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


def generate_cw_signal(fs, points, f0, amplitude=1.0):
    """Простой CW сигнал для тестов."""
    t = np.arange(points) / fs
    return (amplitude * np.exp(1j * 2 * np.pi * f0 * t)).astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════
# Test 1: Нулевая задержка — output ≈ input
# ════════════════════════════════════════════════════════════════════════════

def test_zero_delay():
    """delay=0 -> output == input."""
    print("\n[Test 1] Zero delay...")

    fs = 1e6
    points = 4096
    f0 = 50000.0

    signal = generate_cw_signal(fs, points, f0)

    ctx = gpuworklib.ROCmGPUContext(0)
    proc = gpuworklib.LchFarrowROCm(ctx)
    proc.set_sample_rate(fs)
    proc.set_delays([0.0])
    result = proc.process(signal)

    max_err = np.max(np.abs(result.ravel() - signal))
    print(f"  max_error = {max_err:.2e}")
    assert max_err < 1e-4, f"Zero delay error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 2: Целая задержка (5 сэмплов)
# ════════════════════════════════════════════════════════════════════════════

def test_integer_delay():
    """delay = 5 samples -> output[5:] ≈ input[:N-5], output[:5] = 0."""
    print("\n[Test 2] Integer delay (5 samples)...")

    fs = 1e6
    points = 4096
    f0 = 50000.0
    delay_us = 5.0  # 5 мкс при fs=1MHz = 5 сэмплов

    signal = generate_cw_signal(fs, points, f0)

    ctx = gpuworklib.ROCmGPUContext(0)
    proc = gpuworklib.LchFarrowROCm(ctx)
    proc.set_sample_rate(fs)
    proc.set_delays([delay_us])
    result = proc.process(signal).ravel()

    # Первые 5 должны быть нулями
    zeros_ok = np.max(np.abs(result[:5])) < 1e-6
    print(f"  zeros[0..4] = {'OK' if zeros_ok else 'FAIL'} "
          f"(max={np.max(np.abs(result[:5])):.2e})")

    # NumPy reference
    matrix = load_lagrange_matrix()
    delay_samples = delay_us * 1e-6 * fs
    ref = apply_delay_numpy(signal, delay_samples, matrix)

    max_err = np.max(np.abs(result - ref))
    print(f"  max_error vs NumPy = {max_err:.2e}")
    assert zeros_ok, "First 5 samples should be zero"
    assert max_err < 1e-2, f"Integer delay error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 3: Дробная задержка (2.7 сэмпла)
# ════════════════════════════════════════════════════════════════════════════

def test_fractional_delay():
    """delay = 2.7 samples -> GPU vs NumPy Lagrange."""
    print("\n[Test 3] Fractional delay (2.7 samples)...")

    fs = 1e6
    points = 4096
    f0 = 50000.0
    delay_us = 2.7  # 2.7 мкс при fs=1MHz = 2.7 сэмплов

    signal = generate_cw_signal(fs, points, f0)

    ctx = gpuworklib.ROCmGPUContext(0)
    proc = gpuworklib.LchFarrowROCm(ctx)
    proc.set_sample_rate(fs)
    proc.set_delays([delay_us])
    result = proc.process(signal).ravel()

    # NumPy reference
    matrix = load_lagrange_matrix()
    delay_samples = delay_us * 1e-6 * fs
    ref = apply_delay_numpy(signal, delay_samples, matrix)

    max_err = np.max(np.abs(result - ref))
    print(f"  delay_samples = {delay_samples}")
    print(f"  max_error = {max_err:.2e}")
    assert max_err < 1e-2, f"Fractional delay error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 4: Multi-antenna
# ════════════════════════════════════════════════════════════════════════════

def test_multi_antenna():
    """4 антенны с разными задержками."""
    print("\n[Test 4] Multi-antenna...")

    fs = 1e6
    points = 4096
    f0 = 50000.0
    delays = [0.0, 1.5, 3.0, 5.0]
    antennas = len(delays)

    # Генерируем одинаковый сигнал для всех антенн
    single = generate_cw_signal(fs, points, f0)
    signal = np.tile(single, (antennas, 1))  # (4, 4096)

    ctx = gpuworklib.ROCmGPUContext(0)
    proc = gpuworklib.LchFarrowROCm(ctx)
    proc.set_sample_rate(fs)
    proc.set_delays(delays)
    result = proc.process(signal)

    assert result.shape == (antennas, points), \
        f"Shape mismatch: {result.shape}"

    # Проверяем каждый канал
    matrix = load_lagrange_matrix()
    max_errors = []
    for ch in range(antennas):
        delay_samples = delays[ch] * 1e-6 * fs
        ref = apply_delay_numpy(single, delay_samples, matrix)
        err = np.max(np.abs(result[ch] - ref))
        max_errors.append(err)
        print(f"  ch{ch}: delay={delays[ch]:.1f}us "
              f"({delay_samples:.1f} samp) err={err:.2e}")

    max_err = max(max_errors)
    assert max_err < 1e-2, f"Multi-antenna error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Test 5: LchFarrow + LFM vs LfmAnalyticalDelay
# ════════════════════════════════════════════════════════════════════════════

def test_lch_farrow_vs_analytical():
    """LchFarrow(LFM_signal) vs LfmAnalyticalDelay — оба должны дать ~одинаковый результат."""
    print("\n[Test 5] LchFarrow + LFM vs LfmAnalyticalDelay...")

    fs = 12e6
    length = 4096
    f_start = 1e6
    f_end = 2e6
    delay_us = 0.5  # 0.5 мкс

    ctx = gpuworklib.ROCmGPUContext(0)
    # Вариант 1: Аналитическая задержка (идеальная)
    gen_analytical = gpuworklib.LfmAnalyticalDelayROCm(
        ctx, f_start=f_start, f_end=f_end)
    gen_analytical.set_sampling(fs=fs, length=length)
    gen_analytical.set_delays([delay_us])
    analytical = gen_analytical.generate_gpu().ravel()

    # Вариант 2: Стандартный LFM + LchFarrow
    if not hasattr(gpuworklib, 'SignalGenerator'):
        # ROCm-only build: generate LFM via NumPy
        duration = length / fs
        chirp_rate = (f_end - f_start) / duration
        t = np.arange(length) / fs
        phase = np.pi * chirp_rate * t**2 + 2 * np.pi * f_start * t
        lfm_clean = (np.exp(1j * phase)).astype(np.complex64)
    else:
        sig = gpuworklib.SignalGenerator(ctx)
        lfm_clean = sig.generate_lfm(
            f_start=f_start, f_end=f_end, fs=fs, length=length)

    proc = gpuworklib.LchFarrowROCm(ctx)
    proc.set_sample_rate(fs)
    proc.set_delays([delay_us])
    farrow_result = proc.process(lfm_clean).ravel()

    # Сравнение: Farrow дает интерполяционную задержку,
    # Analytical — идеальную. Разница определяется качеством Lagrange.
    # Пропускаем начало (boundary effects)
    delay_samples = delay_us * 1e-6 * fs
    skip = int(np.ceil(delay_samples)) + 5  # skip boundary

    max_err = np.max(np.abs(farrow_result[skip:] - analytical[skip:]))
    print(f"  delay_samples = {delay_samples:.1f}")
    print(f"  max_error (Farrow vs Analytical, skip {skip}) = {max_err:.2e}")

    # Lagrange 48x5 vs analytical: ожидаем < 0.1 для низкочастотного LFM
    # (Farrow — интерполяция, не идеал)
    assert max_err < 0.1, f"Farrow vs Analytical error too large: {max_err}"
    print("  PASSED!")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("LchFarrow Tests (Standalone Lagrange 48x5 Fractional Delay)")
    print("=" * 70)

    test_zero_delay()
    test_integer_delay()
    test_fractional_delay()
    test_multi_antenna()
    test_lch_farrow_vs_analytical()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
