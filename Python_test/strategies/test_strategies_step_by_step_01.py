#!/usr/bin/env python3
"""
test_strategies_step_by_step_01.py — простой пошаговый тест GPU vs NumPy
=========================================================================

Шаг 0: CPU генерирует сигнал (NumPy формула)
Шаг 1: Загружаем на GPU → скачиваем обратно → сравниваем в цикле
Шаг 2: Добавляем шум, проверяем мощность
Шаг 3: Умножаем на матрицу весов (GEMM) — CPU vs GPU, сравниваем в цикле
Шаг 4: Окно Хэмминга + FFT — ищем пик на f0

PyCharm: Run → Edit Configurations → Script path = этот файл
         Working directory = корень проекта (GPUWorkLib/)
         Можно Debug (F9 → F8 по шагам) или Run (Shift+F10).
"""

import os
import sys
import numpy as np


# =============================================================================
# ⚙️  НАСТРОЙКИ — МЕНЯЙ ЗДЕСЬ
# =============================================================================

# --- Антенная решётка ---

N_ANT = 5
# Количество антенн.
# Увеличь для теста большой решётки. GPU шаг медленнее, NumPy тоже.

N_SAMPLES = 8000
# Отсчётов на антенну.
# Должно совпадать с C++ тестом. nFFT = ближайшая степень 2 × 2 = 16384.

# --- Частоты ---

FS = 12.0e6
# Частота дискретизации, Гц.
# Меняй вместе с F0 — важно только соотношение F0/FS.

F0 = 2.0e6
# Частота сигнала, Гц. Ищем пик именно на этой частоте после FFT.
# Должна быть < FS/2 = 6 МГц, иначе aliasing.

TAU_STEP = 100e-6
# Задержка между соседними антеннами, секунды (100 мкс).
# Чем больше — тем заметнее расхождение Pipeline A vs B для ЛЧМ.

AMPLITUDE = 1.0
# Амплитуда сигнала. Менять не нужно — только если хочешь проверить масштаб.

# --- Допуски сравнения ---

TOLERANCE_SIGNAL = 1e-4
# Максимальное расхождение CPU vs GPU для отсчётов сигнала.
# GPU float32 даёт ошибку ~1e-7, но с накоплением операций — до 1e-4.
# Увеличь до 1e-3 если тест падает из-за архитектурных особенностей GPU.

TOLERANCE_FFT = 1e-3
# Допуск для спектра (после FFT накопленная ошибка больше).

# --- GPU ---

GPU_DEVICE_ID = 0
# Номер GPU (0 = первый). Если несколько GPU — попробуй 1, 2, ...
# Узнать номера: rocm-smi или clinfo.

# --- Путь к gpuworklib.so (собранному C++ модулю) ---

GPUWORKLIB_BUILD_SUBDIR = os.path.join("build", "debian-radeon9070", "python")
# Путь от корня проекта до папки с gpuworklib.so.
# Если собрал в другую директорию — поменяй здесь.
# Примеры:
#   "build/debian-radeon9070/python"  ← AMD GPU, Linux
#   "build/Release/python"            ← Windows/NVIDIA

# --- Шум (для Шага 2) ---

NOISE_SIGMA  = 0.1   # СКО шума (0.1 = SNR ≈ 20 дБ для AMPLITUDE=1)
NOISE_SEED   = 42    # seed для воспроизводимости (любое целое число)

# =============================================================================
# Загрузка gpuworklib — не трогать
# =============================================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, GPUWORKLIB_BUILD_SUBDIR))

try:
    import gpuworklib
    HAS_GPU = (hasattr(gpuworklib, 'AntennaProcessorTest') and
               hasattr(gpuworklib, 'ROCmGPUContext'))
except ImportError:
    HAS_GPU = False

print(f"\n{'='*60}")
print(f"  Параметры: {N_ANT} антенн × {N_SAMPLES} отсчётов")
print(f"  fs={FS/1e6:.0f} МГц,  f0={F0/1e6:.0f} МГц,  tau_step={TAU_STEP*1e6:.0f} мкс")
print(f"  GPU (ROCm): {'ДОСТУПЕН ✓' if HAS_GPU else 'НЕДОСТУПЕН — только NumPy'}")
print(f"{'='*60}\n")

# ===========================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (NumPy эталон)
# ===========================================================================

def cpu_generate_signal():
    """
    Генерирует CW сигнал на NumPy.

    Формула для каждой антенны i:
        tau_i = i * TAU_STEP          ← задержка i-й антенны
        s_i(t) = A * exp(j*2π*f0*(t - tau_i))   ← комплексная экспонента

    Физический смысл: сигнал приходит на каждую антенну с небольшой задержкой.
    Антенна 0 — без задержки, антенна 4 — с задержкой 400 мкс.
    """
    dt = 1.0 / FS                              # шаг по времени
    t  = np.arange(N_SAMPLES) * dt            # ось времени: [0, dt, 2*dt, ...]

    S = np.zeros((N_ANT, N_SAMPLES), dtype=np.complex64)

    for i in range(N_ANT):
        tau = i * TAU_STEP                     # задержка i-й антенны
        t_delayed = t - tau                    # сдвигаем ось времени
        valid = t_delayed >= 0                 # до прихода сигнала — нули

        # Комплексная экспонента: cos(2πf0·t) + j·sin(2πf0·t)
        S[i, valid] = (AMPLITUDE
                       * np.exp(1j * 2 * np.pi * F0 * t_delayed[valid])
                       ).astype(np.complex64)

    return S


def cpu_generate_weight_matrix():
    """
    Генерирует матрицу весов W для delay-and-sum beamforming.

    W[луч][антенна] = (1/sqrt(N)) * exp(-j*2π*f0*tau_антенна)

    Физический смысл: каждый луч "указывает" в сторону, откуда пришёл сигнал.
    После умножения W @ S все антенны сложатся когерентно (в фазе).
    """
    W = np.zeros((N_ANT, N_ANT), dtype=np.complex64)
    inv_sqrt_n = 1.0 / np.sqrt(N_ANT)         # нормировка: сохраняем мощность

    for beam in range(N_ANT):
        for ant in range(N_ANT):
            tau = ant * TAU_STEP               # задержка этой антенны
            # Сопряжённое вращение — компенсируем задержку
            W[beam, ant] = inv_sqrt_n * np.exp(-1j * 2 * np.pi * F0 * tau)

    return W


def cpu_add_noise(S, sigma=NOISE_SIGMA, seed=NOISE_SEED):
    """
    Добавляет комплексный AWGN шум к сигналу.

    Шум: Re ~ N(0, σ/√2),  Im ~ N(0, σ/√2)
    Суммарная мощность шума на антенну ≈ σ².
    """
    rng = np.random.default_rng(seed)

    # Комплексный шум: вещественная + мнимая части
    noise_re = rng.normal(0, sigma / np.sqrt(2), S.shape).astype(np.float32)
    noise_im = rng.normal(0, sigma / np.sqrt(2), S.shape).astype(np.float32)
    noise    = (noise_re + 1j * noise_im).astype(np.complex64)

    return S + noise


def cpu_nfft(n_samples):
    """Размер FFT: следующая степень 2 от n_samples, затем ×2 (zero-padding)."""
    p = 1
    while p < n_samples:
        p <<= 1
    return p * 2    # ×2 для интерполяции в частотной области


# ===========================================================================
# ШАГ 0: CPU генерирует чистый сигнал
# ===========================================================================

def shag_0_cpu_signal():
    """
    ШАГ 0: CPU создаёт CW сигнал без шума.

    Проверяем базовые свойства:
    - Правильная форма массива: (N_ANT, N_SAMPLES)
    - Тип данных: complex64 (как у GPU)
    - Ненулевая амплитуда
    - Единичная амплитуда (после начального участка где tau > 0)
    """
    print("\n" + "─"*60)
    print("  ШАГ 0: CPU генерирует чистый CW сигнал")
    print("─"*60)

    S = cpu_generate_signal()

    print(f"  Форма массива S: {S.shape}  (ожидаем ({N_ANT}, {N_SAMPLES}))")
    print(f"  Тип данных:      {S.dtype}  (ожидаем complex64)")

    if S.shape != (N_ANT, N_SAMPLES):
        raise AssertionError(f"Неверная форма: {S.shape} ≠ ({N_ANT}, {N_SAMPLES})")
    print("  ✓ Форма правильная")

    if S.dtype != np.complex64:
        raise AssertionError(f"Неверный тип: {S.dtype} ≠ complex64")
    print("  ✓ Тип данных правильный (complex64)")

    print(f"\n  Проверяем каждую из {N_ANT} антенн:")
    for i in range(N_ANT):
        start_sample     = int(i * TAU_STEP * FS) + 10
        amplitude_actual = np.max(np.abs(S[i, start_sample:]))
        ok = abs(amplitude_actual - AMPLITUDE) < 0.01
        print(f"    Антенна {i}: амплитуда = {amplitude_actual:.4f}  "
              f"(ожидаем {AMPLITUDE:.1f})  {'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            raise AssertionError(f"Антенна {i}: амплитуда {amplitude_actual:.4f} ≠ {AMPLITUDE}")

    print("\n  ✓ ШАГ 0: ПРОЙДЕН\n")


# ===========================================================================
# ШАГ 1: Загружаем сигнал на GPU → скачиваем → сравниваем в цикле
# ===========================================================================

def shag_1_gpu_roundtrip():
    """ШАГ 1: CPU → GPU: загружаем сигнал, проверяем через GEMM."""
    if not HAS_GPU:
        print("  ШАГ 1: ПРОПУЩЕН (GPU недоступен)")
        return

    print("\n" + "─"*60)
    print("  ШАГ 1: CPU → GPU (данные правильно дошли?)")
    print("─"*60)

    S_cpu = cpu_generate_signal()
    W_cpu = cpu_generate_weight_matrix()
    X_ref = W_cpu @ S_cpu        # NumPy эталон: W @ S

    print("  Создаём GPU контекст...")
    ctx  = gpuworklib.ROCmGPUContext(GPU_DEVICE_ID)
    proc = gpuworklib.AntennaProcessorTest(
        ctx, n_ant=N_ANT, n_samples=N_SAMPLES,
        sample_rate=float(FS), signal_frequency_hz=float(F0),
        debug_mode=True
    )
    proc.step_0_prepare_input(S_cpu, W_cpu)

    print("  Запускаем step_2_gemm() и сравниваем с NumPy W@S...")
    X_gpu = proc.step_2_gemm()

    del proc, ctx   # освобождаем GPU до проверок — иначе segfault при GC

    print(f"\n  Сравниваем каждый из {N_ANT} лучей (допуск={TOLERANCE_SIGNAL}):")
    for i in range(N_ANT):
        diff = float(np.mean(np.abs(X_ref[i] - X_gpu[i])))
        ok   = diff < TOLERANCE_SIGNAL
        print(f"    Луч {i}: расхождение = {diff:.2e}  {'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            raise AssertionError(f"Луч {i}: расхождение {diff:.2e} > {TOLERANCE_SIGNAL}")

    print("\n  ✓ ШАГ 1: ПРОЙДЕН (данные дошли до GPU без искажений)\n")


# ===========================================================================
# ШАГ 2: Добавляем шум, копируем шумный сигнал на CPU
# ===========================================================================

def shag_2_noisy_signal():
    """
    ШАГ 2: Сигнал + шум.

    Сначала проверяем чистый сигнал (Шаг 2а).
    Потом добавляем шум, копируем шумный сигнал на CPU (Шаг 2б).
    Проверяем что мощность шума примерно правильная.

    Зачем: реальные данные всегда содержат шум.
    Если pipeline правильно работает с чистым сигналом — проверяем с шумным.
    """
    print("\n" + "─"*60)
    print("  ШАГ 2: Добавляем шум, проверяем мощность")
    print("─"*60)

    S_clean     = cpu_generate_signal()
    power_clean = float(np.mean(np.abs(S_clean) ** 2))
    print(f"  2а. Чистый сигнал: мощность = {power_clean:.4f}")
    if power_clean <= 0:
        raise AssertionError("Сигнал нулевой!")
    print("  2а. ✓ Чистый сигнал корректный")

    noise_levels = [NOISE_SIGMA * 0.1, NOISE_SIGMA, NOISE_SIGMA * 10]
    print(f"\n  2б. Проверяем {len(noise_levels)} уровня шума:")

    for sigma in noise_levels:
        S_noisy     = cpu_add_noise(S_clean, sigma=sigma, seed=42)
        power_noisy = float(np.mean(np.abs(S_noisy) ** 2))
        snr_db      = 10 * np.log10(power_clean / sigma**2)
        ok          = power_noisy > power_clean * 0.5
        print(f"    σ={sigma:.3f}: мощность={power_noisy:.4f}  SNR={snr_db:.1f} дБ  "
              f"{'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            raise AssertionError(f"σ={sigma:.3f}: мощность {power_noisy:.4f} слишком мала")

    print("\n  ✓ ШАГ 2: ПРОЙДЕН\n")


# ===========================================================================
# ШАГ 3: GEMM — умножение на матрицу весов (CPU vs GPU в цикле)
# ===========================================================================

def shag_3_gemm():
    """
    ШАГ 3: W @ S — beamforming matrix multiply (CPU vs GPU).

    CPU:  X_cpu = W @ S          ← numpy matmul
    GPU:  X_gpu = proc.step_2_gemm()  ← ROCm GPU kernel

    Сравниваем каждый луч (beam) в простом цикле.

    Физический смысл: W @ S складывает все антенны когерентно.
    Сигнал в нужном направлении усиливается в √N раз, шум — нет.
    """
    print("\n" + "─"*60)
    print("  ШАГ 3: GEMM (W @ S) — CPU vs GPU")
    print("─"*60)

    # --- CPU: генерируем сигнал и матрицу ---
    S_cpu = cpu_generate_signal()
    W_cpu = cpu_generate_weight_matrix()

    print(f"  Форма S: {S_cpu.shape}")
    print(f"  Форма W: {W_cpu.shape}")

    # CPU умножение
    X_cpu = W_cpu @ S_cpu    # результат: (N_ANT, N_SAMPLES)
    print(f"  CPU: X = W @ S, форма результата: {X_cpu.shape}")

    # W нормирован (||строка||=1), при когерентном сложении усиление = √N
    expected_gain = np.sqrt(N_ANT)
    gain_cpu      = float(np.mean(np.abs(X_cpu))) / float(np.mean(np.abs(S_cpu)))
    ok_gain       = abs(gain_cpu - expected_gain) < expected_gain * 0.3
    print(f"  CPU: усиление = {gain_cpu:.4f}  (ожидаем ~√{N_ANT} = {expected_gain:.4f})  "
          f"{'✓ OK' if ok_gain else '✗ ОШИБКА'}")

    if not ok_gain:
        raise AssertionError(f"Усиление {gain_cpu:.4f} далеко от √N={expected_gain:.4f}")

    # --- GPU ---
    if not HAS_GPU:
        print("  GPU: ПРОПУЩЕН (GPU недоступен)")
        print("\n  ✓ ШАГ 3: ПРОЙДЕН (только CPU)\n")
        return

    ctx  = gpuworklib.ROCmGPUContext(GPU_DEVICE_ID)
    proc = gpuworklib.AntennaProcessorTest(
        ctx, n_ant=N_ANT, n_samples=N_SAMPLES,
        sample_rate=float(FS), signal_frequency_hz=float(F0),
        debug_mode=True
    )
    proc.step_0_prepare_input(S_cpu, W_cpu)

    print("\n  GPU: запускаем step_2_gemm()...")
    X_gpu = proc.step_2_gemm()
    del proc, ctx   # освобождаем GPU до проверок — иначе segfault при GC
    print(f"  GPU: форма результата: {X_gpu.shape}")

    print(f"\n  Сравниваем каждый из {N_ANT} лучей (допуск={TOLERANCE_SIGNAL}):")
    for beam in range(N_ANT):
        diff = float(np.mean(np.abs(X_cpu[beam] - X_gpu[beam])))
        ok   = diff < TOLERANCE_SIGNAL
        print(f"    Луч {beam}: среднее расхождение = {diff:.2e}  "
              f"{'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            bad_idx = np.where(np.abs(X_cpu[beam] - X_gpu[beam]) > TOLERANCE_SIGNAL)[0]
            print(f"      → Расходятся отсчёты: {bad_idx[:5]} ...")
            raise AssertionError(f"Луч {beam}: расхождение {diff:.2e} > {TOLERANCE_SIGNAL}")

    print("\n  ✓ ШАГ 3: ПРОЙДЕН (GPU GEMM совпадает с NumPy)\n")


# ===========================================================================
# ШАГ 4: FFT — ищем пик на f0=2 МГц (CPU vs GPU)
# ===========================================================================

def shag_4_fft_peak():
    """
    ШАГ 4: FFT + поиск пика.

    После W @ S применяем:
    1. Окно Хэмминга (убирает боковые лепестки)
    2. Zero-padding до nFFT (улучшает точность)
    3. FFT
    4. Ищем пик в первой половине спектра

    Проверяем: пик находится около f0 = 2 МГц (±2 бина).
    """
    print("\n" + "─"*60)
    print("  ШАГ 4: FFT — ищем пик на f0=2 МГц")
    print("─"*60)

    nFFT       = cpu_nfft(N_SAMPLES)
    bin_hz     = FS / nFFT             # разрешение по частоте
    expected_bin = round(F0 / bin_hz)  # ожидаемый бин

    print(f"  nFFT = {nFFT}  (N_SAMPLES={N_SAMPLES} → ближайшая степень 2 × 2)")
    print(f"  Разрешение: {bin_hz/1e3:.2f} кГц на бин")
    print(f"  Ожидаемый бин пика: {expected_bin}  (f0={F0/1e6:.1f} МГц)")

    # --- CPU pipeline ---
    S_cpu = cpu_generate_signal()
    W_cpu = cpu_generate_weight_matrix()
    X_cpu = W_cpu @ S_cpu           # GEMM

    # Окно Хэмминга
    win = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_SAMPLES) / (N_SAMPLES - 1))
           ).astype(np.float32)

    print(f"\n  CPU: FFT-пик для каждого из {N_ANT} лучей:")
    for beam in range(N_ANT):
        X_win      = (X_cpu[beam] * win).astype(np.complex64)
        X_pad      = np.zeros(nFFT, dtype=np.complex64)
        X_pad[:N_SAMPLES] = X_win
        magnitudes = np.abs(np.fft.fft(X_pad))
        half       = nFFT // 2
        peak_bin   = int(np.argmax(magnitudes[1:half])) + 1
        peak_freq  = peak_bin * FS / nFFT
        error_hz   = abs(peak_freq - F0)
        ok         = error_hz < 2 * bin_hz
        print(f"    Луч {beam}: пик {peak_freq/1e6:.4f} МГц  "
              f"ошибка={error_hz/1e3:.2f} кГц  {'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            raise AssertionError(f"CPU луч {beam}: пик {peak_freq/1e6:.4f} МГц ≠ {F0/1e6:.1f} МГц")

    # --- GPU FFT ---
    if not HAS_GPU:
        print("  GPU: ПРОПУЩЕН (GPU недоступен)")
        print("\n  ✓ ШАГ 4: ПРОЙДЕН (только CPU)\n")
        return

    ctx  = gpuworklib.ROCmGPUContext(GPU_DEVICE_ID)
    proc = gpuworklib.AntennaProcessorTest(
        ctx, n_ant=N_ANT, n_samples=N_SAMPLES,
        sample_rate=float(FS), signal_frequency_hz=float(F0),
        debug_mode=True
    )
    proc.step_0_prepare_input(S_cpu, W_cpu)
    proc.step_2_gemm()

    print("\n  GPU: запускаем step_4_window_fft()...")
    spec_gpu  = proc.step_4_window_fft()
    mags_gpu  = np.abs(spec_gpu)
    del proc, ctx   # освобождаем GPU до проверок — иначе segfault при GC

    print("  GPU: FFT-пик для каждого луча:")
    for beam in range(N_ANT):
        half      = nFFT // 2
        peak_bin  = int(np.argmax(mags_gpu[beam, 1:half])) + 1
        peak_freq = peak_bin * FS / nFFT
        error_hz  = abs(peak_freq - F0)
        ok        = error_hz < 2 * bin_hz
        print(f"    Луч {beam}: GPU пик = {peak_freq/1e6:.4f} МГц  "
              f"ошибка={error_hz/1e3:.2f} кГц  {'✓ OK' if ok else '✗ ОШИБКА'}")
        if not ok:
            raise AssertionError(f"GPU луч {beam}: пик {peak_freq/1e6:.4f} МГц ≠ {F0/1e6:.1f} МГц")

    print("\n  ✓ ШАГ 4: ПРОЙДЕН\n")


# ===========================================================================
# Запуск как обычный Python скрипт (без pytest)
# ===========================================================================

if __name__ == "__main__":
    shag_0_cpu_signal()
    shag_1_gpu_roundtrip()
    shag_2_noisy_signal()
    shag_3_gemm()
    shag_4_fft_peak()
    print("=" * 60)
    print("  Все шаги пройдены ✓")
    print("=" * 60)
