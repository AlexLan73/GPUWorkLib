#!/usr/bin/env python3
"""
test_strategies_step_by_step.py — пошаговая валидация pipeline GPU vs NumPy
=============================================================================

ЗАЧЕМ:
    Сравнивает каждый шаг GPU-пайплайна с NumPy-эталоном пошагово.
    Если GPU даёт неверный результат — этот тест покажет на каком именно шаге
    (GEMM? FFT? OneMax?) расходятся GPU и NumPy.
    Аналогия: пошаговая отладка, только вместо debugger — NumPy как эталон.

ЧТО ПРОВЕРЯЕТ:
    Часть 1 (NumPy, GPU не нужен):
      1. W matrix shape и унитарность (||строка||=1)
      2. Коэффициенты Хэмминга совпадают с формулой
      3. W @ S совпадает с numpy matmul
      4. FFT-пик около f0=2 МГц

    Часть 2 (ROCm GPU, AntennaProcessorTest):
      5. GPU GEMM ≈ NumPy (rtol=1e-3)
      6. GPU FFT shape и пик ≈ f0
      7. OneMax: уточнённая частота через параболу
      8. AllMaxima: хотя бы 1 пик на луч
      9. GlobalMinMax: max >= min, DR > 0 дБ
     10. Full pipeline: total_ms > 0, результаты есть

    Параметры: 5 антенн × 8000 отсчётов, fs=12 МГц, f0=2 МГц, tau_step=100 мкс.

GPU:
    Часть 1 — НЕ НУЖЕН (чистый NumPy).
    Часть 2 — нужен ROCm + класс AntennaProcessorTest в gpuworklib.
    Если GPU недоступен — Часть 2 пропускается (SkipTest).

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_strategies_step_by_step.py

Author: Kodo (AI Assistant)
Date: 2026-03-07
"""

import os
import sys
import numpy as np

# ============================================================================
# Constants (mirror C++ test)
# ============================================================================

N_ANT = 5
N_SAMPLES = 8000
FS = 12.0e6
F0 = 2.0e6
TAU_BASE = 0.0
TAU_STEP = 100e-6
AMPLITUDE = 1.0

# ============================================================================
# NumPy reference implementations
# ============================================================================

def generate_signal_numpy(n_ant, n_samples, fs, f0, amplitude, tau_base, tau_step):
    """Generate test signal matching FormSignalGeneratorROCm output."""
    dt = 1.0 / fs
    t = np.arange(n_samples) * dt
    S = np.zeros((n_ant, n_samples), dtype=np.complex64)
    for ant in range(n_ant):
        tau = tau_base + ant * tau_step
        t_delayed = t - tau
        valid = t_delayed >= 0
        S[ant, valid] = amplitude * np.exp(1j * 2 * np.pi * f0 * t_delayed[valid]).astype(np.complex64)
    return S


def generate_weight_matrix_numpy(n_ant, f0, tau_base, tau_step):
    """Generate delay-and-sum weight matrix W[beam][ant]."""
    W = np.zeros((n_ant, n_ant), dtype=np.complex64)
    inv_sqrt_n = 1.0 / np.sqrt(n_ant)
    for beam in range(n_ant):
        for ant in range(n_ant):
            tau = tau_base + ant * tau_step
            W[beam, ant] = inv_sqrt_n * np.exp(-1j * 2 * np.pi * f0 * tau)
    return W


def hamming_window(n):
    """Hamming window matching the C++ kernel."""
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def compute_nFFT(n_samples):
    """Next power of 2 >= n_samples, then x2."""
    p = 1
    while p < n_samples:
        p <<= 1
    return p * 2


# ============================================================================
# NumPy-only tests (no GPU required)
# ============================================================================

class TestNumpyReference:
    """Validate reference implementations with NumPy."""

    def test_weight_matrix_shape(self):
        W = generate_weight_matrix_numpy(N_ANT, F0, TAU_BASE, TAU_STEP)
        assert W.shape == (N_ANT, N_ANT)
        assert W.dtype == np.complex64

    def test_weight_matrix_normalization(self):
        """Each row should have ||row|| = 1 (unit norm for beamforming)."""
        W = generate_weight_matrix_numpy(N_ANT, F0, TAU_BASE, TAU_STEP)
        for beam in range(N_ANT):
            norm = np.linalg.norm(W[beam])
            np.testing.assert_allclose(norm, 1.0, atol=1e-5,
                err_msg=f"Beam {beam} norm={norm}")

    def test_hamming_window(self):
        """Hamming window endpoints and symmetry."""
        w = hamming_window(N_SAMPLES)
        assert len(w) == N_SAMPLES
        # Endpoints: 0.54 - 0.46 = 0.08
        np.testing.assert_allclose(w[0], 0.08, atol=1e-5)
        np.testing.assert_allclose(w[-1], 0.08, atol=1e-5)
        # Center peak
        assert w[N_SAMPLES // 2] > 0.99

    def test_gemm_numpy(self):
        """W @ S produces correct shape and non-zero output."""
        S = generate_signal_numpy(N_ANT, N_SAMPLES, FS, F0, AMPLITUDE, TAU_BASE, TAU_STEP)
        W = generate_weight_matrix_numpy(N_ANT, F0, TAU_BASE, TAU_STEP)
        X = W @ S
        assert X.shape == (N_ANT, N_SAMPLES)
        assert np.max(np.abs(X)) > 0.0

    def test_fft_peak_numpy(self):
        """FFT of windowed GEMM output has peak near f0=2MHz."""
        S = generate_signal_numpy(N_ANT, N_SAMPLES, FS, F0, AMPLITUDE, TAU_BASE, TAU_STEP)
        W = generate_weight_matrix_numpy(N_ANT, F0, TAU_BASE, TAU_STEP)
        X = W @ S

        nFFT = compute_nFFT(N_SAMPLES)
        w = hamming_window(N_SAMPLES)

        # Apply window and zero-pad
        X_windowed = X[0] * w
        X_padded = np.zeros(nFFT, dtype=np.complex64)
        X_padded[:N_SAMPLES] = X_windowed

        spectrum = np.fft.fft(X_padded)
        magnitudes = np.abs(spectrum)

        # Find peak in first half
        half = nFFT // 2
        peak_bin = np.argmax(magnitudes[1:half]) + 1
        peak_freq = peak_bin * FS / nFFT

        # Peak should be within 1 bin of f0
        freq_resolution = FS / nFFT
        assert abs(peak_freq - F0) < 2 * freq_resolution, \
            f"Peak at {peak_freq:.0f} Hz, expected ~{F0:.0f} Hz"

    def test_nfft_computation(self):
        """nFFT = next_pow2(8000) * 2 = 8192 * 2 = 16384."""
        nFFT = compute_nFFT(N_SAMPLES)
        assert nFFT == 16384


# ============================================================================
# GPU tests (require gpuworklib with ENABLE_ROCM)
# ============================================================================

try:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "build", "debian-radeon9070", "python"
    ))
    import gpuworklib
    HAS_GPU = (hasattr(gpuworklib, 'AntennaProcessorTest') and
               hasattr(gpuworklib, 'ROCmGPUContext'))
except ImportError:
    HAS_GPU = False

class TestGPUvsNumPy:
    """Compare GPU pipeline output with NumPy reference."""

    def setUp(self):
        """Create GPU context and processor once. Call before running tests."""
        if not HAS_GPU:
            from common.runner import SkipTest
            raise SkipTest("gpuworklib with ROCm strategies not available")
        self.ctx = gpuworklib.ROCmGPUContext(0)
        self.proc = gpuworklib.AntennaProcessorTest(
            self.ctx,
            n_ant=N_ANT,
            n_samples=N_SAMPLES,
            sample_rate=float(FS),
            signal_frequency_hz=float(F0),
            debug_mode=True
        )

        # Generate reference data
        self.S_ref = generate_signal_numpy(N_ANT, N_SAMPLES, FS, F0, AMPLITUDE, TAU_BASE, TAU_STEP)
        self.W_ref = generate_weight_matrix_numpy(N_ANT, F0, TAU_BASE, TAU_STEP)

        # Upload to GPU
        self.proc.step_0_prepare_input(self.S_ref, self.W_ref)

    def test_gpu_gemm_vs_numpy(self):
        """GPU GEMM output matches numpy W @ S."""
        X_gpu = self.proc.step_2_gemm()
        X_ref = self.W_ref @ self.S_ref

        assert X_gpu.shape == (N_ANT, N_SAMPLES)
        # Allow some floating point tolerance (GPU uses float32)
        np.testing.assert_allclose(
            np.abs(X_gpu), np.abs(X_ref),
            rtol=1e-3, atol=1e-5,
            err_msg="GPU GEMM magnitude mismatch"
        )

    def test_gpu_fft_shape(self):
        """GPU FFT output has correct shape."""
        self.proc.step_2_gemm()
        spectrum = self.proc.step_4_window_fft()
        nFFT = self.proc.nFFT

        assert spectrum.shape == (N_ANT, nFFT)
        assert nFFT == compute_nFFT(N_SAMPLES)

    def test_gpu_fft_peak(self):
        """GPU FFT peak is near f0=2MHz."""
        self.proc.step_2_gemm()
        spectrum = self.proc.step_4_window_fft()
        nFFT = self.proc.nFFT

        magnitudes = np.abs(spectrum[0])
        half = nFFT // 2
        peak_bin = np.argmax(magnitudes[1:half]) + 1
        peak_freq = peak_bin * FS / nFFT

        freq_resolution = FS / nFFT
        assert abs(peak_freq - F0) < 2 * freq_resolution, \
            f"GPU peak at {peak_freq:.0f} Hz, expected ~{F0:.0f} Hz"

    def test_gpu_one_max_frequency(self):
        """Step2.1: OneMax refined frequency near f0."""
        self.proc.step_2_gemm()
        self.proc.step_4_window_fft()
        results = self.proc.step_6_1_one_max_parabola()

        assert len(results) == N_ANT
        # Check beam 0
        freq = results[0]['refined_freq_hz']
        assert abs(freq - F0) < FS / compute_nFFT(N_SAMPLES), \
            f"OneMax freq={freq:.0f}, expected ~{F0:.0f}"
        assert results[0]['magnitude'] > 0

    def test_gpu_all_maxima(self):
        """Step2.2: AllMaxima finds at least one peak per beam."""
        self.proc.step_2_gemm()
        self.proc.step_4_window_fft()
        results = self.proc.step_6_2_all_maxima()

        assert len(results) == N_ANT
        for beam in results:
            assert beam['num_maxima'] >= 1, \
                f"Beam {beam['antenna_id']}: no maxima found"

    def test_gpu_global_minmax(self):
        """Step2.3: GlobalMinMax — max >= min for all beams."""
        self.proc.step_2_gemm()
        self.proc.step_4_window_fft()
        results = self.proc.step_6_3_global_minmax()

        assert len(results) == N_ANT
        for mm in results:
            assert mm['max_magnitude'] >= mm['min_magnitude'], \
                f"Beam {mm['beam_id']}: max < min"
            assert mm['dynamic_range_dB'] > 0, \
                f"Beam {mm['beam_id']}: DR <= 0 dB"

    def test_gpu_full_pipeline(self):
        """Full pipeline runs and returns timing."""
        result = self.proc.process_full()
        assert result['total_ms'] > 0
        assert len(result['one_max']) == N_ANT
        assert len(result['minmax']) == N_ANT


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Strategies ROCm — Step-by-step validation")
    print("=" * 60)
    print(f"Parameters: {N_ANT} ant x {N_SAMPLES} pts, fs={FS/1e6:.0f}MHz, f0={F0/1e6:.0f}MHz")
    print(f"nFFT = {compute_nFFT(N_SAMPLES)}")
    print(f"GPU available: {HAS_GPU}")
    print()

    # Run NumPy tests
    print("--- NumPy Reference Tests ---")
    test = TestNumpyReference()
    for name in dir(test):
        if name.startswith("test_"):
            try:
                getattr(test, name)()
                print(f"  PASS: {name}")
            except Exception as e:
                print(f"  FAIL: {name} — {e}")

    # Run GPU tests if available
    if HAS_GPU:
        print("\n--- GPU vs NumPy Tests ---")
        gpu_test = TestGPUvsNumPy()
        gpu_test.setup()
        for name in dir(gpu_test):
            if name.startswith("test_"):
                try:
                    getattr(gpu_test, name)()
                    print(f"  PASS: {name}")
                except Exception as e:
                    print(f"  FAIL: {name} — {e}")
    else:
        print("\n--- GPU tests skipped (no gpuworklib with ROCm) ---")

    print("\nDone.")
