"""
test_fm_correlator.py — Python тесты FM Correlator (NumPy reference, no GPU)
=============================================================================

Проверяем математику FM-коррелятора через NumPy-эталон:
  - Автокорреляция M-последовательности → пик в 0
  - Кросс-корреляция со сдвигом → пик в нужном лаге
  - Сравнение Python-эталона с gpuworklib (если доступен ROCm)

Тесты 1-5: NumPy-only (работают без GPU)
Тесты 6-8: gpuworklib.FMCorrelatorROCm (только если импорт успешен)

Author: Kodo (AI Assistant)
Date: 2026-03-10
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner
from common.gpu_loader import GPULoader


# ============================================================================
# NumPy reference: LFSR M-sequence
# ============================================================================

def generate_msequence_cpu(n: int, seed: int = 0x12345678,
                           poly: int = 0x00400007) -> np.ndarray:
    """Генерация M-последовательности LFSR на CPU (эталон для C++/GPU).

    Использует LFSR с полиномом poly. Выход: {+1, -1}.
    Совместимо с FMCorrelator::GenerateMSequence().
    """
    seq = np.zeros(n, dtype=np.float32)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        bit = (state >> 31) & 1
        seq[i] = 1.0 if bit else -1.0
        # feedback: XOR bits согласно полиному
        feedback = bin(state & poly).count('1') & 1
        state = ((state << 1) | feedback) & 0xFFFFFFFF
    return seq


def correlate_numpy(ref: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """Кросс-корреляция через FFT (эталон).

    Returns: corr[j] = sum(ref[k] * inp[(k+j) % N]) — циклическая.
    Нормировка: / N (как в FMCorrelator::RunCorrelationPipeline).
    """
    n = len(ref)
    ref_fft = np.fft.rfft(ref, n=n)
    inp_fft = np.fft.rfft(inp, n=n)
    corr_fft = np.conj(ref_fft) * inp_fft
    corr = np.fft.irfft(corr_fft, n=n)
    return np.abs(corr) / n


# ============================================================================
# Tests 1-5: NumPy-only (no GPU required)
# ============================================================================

class TestMSequence:
    """Тесты генератора M-последовательности."""

    def test_values_are_plus_minus_one(self):
        """Все значения M-последовательности должны быть ±1."""
        seq = generate_msequence_cpu(n=1024)
        assert np.all(np.abs(seq) == 1.0), "Все элементы должны быть ±1"

    def test_roughly_half_plus_half_minus(self):
        """~50% единиц и ~50% минус-единиц (свойство M-последовательности)."""
        seq = generate_msequence_cpu(n=4096)
        n_plus = np.sum(seq > 0)
        n_minus = np.sum(seq < 0)
        ratio = n_plus / len(seq)
        assert 0.45 <= ratio <= 0.55, f"Ожидаем ~50% +1, получили {ratio:.2%}"

    def test_different_seeds_produce_different_sequences(self):
        """Разные seed → разные последовательности."""
        seq1 = generate_msequence_cpu(n=256, seed=0x12345678)
        seq2 = generate_msequence_cpu(n=256, seed=0x87654321)
        assert not np.array_equal(seq1, seq2), "Разные seed должны давать разные seq"

    def test_same_seed_reproducible(self):
        """Один и тот же seed → идентичная последовательность."""
        seq1 = generate_msequence_cpu(n=512, seed=0xABCDEF01)
        seq2 = generate_msequence_cpu(n=512, seed=0xABCDEF01)
        assert np.array_equal(seq1, seq2), "Одинаковый seed должен давать одинаковую seq"


class TestCorrelationNumpy:
    """Тесты корреляции через NumPy-эталон."""

    def test_autocorrelation_peak_at_zero(self):
        """Автокорреляция M-последовательности: пик в позиции 0, остальное мало."""
        n = 4096
        ref = generate_msequence_cpu(n=n)
        corr = correlate_numpy(ref, ref)

        peak_pos = np.argmax(corr)
        peak_val = corr[0]  # пик всегда в 0 для автокорреляции
        side_max = np.max(corr[1:])

        snr = peak_val / side_max if side_max > 0 else 1e9

        assert peak_pos == 0, f"Пик должен быть в позиции 0, получили {peak_pos}"
        assert snr > 10, f"SNR должен быть > 10, получили {snr:.2f}"

    def test_cross_correlation_peak_at_shift(self):
        """Кросс-корреляция ref с circshift(ref, d) → пик в позиции d."""
        n = 2048
        shift = 42
        ref = generate_msequence_cpu(n=n)

        # Циклический сдвиг на shift отсчётов
        inp = np.roll(ref, shift)
        corr = correlate_numpy(ref, inp)

        peak_pos = np.argmax(corr[:100])  # ищем пик в первых 100 точках
        assert peak_pos == shift, (
            f"Пик должен быть в позиции {shift}, получили {peak_pos}"
        )

    def test_zero_shift_gives_highest_peak(self):
        """При нулевом сдвиге корреляция максимальна (пик в 0)."""
        n = 1024
        ref = generate_msequence_cpu(n=n)
        corr = correlate_numpy(ref, ref)

        # corr[0] должен быть максимальным среди первых 100 точек
        max_val = float(np.max(corr[:100]))
        assert abs(corr[0] - max_val) < abs(max_val) * 0.01, \
            "corr[0] должен быть максимальным при нулевом сдвиге"

    def test_output_shape_matches_params(self):
        """Форма выхода [S, K, n_kg] соответствует параметрам."""
        S = 3   # signals
        K = 4   # shifts
        n_kg = 50  # output points
        n = 512

        ref = generate_msequence_cpu(n=n)
        peaks = np.zeros((S, K, n_kg), dtype=np.float32)

        for s in range(S):
            inp = np.roll(ref, s * 10)
            for k in range(K):
                ref_shifted = np.roll(ref, k * 5)
                corr = correlate_numpy(ref_shifted, inp)
                peaks[s, k, :] = corr[:n_kg]

        assert peaks.shape == (S, K, n_kg), \
            f"Форма {peaks.shape} != ожидаемой ({S}, {K}, {n_kg})"


# ============================================================================
# Tests 6-8: gpuworklib.FMCorrelatorROCm (requires ROCm GPU)
# ============================================================================

# gpuworklib.FMCorrelatorROCm — используем GPULoader для кросс-платформенного поиска
_gw = GPULoader.get()
HAS_FM_CORRELATOR = _gw is not None and hasattr(_gw, 'FMCorrelatorROCm')
if _gw is not None:
    gpuworklib = _gw

class TestFMCorrelatorROCm:
    """Тесты GPU-реализации через gpuworklib.FMCorrelatorROCm."""

    def setUp(self):
        if not HAS_FM_CORRELATOR:
            raise SkipTest("gpuworklib.FMCorrelatorROCm не найден (нужна пересборка с ENABLE_ROCM=ON)")
        self._ctx = gpuworklib.ROCmGPUContext(0)

    def test_generate_msequence_values(self):
        """GPU: generate_msequence возвращает ±1 массив."""
        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=1024, num_shifts=1, num_signals=1,
                        num_output_points=50)
        seq = corr.generate_msequence(seed=0x12345678)

        assert seq.dtype == np.float32
        assert len(seq) == 1024
        assert np.all(np.abs(seq) == 1.0), "Все элементы должны быть ±1"

    def test_gpu_autocorrelation_snr(self):
        """GPU: автокорреляция M-sequence → SNR > 10."""
        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=4096, num_shifts=1, num_signals=1,
                        num_output_points=200)

        # Генерируем ref на CPU, загружаем как inp тоже
        ref = corr.generate_msequence(seed=0x12345678)
        corr.prepare_reference_from_data(ref)

        # Коррелируем ref с самим собой
        inp = np.tile(ref, 1).reshape(1, -1)  # [1, N]
        peaks = corr.process(inp)  # [S=1, K=1, n_kg=200]

        assert peaks.shape == (1, 1, 200), f"Форма {peaks.shape} != (1, 1, 200)"

        peak_val = peaks[0, 0, 0]
        side_max = np.max(peaks[0, 0, 1:])
        snr = peak_val / side_max if side_max > 0 else 1e9

        assert snr > 10, f"SNR={snr:.2f} < 10"

    def test_gpu_shift_peak_position(self):
        """GPU: test_pattern с shift_step=2 → пики на ожидаемых позициях."""
        S = 3
        K = 4
        shift_step = 2
        n_kg = 100

        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=1024, num_shifts=K, num_signals=S,
                        num_output_points=n_kg)
        corr.prepare_reference()

        peaks = corr.run_test_pattern(shift_step=shift_step)

        assert peaks.shape == (S, K, n_kg), \
            f"Форма {peaks.shape} != ({S}, {K}, {n_kg})"

        # Для сигнала s=0, сдвига k=0: пик должен быть в позиции 0
        peak_k0 = np.argmax(peaks[0, 0, :])
        assert peak_k0 == 0, f"s=0, k=0: пик в {peak_k0}, ожидали 0"

    def test_gpu_vs_numpy_correlation(self):
        """GPU результат совпадает с NumPy эталоном (max_error < 0.05)."""
        n = 1024

        corr = gpuworklib.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=n, num_shifts=1, num_signals=1,
                        num_output_points=100)

        # Одна и та же seed — CPU и GPU должны дать одинаковую M-seq
        ref = corr.generate_msequence(seed=0x12345678)
        corr.prepare_reference_from_data(ref)

        # Вход = ref (нулевой сдвиг)
        inp = ref.reshape(1, -1)
        peaks_gpu = corr.process(inp)  # [1, 1, 100]

        # NumPy эталон
        corr_np = correlate_numpy(ref, ref)[:100]

        max_err = np.max(np.abs(peaks_gpu[0, 0] - corr_np))
        assert max_err < 0.05, f"max_error={max_err:.6f} >= 0.05 (GPU vs NumPy)"


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestMSequence())
    results += runner.run(TestCorrelationNumpy())
    results += runner.run(TestFMCorrelatorROCm())
    runner.print_summary(results)
