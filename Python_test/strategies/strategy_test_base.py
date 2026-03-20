"""
strategy_test_base.py — StrategyTestBase (Template Method GoF)
==============================================================

Наследует common.TestBase (Template Method).
Специализирован для тестов антенной стратегии:
  - Генерация сигнала через ISignalStrategy (Strategy GoF)
  - Подготовка матрицы W (identity-like)
  - NumPy reference pipeline (GEMM + FFT + peak finding)

Подклассы реализуют process() и validate().

Пример:
    class MyTest(StrategyTestBase):
        def process(self, data, ctx):
            ...
        def validate(self, result, params):
            ...

    test = MyTest(SinSignalStrategy(), AntennaTestParams.small())
    tr = test.run()
    print(tr.summary())
"""

from abc import abstractmethod
import math
import numpy as np
import sys
import os

# Добавить strategies/ в sys.path
_STRAT_DIR = os.path.dirname(os.path.abspath(__file__))
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

# Добавить Python_test/ в sys.path (чтобы common был пакетом)
_PYTHON_TEST_DIR = os.path.dirname(_STRAT_DIR)
if _PYTHON_TEST_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_DIR)

from common.test_base import TestBase
from common.result import TestResult, ValidationResult
from test_params import AntennaTestParams, SignalVariant
from signal_generators_strategy import ISignalStrategy


class StrategyTestBase(TestBase):
    """Template Method для тестов антенной стратегии.

    Скелет:
      run() → setup() → get_params() → generate_data()
            → process() → validate() → teardown()

    Добавляет:
      - generate_data() автоматически вызывает signal_strategy.generate()
      - Готовит матрицу W (identity / delay-and-sum) через _make_weight_matrix()
      - Хранит промежуточные данные (W, X_gemm, spectrum) для подклассов

    Args:
        signal_strategy:  ISignalStrategy — источник тестовых сигналов
        params:           AntennaTestParams — параметры теста
    """

    def __init__(self,
                 signal_strategy: ISignalStrategy,
                 params: AntennaTestParams):
        super().__init__(name=f"{self.__class__.__name__}_{signal_strategy.name}")
        self.signal_strategy = signal_strategy
        self.params          = params
        # Промежуточные данные (доступны подклассам после process())
        self.W:        np.ndarray | None = None   # [n_ant, n_ant] complex64
        self.X_gemm:   np.ndarray | None = None   # [n_ant, n_samples] complex64
        self.spectrum: np.ndarray | None = None   # [n_ant, nFFT] complex64
        self.magnitudes: np.ndarray | None = None # [n_ant, nFFT] float32

    # ── TestBase hooks ────────────────────────────────────────────────────────

    def get_params(self) -> AntennaTestParams:
        return self.params

    def generate_data(self, params: AntennaTestParams) -> np.ndarray:
        """Генерировать входной сигнал через ISignalStrategy.

        Returns:
            [n_ant, n_samples] complex64
        """
        return self.signal_strategy.generate(params)

    @abstractmethod
    def process(self, data: np.ndarray, ctx) -> np.ndarray:
        """Выполнить обработку (NumPy reference или GPU вызов).

        Args:
            data: [n_ant, n_samples] complex64 — входной сигнал
            ctx:  GPU-контекст (None для NumPy тестов)

        Returns:
            np.ndarray — результат (форма зависит от теста)
        """
        ...

    @abstractmethod
    def validate(self, result: np.ndarray, params: AntennaTestParams) -> TestResult:
        """Проверить результат.

        Returns:
            TestResult с набором ValidationResult
        """
        ...

    # ── NumPy reference helpers (доступны подклассам) ────────────────────────

    def _make_weight_matrix(self, params: AntennaTestParams) -> np.ndarray:
        """Identity-like матрица весов [n_ant, n_ant] complex64.

        Note:
            Для non-square (n_ant×n_beams) — нужно добавить n_beams.
            Сейчас: W = I / sqrt(n_ant) (identity-like, delay=0).
        """
        W = np.eye(params.n_ant, dtype=np.complex64) / np.sqrt(params.n_ant)
        return W

    def _apply_gemm(self, W: np.ndarray, S: np.ndarray) -> np.ndarray:
        """X = W @ S (матричное умножение, NumPy).

        Args:
            W: [n_ant, n_ant] complex64
            S: [n_ant, n_samples] complex64

        Returns:
            [n_ant, n_samples] complex64
        """
        return (W @ S).astype(np.complex64)

    def _apply_window_fft(self,
                          X: np.ndarray,
                          nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Hamming window + zero-pad + FFT → spectrum + magnitudes.

        Args:
            X:    [n_ant, n_samples] complex64
            nfft: размер FFT (None = следующая степень 2)

        Returns:
            (spectrum [n_ant, nFFT] complex64, magnitudes [n_ant, nFFT] float32)
        """
        n_samples = X.shape[1]
        if nfft is None:
            nfft = 2 ** math.ceil(math.log2(n_samples))

        window   = np.hamming(n_samples).astype(np.float32)
        X_win    = (X * window[np.newaxis, :]).astype(np.complex64)
        X_pad    = np.zeros((X.shape[0], nfft), dtype=np.complex64)
        X_pad[:, :n_samples] = X_win

        spec     = np.fft.fft(X_pad, axis=1).astype(np.complex64)
        mags     = np.abs(spec).astype(np.float32)
        return spec, mags

    def _find_peak_freq(self, magnitudes: np.ndarray,
                        beam: int = 0) -> float:
        """Найти частоту пика спектра для луча `beam`.

        Returns:
            Частота пика (Гц)
        """
        nfft   = magnitudes.shape[1]
        bin_hz = self.params.fs / nfft
        peak_bin = int(np.argmax(magnitudes[beam, :nfft // 2]))
        return peak_bin * bin_hz

    def _run_numpy_pipeline(self, S: np.ndarray) -> dict:
        """Запустить полный NumPy reference pipeline.

        Returns:
            dict с ключами: W, X_gemm, spectrum, magnitudes, peak_freq_hz
        """
        W       = self._make_weight_matrix(self.params)
        X       = self._apply_gemm(W, S)
        spec, mags = self._apply_window_fft(X)
        freq_hz = self._find_peak_freq(mags)

        self.W          = W
        self.X_gemm     = X
        self.spectrum   = spec
        self.magnitudes = mags

        return {
            "W":          W,
            "X_gemm":     X,
            "spectrum":   spec,
            "magnitudes": mags,
            "peak_freq_hz": freq_hz,
        }
