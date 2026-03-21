"""
signal_test_base.py — SignalTestBase (Template Method для генераторов)
======================================================================

Наследует TestBase. Специализирован для тестирования GPU-генераторов сигналов.

Template Method hooks:
    get_params()         → конфигурация (SignalConfig или LfmParams)
    get_generator()      → GPU-генератор (SignalGenerator, FormSignalGenerator, ...)
    generate_data()      → опционально (по умолчанию пустой массив)
    process()            → вызов GPU-генератора → numpy сигнал
    compute_reference()  → numpy эталон
    validate()           → сравнение GPU vs numpy

Пример использования:
    class TestCW(SignalTestBase):
        filter_name = "CW_generator"

        def get_params(self):
            return SignalConfig(fs=12e6, n_samples=8192, f0_hz=2e6)

        def process(self, data, ctx):
            gen = gpuworklib.SignalGenerator(ctx)
            return gen.generate_cw(freq=self.params.f0_hz,
                                   fs=self.params.fs,
                                   length=self.params.n_samples)

        def compute_reference(self, params):
            t = np.arange(params.n_samples) / params.fs
            return np.exp(1j * 2 * np.pi * params.f0_hz * t).astype(np.complex64)

        def validate(self, result, params):
            ref = self.compute_reference(params)
            return self._validate_vs_reference(result, ref)
"""

import numpy as np
from abc import abstractmethod
from typing import Optional

from common.test_base import TestBase
from common.result import TestResult
from common.validators import DataValidator


class SignalTestBase(TestBase):
    """Базовый класс для тестов GPU-генераторов сигналов.

    Расширяет TestBase:
      - хранит _last_params для использования в validate()
      - добавляет hook compute_reference()
      - предоставляет _validate_vs_reference() для удобной валидации

    По умолчанию generate_data() возвращает None — для генераторов
    входные данные не нужны (генератор создаёт сигнал из параметров).
    """

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._last_params = None

    def generate_data(self, params) -> Optional[np.ndarray]:
        """Нет входных данных — генераторы создают сигнал из параметров."""
        self._last_params = params
        return None

    def run(self) -> TestResult:
        """Переопределён: сохраняет params перед process()."""
        from common.gpu_context import GPUContextManager
        result = TestResult(test_name=self.name)
        try:
            self.setup()
            ctx = GPUContextManager.get()
            params = self.get_params()
            self._last_params = params
            data = self.generate_data(params)
            output = self.process(data, ctx)
            result = self.validate(output, params)
            result.test_name = self.name
        except Exception as e:
            result.error = e
        finally:
            self.teardown()
        return result

    @abstractmethod
    def compute_reference(self, params) -> np.ndarray:
        """Вычислить numpy-эталон для GPU-сигнала.

        Args:
            params: конфигурация (SignalConfig или LfmParams)

        Returns:
            np.ndarray — эталонный сигнал
        """
        ...

    def _validate_vs_reference(self, gpu_output: np.ndarray,
                                reference: np.ndarray,
                                tolerance: float = 1e-4) -> TestResult:
        """Сравнить GPU-выход с numpy-эталоном.

        Args:
            gpu_output:  сигнал от GPU-генератора
            reference:   numpy эталон
            tolerance:   допустимая относительная погрешность

        Returns:
            TestResult с ValidationResult
        """
        validator = DataValidator(tolerance=tolerance, metric="max_rel")
        vr = validator.validate(gpu_output, reference, name="gpu_vs_numpy")
        return TestResult(self.name).add(vr)

    def _check_peak_frequency(self, signal: np.ndarray, fs: float,
                               expected_hz: float,
                               tolerance_hz: float = 1e3) -> bool:
        """Проверить что пик спектра находится на ожидаемой частоте.

        Args:
            signal:       комплексный сигнал
            fs:           частота дискретизации
            expected_hz:  ожидаемая частота пика
            tolerance_hz: допустимое отклонение

        Returns:
            True если пик в пределах tolerance_hz
        """
        n = len(signal)
        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        peak_idx = np.argmax(spectrum)
        peak_freq = abs(float(freqs[peak_idx]))
        return abs(peak_freq - expected_hz) <= tolerance_hz
