"""
filter_test_base.py — FilterTestBase (Template Method для фильтров)
====================================================================

Наследует TestBase и специализирует для тестирования GPU-фильтров.

Template Method (GoF) — добавляет hook compute_reference() для вычисления
scipy-эталона, который используется в validate() по умолчанию.

Паттерн использования:
    class TestFirLowpass(FilterTestBase):
        filter_name = "FirFilterROCm_lowpass"

        def get_params(self):
            return FilterConfig(filter_type="fir", cutoff_hz=1000, fs=50e3)

        def generate_data(self, params):
            rng = np.random.default_rng(42)
            return (rng.standard_normal(4096) + 1j*rng.standard_normal(4096)).astype(np.complex64)

        def process(self, data, ctx):
            coeffs = scipy.signal.firwin(64, params.normalized_cutoff())
            f = gpuworklib.FirFilterROCm(ctx, coeffs.tolist())
            return f.process(data)

        def compute_reference(self, data, params):
            coeffs = scipy.signal.firwin(64, params.normalized_cutoff())
            return scipy.signal.lfilter(coeffs, [1.0], data).astype(np.complex64)

        def validate(self, result, params):
            ref = self.compute_reference(self._last_data, params)
            v = DataValidator(tolerance=1e-4, metric="max_rel")
            return TestResult(self.name).add(v.validate(result, ref))
"""

import numpy as np
from abc import abstractmethod
from typing import Optional

from common.test_base import TestBase
from common.result import TestResult, ValidationResult
from common.validators import DataValidator
from common.configs import FilterConfig


class FilterTestBase(TestBase):
    """Базовый класс для тестов GPU-фильтров.

    Расширяет TestBase:
      - хранит _last_data для использования в validate()
      - добавляет hook compute_reference()
      - предоставляет _validate_with_scipy() для удобной валидации

    Подклассы реализуют:
      - get_params()        → FilterConfig
      - generate_data()     → np.ndarray
      - process()           → np.ndarray (GPU output)
      - compute_reference() → np.ndarray (scipy reference)
    """

    filter_name: str = ""  # переопределить в подклассе

    def __init__(self, name: str = ""):
        super().__init__(name or self.filter_name)
        self._last_data: Optional[np.ndarray] = None
        self._last_params: Optional[FilterConfig] = None

    def generate_data(self, params) -> np.ndarray:
        """По умолчанию — случайный комплексный сигнал.

        Переопределить для специфичных сигналов (two-tone, swept, ...).
        """
        n = getattr(params, "n_samples", 4096)
        seed = getattr(params, "seed", 42)
        rng = np.random.default_rng(seed)
        data = (rng.standard_normal(n) +
                1j * rng.standard_normal(n)).astype(np.complex64)
        self._last_data = data
        return data

    def run(self) -> TestResult:
        """Переопределён: сохраняет _last_data перед process()."""
        from common.gpu_context import GPUContextManager
        result = TestResult(test_name=self.name)
        try:
            self.setup()
            ctx = GPUContextManager.get()
            params = self.get_params()
            self._last_params = params
            data = self.generate_data(params)
            self._last_data = data
            output = self.process(data, ctx)
            result = self.validate(output, params)
            result.test_name = self.name
        except Exception as e:
            result.error = e
        finally:
            self.teardown()
        return result

    @abstractmethod
    def compute_reference(self, data: np.ndarray, params) -> np.ndarray:
        """Вычислить эталонный результат (scipy/numpy).

        Args:
            data:   входной сигнал (тот же что передан в process())
            params: конфигурация теста

        Returns:
            np.ndarray — эталонный выход фильтра
        """
        ...

    def _validate_with_scipy(self, gpu_output: np.ndarray, params,
                              tolerance: float = 1e-4) -> TestResult:
        """Сравнить GPU-выход с scipy-эталоном используя DataValidator.

        Convenience метод — чтобы не дублировать в каждом подклассе.

        Args:
            gpu_output: выход GPU-фильтра
            params:     конфигурация (передаётся в compute_reference)
            tolerance:  допустимая относительная погрешность

        Returns:
            TestResult с результатом сравнения
        """
        reference = self.compute_reference(self._last_data, params)
        validator = DataValidator(tolerance=tolerance, metric="max_rel")
        vr = validator.validate(gpu_output, reference, name="gpu_vs_scipy")
        return TestResult(self.name).add(vr)
