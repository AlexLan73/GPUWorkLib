"""
test_base.py — TestBase (Template Method)
==========================================

Template Method (GoF):
  TestBase.run() определяет неизменный скелет теста.
  Подклассы переопределяют hooks: get_params, generate_data, process, validate.

GRASP Controller:
  TestBase координирует шаги теста (не сами тесты вызывают ctx напрямую).

Usage:
    class MyFilterTest(TestBase):
        def get_params(self):
            return FilterConfig(cutoff_hz=1e3, fs=50e3)

        def generate_data(self, params):
            return np.random.randn(4096).astype(np.complex64)

        def process(self, data, ctx):
            f = gpuworklib.FirFilterROCm(ctx, coeffs)
            return f.process(data)

        def validate(self, result, params):
            v = DataValidator(tolerance=0.01, metric="max_rel")
            vr = v.validate(result, scipy_reference, name="filter_check")
            tr = TestResult("my_test")
            return tr.add(vr)

    test = MyFilterTest()
    result = test.run()
    print(result.summary())
"""

from abc import ABC, abstractmethod
import numpy as np

from .result import TestResult, ValidationResult
from .gpu_context import GPUContextManager


class TestBase(ABC):
    """Абстрактный базовый класс для GPU-тестов.

    Template Method (GoF) — run() задаёт скелет, подклассы реализуют hooks.

    Hooks (переопределяются в подклассах):
        get_params()         → конфигурация теста (dataclass)
        generate_data()      → тестовые данные (numpy array)
        process()            → GPU-обработка → результат
        validate()           → сравнение с эталоном → TestResult

    Optional hooks (переопределяются при необходимости):
        setup()              → дополнительная инициализация
        teardown()           → очистка ресурсов
    """

    # Имя теста (переопределить в подклассе или передать в конструктор)
    name: str = ""

    def __init__(self, name: str = ""):
        if name:
            self.name = name
        if not self.name:
            self.name = self.__class__.__name__

    def run(self) -> TestResult:
        """Неизменный скелет GPU-теста (Template Method).

        Шаги:
          1. setup()          — инициализация
          2. get_params()     — конфигурация
          3. generate_data()  — входные данные
          4. process()        — GPU-обработка
          5. validate()       — проверка результатов
          6. teardown()       — очистка

        Returns:
            TestResult с результатами всех проверок.
        """
        result = TestResult(test_name=self.name)
        try:
            self.setup()
            ctx = GPUContextManager.get()
            params = self.get_params()
            data = self.generate_data(params)
            output = self.process(data, ctx)
            result = self.validate(output, params)
            result.test_name = self.name
        except Exception as e:
            result.error = e
        finally:
            self.teardown()
        return result

    def setup(self) -> None:
        """Хук инициализации — переопределить при необходимости."""
        pass

    def teardown(self) -> None:
        """Хук очистки — переопределить при необходимости."""
        pass

    @abstractmethod
    def get_params(self):
        """Вернуть конфигурацию теста (dataclass).

        Returns:
            Любой dataclass (SignalConfig, FilterConfig, ...)
        """
        ...

    @abstractmethod
    def generate_data(self, params) -> np.ndarray:
        """Сгенерировать входные данные теста.

        Args:
            params: конфигурация (результат get_params())

        Returns:
            np.ndarray — входной сигнал
        """
        ...

    @abstractmethod
    def process(self, data: np.ndarray, ctx) -> np.ndarray:
        """Выполнить GPU-обработку.

        Args:
            data: входной сигнал
            ctx:  GPU-контекст (GPUContext или None)

        Returns:
            np.ndarray — выход GPU-алгоритма
        """
        ...

    @abstractmethod
    def validate(self, result: np.ndarray, params) -> TestResult:
        """Сравнить GPU-результат с эталоном.

        Args:
            result: выход GPU-алгоритма
            params: конфигурация (для вычисления эталона)

        Returns:
            TestResult с набором ValidationResult
        """
        ...
