# TASK_PythonArch_01 — Core/generators/

> **Фаза**: 1 (приоритет HIGH)
> **Зависимости**: — (первый в цепочке)
> **Статус**: ⬜ TODO
> **Оценка**: ~2 часа
> **Паттерны**: GoF Adapter, GoF Factory Method (Registry), GRASP Creator, SOLID ISP, OCP

---

## 🎯 Цель

Создать `Python_test/Core/generators/` — переиспользуемые GPU-генераторы сигналов.

**Проблема сейчас**: Каждый тест создаёт `gw.SignalGenerator(ctx)` вручную.
Если API gpuworklib изменится — нужно исправлять в 20 местах.

**Решение**: Adapter поверх gpuworklib + Registry Factory.
Тест делает `gen = GeneratorFactory.create("cw", ctx, params)` — и не зависит от gpuworklib напрямую.

---

## 📁 Создаваемые файлы (7 штук)

```
Python_test/
├── Core/
│   ├── __init__.py                 ← 1. пакет
│   └── generators/
│       ├── __init__.py             ← 2. регистрация + export
│       ├── base.py                 ← 3. ISignalGenerator (ABC)
│       ├── cw.py                   ← 4. CwGenerator (Adapter)
│       ├── lfm.py                  ← 5. LfmGenerator (Adapter)
│       ├── noise.py                ← 6. NoiseGenerator (Adapter)
│       └── factory.py              ← 7. GeneratorFactory (Registry)
```

---

## 📝 Детальное ТЗ для каждого файла

### 1. `Core/__init__.py`

```python
"""
Core — репозиторий готовых GPU-объектов.
Протестированные адаптеры над gpuworklib.

Использование:
    from Core.generators import GeneratorFactory
    gen = GeneratorFactory.create("cw", ctx, params)
"""
```

---

### 2. `Core/generators/__init__.py`

```python
"""Регистрирует все доступные генераторы в Factory при импорте."""

from .base import ISignalGenerator
from .cw import CwGenerator
from .lfm import LfmGenerator
from .noise import NoiseGenerator
from .factory import GeneratorFactory

# OCP: добавить новый генератор = register() здесь, не менять Factory
GeneratorFactory.register("cw", CwGenerator)
GeneratorFactory.register("lfm", LfmGenerator)
GeneratorFactory.register("noise", NoiseGenerator)

__all__ = [
    "ISignalGenerator",
    "CwGenerator", "LfmGenerator", "NoiseGenerator",
    "GeneratorFactory",
]
```

---

### 3. `Core/generators/base.py` — ISignalGenerator (ABC)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.configs import SignalConfig

class ISignalGenerator(ABC):
    """
    Интерфейс GPU-генератора сигнала.

    SOLID ISP: только генерация сигнала, ничего лишнего.
    GoF: общий интерфейс для всех Adapter-реализаций.
    GRASP LSP: CwGenerator/LfmGenerator/NoiseGenerator взаимозаменяемы.

    OCP: каждый адаптер реализует from_config() — Factory не знает
    про конкретные параметры, вызывает cls.from_config(ctx, params).
    """

    @classmethod
    @abstractmethod
    def from_config(cls, ctx, params: "SignalConfig") -> "ISignalGenerator":
        """
        Создаёт генератор из SignalConfig (OCP-точка расширения).
        Каждый адаптер сам извлекает нужные поля из params.

        Args:
            ctx:    GPU контекст (GPUContext или ROCmGPUContext)
            params: SignalConfig с fs, f0_hz, fdev_hz, amplitude, seed
        """

    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Генерирует сигнал длиной n_samples на GPU.

        Returns:
            np.ndarray: dtype=complex64, shape=(n_samples,)
        """

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Обновляет параметры без пересоздания GPU-объекта.
        Конкретные kwargs зависят от типа генератора.
        """

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Частота дискретизации (Гц)."""

    @property
    @abstractmethod
    def generator_type(self) -> str:
        """Строковый тип: 'cw', 'lfm', 'noise'."""
```

---

### 4. `Core/generators/cw.py` — CwGenerator

```python
import numpy as np
from .base import ISignalGenerator
from common import GPULoader

class CwGenerator(ISignalGenerator):
    """
    GPU CW-генератор (непрерывная синусоида).

    GoF Adapter: оборачивает gpuworklib.SignalGenerator
    в стандартный интерфейс ISignalGenerator.

    Args:
        ctx:        GPU контекст (GPUContext или ROCmGPUContext)
        fs:         Частота дискретизации (Гц)
        f0:         Несущая частота (Гц)
        amplitude:  Амплитуда (default=1.0)
        phase:      Начальная фаза (рад, default=0.0)
    """

    def __init__(self, ctx, fs: float, f0: float,
                 amplitude: float = 1.0, phase: float = 0.0):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._f0 = f0
        self._amplitude = amplitude
        self._phase = phase

    @classmethod
    def from_config(cls, ctx, params) -> "CwGenerator":
        """OCP: Factory вызывает этот метод, не зная про f0/amplitude."""
        return cls(ctx, fs=params.fs, f0=params.f0_hz,
                   amplitude=params.amplitude)

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_cw(
            freq=self._f0,
            fs=self._fs,
            length=n_samples,
            amplitude=self._amplitude,
        )

    def set_params(self, f0: float | None = None,
                   amplitude: float | None = None,
                   phase: float | None = None) -> None:
        if f0 is not None:         self._f0 = f0
        if amplitude is not None:  self._amplitude = amplitude
        if phase is not None:      self._phase = phase

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "cw"
```

---

### 5. `Core/generators/lfm.py` — LfmGenerator

```python
import numpy as np
from .base import ISignalGenerator
from common import GPULoader

class LfmGenerator(ISignalGenerator):
    """
    GPU LFM-генератор (линейная частотная модуляция).

    GoF Adapter: gpuworklib.SignalGenerator.generate_lfm → ISignalGenerator.

    Args:
        ctx:        GPU контекст
        fs:         Частота дискретизации (Гц)
        f_start:    Начальная частота (Гц)
        f_end:      Конечная частота (Гц)
        amplitude:  Амплитуда (default=1.0)
        phase:      Начальная фаза (рад, default=0.0)
    """

    def __init__(self, ctx, fs: float, f_start: float, f_end: float,
                 amplitude: float = 1.0, phase: float = 0.0):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._f_start = f_start
        self._f_end = f_end
        self._amplitude = amplitude
        self._phase = phase

    @classmethod
    def from_config(cls, ctx, params) -> "LfmGenerator":
        """OCP: SignalConfig.f0_hz=f_start, fdev_hz=bandwidth (f_end-f_start)."""
        return cls(ctx, fs=params.fs,
                   f_start=params.f0_hz,
                   f_end=params.f0_hz + params.fdev_hz,
                   amplitude=params.amplitude)

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_lfm(
            f_start=self._f_start,
            f_end=self._f_end,
            fs=self._fs,
            length=n_samples,
            amplitude=self._amplitude,
        )

    def set_params(self, f_start: float | None = None,
                   f_end: float | None = None,
                   amplitude: float | None = None) -> None:
        if f_start is not None:   self._f_start = f_start
        if f_end is not None:     self._f_end = f_end
        if amplitude is not None: self._amplitude = amplitude

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "lfm"

    @property
    def chirp_rate(self) -> float:
        """Скорость изменения частоты (Гц/с)."""
        raise NotImplementedError("нужно знать n_samples для расчёта")
```

---

### 6. `Core/generators/noise.py` — NoiseGenerator

```python
import numpy as np
from .base import ISignalGenerator
from common import GPULoader

class NoiseGenerator(ISignalGenerator):
    """
    GPU Noise-генератор (Гауссов шум, Philox PRNG).

    GoF Adapter: gpuworklib.SignalGenerator.generate_noise → ISignalGenerator.

    Args:
        ctx:        GPU контекст
        fs:         Частота дискретизации (Гц)
        amplitude:  Амплитуда (default=1.0)
        seed:       Начальное значение PRNG (default=42, воспроизводимость)
    """

    def __init__(self, ctx, fs: float,
                 amplitude: float = 1.0, seed: int = 42):
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        self._gen = gw.SignalGenerator(ctx)
        self._fs = fs
        self._amplitude = amplitude
        self._seed = seed

    @classmethod
    def from_config(cls, ctx, params) -> "NoiseGenerator":
        """OCP: seed из SignalConfig.seed (default=42)."""
        return cls(ctx, fs=params.fs, amplitude=params.amplitude,
                   seed=getattr(params, "seed", 42))

    def generate(self, n_samples: int) -> np.ndarray:
        return self._gen.generate_noise(
            length=n_samples,
            amplitude=self._amplitude,
            seed=self._seed,
        )

    def set_params(self, amplitude: float | None = None,
                   seed: int | None = None) -> None:
        if amplitude is not None: self._amplitude = amplitude
        if seed is not None:      self._seed = seed

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def generator_type(self) -> str:
        return "noise"
```

---

### 7. `Core/generators/factory.py` — GeneratorFactory

```python
from __future__ import annotations
from typing import Type, TYPE_CHECKING
import numpy as np

from .base import ISignalGenerator
if TYPE_CHECKING:
    from common import SignalConfig

class GeneratorFactory:
    """
    Создаёт GPU-генераторы по строковому типу.

    GoF Factory Method + Registry: OCP — добавить новый тип =
    GeneratorFactory.register("mytype", MyClass), не менять этот класс.
    GRASP Creator: знает как создать ISignalGenerator.

    OCP гарантия: Factory НЕ содержит if/elif по типам.
    Каждый адаптер реализует from_config(ctx, params) — Factory
    вызывает его единообразно.

    Типы: "cw" | "lfm" | "noise"  (регистрируются в generators/__init__.py)
    """

    _registry: dict[str, Type[ISignalGenerator]] = {}

    @classmethod
    def register(cls, type_name: str, generator_class: Type[ISignalGenerator]) -> None:
        """
        Регистрирует тип генератора.

        Args:
            type_name:       Строка-ключ ("cw", "lfm", "noise", ...)
            generator_class: Класс, реализующий ISignalGenerator
        """
        cls._registry[type_name] = generator_class

    @classmethod
    def create(cls, type_name: str, ctx, params: "SignalConfig") -> ISignalGenerator:
        """
        Создаёт генератор по типу и конфигурации.

        Args:
            type_name: "cw" | "lfm" | "noise"
            ctx:       GPU контекст (GPUContext или ROCmGPUContext)
            params:    SignalConfig с fs, f0_hz, fdev_hz, amplitude, seed

        Returns:
            ISignalGenerator: готов к generate(n_samples)

        Raises:
            ValueError: если type_name не зарегистрирован
        """
        if type_name not in cls._registry:
            available = sorted(cls._registry)
            raise ValueError(
                f"Неизвестный тип генератора: '{type_name}'. "
                f"Доступные: {available}"
            )
        # OCP: вызываем from_config() — Factory не знает про конкретные параметры
        return cls._registry[type_name].from_config(ctx, params)

    @classmethod
    def available(cls) -> list[str]:
        """Список зарегистрированных типов."""
        return sorted(cls._registry)
```

---

## ✅ Критерии завершения

- [ ] Все 7 файлов созданы
- [ ] `from Core.generators import GeneratorFactory` работает
- [ ] `GeneratorFactory.available()` → `['cw', 'lfm', 'noise']`
- [ ] `gen = GeneratorFactory.create("cw", ctx, params)` → `ISignalGenerator`
- [ ] `signal = gen.generate(4096)` → `ndarray, dtype=complex64, shape=(4096,)`
- [ ] `gen.set_params(f0=3e6)` изменяет параметры без пересоздания
- [ ] Добавить пример использования в `Core/generators/__init__.py` (docstring)
- [ ] pytest НЕ используется

## 🧪 Мини-тест после реализации

```python
# python Python_test/Core/generators/test_generators_smoke.py
import numpy as np
from common import GPUContextManager, SkipTest
from common.configs import SignalConfig
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from Core.generators import GeneratorFactory

class TestGeneratorsSmoke:
    """Smoke-тест: все зарегистрированные генераторы создаются и генерируют."""

    def test_all_generators(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")

        result = TestResult(test_name="generators_smoke")
        # LFM нуждается в fdev_hz > 0
        params = SignalConfig(fs=12e6, f0_hz=2e6, fdev_hz=1e6, n_samples=4096)
        for gen_type in GeneratorFactory.available():
            gen = GeneratorFactory.create(gen_type, ctx, params)
            sig = gen.generate(params.n_samples)
            ok_dtype = sig.dtype == np.complex64
            ok_shape = sig.shape == (params.n_samples,)
            result.add(ValidationResult(
                passed=ok_dtype and ok_shape,
                metric_name=f"{gen_type}_shape_dtype",
                actual_value=1.0 if ok_dtype and ok_shape else 0.0,
                threshold=1.0,
                message=f"dtype={sig.dtype}, shape={sig.shape}, max={np.abs(sig).max():.4f}"
            ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestGeneratorsSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 1*
