# TASK_PythonArch_02 — Core/processing/

> **Фаза**: 1 (приоритет HIGH)
> **Зависимости**: TASK_Arch_01 (нужен `Core/__init__.py`, можно создать заранее)
> **Статус**: ⬜ TODO
> **Оценка**: ~2 часа
> **Паттерны**: GoF Adapter, GRASP Creator, SOLID ISP

---

## 🎯 Цель

Создать `Python_test/Core/processing/` — переиспользуемые GPU-процессоры.

**Проблема сейчас**: Каждый тест вручную создаёт `gw.StatisticsProcessor(ctx)`,
`gw.HeterodyneDechirp(ctx, fs, f_start, ...)` с разным набором аргументов.

**Решение**: Adapter-классы со стандартным интерфейсом `IProcessor`:
```python
proc = StatisticsAdapter(ctx)
result = proc.process(data)  # ndarray → dict
```

---

## 📁 Создаваемые файлы (6 штук)

```
Python_test/Core/processing/
├── __init__.py          ← 1. пакет + export
├── base.py              ← 2. IProcessor (ABC)
├── statistics.py        ← 3. StatisticsAdapter
├── heterodyne.py        ← 4. HeterodyneAdapter
└── fft.py               ← 5. FftAdapter
```
(+обновить `Core/__init__.py` — добавить импорт processing)

---

## 📝 Детальное ТЗ

### 2. `Core/processing/base.py` — GpuProcessorMixin

```python
"""
Базовый миксин для GPU-процессоров.

Почему НЕ ABC с process() -> ndarray|dict:
  Statistics возвращает dict, FFT/Heterodyne — ndarray.
  Union return type нарушает LSP (код вроде np.abs(result) падает для dict).
  Каждый адаптер имеет СВОЙ типизированный process().
  Миксин даёт общий __repr__ и проверку GPULoader — без навязывания сигнатуры.

Масштабирование:
  Новые адаптеры (FilterAdapter, CorrelatorAdapter) наследуют миксин
  и определяют свой process() с нужным return type.
"""

from common import GPULoader

class GpuProcessorMixin:
    """Миксин для GPU-адаптеров: загрузка gpuworklib + repr."""

    _processor_name: str = "GpuProcessor"

    @staticmethod
    def _load_gw():
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        return gw

    @property
    def name(self) -> str:
        return self._processor_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
```

---

### 3. `Core/processing/statistics.py` — StatisticsAdapter

```python
import numpy as np
from .base import GpuProcessorMixin

class StatisticsAdapter(GpuProcessorMixin):
    """
    GPU Статистика: mean, std, median.

    GoF Adapter: gpuworklib.StatisticsProcessor → удобный Python-интерфейс.

    Входные данные:
        np.ndarray shape=(n_channels, n_samples), dtype=complex64 или float32

    Результат process():
        dict с ключами: "mean", "std", "median"
        Каждый ключ → np.ndarray shape=(n_channels,)

    Args:
        ctx: GPU контекст (ROCmGPUContext)
    """

    _processor_name = "StatisticsAdapter"

    def __init__(self, ctx):
        gw = self._load_gw()
        self._proc = gw.StatisticsProcessor(ctx)

    def process(self, data: np.ndarray) -> dict:
        """
        Вычисляет mean, std, median по каналам.

        Returns:
            {"mean": np.ndarray, "std": np.ndarray, "median": np.ndarray}
            Каждый массив shape=(n_channels,)
        """
        raw = self._proc.compute_statistics(data)
        # raw — List[(mean, std, median)] по каналам
        n = len(raw)
        means   = np.array([raw[i][0] for i in range(n)], dtype=np.float32)
        stds    = np.array([raw[i][1] for i in range(n)], dtype=np.float32)
        medians = np.array([raw[i][2] for i in range(n)], dtype=np.float32)
        return {"mean": means, "std": stds, "median": medians}

    def compute_mean_std(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Упрощённый вызов — только mean и std (без median)."""
        result = self.process(data)
        return result["mean"], result["std"]
```

---

### 3b. `common/configs.py` — добавить HeterodyneConfig (расширение SignalConfig)

```python
# Добавить в common/configs.py ПОСЛЕ SignalConfig:

@dataclass
class HeterodyneConfig(SignalConfig):
    """SignalConfig + вычисляемые LFM/dechirp свойства.

    Маппинг полей:
        f0_hz   = f_start (начальная частота ЛЧМ)
        fdev_hz = bandwidth (f_end - f_start)
        f_end   = f0_hz + fdev_hz  (вычисляемое)

    Использование:
        cfg = HeterodyneConfig(fs=12e6, f0_hz=0.0, fdev_hz=2e6,
                               n_samples=8000, n_antennas=5)
        print(cfg.chirp_rate)             # → 3e12
        print(cfg.fbeat_from_delay(1e-4)) # → 300e3
    """
    n_antennas: int = 5
    c_light: float = 3e8

    @property
    def f_start(self) -> float:
        return self.f0_hz

    @property
    def f_end(self) -> float:
        return self.f0_hz + self.fdev_hz

    @property
    def bandwidth(self) -> float:
        return self.fdev_hz

    @property
    def chirp_rate(self) -> float:
        """Скорость изменения частоты (Гц/с)."""
        return self.fdev_hz / self.duration_s()

    def range_from_delay(self, delay_s: float) -> float:
        """Дальность из задержки (м)."""
        return self.c_light * delay_s / 2.0

    def fbeat_from_delay(self, delay_s: float) -> float:
        """Частота биения из задержки (Гц)."""
        return self.chirp_rate * delay_s
```

Также добавить в `common/configs.py` экспорт и в `common/__init__.py`:
```python
from .configs import SignalConfig, FilterConfig, HeterodyneConfig
```

---

### 4. `Core/processing/heterodyne.py` — HeterodyneAdapter

```python
import numpy as np
from .base import GpuProcessorMixin
from common.configs import HeterodyneConfig

class HeterodyneAdapter(GpuProcessorMixin):
    """
    GPU Гетеродин/Дечирп для ЛЧМ.

    GoF Adapter: gpuworklib.HeterodyneDechirp → удобный Python-интерфейс.

    Входные данные:
        np.ndarray shape=(n_antennas, n_samples), dtype=complex64

    Результат:
        np.ndarray shape=(n_antennas, n_samples), dtype=complex64
        Дечирпованный сигнал: s_dc = s_rx * conj(s_ref)

    Args:
        ctx:     GPU контекст
        params:  HeterodyneConfig (расширяет SignalConfig + chirp_rate, n_antennas)
    """

    _processor_name = "HeterodyneAdapter"

    def __init__(self, ctx, params: HeterodyneConfig):
        gw = self._load_gw()
        self._proc = gw.HeterodyneDechirp(
            ctx, params.fs, params.f_start, params.f_end,
            params.n_samples, params.n_antennas
        )
        self._params = params

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Выполняет дечирп на GPU.

        Args:
            data: shape=(n_antennas, n_samples), dtype=complex64

        Returns:
            np.ndarray: shape=(n_antennas, n_samples), dtype=complex64
        """
        return self._proc.dechirp(data)

    @property
    def n_antennas(self) -> int:
        return self._params.n_antennas

    @property
    def params(self) -> HeterodyneConfig:
        """Параметры дечирпа (chirp_rate, fbeat_from_delay и т.д.)."""
        return self._params
```

---

### 5. `Core/processing/fft.py` — FftAdapter

```python
import numpy as np
from .base import GpuProcessorMixin

class FftAdapter(GpuProcessorMixin):
    """
    GPU FFT-процессор.

    GoF Adapter: gpuworklib.FFTProcessorROCm → удобный Python-интерфейс.

    ⚠️ Ветка main — только ROCm (FFTProcessorROCm).
    OpenCL/clFFT вынесен в legacy/opencl-clfft.

    Входные данные:
        np.ndarray shape=(n_samples,) или (n_channels, n_samples), dtype=complex64

    Результат (зависит от mode):
        mode="complex"  → complex64 ndarray (FFT bins)
        mode="magnitude"→ float32 ndarray (|FFT|)

    Args:
        ctx:     GPU контекст (ROCmGPUContext)
        n_fft:   Размер FFT (степень 2, например 4096)
        mode:    "complex" | "magnitude" (default="complex")
    """

    def __init__(self, ctx, n_fft: int, mode: str = "complex"):
        gw = self._load_gw()
        if mode not in ("complex", "magnitude"):
            raise ValueError(f"mode должен быть 'complex' или 'magnitude', не '{mode}'")
        self._proc = gw.FFTProcessorROCm(ctx, n_fft)
        self._n_fft = n_fft
        self._mode = mode
        self._processor_name = f"FftAdapter(n_fft={n_fft}, mode={mode})"

    def process(self, data: np.ndarray) -> np.ndarray:
        """Выполняет FFT на GPU. Возвращает complex64 или float32."""
        if self._mode == "complex":
            return self._proc.process_complex(data)
        else:
            return self._proc.process_magnitude(data)

    @property
    def n_fft(self) -> int:
        return self._n_fft
```

---

### 1. `Core/processing/__init__.py`

```python
"""GPU-процессоры как переиспользуемые Adapter'ы."""

from .base import GpuProcessorMixin
from .statistics import StatisticsAdapter
from .heterodyne import HeterodyneAdapter
from .fft import FftAdapter

__all__ = [
    "GpuProcessorMixin",
    "StatisticsAdapter",
    "HeterodyneAdapter",
    "FftAdapter",
]
```

---

### Обновить `Core/__init__.py`

```python
"""
Core — репозиторий готовых GPU-объектов GPUWorkLib.

Протестированные адаптеры над gpuworklib.

Использование:
    from Core.generators import GeneratorFactory
    from Core.processing import StatisticsAdapter, FftAdapter

    gen = GeneratorFactory.create("cw", ctx, params)
    signal = gen.generate(n_samples)

    stats = StatisticsAdapter(ctx)
    result = stats.process(signal.reshape(1, -1))
"""
from . import generators
from . import processing
```

---

## ✅ Критерии завершения

- [ ] Все файлы созданы (base.py + 3 адаптера + __init__.py)
- [ ] `HeterodyneConfig` добавлен в `common/configs.py`
- [ ] `from Core.processing import StatisticsAdapter, HeterodyneAdapter, FftAdapter`
- [ ] `StatisticsAdapter(ctx).process(data)` → dict с "mean", "std", "median"
- [ ] `HeterodyneAdapter(ctx, params).process(data)` → ndarray complex64 (params=HeterodyneConfig)
- [ ] `FftAdapter(ctx, 4096).process(data)` → ndarray complex64
- [ ] `FftAdapter(ctx, 4096, mode="magnitude").process(data)` → ndarray float32

## 🧪 Мини-тест после реализации

```python
# Запуск: python Python_test/Core/processing/test_processing_smoke.py
import numpy as np
from common import GPUContextManager, SkipTest
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from Core.processing import StatisticsAdapter

class TestProcessingSmoke:
    def test_statistics_adapter(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")

        result = TestResult(test_name="statistics_adapter_smoke")
        adapter = StatisticsAdapter(ctx)
        data = np.random.randn(4, 1024).astype(np.float32)
        stats = adapter.process(data)

        has_keys = all(k in stats for k in ("mean", "std", "median"))
        ok_shape = stats["mean"].shape == (4,)
        result.add(ValidationResult(
            passed=has_keys and ok_shape,
            metric_name="stats_structure",
            actual_value=1.0 if has_keys and ok_shape else 0.0,
            threshold=1.0,
            message=f"keys={list(stats.keys())}, mean_shape={stats['mean'].shape}"
        ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestProcessingSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 1*
