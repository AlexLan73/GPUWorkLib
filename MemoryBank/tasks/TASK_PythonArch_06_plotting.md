# TASK_PythonArch_06 — common/plotting/ (PlotterFactory)

> **Фаза**: 3 (приоритет LOW — полировка)
> **Зависимости**: TASK_Arch_03 (references — для данных), TASK_Arch_05 (io — для путей)
> **Статус**: ✅ DONE 2026-04-09
> **Оценка**: ~1.5 часа
> **Паттерны**: GoF Factory Method, GoF Strategy, GRASP Information Expert

## ✅ Итог реализации (2026-04-09)

- `plotter_base.py` расширен: `_slugify()` + `save_fig()` с lazy matplotlib
  import, показом и закрытием figure.
- 3 новых файла: `spectrum_plotter.py`, `time_plotter.py`, `factory.py`.
- **SpectrumPlotter / TimePlotter вызывают `super().__init__(config)`** —
  `self.config` доступен в унаследованном `save_fig()`.
- **`plt.style.use()` вызывается ДО `plt.subplots()`** — стиль применяется.
- `PlotterFactory(out_dir=...)` принимает override базового пути; в тестах
  больше НЕ нужно мутировать приватный `_config`.
- `_with_subdir()` использует `dataclasses.replace` — устойчиво к добавлению
  новых полей в `PlotConfig`.
- Slugify имён файлов: заголовки со спецсимволами (`:`, `/`, пробелы)
  корректно превращаются в `spectrum_CW_2MHz_f0_2e6_test.png`.
- `PlotterFactory.config` — `@property` (read-only).
- Smoke-тесты: `Python_test/common/plotting/test_smoke.py` — **6/6 PASS**
  (matplotlib 3.10.7 верифицирован).

---

## 🎯 Цель

Расширить `Python_test/common/plotting/` — добавить Factory и специализированные плоттеры.

**Проблема сейчас**:
- Каждый тест сам настраивает `PlotConfig(out_dir=...)` — дублирование путей
- Нет стандартных плоттеров для спектра, временного ряда
- Нет единой точки создания плоттеров для конкретного модуля

**Решение**: `PlotterFactory(module_name)` → плоттер с правильным путём автоматически.

---

## 📁 Создаваемые/изменяемые файлы

```
Python_test/common/plotting/
├── plotter_base.py      ← [ИЗМЕНИТЬ] IPlotter.save_fig — добавить show/close
├── factory.py           ← 1. PlotterFactory(module_name)  [НОВОЕ]
├── spectrum_plotter.py  ← 2. SpectrumPlotter              [НОВОЕ]
└── time_plotter.py      ← 3. TimePlotter                  [НОВОЕ]
```

### 0. Расширить `plotter_base.py` — IPlotter.save_fig() + slugify

> R-14: вместо дублирования save_fig() в каждом плоттере — расширяем базовый.

```python
# В IPlotter — ЗАМЕНИТЬ существующий save_fig:
import re

def _slugify(name: str) -> str:
    """Безопасное имя файла: убирает пробелы, слэши, двоеточия и пр.

    Нужно для Windows — запрещены: < > : " / \\ | ? *
    """
    s = re.sub(r"[^\w\-.]+", "_", name)
    return s.strip("_") or "unnamed"


class IPlotter(ABC):
    # ... (без изменений)

    def save_fig(self, fig, name: str) -> str:
        """Сохранить фигуру, показать (опционально), закрыть.

        Имя автоматически slugify-тся — в title можно писать любые символы.
        """
        import matplotlib.pyplot as plt
        safe = _slugify(name)
        path = self.config.filepath(safe)
        if self.config.save:
            fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"[Plotter] Saved: {path}")
        if self.config.show:
            plt.show()
        plt.close(fig)
        return path
```

После этого SpectrumPlotter и TimePlotter **НЕ переопределяют** save_fig() —
используют базовый. Дополнительно: они **ОБЯЗАНЫ** вызывать `super().__init__(config)`,
чтобы установился `self.config` (он используется в `save_fig`).

---

## 📝 Детальное ТЗ

### 1. `common/plotting/factory.py` — PlotterFactory

```python
"""
PlotterFactory: создаёт плоттеры с правильными путями для модуля.
GoF Factory Method. GRASP Information Expert: знает пути Results/Plots/.
"""

from dataclasses import replace
from pathlib import Path
from .plotter_base import PlotConfig, IPlotter
from .spectrum_plotter import SpectrumPlotter
from .time_plotter import TimePlotter

class PlotterFactory:
    """
    Фабрика плоттеров для конкретного модуля.

    Использование:
        factory = PlotterFactory("signal_generators")
        sp = factory.spectrum()   # SpectrumPlotter → Results/Plots/signal_generators/
        tp = factory.timeseries() # TimePlotter     → Results/Plots/signal_generators/

        sp.plot(signal, fs=12e6, title="CW 2MHz")

    Для тестов — можно подменить out_dir через параметр конструктора
    (без мутации приватного _config):
        factory = PlotterFactory("test_module", out_dir=tmp_dir, save=True)
    """

    BASE_PLOTS_DIR = "Results/Plots"

    def __init__(self, module_name: str,
                 out_dir: str | Path | None = None,
                 dpi: int = 120,
                 style: str = "dark_background",
                 show: bool = False,
                 save: bool = True):
        """
        Args:
            module_name: имя модуля (signal_generators, heterodyne, ...).
            out_dir:     override базового пути. Если None — берётся
                         `Results/Plots/{module_name}`.
        """
        self._module = module_name
        if out_dir is None:
            resolved = Path(self.BASE_PLOTS_DIR) / module_name
        else:
            resolved = Path(out_dir)
        self._config = PlotConfig(
            out_dir=str(resolved),
            dpi=dpi,
            style=style,
            show=show,
            save=save,
        )

    def spectrum(self, subdir: str = "") -> "SpectrumPlotter":
        """
        Создаёт SpectrumPlotter для модуля.

        Args:
            subdir: подпапка внутри модуля (например "FormSignal")
        """
        return SpectrumPlotter(self._with_subdir(subdir))

    def timeseries(self, subdir: str = "") -> "TimePlotter":
        """Создаёт TimePlotter для модуля."""
        return TimePlotter(self._with_subdir(subdir))

    @property
    def config(self) -> PlotConfig:
        """Возвращает текущую PlotConfig (read-only)."""
        return self._config

    def _with_subdir(self, subdir: str) -> PlotConfig:
        """Создаёт копию PlotConfig с новой подпапкой.

        Используется `dataclasses.replace` — устойчиво к добавлению
        новых полей в PlotConfig.
        """
        if not subdir:
            # Возвращаем копию, чтобы плоттеры не делили один инстанс
            return replace(self._config)
        return replace(
            self._config,
            out_dir=str(Path(self._config.out_dir) / subdir),
        )
```

---

### 2. `common/plotting/spectrum_plotter.py` — SpectrumPlotter

```python
"""
SpectrumPlotter: строит амплитудный спектр сигнала.
GoF Strategy: реализует IPlotter для спектральных данных.
"""

import numpy as np
from .plotter_base import IPlotter, PlotConfig


class SpectrumPlotter(IPlotter):
    """
    Строит амплитудный спектр (|FFT|) сигнала.

    Использование:
        from common.plotting import PlotterFactory

        factory = PlotterFactory("signal_generators")
        plotter = factory.spectrum()
        path = plotter.plot(cw_signal, fs=12e6, title="CW 2MHz")
        # → Results/Plots/signal_generators/spectrum_CW_2MHz.png

    Или напрямую (два сигнала для сравнения):
        path = plotter.plot_compare(
            gpu_signal, ref_signal, fs=12e6,
            labels=("GPU", "NumPy"), title="GPU vs NumPy"
        )
    """

    def __init__(self, config: PlotConfig):
        # КРИТИЧНО: super().__init__ ставит self.config, который нужен save_fig()
        super().__init__(config)

    def plot(self, signal: np.ndarray, fs: float,
             title: str = "Spectrum",
             n_fft: int | None = None,
             db_scale: bool = True) -> str:
        """
        Строит и сохраняет спектр.

        Args:
            signal:   1D или 2D (channels × samples) complex64
            fs:       Частота дискретизации (Гц)
            title:    Заголовок и имя файла
            n_fft:    Размер FFT (None = len(signal))
            db_scale: True → ось Y в дБ

        Returns:
            str: путь к сохранённому файлу
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            # Берём первый канал для простого просмотра
            sig = sig[0]

        spec = np.abs(np.fft.fftshift(np.fft.fft(sig, n=n_fft)))
        n = len(spec)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) / 1e6  # МГц

        # СТИЛЬ ДО subplots — иначе не применяется к созданной figure
        if self.config.style:
            plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figsize)

        y = 20 * np.log10(spec + 1e-12) if db_scale else spec
        ax.plot(freqs, y, linewidth=0.8)
        ax.set_xlabel("Частота (МГц)")
        ax.set_ylabel("Амплитуда (дБ)" if db_scale else "Амплитуда")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # save_fig сам делает slugify имени — title может содержать любые символы
        return self.save_fig(fig, f"spectrum_{title}")

    def plot_compare(self, gpu_signal: np.ndarray,
                     ref_signal: np.ndarray,
                     fs: float,
                     labels: tuple = ("GPU", "NumPy"),
                     title: str = "GPU vs Reference",
                     n_fft: int | None = None) -> str:
        """
        Сравнивает два спектра на одном графике.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # СТИЛЬ ДО subplots
        if self.config.style:
            plt.style.use(self.config.style)
        fig, axes = plt.subplots(
            2, 1,
            figsize=(self.config.figsize[0], self.config.figsize[1] * 1.2),
        )

        for ax, sig, label in zip(axes, [gpu_signal, ref_signal], labels):
            spec = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(sig), n=n_fft)))
            n = len(spec)
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) / 1e6
            y = 20 * np.log10(spec + 1e-12)
            ax.plot(freqs, y, linewidth=0.8)
            ax.set_title(f"{label}")
            ax.set_xlabel("Частота (МГц)")
            ax.set_ylabel("дБ")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"compare_{title}")

    # save_fig() наследуется от IPlotter — НЕ дублируем (в базе уже есть slugify)
```

---

### 3. `common/plotting/time_plotter.py` — TimePlotter

```python
"""
TimePlotter: строит временной ряд сигнала.
GoF Strategy: реализует IPlotter для временных данных.
"""

import numpy as np
from .plotter_base import IPlotter, PlotConfig


class TimePlotter(IPlotter):
    """
    Строит I/Q компоненты сигнала во времени.

    Использование:
        from common.plotting import PlotterFactory

        factory = PlotterFactory("heterodyne")
        plotter = factory.timeseries()
        path = plotter.plot(dechirped, fs=12e6, title="Dechirp Channel 0")
    """

    def __init__(self, config: PlotConfig):
        # КРИТИЧНО: super().__init__ ставит self.config, который нужен save_fig()
        super().__init__(config)

    def plot(self, signal: np.ndarray, fs: float,
             title: str = "Signal",
             channel: int = 0,
             max_samples: int = 2048) -> str:
        """
        Строит I (real) и Q (imag) компоненты.

        Args:
            signal:      1D или 2D complex64
            fs:          Частота дискретизации (Гц)
            title:       Заголовок и имя файла
            channel:     Канал для 2D сигнала
            max_samples: Максимум точек (для быстрой отрисовки)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            sig = sig[channel]
        sig = sig[:max_samples]
        n = len(sig)
        t_us = np.arange(n) / fs * 1e6  # мкс

        # СТИЛЬ ДО subplots
        if self.config.style:
            plt.style.use(self.config.style)
        fig, (ax_i, ax_q) = plt.subplots(
            2, 1, figsize=self.config.figsize, sharex=True,
        )

        ax_i.plot(t_us, sig.real, linewidth=0.8, color="cyan")
        ax_i.set_ylabel("I (real)")
        ax_i.grid(True, alpha=0.3)

        ax_q.plot(t_us, sig.imag, linewidth=0.8, color="orange")
        ax_q.set_ylabel("Q (imag)")
        ax_q.set_xlabel("Время (мкс)")
        ax_q.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"time_{title}")

    def plot_magnitude(self, signal: np.ndarray, fs: float,
                       title: str = "Magnitude") -> str:
        """Строит |signal| во времени."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            sig = sig[0]
        n = len(sig)
        t_us = np.arange(n) / fs * 1e6

        # СТИЛЬ ДО subplots
        if self.config.style:
            plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figsize)

        ax.plot(t_us, np.abs(sig), linewidth=0.8)
        ax.set_xlabel("Время (мкс)")
        ax.set_ylabel("|A|")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.save_fig(fig, f"mag_{title}")

    # save_fig() наследуется от IPlotter — НЕ дублируем (в базе уже есть slugify)
```

---

## 📋 Обновить `common/plotting/__init__.py`:

```python
from .plotter_base import IPlotter, PlotConfig
from .spectrum_plotter import SpectrumPlotter
from .time_plotter import TimePlotter
from .factory import PlotterFactory

__all__ = [
    "IPlotter", "PlotConfig",
    "SpectrumPlotter", "TimePlotter",
    "PlotterFactory",
]
```

---

## ✅ Критерии завершения

- [ ] Шаг 0 выполнен: `IPlotter.save_fig()` расширен + `_slugify()` добавлен
- [ ] Все 3 файла созданы
- [ ] `from common.plotting import PlotterFactory` работает
- [ ] `SpectrumPlotter`/`TimePlotter` вызывают `super().__init__(config)` → `self.config` доступен
- [ ] `plt.style.use()` вызывается ДО `plt.subplots()` (стиль применяется)
- [ ] `PlotterFactory("signal_generators").spectrum()` → SpectrumPlotter с правильным путём
- [ ] `PlotterFactory("test", out_dir=tmp).spectrum()` → пишет в `tmp` (без мутации приватов)
- [ ] `_with_subdir()` использует `dataclasses.replace` (устойчив к новым полям)
- [ ] `SpectrumPlotter(config).plot(signal, 12e6)` → сохраняет PNG
- [ ] `TimePlotter(config).plot(signal, 12e6)` → сохраняет PNG
- [ ] `plot_compare(gpu, ref, 12e6)` → PNG с двумя спектрами
- [ ] Title со спецсимволами (`: / *`) → корректно slug-ится, файл создаётся
- [ ] matplotlib импортируется ТОЛЬКО внутри методов (lazy import)
- [ ] Пути правильные: `Results/Plots/{module}/spectrum_*.png`

## 🧪 Проверка

```python
# python Python_test/common/plotting/test_plotting_smoke.py
import sys
from pathlib import Path
# Bootstrap: Python_test/ в sys.path (файл живёт внутри common/plotting/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tempfile
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from common.plotting import PlotterFactory

class TestPlottingSmoke:
    """Smoke-тест плоттеров — без GPU, с временной директорией."""

    @staticmethod
    def _cw(fs: float, n: int, f: float) -> np.ndarray:
        t = np.arange(n) / fs
        return np.exp(2j * np.pi * f * t).astype(np.complex64)

    @staticmethod
    def _lfm(fs: float, n: int, f0: float, f1: float) -> np.ndarray:
        t = np.arange(n) / fs
        k = (f1 - f0) / (n / fs)
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
        return np.exp(1j * phase).astype(np.complex64)

    def test_spectrum_plotter(self):
        result = TestResult(test_name="spectrum_plotter")
        cw = self._cw(12e6, 4096, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            # out_dir передаётся в конструктор — без мутации приватных атрибутов!
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            # Заголовок со спецсимволами — проверяем slugify
            p1 = factory.spectrum().plot(cw, 12e6, title="CW 2MHz: f0=2e6")
            ok = Path(p1).exists()
            result.add(ValidationResult(
                passed=ok,
                metric_name="spectrum_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={p1}"
            ))
        return result

    def test_timeseries_plotter(self):
        result = TestResult(test_name="timeseries_plotter")
        lfm = self._lfm(12e6, 4096, 0.0, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            p2 = factory.timeseries().plot(lfm, 12e6, title="LFM")
            ok = Path(p2).exists()
            result.add(ValidationResult(
                passed=ok,
                metric_name="time_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={p2}"
            ))
        return result

    def test_subdir_creates_new_config(self):
        """factory.spectrum('FormSignal') должен писать в подпапку."""
        result = TestResult(test_name="subdir")
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            cw = self._cw(12e6, 1024, 1e6)
            p = factory.spectrum(subdir="FormSignal").plot(cw, 12e6, title="t")
            ok = "FormSignal" in p and Path(p).exists()
            result.add(ValidationResult(
                passed=ok,
                metric_name="subdir_ok",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={p}",
            ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestPlottingSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 3*
