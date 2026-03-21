# TASK_PythonArch_06 — common/plotting/ (PlotterFactory)

> **Фаза**: 3 (приоритет LOW — полировка)
> **Зависимости**: TASK_Arch_03 (references — для данных), TASK_Arch_05 (io — для путей)
> **Статус**: ⬜ TODO
> **Оценка**: ~1.5 часа
> **Паттерны**: GoF Factory Method, GoF Strategy, GRASP Information Expert

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

### 0. Расширить `plotter_base.py` — IPlotter.save_fig()

> R-14: вместо дублирования save_fig() в каждом плоттере — расширяем базовый.

```python
# В IPlotter — ЗАМЕНИТЬ существующий save_fig:
def save_fig(self, fig, name: str) -> str:
    """Сохранить фигуру, показать (опционально), закрыть."""
    import matplotlib.pyplot as plt
    path = self.config.filepath(name)
    if self.config.save:
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        print(f"[Plotter] Saved: {path}")
    if self.config.show:
        plt.show()
    plt.close(fig)
    return path
```

После этого SpectrumPlotter и TimePlotter **НЕ переопределяют** save_fig() — используют базовый.

---

## 📝 Детальное ТЗ

### 1. `common/plotting/factory.py` — PlotterFactory

```python
"""
PlotterFactory: создаёт плоттеры с правильными путями для модуля.
GoF Factory Method. GRASP Information Expert: знает пути Results/Plots/.
"""

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
    """

    BASE_PLOTS_DIR = "Results/Plots"

    def __init__(self, module_name: str,
                 dpi: int = 120,
                 style: str = "dark_background",
                 show: bool = False,
                 save: bool = True):
        self._module = module_name
        self._config = PlotConfig(
            out_dir=str(Path(self.BASE_PLOTS_DIR) / module_name),
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
        config = self._with_subdir(subdir)
        return SpectrumPlotter(config)

    def timeseries(self, subdir: str = "") -> "TimePlotter":
        """Создаёт TimePlotter для модуля."""
        config = self._with_subdir(subdir)
        return TimePlotter(config)

    def config(self) -> PlotConfig:
        """Возвращает текущую PlotConfig (для расширенной настройки)."""
        return self._config

    def _with_subdir(self, subdir: str) -> PlotConfig:
        if not subdir:
            return self._config
        # Явное создание нового PlotConfig (без скрытого dataclasses.replace)
        return PlotConfig(
            out_dir=str(Path(self._config.out_dir) / subdir),
            dpi=self._config.dpi,
            style=self._config.style,
            figsize=self._config.figsize,
            show=self._config.show,
            save=self._config.save,
            fmt=self._config.fmt,
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
from pathlib import Path
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
        self._cfg = config

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
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0/fs)) / 1e6  # МГц

        fig, ax = plt.subplots(figsize=self._cfg.figsize)
        if self._cfg.style:
            plt.style.use(self._cfg.style)

        y = 20 * np.log10(spec + 1e-12) if db_scale else spec
        ax.plot(freqs, y, linewidth=0.8)
        ax.set_xlabel("Частота (МГц)")
        ax.set_ylabel("Амплитуда (дБ)" if db_scale else "Амплитуда")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self.save_fig(fig, f"spectrum_{title.replace(' ', '_')}")

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

        fig, axes = plt.subplots(2, 1, figsize=(self._cfg.figsize[0],
                                                self._cfg.figsize[1] * 1.2))
        if self._cfg.style:
            plt.style.use(self._cfg.style)

        for ax, sig, label in zip(axes, [gpu_signal, ref_signal], labels):
            spec = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(sig), n=n_fft)))
            n = len(spec)
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0/fs)) / 1e6
            y = 20 * np.log10(spec + 1e-12)
            ax.plot(freqs, y, linewidth=0.8)
            ax.set_title(f"{label}")
            ax.set_xlabel("Частота (МГц)")
            ax.set_ylabel("дБ")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"compare_{title.replace(' ', '_')}")

    # save_fig() наследуется от IPlotter — НЕ дублируем
```

---

### 3. `common/plotting/time_plotter.py` — TimePlotter

```python
"""
TimePlotter: строит временной ряд сигнала.
GoF Strategy: реализует IPlotter для временных данных.
"""

import numpy as np
from pathlib import Path
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
        self._cfg = config

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

        fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=self._cfg.figsize,
                                          sharex=True)
        if self._cfg.style:
            plt.style.use(self._cfg.style)

        ax_i.plot(t_us, sig.real, linewidth=0.8, color="cyan")
        ax_i.set_ylabel("I (real)")
        ax_i.grid(True, alpha=0.3)

        ax_q.plot(t_us, sig.imag, linewidth=0.8, color="orange")
        ax_q.set_ylabel("Q (imag)")
        ax_q.set_xlabel("Время (мкс)")
        ax_q.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"time_{title.replace(' ', '_')}")

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

        fig, ax = plt.subplots(figsize=self._cfg.figsize)
        ax.plot(t_us, np.abs(sig), linewidth=0.8)
        ax.set_xlabel("Время (мкс)")
        ax.set_ylabel("|A|")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.save_fig(fig, f"mag_{title.replace(' ', '_')}")

    # save_fig() наследуется от IPlotter — НЕ дублируем
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

- [ ] Все 3 файла созданы
- [ ] `from common.plotting import PlotterFactory` работает
- [ ] `PlotterFactory("signal_generators").spectrum()` → SpectrumPlotter с правильным путём
- [ ] `SpectrumPlotter(config).plot(signal, 12e6)` → сохраняет PNG
- [ ] `TimePlotter(config).plot(signal, 12e6)` → сохраняет PNG
- [ ] `plot_compare(gpu, ref, 12e6)` → PNG с двумя спектрами
- [ ] matplotlib импортируется ТОЛЬКО внутри методов (lazy import)
- [ ] Пути правильные: `Results/Plots/{module}/spectrum_*.png`

## 🧪 Проверка

```python
# python Python_test/common/plotting/test_plotting_smoke.py
import numpy as np
import tempfile
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from common.references import SignalReferences
from common.plotting import PlotterFactory

class TestPlottingSmoke:
    """Smoke-тест плоттеров — без GPU, с временной директорией."""

    def test_spectrum_plotter(self):
        result = TestResult(test_name="spectrum_plotter")
        cw = SignalReferences.cw(12e6, 4096, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory("test_module", save=True, show=False)
            factory._config.out_dir = tmp  # временная директория
            p1 = factory.spectrum().plot(cw, 12e6, title="CW_2MHz")
            from pathlib import Path
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
        lfm = SignalReferences.lfm(12e6, 4096, 0.0, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory("test_module", save=True, show=False)
            factory._config.out_dir = tmp
            p2 = factory.timeseries().plot(lfm, 12e6, title="LFM")
            from pathlib import Path
            ok = Path(p2).exists()
            result.add(ValidationResult(
                passed=ok,
                metric_name="time_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={p2}"
            ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestPlottingSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 3*
