"""
time_plotter.py — TimePlotter (GoF Strategy)
==============================================

Строит временной ряд: I (real), Q (imag), |A| (magnitude).

    from common.plotting import PlotterFactory
    factory = PlotterFactory("heterodyne")
    plotter = factory.timeseries()
    plotter.plot(dechirped, fs=12e6, title="Dechirp Channel 0")
"""

from __future__ import annotations

import numpy as np

from .plotter_base import IPlotter, PlotConfig


class TimePlotter(IPlotter):
    """Плоттер временных рядов (I/Q + magnitude)."""

    def __init__(self, config: PlotConfig):
        # КРИТИЧНО: super().__init__ ставит self.config, нужный save_fig()
        super().__init__(config)

    def plot(self, *args,
             title: str = "Signal",
             channel: int = 0,
             max_samples: int = 2048,
             **kwargs) -> str:
        """Построить I (real) и Q (imag) компоненты в субплотах.

        Args:
            signal (args[0]):  1D или 2D complex/real массив.
            fs     (args[1] or kwargs["fs"]): частота дискретизации (Гц).
            title:       заголовок + основа имени файла.
            channel:     канал для 2D-сигнала.
            max_samples: лимит точек (для скорости отрисовки).
        """
        if len(args) < 1:
            raise TypeError("TimePlotter.plot: signal обязателен")
        signal = args[0]
        fs = args[1] if len(args) > 1 else kwargs.get("fs")
        if fs is None:
            raise TypeError("TimePlotter.plot: fs обязателен")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            sig = sig[channel]
        sig = sig[:max_samples]
        n = len(sig)
        t_us = np.arange(n) / float(fs) * 1e6  # мкс

        # КРИТИЧНО: стиль ДО plt.subplots
        if self.config.style:
            plt.style.use(self.config.style)
        fig, (ax_i, ax_q) = plt.subplots(
            2, 1, figsize=self.config.figsize, sharex=True,
        )

        # Для real-сигнала .imag == 0 — всё равно покажем оба канала
        real = np.real(sig)
        imag = np.imag(sig) if np.iscomplexobj(sig) else np.zeros_like(real)

        ax_i.plot(t_us, real, linewidth=0.8, color="cyan")
        ax_i.set_ylabel("I (real)")
        ax_i.grid(True, alpha=0.3)

        ax_q.plot(t_us, imag, linewidth=0.8, color="orange")
        ax_q.set_ylabel("Q (imag)")
        ax_q.set_xlabel("Время (мкс)")
        ax_q.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"time_{title}")

    def plot_magnitude(self, *args,
                       title: str = "Magnitude",
                       channel: int = 0,
                       max_samples: int = 2048,
                       **kwargs) -> str:
        """Построить |signal| во времени.

        Сигнатура согласована с ``IPlotter.plot`` / ``TimePlotter.plot``:

            plotter.plot_magnitude(signal, fs, title="…")
            plotter.plot_magnitude(signal, fs=12e6, title="…")

        Args:
            signal (args[0]):  1D или 2D complex/real массив.
            fs     (args[1] or kwargs["fs"]): частота дискретизации (Гц).
            title:       заголовок + основа имени файла.
            channel:     канал для 2D-сигнала.
            max_samples: лимит точек (для скорости отрисовки).
        """
        if len(args) < 1:
            raise TypeError("TimePlotter.plot_magnitude: signal обязателен")
        signal = args[0]
        fs = args[1] if len(args) > 1 else kwargs.get("fs")
        if fs is None:
            raise TypeError("TimePlotter.plot_magnitude: fs обязателен")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            sig = sig[channel]
        sig = sig[:max_samples]
        n = len(sig)
        t_us = np.arange(n) / float(fs) * 1e6

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
