"""
spectrum_plotter.py — SpectrumPlotter (GoF Strategy)
======================================================

Строит амплитудный спектр |FFT| сигнала. Используется для валидации
генераторов, фильтров, гетеродина.

    from common.plotting import PlotterFactory
    factory = PlotterFactory("signal_generators")
    plotter = factory.spectrum()
    plotter.plot(cw_signal, fs=12e6, title="CW 2MHz")
    # → Results/Plots/signal_generators/spectrum_CW_2MHz.png
"""

from __future__ import annotations

import numpy as np

from .plotter_base import IPlotter, PlotConfig


class SpectrumPlotter(IPlotter):
    """Плоттер амплитудных спектров."""

    def __init__(self, config: PlotConfig):
        # КРИТИЧНО: super().__init__ ставит self.config, нужный save_fig()
        super().__init__(config)

    def plot(self, *args,
             title: str = "Spectrum",
             n_fft: int | None = None,
             db_scale: bool = True,
             **kwargs) -> str:
        """Построить и сохранить одиночный спектр.

        Args:
            signal (args[0]):   1D (или 2D — берётся первый канал) complex/real.
            fs     (args[1] or kwargs["fs"]): частота дискретизации (Гц).
            title:    заголовок и (слагифицированная) основа имени файла.
            n_fft:    размер FFT (None = len(signal)).
            db_scale: True → амплитуда в дБ.

        Returns:
            Путь к сохранённому PNG (или "" при save=False).
        """
        if len(args) < 1:
            raise TypeError("SpectrumPlotter.plot: signal обязателен")
        signal = args[0]
        fs = args[1] if len(args) > 1 else kwargs.get("fs")
        if fs is None:
            raise TypeError("SpectrumPlotter.plot: fs обязателен")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = np.asarray(signal)
        if sig.ndim == 2:
            sig = sig[0]  # берём первый канал

        spec = np.abs(np.fft.fftshift(np.fft.fft(sig, n=n_fft)))
        n = len(spec)
        freqs_mhz = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(fs))) / 1e6

        # КРИТИЧНО: стиль ДО plt.subplots, иначе не применится к figure
        if self.config.style:
            plt.style.use(self.config.style)
        fig, ax = plt.subplots(figsize=self.config.figsize)

        y = 20 * np.log10(spec + 1e-12) if db_scale else spec
        ax.plot(freqs_mhz, y, linewidth=0.8)
        ax.set_xlabel("Частота (МГц)")
        ax.set_ylabel("Амплитуда (дБ)" if db_scale else "Амплитуда")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # save_fig сам slugify-т имя — title может содержать любые символы
        return self.save_fig(fig, f"spectrum_{title}")

    def plot_compare(self,
                     gpu_signal: np.ndarray,
                     ref_signal: np.ndarray,
                     fs: float,
                     labels: tuple[str, str] = ("GPU", "NumPy"),
                     title: str = "GPU vs Reference",
                     n_fft: int | None = None) -> str:
        """Построить два спектра на одной figure (разные субплоты)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.config.style:
            plt.style.use(self.config.style)
        fig, axes = plt.subplots(
            2, 1,
            figsize=(self.config.figsize[0], self.config.figsize[1] * 1.2),
        )

        for ax, sig, label in zip(axes, (gpu_signal, ref_signal), labels):
            spec = np.abs(np.fft.fftshift(np.fft.fft(np.asarray(sig), n=n_fft)))
            n = len(spec)
            freqs_mhz = np.fft.fftshift(
                np.fft.fftfreq(n, d=1.0 / float(fs))
            ) / 1e6
            y = 20 * np.log10(spec + 1e-12)
            ax.plot(freqs_mhz, y, linewidth=0.8)
            ax.set_title(label)
            ax.set_xlabel("Частота (МГц)")
            ax.set_ylabel("Амплитуда (дБ)")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()
        return self.save_fig(fig, f"compare_{title}")
