"""
factory.py — PlotterFactory (GoF Factory Method, GRASP Information Expert)
============================================================================

Создаёт плоттеры с правильными путями для конкретного модуля.

    factory = PlotterFactory("signal_generators")
    factory.spectrum().plot(signal, fs=12e6, title="CW 2MHz")
    # → Results/Plots/signal_generators/spectrum_CW_2MHz.png

    # Тестовый override базового пути (без мутации приватов!):
    factory = PlotterFactory("test_mod", out_dir=tmp_dir, save=True)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .plotter_base import PlotConfig
from .spectrum_plotter import SpectrumPlotter
from .time_plotter import TimePlotter


class PlotterFactory:
    """Фабрика плоттеров для модуля GPUWorkLib."""

    BASE_PLOTS_DIR = "Results/Plots"

    def __init__(self,
                 module_name: str,
                 out_dir: str | Path | None = None,
                 dpi: int = 120,
                 style: str = "dark_background",
                 show: bool = False,
                 save: bool = True,
                 verbose: bool = True):
        """
        Args:
            module_name: имя модуля (signal_generators, heterodyne, …).
            out_dir:     override базового пути. По умолчанию —
                         ``Results/Plots/{module_name}``.
            dpi:         DPI для сохранения.
            style:       matplotlib style.
            show:        показывать ли figure после сохранения.
            save:        сохранять ли файл вообще.
            verbose:     печатать ли "[Plotter] Saved: …" в stdout.
                         Отключить при параллельном прогоне тестов.
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
            verbose=verbose,
        )

    def spectrum(self, subdir: str = "") -> SpectrumPlotter:
        """Создать SpectrumPlotter для модуля.

        Args:
            subdir: необязательная подпапка внутри модуля (напр. "FormSignal").
        """
        return SpectrumPlotter(self._with_subdir(subdir))

    def timeseries(self, subdir: str = "") -> TimePlotter:
        """Создать TimePlotter для модуля."""
        return TimePlotter(self._with_subdir(subdir))

    @property
    def config(self) -> PlotConfig:
        """Текущая PlotConfig фабрики (read-only)."""
        return self._config

    def _with_subdir(self, subdir: str) -> PlotConfig:
        """Создать копию PlotConfig с новой подпапкой.

        ``dataclasses.replace`` — идиоматический способ, устойчивый к
        добавлению новых полей в PlotConfig.
        """
        if not subdir:
            # Возвращаем копию, чтобы плоттеры не делили один инстанс
            return replace(self._config)
        return replace(
            self._config,
            out_dir=str(Path(self._config.out_dir) / subdir),
        )
