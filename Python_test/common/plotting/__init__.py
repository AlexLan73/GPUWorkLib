"""
plotting — абстракции и реализации визуализации
=================================================

Отделяет слой визуализации от слоя тестирования.
Графики вызываются только из __main__ и plot_*() методов.

Классы:
    IPlotter         — абстрактный интерфейс (Strategy)
    PlotConfig       — настройки стиля графиков
    SpectrumPlotter  — амплитудный спектр (|FFT|)
    TimePlotter      — временной ряд (I/Q + magnitude)
    PlotterFactory   — Factory Method для создания плоттеров модуля
"""

from .factory import PlotterFactory
from .plotter_base import IPlotter, PlotConfig
from .spectrum_plotter import SpectrumPlotter
from .time_plotter import TimePlotter

__all__ = [
    "IPlotter", "PlotConfig",
    "SpectrumPlotter", "TimePlotter",
    "PlotterFactory",
]
