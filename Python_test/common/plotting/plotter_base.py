"""
plotter_base.py — IPlotter (Strategy ABC) + PlotConfig
=======================================================

Strategy (GoF):
  IPlotter — интерфейс, конкретные реализации определяют что рисовать.
  Позволяет легко менять способ визуализации без изменения тестов.

Правило: matplotlib импортируется ТОЛЬКО здесь и в конкретных реализациях.
Тесты (test_*.py) никогда не импортируют matplotlib напрямую.

Usage:
    class SpectrumPlotter(IPlotter):
        def plot(self, data, title="Spectrum"):
            fig, ax = plt.subplots()
            ax.plot(np.abs(np.fft.fft(data)))
            self.save(fig, title)

    plotter = SpectrumPlotter(PlotConfig(out_dir="Results/Plots/filters"))
    plotter.plot(signal, title="FIR output")
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlotConfig:
    """Настройки стиля и вывода графиков.

    Attributes:
        out_dir:    директория для сохранения файлов
        dpi:        разрешение графика
        style:      matplotlib style ("dark_background" / "default" / ...)
        figsize:    размер фигуры (ширина, высота) в дюймах
        show:       показывать ли интерактивное окно
        save:       сохранять ли файл
        fmt:        формат файла ("png" / "svg" / "pdf")
    """
    out_dir: str = "Results/Plots"
    dpi: int = 120
    style: str = "dark_background"
    figsize: tuple = (14, 8)
    show: bool = False
    save: bool = True
    fmt: str = "png"

    def filepath(self, name: str) -> str:
        """Полный путь к файлу графика."""
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, f"{name}.{self.fmt}")


class IPlotter(ABC):
    """Абстрактный плоттер — Strategy interface.

    Конкретные реализации:
      SpectrumPlotter  — спектр сигнала
      TimePlotter      — временная развёртка
      ComparisonPlotter — сравнение GPU vs reference
      FilterPlotter    — АЧХ и ФЧХ фильтра
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()

    @abstractmethod
    def plot(self, *args, title: str = "", **kwargs) -> str:
        """Построить и сохранить график.

        Args:
            *args:  данные для визуализации
            title:  заголовок и имя файла
            **kwargs: дополнительные параметры

        Returns:
            Путь к сохранённому файлу (или "" если save=False).
        """
        ...

    def save_fig(self, fig, name: str) -> str:
        """Сохранить фигуру matplotlib в файл.

        Args:
            fig:  matplotlib Figure
            name: имя файла (без расширения)

        Returns:
            Путь к файлу.
        """
        path = self.config.filepath(name)
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        print(f"[Plotter] Saved: {path}")
        return path

    def _apply_style(self):
        """Применить стиль matplotlib перед построением."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if self.config.style:
                plt.style.use(self.config.style)
            return plt
        except ImportError:
            raise ImportError(
                "matplotlib не установлен. "
                "Установите: pip install matplotlib"
            )
