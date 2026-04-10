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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def _slugify(name: str) -> str:
    """Безопасное имя файла: убирает спец-символы, запрещённые в Windows.

    Windows запрещает: < > : " / \\ | ? *   — плюс пробелы нежелательны.
    Всё, кроме букв/цифр/подчёркиваний/точек/дефисов, заменяется на "_".
    Возвращает "unnamed" для пустого результата.
    """
    s = re.sub(r"[^\w\-.]+", "_", name)
    return s.strip("_") or "unnamed"


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
        verbose:    печатать ли "[Plotter] Saved: …" при сохранении.
                    Отключить при параллельном прогоне тестов, чтобы
                    stdout не смешивался между потоками.
    """
    out_dir: str = "Results/Plots"
    dpi: int = 120
    style: str = "dark_background"
    figsize: tuple = (14, 8)
    show: bool = False
    save: bool = True
    fmt: str = "png"
    verbose: bool = True

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
        """Сохранить фигуру, показать (опционально), закрыть.

        Имя автоматически slugify-тся — в name можно передавать любые
        заголовки (даже с ``:``, ``/``, пробелами и юникодом).

        Args:
            fig:  matplotlib Figure
            name: имя файла (без расширения), будет слагифицировано.

        Returns:
            Абсолютный путь к сохранённому файлу (или пустая строка если
            ``config.save=False``).
        """
        import matplotlib.pyplot as plt  # lazy import

        safe = _slugify(name)
        path = self.config.filepath(safe)
        if self.config.save:
            fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
            if self.config.verbose:
                print(f"[Plotter] Saved: {path}")
        if self.config.show:
            plt.show()
        plt.close(fig)
        return path if self.config.save else ""
