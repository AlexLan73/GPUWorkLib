"""
base.py — IDataStore (Strategy interface)
==========================================

GoF Strategy: единый интерфейс для хранилища данных.
SOLID ISP: только операции I/O, ничего лишнего.

Реализации:
    NumpyStore — .npy / .npz (numpy arrays)
    JsonStore  — .json (dict / TestResult)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class IDataStore(ABC):
    """Абстрактное хранилище данных."""

    @abstractmethod
    def save(self, data, name: str, subdir: str = "") -> Path:
        """Сохранить данные.

        Args:
            data:   данные для сохранения (тип зависит от реализации).
            name:   имя файла (без расширения).
            subdir: подкаталог внутри base_dir (например, "signal_generators").

        Returns:
            Path: абсолютный путь к сохранённому файлу.
        """

    @abstractmethod
    def load(self, name: str, subdir: str = ""):
        """Загрузить данные по имени."""

    @abstractmethod
    def exists(self, name: str, subdir: str = "") -> bool:
        """True если данные уже сохранены."""

    @abstractmethod
    def list(self, subdir: str = "") -> list[str]:
        """Список всех имён в subdir (без расширения)."""
