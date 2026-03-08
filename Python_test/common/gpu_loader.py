"""
gpu_loader.py — GPULoader Singleton
=====================================

Singleton (GoF) + Protected Variations (GRASP):
  Находит gpuworklib.so/.pyd один раз для всей pytest-сессии.
  Все тесты получают модуль через GPULoader.get() вместо хардкода путей.

Порядок поиска (от приоритетного к резервному):
  1. build/python/Release          ← MSVC Release
  2. build/python/Debug            ← MSVC Debug
  3. build/debian-radeon9070/python ← Linux ROCm (Alex's workstation)
  4. build/Release                 ← альтернатива
  5. build/Debug
  6. build/python                  ← общая сборка

Usage:
    gw = GPULoader.get()           # модуль gpuworklib или None
    if gw is None:
        pytest.skip("gpuworklib not found")

    ctx = gw.GPUContext()
"""

import os
import sys
from pathlib import Path
from typing import Optional


# Корень проекта GPUWorkLib (2 уровня вверх от common/)
_PROJECT_ROOT = Path(__file__).parents[2]

_SEARCH_PATHS = [
    "build/python/Release",
    "build/python/Debug",
    "build/debian-radeon9070/python",
    "build/Release",
    "build/Debug",
    "build/python",
]


class GPULoader:
    """Singleton — загружает gpuworklib один раз для всей сессии.

    Attributes:
        _instance:   единственный экземпляр (Singleton)
        _gpuworklib: загруженный модуль или None
        _loaded_from: путь откуда загружен модуль
    """

    _instance: Optional["GPULoader"] = None
    _gpuworklib = None
    _loaded_from: Optional[str] = None
    _load_attempted: bool = False

    @classmethod
    def get(cls):
        """Получить модуль gpuworklib.

        Первый вызов — ищет .so/.pyd по _SEARCH_PATHS и импортирует.
        Последующие вызовы — возвращают кешированный результат.

        Returns:
            Модуль gpuworklib или None если не найден.
        """
        if not cls._load_attempted:
            cls._load_attempted = True
            cls._try_load()
        return cls._gpuworklib

    @classmethod
    def _try_load(cls) -> None:
        """Найти и загрузить gpuworklib."""
        # Попробовать уже добавленные пути (вдруг уже доступен)
        try:
            import gpuworklib as gw
            cls._gpuworklib = gw
            cls._loaded_from = "already in sys.path"
            return
        except ImportError:
            pass

        # Перебрать пути поиска
        for rel_path in _SEARCH_PATHS:
            candidate = _PROJECT_ROOT / rel_path
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                try:
                    import gpuworklib as gw
                    cls._gpuworklib = gw
                    cls._loaded_from = str(candidate)
                    return
                except ImportError:
                    # Этот путь не подошёл — убрать его из sys.path
                    sys.path.pop(0)

        # Ничего не нашли
        cls._gpuworklib = None

    @classmethod
    def loaded_from(cls) -> Optional[str]:
        """Вернуть путь откуда загружен модуль (для диагностики)."""
        return cls._loaded_from

    @classmethod
    def is_available(cls) -> bool:
        """True если gpuworklib доступен."""
        return cls.get() is not None

    @classmethod
    def reset(cls) -> None:
        """Сбросить Singleton (для тестирования GPULoader)."""
        cls._gpuworklib = None
        cls._loaded_from = None
        cls._load_attempted = False
