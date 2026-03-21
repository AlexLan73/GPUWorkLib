"""
gpu_loader.py — GPULoader Singleton
=====================================

Singleton (GoF) + Protected Variations (GRASP):
  Находит gpuworklib.so/.pyd один раз для всей сессии.
  Все тесты получают модуль через GPULoader.get() вместо хардкода путей.

Порядок поиска (от приоритетного к резервному):
  0. GPUWORKLIB_BUILD_DIR (переменная окружения)  ← высший приоритет
  1. build/python/Release          ← MSVC Release
  2. build/python/Debug            ← MSVC Debug
  3. build/debian-radeon9070/python ← Linux ROCm (RDNA4, gfx1201)
  4. build/debian-mi100/python     ← Linux ROCm (CDNA1, gfx908)
  5. build/Release                 ← альтернатива
  6. build/Debug
  7. build/python                  ← общая сборка
  8. build/**/gpuworklib.*         ← авто-поиск по всему build/

Usage:
    gw = GPULoader.get()           # модуль gpuworklib или None
    if gw is None:
        raise SkipTest("gpuworklib not found")

    ctx = gw.GPUContext()

Cross-platform:
    Windows: gpuworklib.cpXXX-win_amd64.pyd
    Linux:   gpuworklib.cpython-XXX-x86_64-linux-gnu.so
    Python import system resolves the extension automatically.

    Override build path:
        GPUWORKLIB_BUILD_DIR=/path/to/build/python python Python_test/module/test_xxx.py
"""

import glob
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
    "build/debian-mi100/python",
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

        def _try_path(path: Path, label: str) -> bool:
            """Добавить path в sys.path и попробовать import. True если успешно."""
            sys.path.insert(0, str(path))
            try:
                import gpuworklib as gw
                cls._gpuworklib = gw
                cls._loaded_from = label
                return True
            except ImportError:
                sys.path.pop(0)
                return False

        # 0. Переменная окружения GPUWORKLIB_BUILD_DIR (высший приоритет)
        env_dir = os.environ.get("GPUWORKLIB_BUILD_DIR")
        if env_dir:
            candidate = Path(env_dir)
            if not candidate.is_absolute():
                candidate = _PROJECT_ROOT / candidate
            if candidate.exists() and _try_path(candidate, f"env: {candidate}"):
                return

        # 1. Попробовать уже добавленные пути (вдруг уже доступен)
        try:
            import gpuworklib as gw
            cls._gpuworklib = gw
            cls._loaded_from = "already in sys.path"
            return
        except ImportError:
            pass

        # 2. Перебрать статические пути поиска
        for rel_path in _SEARCH_PATHS:
            candidate = _PROJECT_ROOT / rel_path
            if candidate.exists() and _try_path(candidate, str(candidate)):
                return

        # 3. Авто-поиск: найти gpuworklib.* где угодно внутри build/
        build_dir = _PROJECT_ROOT / "build"
        if build_dir.exists():
            pattern = str(build_dir / "**" / "gpuworklib*")
            for found in sorted(glob.glob(pattern, recursive=True)):
                fp = Path(found)
                # Только .pyd (Windows) или .so (Linux), пропустить .so.debug и т.п.
                if fp.suffix in (".pyd", ".so") or ".cpython" in fp.name:
                    parent = fp.parent
                    if _try_path(parent, f"auto: {parent}"):
                        return

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
