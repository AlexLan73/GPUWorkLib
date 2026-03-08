"""
gpu_context.py — GPUContextManager Singleton
=============================================

Singleton (GoF):
  Создаёт GPU-контекст один раз для всей pytest-сессии.
  Переиспользование контекста критично — создание занимает ~1-2 сек.

Usage:
    ctx = GPUContextManager.get()          # создаёт или возвращает существующий
    ctx = GPUContextManager.get(device=1)  # выбрать GPU #1

    # В pytest conftest.py:
    @pytest.fixture(scope="session")
    def gpu_ctx():
        return GPUContextManager.get()
"""

from typing import Optional
from .gpu_loader import GPULoader


class GPUContextManager:
    """Singleton — хранит GPUContext для всей сессии.

    Attributes:
        _context:      единственный GPU-контекст
        _device_index: индекс GPU на котором создан контекст
    """

    _context = None
    _device_index: int = 0
    _create_attempted: bool = False

    @classmethod
    def get(cls, device: int = 0):
        """Получить или создать GPU-контекст.

        Args:
            device: индекс GPU (0 = первый). Игнорируется если контекст
                    уже создан — переиспользуется существующий.

        Returns:
            GPUContext или None если gpuworklib недоступен.
        """
        if cls._context is None and not cls._create_attempted:
            cls._create_attempted = True
            cls._device_index = device
            cls._try_create(device)
        return cls._context

    @classmethod
    def _try_create(cls, device: int) -> None:
        """Создать GPUContext через gpuworklib."""
        gw = GPULoader.get()
        if gw is None:
            return
        try:
            cls._context = gw.GPUContext(device)
        except Exception as e:
            print(f"[GPUContextManager] Failed to create context on device {device}: {e}")
            cls._context = None

    @classmethod
    def is_available(cls) -> bool:
        """True если контекст создан."""
        return cls.get() is not None

    @classmethod
    def device_index(cls) -> int:
        """Индекс GPU на котором создан контекст."""
        return cls._device_index

    @classmethod
    def reset(cls) -> None:
        """Сбросить контекст (для тестирования GPUContextManager)."""
        cls._context = None
        cls._create_attempted = False
