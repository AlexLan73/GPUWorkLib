"""
GeneratorFactory -- создаёт GPU-генераторы по строковому типу.

GoF Factory Method + Registry: OCP -- добавить новый тип =
GeneratorFactory.register("mytype", MyClass), не менять этот класс.
GRASP Creator: знает как создать ISignalGenerator.

OCP гарантия: Factory НЕ содержит if/elif по типам.
Каждый адаптер реализует from_config(ctx, params) -- Factory
вызывает его единообразно.
"""

from __future__ import annotations
from typing import Dict, Type, TYPE_CHECKING

from .base import ISignalGenerator

if TYPE_CHECKING:
    from common.configs import SignalConfig


class GeneratorFactory:
    """
    Создаёт GPU-генераторы по строковому типу.

    Типы: "cw" | "lfm" | "noise"  (регистрируются в generators/__init__.py)

    Использование:
        gen = GeneratorFactory.create("cw", ctx, params)
        signal = gen.generate(4096)
    """

    _registry: Dict[str, Type[ISignalGenerator]] = {}

    @classmethod
    def register(cls, type_name: str, generator_class: Type[ISignalGenerator]) -> None:
        """
        Регистрирует тип генератора.

        Args:
            type_name:       Строка-ключ ("cw", "lfm", "noise", ...)
            generator_class: Класс, реализующий ISignalGenerator
        """
        cls._registry[type_name] = generator_class

    @classmethod
    def create(cls, type_name: str, ctx, params: "SignalConfig") -> ISignalGenerator:
        """
        Создаёт генератор по типу и конфигурации.

        Args:
            type_name: "cw" | "lfm" | "noise"
            ctx:       GPU контекст (GPUContext или ROCmGPUContext)
            params:    SignalConfig с fs, f0_hz, fdev_hz, amplitude, seed

        Returns:
            ISignalGenerator: готов к generate(n_samples)

        Raises:
            ValueError: если type_name не зарегистрирован
        """
        if type_name not in cls._registry:
            available = sorted(cls._registry)
            raise ValueError(
                f"Неизвестный тип генератора: '{type_name}'. "
                f"Доступные: {available}"
            )
        # OCP: вызываем from_config() -- Factory не знает про конкретные параметры
        return cls._registry[type_name].from_config(ctx, params)

    @classmethod
    def available(cls) -> list:
        """Список зарегистрированных типов."""
        return sorted(cls._registry)
