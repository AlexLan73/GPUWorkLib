"""
Базовый миксин для GPU-процессоров.

Почему НЕ ABC с process() -> ndarray|dict:
  Statistics возвращает dict, FFT/Heterodyne -- ndarray.
  Union return type нарушает LSP (код вроде np.abs(result) падает для dict).
  Каждый адаптер имеет СВОЙ типизированный process().
  Миксин даёт общий __repr__ и проверку GPULoader -- без навязывания сигнатуры.

Масштабирование:
  Новые адаптеры (FilterAdapter, CorrelatorAdapter) наследуют миксин
  и определяют свой process() с нужным return type.
"""

from common import GPULoader


class GpuProcessorMixin:
    """Миксин для GPU-адаптеров: загрузка gpuworklib + repr."""

    _processor_name: str = "GpuProcessor"

    @staticmethod
    def _load_gw():
        gw = GPULoader.get()
        if gw is None:
            raise RuntimeError("gpuworklib не загружен")
        return gw

    @property
    def name(self) -> str:
        return self._processor_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
