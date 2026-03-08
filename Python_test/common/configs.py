"""
configs.py — конфигурационные dataclasses
==========================================

Information Expert (GRASP) — конфиг сам вычисляет производные параметры.
ISP (SOLID) — мелкие специализированные конфиги вместо одного большого dict.

Classes:
    SignalConfig   — параметры сигнала (fs, n_samples, f0, ...)
    FilterConfig   — параметры фильтра (type, cutoff, order, backend)
    ProcessorConfig — параметры GPU-обработки (device, batch_size)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import os


@dataclass
class SignalConfig:
    """Параметры тестового сигнала.

    Attributes:
        fs:         частота дискретизации, Гц
        n_samples:  число отсчётов
        f0_hz:      несущая частота, Гц
        fdev_hz:    девиация (для ЛЧМ), Гц
        amplitude:  амплитуда сигнала
        seed:       seed генератора случайных чисел (воспроизводимость)
    """
    fs: float = 12e6
    n_samples: int = 4096
    f0_hz: float = 2e6
    fdev_hz: float = 0.0
    amplitude: float = 1.0
    seed: int = 42

    def duration_s(self) -> float:
        """Длительность сигнала в секундах."""
        return self.n_samples / self.fs

    def duration_ms(self) -> float:
        """Длительность сигнала в миллисекундах."""
        return self.duration_s() * 1e3

    def freq_resolution_hz(self, nfft: Optional[int] = None) -> float:
        """Разрешение по частоте для FFT (Гц/бин)."""
        n = nfft or self.n_samples
        return self.fs / n

    def nyquist_hz(self) -> float:
        """Частота Найквиста."""
        return self.fs / 2.0


@dataclass
class FilterConfig:
    """Параметры GPU-фильтра.

    Attributes:
        filter_type:  тип фильтра — "fir" | "iir" | "kalman" | "kaufman"
        cutoff_hz:    частота среза (или [f_low, f_high] для полосовых)
        fs:           частота дискретизации
        order:        порядок фильтра
        window:       окно для FIR (kaiser/hamming/hann/blackman)
        ripple_db:    затухание в полосе задерживания (дБ)
        backend:      бэкенд — "rocm" | "opencl"
    """
    filter_type: str = "fir"
    cutoff_hz: Union[float, Tuple[float, float]] = 1e3
    fs: float = 12e6
    order: int = 4
    window: str = "kaiser"
    ripple_db: float = 60.0
    backend: str = "rocm"

    def normalized_cutoff(self) -> Union[float, list]:
        """Нормированная частота среза (0..1, Nyquist=1)."""
        nyq = self.fs / 2.0
        if isinstance(self.cutoff_hz, (list, tuple)):
            return [f / nyq for f in self.cutoff_hz]
        return self.cutoff_hz / nyq


@dataclass
class ProcessorConfig:
    """Параметры GPU-обработки.

    Attributes:
        device_index: индекс GPU (0 = первый)
        batch_size:   размер батча для обработки
        plot_dir:     директория для сохранения графиков
    """
    device_index: int = 0
    batch_size: int = 1024
    plot_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "Results", "Plots"
    ))

    def module_plot_dir(self, module_name: str) -> str:
        """Полный путь к директории графиков модуля."""
        return os.path.join(self.plot_dir, module_name)
