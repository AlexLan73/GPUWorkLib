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
from typing import List, Optional, Tuple, Union
import json
import os
from pathlib import Path


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
class HeterodyneConfig(SignalConfig):
    """SignalConfig + вычисляемые LFM/dechirp свойства.

    Маппинг полей:
        f0_hz   = f_start (начальная частота ЛЧМ)
        fdev_hz = bandwidth (f_end - f_start)
        f_end   = f0_hz + fdev_hz  (вычисляемое)

    Использование:
        cfg = HeterodyneConfig(fs=12e6, f0_hz=0.0, fdev_hz=2e6,
                               n_samples=8000, n_antennas=5)
        print(cfg.chirp_rate)             # -> 3e9
        print(cfg.fbeat_from_delay(1e-4)) # -> 300e3
    """
    n_antennas: int = 5
    c_light: float = 3e8

    @property
    def f_start(self) -> float:
        return self.f0_hz

    @property
    def f_end(self) -> float:
        return self.f0_hz + self.fdev_hz

    @property
    def bandwidth(self) -> float:
        return self.fdev_hz

    @property
    def chirp_rate(self) -> float:
        """Скорость изменения частоты (Гц/с)."""
        return self.fdev_hz / self.duration_s()

    def range_from_delay(self, delay_s: float) -> float:
        """Дальность из задержки (м)."""
        return self.c_light * delay_s / 2.0

    def fbeat_from_delay(self, delay_s: float) -> float:
        """Частота биения из задержки (Гц)."""
        return self.chirp_rate * delay_s


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


# ─────────────────────────────────────────────────────────────────────────────
# GPU Config — configGPU.json
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuEntry:
    """Одна запись GPU из configGPU.json."""
    id: int = 0
    is_active: bool = False
    name: str = ""


def load_gpu_config(config_path: Union[str, Path]) -> List[GpuEntry]:
    """Прочитать configGPU.json и вернуть список GpuEntry.

    Args:
        config_path: путь к configGPU.json (рядом с бинарником/либо).

    Returns:
        Список GpuEntry из секции "gpus". Пустой список если файл не найден.
    """
    path = Path(config_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        for gpu in data.get("gpus", []):
            entries.append(GpuEntry(
                id=int(gpu.get("id", 0)),
                is_active=bool(gpu.get("is_active", False)),
                name=str(gpu.get("name", "")),
            ))
        return entries
    except Exception:
        return []


def active_gpu_ids(config_path: Union[str, Path]) -> List[int]:
    """Вернуть список id активных GPU из configGPU.json."""
    return [e.id for e in load_gpu_config(config_path) if e.is_active]


def first_active_gpu_id(config_path: Union[str, Path], default: int = 0) -> int:
    """Вернуть id первого активного GPU (default если конфиг не найден/пустой)."""
    ids = active_gpu_ids(config_path)
    return ids[0] if ids else default
