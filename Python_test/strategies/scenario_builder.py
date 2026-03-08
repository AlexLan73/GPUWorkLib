#!/usr/bin/env python3
"""
ScenarioBuilder -- физически корректный генератор тестовых сценариев
====================================================================

Генерирует сигнальные матрицы S [n_ant x n_samples] и весовые
матрицы W [n_ant x n_ant] для тестирования AntennaProcessor pipeline.

Физическая модель:
  - ULA (Uniform Linear Array) с шагом d между антеннами
  - Задержка i-й антенны: tau_i = i * d * sin(theta) / c
  - ЛЧМ формула совместима с C++ FormSignalGeneratorROCm:
    phase = 2*pi*f0*t_d + pi*(fdev/Ti)*(t_d - Ti/2)^2 + phi

Использование:
  builder = ScenarioBuilder(
      array=ULAGeometry(n_ant=8, d_ant_m=0.05),
      fs=12e6,
      n_samples=8000
  )
  builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=1e6)
  builder.add_jammer(theta_deg=-20, f0_hz=2e6, fdev_hz=500e3, amplitude=0.5)
  builder.set_noise(sigma=0.1)

  scenario = builder.build()
  S = scenario['S']   # [8 x 8000] complex64
  W = builder.generate_weight_matrix(steer_theta_deg=30)

Author: Кодо (AI Assistant)
Date: 2026-03-08
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# ULAGeometry -- геометрия антенной решётки
# ============================================================================

@dataclass
class ULAGeometry:
    """Uniform Linear Array -- геометрия линейной антенной решётки.

    Attributes:
        n_ant:    количество антенн
        d_ant_m:  шаг между соседними антеннами (м)
        c:        скорость распространения (м/с), 3e8 для РЛС
    """
    n_ant: int
    d_ant_m: float
    c: float = 3e8

    def compute_delays(self, theta_deg: float) -> np.ndarray:
        """Задержки [n_ant] в секундах для плоской волны под углом theta.

        tau_i = i * d * sin(theta) / c

        Args:
            theta_deg: угол прихода относительно нормали решётки (градусы)

        Returns:
            np.ndarray shape [n_ant], float64
        """
        theta_rad = np.deg2rad(theta_deg)
        return np.arange(self.n_ant) * self.d_ant_m * np.sin(theta_rad) / self.c

    def max_unambiguous_angle(self, freq_hz: float) -> float:
        """Максимальный однозначный угол (градусы) при d <= lambda/2.

        Если d > lambda/2 — возможны grating lobes.
        """
        wavelength = self.c / freq_hz
        ratio = wavelength / (2.0 * self.d_ant_m)
        if ratio >= 1.0:
            return 90.0
        return np.rad2deg(np.arcsin(ratio))

    @staticmethod
    def from_lambda_half(n_ant: int, carrier_freq_hz: float, c: float = 3e8):
        """Создать решётку с d = lambda/2 для заданной несущей.

        Args:
            n_ant: количество антенн
            carrier_freq_hz: несущая частота (Гц) -- РЕАЛЬНАЯ, не промежуточная
            c: скорость распространения
        """
        wavelength = c / carrier_freq_hz
        return ULAGeometry(n_ant=n_ant, d_ant_m=wavelength / 2.0, c=c)


# ============================================================================
# EmitterSignal -- один источник излучения
# ============================================================================

@dataclass
class EmitterSignal:
    """Описание одного источника (цель или помеха).

    Attributes:
        theta_deg:  угол прихода относительно нормали (градусы)
        f0_hz:      несущая частота (Гц)
        fdev_hz:    девиация частоты ЛЧМ (Гц), 0 = CW
        amplitude:  амплитуда сигнала
        phase_rad:  начальная фаза (рад)
        label:      метка для логов ("target_0", "jammer_0")
    """
    theta_deg: float
    f0_hz: float
    fdev_hz: float = 0.0
    amplitude: float = 1.0
    phase_rad: float = 0.0
    label: str = ""

    def to_dict(self) -> dict:
        """Сериализация в dict для JSON."""
        return {
            'theta_deg': self.theta_deg,
            'f0_hz': self.f0_hz,
            'fdev_hz': self.fdev_hz,
            'amplitude': self.amplitude,
            'phase_rad': self.phase_rad,
            'label': self.label,
        }


# ============================================================================
# ScenarioBuilder -- строитель тестовых сценариев
# ============================================================================

class ScenarioBuilder:
    """Строитель тестовых сценариев для AntennaProcessor.

    Генерирует сигнальную матрицу S [n_ant x n_samples] как суммму
    нескольких источников (целей + помех) + AWGN шума.

    Формула ЛЧМ (совместимая с C++ FormSignalGeneratorROCm):
      Ti = n_samples / fs
      t_d = t - tau_ant
      phase = 2*pi*f0*t_d + pi*(fdev/Ti)*(t_d - Ti/2)^2 + phi
      x = A * norm * exp(j * phase)   при 0 <= t_d < Ti

    Args:
        array:     ULAGeometry -- геометрия решётки
        fs:        частота дискретизации (Гц)
        n_samples: количество отсчётов на антенну
    """

    def __init__(self, array: ULAGeometry, fs: float, n_samples: int):
        self.array = array
        self.fs = fs
        self.n_samples = n_samples
        self._targets: List[EmitterSignal] = []
        self._jammers: List[EmitterSignal] = []
        self._noise_sigma: float = 0.0
        self._noise_seed: int = 42

    # ---- Fluent API ----

    def add_target(self, theta_deg: float, f0_hz: float,
                   fdev_hz: float = 0.0, amplitude: float = 1.0,
                   phase_rad: float = 0.0) -> 'ScenarioBuilder':
        """Добавить цель (ЛЧМ или CW сигнал).

        Args:
            theta_deg:  угол прихода (градусы от нормали)
            f0_hz:      несущая частота (Гц)
            fdev_hz:    девиация ЛЧМ (0 = CW)
            amplitude:  амплитуда
            phase_rad:  начальная фаза
        """
        sig = EmitterSignal(
            theta_deg=theta_deg,
            f0_hz=f0_hz,
            fdev_hz=fdev_hz,
            amplitude=amplitude,
            phase_rad=phase_rad,
            label=f"target_{len(self._targets)}"
        )
        self._targets.append(sig)
        return self

    def add_jammer(self, theta_deg: float, f0_hz: float,
                   fdev_hz: float = 0.0, amplitude: float = 1.0,
                   phase_rad: float = 0.0) -> 'ScenarioBuilder':
        """Добавить помеху (ЛЧМ или CW).

        Args:
            theta_deg:  угол прихода помехи (градусы)
            f0_hz:      частота помехи (Гц)
            fdev_hz:    девиация ЛЧМ помехи (0 = CW)
            amplitude:  амплитуда помехи
            phase_rad:  начальная фаза
        """
        sig = EmitterSignal(
            theta_deg=theta_deg,
            f0_hz=f0_hz,
            fdev_hz=fdev_hz,
            amplitude=amplitude,
            phase_rad=phase_rad,
            label=f"jammer_{len(self._jammers)}"
        )
        self._jammers.append(sig)
        return self

    def set_noise(self, sigma: float, seed: int = 42) -> 'ScenarioBuilder':
        """Установить уровень AWGN шума.

        Args:
            sigma:  СКО шума (амплитуда)
            seed:   seed для воспроизводимости
        """
        self._noise_sigma = sigma
        self._noise_seed = seed
        return self

    # ---- Генерация ----

    def _generate_emitter(self, emitter: EmitterSignal) -> np.ndarray:
        """Генерировать [n_ant x n_samples] для одного источника.

        Формула ЛЧМ (совместимая с C++ FormSignalGeneratorROCm):
          Ti = n_samples / fs
          t_d = t - tau[ant]
          phase = 2*pi*f0*t_d + pi*(fdev/Ti)*(t_d - Ti/2)^2 + phi
          x = A * (1/sqrt(2)) * exp(j * phase)

        При fdev=0 формула вырождается в CW: phase = 2*pi*f0*t_d + phi
        """
        dt = 1.0 / self.fs
        t = np.arange(self.n_samples) * dt     # [n_samples]
        Ti = self.n_samples * dt               # длительность сигнала

        delays = self.array.compute_delays(emitter.theta_deg)  # [n_ant]

        S = np.zeros((self.array.n_ant, self.n_samples), dtype=np.complex64)
        norm = 1.0 / np.sqrt(2.0)  # совместимо с C++ norm = 0.7071...

        for ant in range(self.array.n_ant):
            t_d = t - delays[ant]
            valid = (t_d >= 0) & (t_d < Ti)
            t_v = t_d[valid]

            # ЛЧМ фаза с центрированием (t_d - Ti/2)^2
            phase = (2.0 * np.pi * emitter.f0_hz * t_v
                     + np.pi * (emitter.fdev_hz / Ti) * (t_v - Ti / 2.0) ** 2
                     + emitter.phase_rad)

            S[ant, valid] = (emitter.amplitude * norm
                             * np.exp(1j * phase)).astype(np.complex64)

        return S

    def build(self) -> dict:
        """Построить сигнальную матрицу S как суммму всех источников + шума.

        Returns:
            dict с ключами:
              'S':           np.ndarray [n_ant x n_samples] complex64
              'targets':     list[EmitterSignal]
              'jammers':     list[EmitterSignal]
              'noise_sigma': float
              'array':       ULAGeometry
              'fs':          float
              'n_samples':   int
        """
        S = np.zeros((self.array.n_ant, self.n_samples), dtype=np.complex64)

        # Суммируем цели
        for target in self._targets:
            S += self._generate_emitter(target)

        # Суммируем помехи
        for jammer in self._jammers:
            S += self._generate_emitter(jammer)

        # Добавляем AWGN
        if self._noise_sigma > 0:
            rng = np.random.default_rng(self._noise_seed)
            noise = (self._noise_sigma / np.sqrt(2.0)
                     * (rng.standard_normal(S.shape)
                        + 1j * rng.standard_normal(S.shape)))
            S += noise.astype(np.complex64)

        return {
            'S': S,
            'targets': list(self._targets),
            'jammers': list(self._jammers),
            'noise_sigma': self._noise_sigma,
            'array': self.array,
            'fs': self.fs,
            'n_samples': self.n_samples,
        }

    def generate_weight_matrix(self, steer_theta_deg: float,
                                steer_freq_hz: Optional[float] = None
                                ) -> np.ndarray:
        """Генерация Delay-and-sum весовой матрицы W.

        W[beam, ant] = (1/sqrt(N)) * exp(-j * 2*pi * f * tau_ant(theta_steer))

        В текущей версии все beam-ы наводятся на один угол (single-steer).

        Args:
            steer_theta_deg: угол наведения (градусы)
            steer_freq_hz:   частота для фазового сдвига (Гц),
                             если None — берётся f0 первой цели

        Returns:
            np.ndarray [n_ant x n_ant] complex64
        """
        n = self.array.n_ant

        if steer_freq_hz is None:
            if self._targets:
                steer_freq_hz = self._targets[0].f0_hz
            else:
                raise ValueError("steer_freq_hz не задан и нет целей")

        delays = self.array.compute_delays(steer_theta_deg)
        inv_sqrt_n = 1.0 / np.sqrt(n)

        W = np.zeros((n, n), dtype=np.complex64)
        for beam in range(n):
            for ant in range(n):
                W[beam, ant] = inv_sqrt_n * np.exp(
                    -1j * 2.0 * np.pi * steer_freq_hz * delays[ant]
                )

        return W

    def generate_scan_weight_matrix(self, steer_angles_deg: List[float],
                                     steer_freq_hz: Optional[float] = None
                                     ) -> np.ndarray:
        """Генерация W со сканированием по углу (каждый beam -- свой угол).

        W[beam, ant] = (1/sqrt(N)) * exp(-j * 2*pi * f * tau_ant(theta_beam))

        Args:
            steer_angles_deg: список углов наведения [n_beams]
            steer_freq_hz:    частота для фазового сдвига

        Returns:
            np.ndarray [n_beams x n_ant] complex64
        """
        n_ant = self.array.n_ant
        n_beams = len(steer_angles_deg)

        if steer_freq_hz is None:
            if self._targets:
                steer_freq_hz = self._targets[0].f0_hz
            else:
                raise ValueError("steer_freq_hz не задан и нет целей")

        inv_sqrt_n = 1.0 / np.sqrt(n_ant)
        W = np.zeros((n_beams, n_ant), dtype=np.complex64)

        for beam, theta in enumerate(steer_angles_deg):
            delays = self.array.compute_delays(theta)
            for ant in range(n_ant):
                W[beam, ant] = inv_sqrt_n * np.exp(
                    -1j * 2.0 * np.pi * steer_freq_hz * delays[ant]
                )

        return W

    def summary(self) -> str:
        """Краткое текстовое описание сценария."""
        lines = [
            f"ScenarioBuilder Summary:",
            f"  Array: {self.array.n_ant} ant, d={self.array.d_ant_m*100:.1f} cm, "
            f"c={self.array.c:.0e} m/s",
            f"  Sampling: fs={self.fs/1e6:.1f} MHz, N={self.n_samples}",
            f"  Duration: {self.n_samples/self.fs*1e3:.2f} ms",
        ]
        if self._targets:
            lines.append(f"  Targets ({len(self._targets)}):")
            for t in self._targets:
                chirp = f", fdev={t.fdev_hz/1e3:.0f} kHz" if t.fdev_hz else " (CW)"
                lines.append(
                    f"    {t.label}: theta={t.theta_deg:.1f} deg, "
                    f"f0={t.f0_hz/1e6:.2f} MHz{chirp}, A={t.amplitude}"
                )
        if self._jammers:
            lines.append(f"  Jammers ({len(self._jammers)}):")
            for j in self._jammers:
                chirp = f", fdev={j.fdev_hz/1e3:.0f} kHz" if j.fdev_hz else " (CW)"
                lines.append(
                    f"    {j.label}: theta={j.theta_deg:.1f} deg, "
                    f"f0={j.f0_hz/1e6:.2f} MHz{chirp}, A={j.amplitude}"
                )
        if self._noise_sigma > 0:
            lines.append(f"  Noise: sigma={self._noise_sigma}, seed={self._noise_seed}")
        return "\n".join(lines)


# ============================================================================
# Готовые сценарии-фабрики
# ============================================================================

def make_single_target(n_ant: int = 8,
                       d_ant_m: float = 0.05,
                       theta_deg: float = 30.0,
                       f0_hz: float = 2e6,
                       fdev_hz: float = 1e6,
                       amplitude: float = 1.0,
                       noise_sigma: float = 0.1,
                       fs: float = 12e6,
                       n_samples: int = 8000) -> dict:
    """Сценарий: одна ЛЧМ цель + AWGN.

    Returns:
        dict: {'S', 'W', 'builder', ...}
    """
    builder = ScenarioBuilder(
        array=ULAGeometry(n_ant=n_ant, d_ant_m=d_ant_m),
        fs=fs,
        n_samples=n_samples
    )
    builder.add_target(theta_deg, f0_hz, fdev_hz, amplitude)
    builder.set_noise(noise_sigma)

    scenario = builder.build()
    scenario['W'] = builder.generate_weight_matrix(steer_theta_deg=theta_deg)
    scenario['builder'] = builder
    return scenario


def make_target_and_jammer(n_ant: int = 8,
                           d_ant_m: float = 0.05,
                           target_theta: float = 30.0,
                           target_f0: float = 2e6,
                           target_fdev: float = 1e6,
                           jammer_theta: float = -20.0,
                           jammer_f0: float = 2e6,
                           jammer_fdev: float = 500e3,
                           jammer_amplitude: float = 0.5,
                           noise_sigma: float = 0.1,
                           fs: float = 12e6,
                           n_samples: int = 8000) -> dict:
    """Сценарий: 1 ЛЧМ цель + 1 ЛЧМ помеха + AWGN.

    Returns:
        dict: {'S', 'W', 'builder', ...}
    """
    builder = ScenarioBuilder(
        array=ULAGeometry(n_ant=n_ant, d_ant_m=d_ant_m),
        fs=fs,
        n_samples=n_samples
    )
    builder.add_target(target_theta, target_f0, target_fdev)
    builder.add_jammer(jammer_theta, jammer_f0, jammer_fdev, jammer_amplitude)
    builder.set_noise(noise_sigma)

    scenario = builder.build()
    scenario['W'] = builder.generate_weight_matrix(steer_theta_deg=target_theta)
    scenario['builder'] = builder
    return scenario


def make_multi_target(n_ant: int = 8,
                      d_ant_m: float = 0.05,
                      thetas: Optional[List[float]] = None,
                      f0s: Optional[List[float]] = None,
                      fdevs: Optional[List[float]] = None,
                      amplitudes: Optional[List[float]] = None,
                      noise_sigma: float = 0.1,
                      fs: float = 12e6,
                      n_samples: int = 8000) -> dict:
    """Сценарий: несколько ЛЧМ целей + AWGN.

    Args:
        thetas:     углы целей [deg], default [20, 45]
        f0s:        несущие частоты [Hz], default [2e6, 3e6]
        fdevs:      девиации ЛЧМ [Hz], default [1e6, 1e6]
        amplitudes: амплитуды, default [1.0, 0.5]

    Returns:
        dict: {'S', 'W', 'builder', ...}
    """
    if thetas is None:
        thetas = [20.0, 45.0]
    if f0s is None:
        f0s = [2e6, 3e6]
    if fdevs is None:
        fdevs = [1e6, 1e6]
    if amplitudes is None:
        amplitudes = [1.0, 0.5]

    builder = ScenarioBuilder(
        array=ULAGeometry(n_ant=n_ant, d_ant_m=d_ant_m),
        fs=fs,
        n_samples=n_samples
    )

    for theta, f0, fdev, amp in zip(thetas, f0s, fdevs, amplitudes):
        builder.add_target(theta, f0, fdev, amp)

    builder.set_noise(noise_sigma)

    scenario = builder.build()
    # W наводим на первую цель
    scenario['W'] = builder.generate_weight_matrix(steer_theta_deg=thetas[0])
    scenario['builder'] = builder
    return scenario
