"""
signal_factory.py — ISignalSource + SignalSourceFactory
========================================================

Фабрика тестовых сигналов: 5 вариантов (V1–V5).

Паттерны:
    Strategy (GoF):      ISignalSource — интерфейс, каждая реализация = один вариант
    Factory Method (GoF): SignalSourceFactory.create(variant) — создаёт нужную реализацию
    OCP (SOLID):         добавить V6 = новый класс, без изменений существующего кода

Важно: step_0_prepare_input() принимает NumPy-массивы напрямую (pybind11 делает
hipMemcpy внутри). Поэтому d_S/d_W в SignalData — это CPU-массивы complex64,
которые передаются в step_0_prepare_input.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Дефолтные параметры тестового сценария
N_ANT     = 5
N_SAMPLES = 8000
FS        = 12e6     # Гц
F0        = 2e6      # Гц (CW тон)
TAU_STEP  = 100e-6   # сек (задержка между антеннами, для V3/V4)
N_FFT     = 8192     # следующая степень 2 >= N_SAMPLES
SNR_DB    = 20.0     # дБ (для V2/V4)


class SignalVariant(Enum):
    """Выбор сценария перед стартом теста.

    Information Expert (GRASP): знает параметры каждого варианта.

    V1–V5: варианты CW pipeline (используются SignalSourceFactory + ISignalSource).
    SIN/LFM_*: варианты LFM/SIN стратегий (используются SignalStrategyFactory + ISignalStrategy).
    """
    # CW pipeline variants (для test_strategies_pipeline.py)
    V1_CW_CLEAN        = 1   # CW без шума,   W = Identity
    V2_CW_NOISE        = 2   # CW + AWGN,     W = Identity
    V3_CW_PHASE_DELAY  = 3   # CW + задержки, W = delay_and_sum, без шума
    V4_CW_PHASE_NOISE  = 4   # CW + задержки, W = delay_and_sum, + AWGN
    V5_FROM_FILE       = 5   # Загрузка из файла → GPU (заглушка)

    # LFM/SIN variants (для test_base_pipeline.py, test_debug_steps.py)
    SIN            = 10  # синус (fdev=0)
    LFM_NO_DELAY   = 11  # ЛЧМ без задержек (2.1)
    LFM_WITH_DELAY = 12  # ЛЧМ + целочисленные задержки (2.2.1)
    LFM_FARROW     = 13  # ЛЧМ + дробные задержки (2.2.2)


@dataclass
class SignalConfig:
    """Конфигурация тестового сигнала."""
    n_ant    : int   = N_ANT
    n_samples: int   = N_SAMPLES
    fs       : float = FS
    f0       : float = F0
    tau_step : float = TAU_STEP  # для V3/V4
    snr_db   : float = SNR_DB    # для V2/V4
    n_fft    : int   = N_FFT
    file_path: str   = ""        # для V5


@dataclass
class SignalData:
    """Результат генерации: NumPy массивы + параметры.

    Creator (GRASP): создаётся ISignalSource.generate().

    d_S и d_W — NumPy-массивы complex64 (CPU).
    Передаются напрямую в proc.step_0_prepare_input(d_S, d_W).
    pybind11 делает hipMemcpy в GPU внутри этого метода.
    """
    d_S  : np.ndarray   # [n_ant * n_samples] complex64  (flat, CPU → GPU через step_0)
    d_W  : np.ndarray   # [n_ant * n_ant] complex64      (flat, CPU → GPU через step_0)
    S_ref: np.ndarray   # [n_ant, n_samples] complex64   CPU-эталон
    W_ref: np.ndarray   # [n_ant, n_ant] complex64       CPU-эталон
    cfg  : SignalConfig  # параметры которые использовались
    variant: SignalVariant  # какой вариант сгенерирован


class ISignalSource(ABC):
    """Strategy: источник тестового сигнала.

    Каждая реализация — отдельный вариант V1..V5.
    Интерфейс стабилен — добавление V6 не ломает существующий код (OCP).
    """

    @abstractmethod
    def generate(self, cfg: SignalConfig) -> SignalData:
        """Создать сигнал: CPU-массивы + CPU-эталоны.

        Args:
            cfg: конфигурация сигнала

        Returns:
            SignalData с d_S, d_W (CPU, flat) и S_ref, W_ref (CPU, 2D)

        Note:
            d_S и d_W передаются потом в proc.step_0_prepare_input()
            который делает upload на GPU внутри pybind11.
        """
        ...


class CwCleanSignalSource(ISignalSource):
    """V1: CW без шума, весовая матрица = Identity.

    GEMM тривиален: X = I @ S = S.
    Используется для проверки базового pipeline без алгоритмической нагрузки.
    """

    def generate(self, cfg: SignalConfig) -> SignalData:
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs
        cw = np.exp(1j * 2 * np.pi * cfg.f0 * t).astype(np.complex64)
        S_ref = np.tile(cw, (cfg.n_ant, 1))  # [n_ant, n_samples]
        W_ref = np.eye(cfg.n_ant, dtype=np.complex64)

        return SignalData(
            d_S=S_ref.ravel(),
            d_W=W_ref.ravel(),
            S_ref=S_ref,
            W_ref=W_ref,
            cfg=cfg,
            variant=SignalVariant.V1_CW_CLEAN,
        )


class CwNoiseSignalSource(ISignalSource):
    """V2: CW + белый гауссов шум (AWGN), W = Identity.

    S[k,n] = exp(j*2π*f0*n/fs) + noise[k,n]
    Шум добавляется на CPU.
    SNR вычисляется через cfg.snr_db.
    seed=42 → воспроизводимые результаты.
    """

    def generate(self, cfg: SignalConfig) -> SignalData:
        rng = np.random.default_rng(seed=42)  # воспроизводимость!
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs
        cw = np.exp(1j * 2 * np.pi * cfg.f0 * t).astype(np.complex64)
        S_ref = np.tile(cw, (cfg.n_ant, 1))

        # Добавить шум с нужным SNR
        signal_power = 1.0  # |exp(jx)|^2 = 1
        noise_power = signal_power / (10 ** (cfg.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)  # / 2 потому что I+Q
        noise = (rng.normal(0, noise_std, S_ref.shape) +
                 1j * rng.normal(0, noise_std, S_ref.shape)).astype(np.complex64)
        S_ref = (S_ref + noise).astype(np.complex64)

        W_ref = np.eye(cfg.n_ant, dtype=np.complex64)

        return SignalData(
            d_S=S_ref.ravel(),
            d_W=W_ref.ravel(),
            S_ref=S_ref,
            W_ref=W_ref,
            cfg=cfg,
            variant=SignalVariant.V2_CW_NOISE,
        )


class CwDelayedSignalSource(ISignalSource):
    """V3: CW с межантенной задержкой, W = delay_and_sum (без шума).

    Задержка k-й антенны: tau[k] = tau_step * k
    Сигнал: S[k,n] = exp(j*2π*f0*(n/fs - tau[k]))
    Весовая матрица W компенсирует эти задержки (beamforming).
    """

    def generate(self, cfg: SignalConfig) -> SignalData:
        # Задержки антенн
        tau = np.arange(cfg.n_ant, dtype=np.float32) * cfg.tau_step  # [n_ant]
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs       # [n_samples]

        # S[k,n] = exp(j*2π*f0*(t[n] - tau[k]))
        phase = 2 * np.pi * cfg.f0 * (t[np.newaxis, :] - tau[:, np.newaxis])
        S_ref = np.exp(1j * phase).astype(np.complex64)  # [n_ant, n_samples]

        # Весовая матрица delay_and_sum
        # W[b,k] = exp(-j*2π*f0*tau[k]) / sqrt(n_ant)
        w_row = np.exp(-1j * 2 * np.pi * cfg.f0 * tau) / np.sqrt(cfg.n_ant)
        # Каждый луч b имеет одинаковые веса
        W_ref = np.tile(w_row[np.newaxis, :], (cfg.n_ant, 1)).astype(np.complex64)

        return SignalData(
            d_S=S_ref.ravel(),
            d_W=W_ref.ravel(),
            S_ref=S_ref,
            W_ref=W_ref,
            cfg=cfg,
            variant=SignalVariant.V3_CW_PHASE_DELAY,
        )


class CwPhaseNoiseSignalSource(ISignalSource):
    """V4: V3 + AWGN шум. Полный реальный сценарий."""

    def generate(self, cfg: SignalConfig) -> SignalData:
        # Взять V3 за основу
        v3_source = CwDelayedSignalSource()
        data = v3_source.generate(cfg)

        # Добавить шум к S_ref
        rng = np.random.default_rng(seed=42)
        signal_power = 1.0
        noise_power = signal_power / (10 ** (cfg.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        noise = (rng.normal(0, noise_std, data.S_ref.shape) +
                 1j * rng.normal(0, noise_std, data.S_ref.shape)).astype(np.complex64)
        S_ref_noisy = (data.S_ref + noise).astype(np.complex64)

        return SignalData(
            d_S=S_ref_noisy.ravel(),
            d_W=data.d_W,
            S_ref=S_ref_noisy,
            W_ref=data.W_ref,
            cfg=cfg,
            variant=SignalVariant.V4_CW_PHASE_NOISE,
        )


class FileSignalSource(ISignalSource):
    """V5: Загрузка данных из файла → GPU. ЗАГЛУШКА для будущего.

    Когда появятся реальные тестовые данные — реализовать полностью.
    Сейчас: бросает SkipTest если файл не найден.

    Формат файла (планируемый):
        .npz с ключами "S" (complex64) и "W" (complex64)
    """

    def generate(self, cfg: SignalConfig) -> SignalData:
        from common.runner import SkipTest

        if not cfg.file_path:
            raise SkipTest("V5: file_path не задан (реальные данные ещё не готовы)")

        if not os.path.exists(cfg.file_path):
            raise SkipTest(f"V5: файл не найден: {cfg.file_path}")

        # Загрузить из .npz
        npz = np.load(cfg.file_path)
        S_ref = npz["S"].astype(np.complex64)
        W_ref = (npz["W"].astype(np.complex64) if "W" in npz else
                 np.eye(S_ref.shape[0], dtype=np.complex64))

        return SignalData(
            d_S=S_ref.ravel(),
            d_W=W_ref.ravel(),
            S_ref=S_ref,
            W_ref=W_ref,
            cfg=cfg,
            variant=SignalVariant.V5_FROM_FILE,
        )


class SignalSourceFactory:
    """Factory Method (GoF): создаёт ISignalSource по SignalVariant.

    Creator (GRASP): отвечает за создание правильного источника сигнала.

    Usage:
        source = SignalSourceFactory.create(SignalVariant.V3_CW_PHASE_DELAY)
        data = source.generate(cfg)
    """

    _registry: dict = {
        SignalVariant.V1_CW_CLEAN       : CwCleanSignalSource,
        SignalVariant.V2_CW_NOISE       : CwNoiseSignalSource,
        SignalVariant.V3_CW_PHASE_DELAY : CwDelayedSignalSource,
        SignalVariant.V4_CW_PHASE_NOISE : CwPhaseNoiseSignalSource,
        SignalVariant.V5_FROM_FILE      : FileSignalSource,
    }

    @classmethod
    def create(cls, variant: SignalVariant) -> ISignalSource:
        """Создать источник сигнала по варианту."""
        if variant not in cls._registry:
            raise ValueError(f"Неизвестный вариант: {variant}")
        return cls._registry[variant]()
