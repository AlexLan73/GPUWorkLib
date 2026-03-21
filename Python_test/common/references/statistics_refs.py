"""
NumPy-эталоны для статистических вычислений. GRASP Information Expert.
"""

import numpy as np


class StatisticsReferences:
    """
    NumPy-реализации GPU-статистики.

    Сценарий валидации:
        GPU генерирует данные -> копия на CPU -> ОДНИ И ТЕ ЖЕ данные
        обрабатываются GPU StatisticsProcessor и этим классом.
        Сравниваем результаты (DataValidator).

    Complex данные:
        Для complex64 вычисляем статистику по МОЩНОСТИ: |x|^2.
        Убедись что GPU StatisticsProcessor делает так же!
        Если GPU считает по амплитуде |x| -- замени np.abs(data)**2 на np.abs(data).

    Использование:
        from common.references import StatisticsReferences as StatsRef

        ref_mean   = StatsRef.mean(data)        # shape=(n_channels,)
        ref_std    = StatsRef.std(data)
        ref_median = StatsRef.median(data)
    """

    @staticmethod
    def _to_real(data: np.ndarray) -> np.ndarray:
        """Для complex -> мощность |x|^2, иначе как есть."""
        if np.iscomplexobj(data):
            return np.abs(data) ** 2
        return data

    @staticmethod
    def mean(data: np.ndarray) -> np.ndarray:
        """
        Среднее по столбцам (вдоль оси samples).

        Args:
            data: shape=(n_channels, n_samples) или (n_samples,)

        Returns: float32, shape=(n_channels,) или scalar
        """
        real = StatisticsReferences._to_real(data)
        if data.ndim == 1:
            return np.float32(np.mean(real))
        return np.mean(real, axis=1).astype(np.float32)

    @staticmethod
    def std(data: np.ndarray) -> np.ndarray:
        """Стандартное отклонение по каналам (Welford-совместимо)."""
        real = StatisticsReferences._to_real(data)
        if data.ndim == 1:
            return np.float32(np.std(real))
        return np.std(real, axis=1).astype(np.float32)

    @staticmethod
    def median(data: np.ndarray) -> np.ndarray:
        """Медиана по каналам."""
        real = StatisticsReferences._to_real(data)
        if data.ndim == 1:
            return np.float32(np.median(real))
        return np.median(real, axis=1).astype(np.float32)

    @staticmethod
    def mean_std_median(data: np.ndarray) -> dict:
        """Все три метрики в одном вызове."""
        return {
            "mean":   StatisticsReferences.mean(data),
            "std":    StatisticsReferences.std(data),
            "median": StatisticsReferences.median(data),
        }
