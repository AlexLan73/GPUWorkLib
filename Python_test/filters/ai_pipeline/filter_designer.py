"""
filter_designer.py — FilterDesigner (SRP: только дизайн фильтра через scipy)
=============================================================================

Single Responsibility (SOLID):
  FilterDesigner отвечает ТОЛЬКО за вычисление коэффициентов фильтра.
  Ни GPU, ни графики, ни AI — только scipy.signal.

Information Expert (GRASP):
  FilterDesigner знает как сделать FIR и IIR фильтры.

Usage:
    from filters.ai_pipeline.llm_parser import FilterSpec, MockParser
    from filters.ai_pipeline.filter_designer import FilterDesigner

    spec = MockParser().parse("FIR lowpass 1kHz, fs=50kHz", fs=50_000)
    designer = FilterDesigner()
    design = designer.design(spec)

    print(design.coeffs_b)          # FIR: коэффициенты
    print(design.coeffs_b, design.coeffs_a)  # IIR: числитель/знаменатель
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .llm_parser import FilterSpec


@dataclass
class FilterDesign:
    """Результат дизайна фильтра — коэффициенты и метаданные.

    Attributes:
        spec:      исходная спецификация
        coeffs_b:  коэффициенты числителя (FIR: полный фильтр, IIR: b)
        coeffs_a:  коэффициенты знаменателя (FIR: [1.0], IIR: a)
        n_taps:    число отводов FIR или порядок IIR
        method:    метод расчёта ("firwin" / "butter" / "cheby1" / "ellip")
        sos:       SOS-форма для IIR (устойчивее чем b/a для высоких порядков)
    """
    spec: FilterSpec
    coeffs_b: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    coeffs_a: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    n_taps: int = 0
    method: str = ""
    sos: Optional[np.ndarray] = None

    @property
    def is_fir(self) -> bool:
        return self.spec.filter_class == "fir"

    def apply_scipy(self, signal: np.ndarray) -> np.ndarray:
        """Применить фильтр к сигналу через scipy (для сравнения с GPU).

        Args:
            signal: входной сигнал [n_samples] или [n_channels, n_samples]

        Returns:
            Отфильтрованный сигнал той же формы
        """
        try:
            import scipy.signal as ss
        except ImportError:
            raise ImportError("pip install scipy")

        def _apply_1d(x: np.ndarray) -> np.ndarray:
            if self.sos is not None:
                return ss.sosfilt(self.sos, x).astype(x.dtype)
            return ss.lfilter(self.coeffs_b, self.coeffs_a, x).astype(x.dtype)

        if signal.ndim == 1:
            return _apply_1d(signal)

        # multi-channel: apply per row
        out = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            out[ch] = _apply_1d(signal[ch])
        return out


class FilterDesigner:
    """Вычисляет коэффициенты фильтра из FilterSpec через scipy.

    Поддерживает:
      FIR: firwin (window method) + firwin2 (arbitrary frequency response)
      IIR: Butterworth, Chebyshev I/II, Elliptic
    """

    # Минимальное число отводов FIR (если order=0, вычисляется автоматически)
    _MIN_FIR_TAPS = 32
    _DEFAULT_FIR_TAPS = 64

    def design(self, spec: FilterSpec) -> FilterDesign:
        """Рассчитать фильтр по спецификации.

        Args:
            spec: FilterSpec (от MockParser / GroqParser / ...)

        Returns:
            FilterDesign с коэффициентами
        """
        if spec.filter_class == "fir":
            return self._design_fir(spec)
        return self._design_iir(spec)

    def _design_fir(self, spec: FilterSpec) -> FilterDesign:
        """Рассчитать FIR через scipy.signal.firwin."""
        try:
            import scipy.signal as ss
        except ImportError:
            raise ImportError("pip install scipy")

        n_taps = spec.order if spec.order > 0 else self._DEFAULT_FIR_TAPS
        # firwin требует нечётное число отводов для некоторых типов
        if spec.filter_type in ("highpass", "bandstop") and n_taps % 2 == 0:
            n_taps += 1

        cutoff = spec.normalized_cutoff()
        window = (spec.window, spec.ripple_db) if spec.window == "kaiser" else spec.window

        b = ss.firwin(
            n_taps, cutoff,
            window=window,
            pass_zero=(spec.filter_type in ("lowpass", "bandstop")),
        ).astype(np.float64)

        return FilterDesign(
            spec=spec,
            coeffs_b=b,
            coeffs_a=np.array([1.0]),
            n_taps=n_taps,
            method="firwin",
        )

    def _design_iir(self, spec: FilterSpec) -> FilterDesign:
        """Рассчитать IIR Butterworth через scipy.signal.butter.

        Использует SOS-форму для численной устойчивости.
        """
        try:
            import scipy.signal as ss
        except ImportError:
            raise ImportError("pip install scipy")

        order = max(1, spec.order)
        cutoff = spec.normalized_cutoff()

        btype_map = {
            "lowpass": "lowpass",
            "highpass": "highpass",
            "bandpass": "bandpass",
            "bandstop": "bandstop",
        }
        btype = btype_map.get(spec.filter_type, "lowpass")

        sos = ss.butter(order, cutoff, btype=btype, output="sos")
        b, a = ss.butter(order, cutoff, btype=btype, output="ba")

        return FilterDesign(
            spec=spec,
            coeffs_b=b.astype(np.float64),
            coeffs_a=a.astype(np.float64),
            n_taps=order,
            method="butter",
            sos=sos.astype(np.float64),
        )
