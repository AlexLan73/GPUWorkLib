# TASK_PythonArch_03 — common/references/

> **Фаза**: 1 (приоритет HIGH — устраняет дублирование немедленно)
> **Зависимости**: — (не зависит ни от чего)
> **Статус**: ⬜ TODO
> **Оценка**: ~1.5 часа
> **Паттерны**: GRASP Information Expert, DRY, SOLID SRP

---

## 🎯 Цель

Создать `Python_test/common/references/` — единое место для всех NumPy-эталонов.

**Проблема сейчас**:
- `signal_generators/conftest.py` — `cw_numpy()`, `lfm_numpy()`, `noise_numpy()`
- `heterodyne/conftest.py` — `make_lfm_srx()`, `make_s_ref()`
- `fft_func/conftest.py` — свои helpers
- Если формула меняется — нужно исправлять в **5+ местах**

**Решение**: Один класс = одна ответственность = один источник истины (DRY).

После задачи: `from common.references import SignalReferences` — везде.

---

## 📁 Создаваемые файлы (5 штук)

```
Python_test/common/references/
├── __init__.py           ← 1. пакет + export
├── signal_refs.py        ← 2. SignalReferences (CW, LFM, Noise)
├── filter_refs.py        ← 3. FilterReferences (FIR, IIR через SciPy)
├── statistics_refs.py    ← 4. StatisticsReferences (mean, std, median)
└── fft_refs.py           ← 5. FftReferences (FFT, magnitude, phase)
```

---

## 📝 Детальное ТЗ для каждого файла

### 2. `common/references/signal_refs.py` — SignalReferences

```python
"""
NumPy-эталоны для сигналов. Единая точка истины (DRY).
GRASP Information Expert: знает все формулы генерации сигналов.
"""

import numpy as np
from math import pi

class SignalReferences:
    """
    Статические NumPy-реализации GPU-генераторов.

    Использование:
        from common.references import SignalReferences

        ref_cw  = SignalReferences.cw(fs=12e6, n_samples=4096, f0=2e6)
        ref_lfm = SignalReferences.lfm(fs=12e6, n_samples=4096, f_start=0, f_end=2e6)
    """

    @staticmethod
    def cw(fs: float, n_samples: int, f0: float,
           amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        CW сигнал (непрерывная синусоида).

        Returns: complex64, shape=(n_samples,)
        """
        t = np.arange(n_samples) / fs
        return (amplitude * np.exp(1j * (2*pi*f0*t + phase))).astype(np.complex64)

    @staticmethod
    def lfm(fs: float, n_samples: int, f_start: float, f_end: float,
            amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        """
        ЛЧМ (линейная частотная модуляция).

        Returns: complex64, shape=(n_samples,)
        """
        t = np.arange(n_samples) / fs
        duration = n_samples / fs
        rate = (f_end - f_start) / duration
        phi = 2*pi * (f_start*t + 0.5*rate*t**2) + phase
        return (amplitude * np.exp(1j * phi)).astype(np.complex64)

    @staticmethod
    def lfm_with_delay(fs: float, n_samples: int, f_start: float, f_end: float,
                       delay_s: float, amplitude: float = 1.0) -> np.ndarray:
        """
        ЛЧМ с задержкой (для тестов гетеродина/дечирпа).

        Returns: complex64, shape=(n_samples,)
        Сигнал начинается с t=delay_s, до этого — нули.
        """
        t = np.arange(n_samples) / fs
        duration = n_samples / fs
        rate = (f_end - f_start) / duration
        result = np.zeros(n_samples, dtype=np.complex64)
        mask = t >= delay_s
        t_local = t[mask] - delay_s
        phi = 2*pi * (f_start*t_local + 0.5*rate*t_local**2)
        result[mask] = (amplitude * np.exp(1j * phi)).astype(np.complex64)
        return result

    @staticmethod
    def lfm_multi_antenna(fs: float, n_samples: int, f_start: float, f_end: float,
                          delays_s: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Несколько ЛЧМ с разными задержками (массив антенн).

        Args:
            delays_s: задержки по каждой антенне, shape=(n_antennas,)

        Returns: complex64, shape=(n_antennas, n_samples)
        """
        n_ant = len(delays_s)
        result = np.zeros((n_ant, n_samples), dtype=np.complex64)
        for i, tau in enumerate(delays_s):
            result[i] = SignalReferences.lfm_with_delay(
                fs, n_samples, f_start, f_end, tau, amplitude
            )
        return result

    @staticmethod
    def noise(n_samples: int, seed: int = 42, amplitude: float = 1.0) -> np.ndarray:
        """
        Гауссов шум (воспроизводимый через seed).

        ⚠️ GPU (Philox PRNG) и NumPy (PCG64) дают РАЗНЫЕ числа при одном seed!
        Этот метод полезен ТОЛЬКО для чисто Python-тестов без GPU.

        Для валидации GPU statistics:
            1. GPU генерирует сигнал+шум → копия на CPU (ndarray)
            2. GPU StatisticsProcessor считает stats из этих данных
            3. NumPy считает stats из ТЕХ ЖЕ скопированных данных
            4. Сравниваем GPU stats vs NumPy stats (DataValidator)

        Returns: complex64, shape=(n_samples,)
        """
        rng = np.random.default_rng(seed)
        sig = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        return (sig * amplitude / 2**0.5).astype(np.complex64)

    @staticmethod
    def form_signal(fs: float, points: int, f0: float, amplitude: float,
                    phase: float, fdev: float, norm_val: float,
                    tau: float = 0.0) -> np.ndarray:
        """
        CPU reference FormSignal (формула getX без шума).

        Воспроизводит GPU FormSignalGenerator: окно + центрированная фаза.

        Args:
            fs:        частота дискретизации (Гц)
            points:    число отсчётов
            f0:        несущая частота (Гц)
            amplitude: амплитуда
            phase:     начальная фаза (рад)
            fdev:      девиация частоты (Гц)
            norm_val:  нормировочный коэффициент
            tau:       задержка (с), default=0.0

        Returns: complex64, shape=(points,)
        """
        dt = 1.0 / fs
        ti = points * dt
        t = np.arange(points, dtype=np.float64) * dt + tau
        in_window = (t >= 0.0) & (t <= ti - dt)
        t_centered = t - ti / 2.0
        ph = 2.0 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase
        X = amplitude * norm_val * np.exp(1j * ph)
        X[~in_window] = 0.0
        return X.astype(np.complex64)

    @staticmethod
    def dechirp(s_rx: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
        """
        Дечирп: s_dc = s_rx * conj(s_ref).
        NumPy-эталон для HeterodyneAdapter.

        Returns: complex64, shape=s_rx.shape
        """
        return (s_rx * np.conj(s_ref)).astype(np.complex64)
```

---

### 3. `common/references/filter_refs.py` — FilterReferences

```python
"""NumPy/SciPy-эталоны для фильтров. GRASP Information Expert."""

import numpy as np
try:
    from scipy import signal as scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

class FilterReferences:
    """
    SciPy-реализации GPU-фильтров.

    Использование:
        from common.references import FilterReferences

        ref = FilterReferences.fir_lowpass(data, fs=50e3, cutoff_hz=1e3, n_taps=64)
    """

    @staticmethod
    def _check_scipy():
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy требуется для FilterReferences. pip install scipy")

    @staticmethod
    def fir_lowpass(data: np.ndarray, fs: float, cutoff_hz: float,
                    n_taps: int = 64, window: str = "hamming") -> np.ndarray:
        """
        FIR lowpass фильтр через scipy.

        Returns: complex64, shape=data.shape
        """
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        coeffs = scipy_signal.firwin(n_taps, cutoff_hz / nyq, window=window)
        result = scipy_signal.lfilter(coeffs, [1.0], data)
        return result.astype(np.complex64)

    @staticmethod
    def fir_bandpass(data: np.ndarray, fs: float,
                     f_low: float, f_high: float,
                     n_taps: int = 64) -> np.ndarray:
        """FIR bandpass фильтр через scipy."""
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        coeffs = scipy_signal.firwin(
            n_taps, [f_low / nyq, f_high / nyq],
            pass_zero=False
        )
        return scipy_signal.lfilter(coeffs, [1.0], data).astype(np.complex64)

    @staticmethod
    def fir_from_coeffs(data: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """FIR с произвольными коэффициентами."""
        FilterReferences._check_scipy()
        return scipy_signal.lfilter(coeffs, [1.0], data).astype(np.complex64)

    @staticmethod
    def iir_lowpass(data: np.ndarray, fs: float, cutoff_hz: float,
                    order: int = 4, ftype: str = "butter") -> np.ndarray:
        """
        IIR lowpass фильтр через scipy.sosfilt (численно стабильный).
        """
        FilterReferences._check_scipy()
        nyq = fs / 2.0
        sos = scipy_signal.iirfilter(
            order, cutoff_hz / nyq,
            btype="low", ftype=ftype, output="sos"
        )
        return scipy_signal.sosfilt(sos, data).astype(np.complex64)
```

---

### 4. `common/references/statistics_refs.py` — StatisticsReferences

```python
"""NumPy-эталоны для статистических вычислений. GRASP Information Expert."""

import numpy as np

class StatisticsReferences:
    """
    NumPy-реализации GPU-статистики.

    ⚠️ Сценарий валидации:
        GPU генерирует данные → копия на CPU → ОДНИ И ТЕ ЖЕ данные
        обрабатываются GPU StatisticsProcessor и этим классом.
        Сравниваем результаты (DataValidator).

    ⚠️ Complex данные:
        Для complex64 вычисляем статистику по МОЩНОСТИ: |x|².
        Убедись что GPU StatisticsProcessor делает так же!
        Если GPU считает по амплитуде |x| — замени np.abs(data)**2 на np.abs(data).

    Использование:
        from common.references import StatisticsReferences as StatsRef

        ref_mean   = StatsRef.mean(data)        # shape=(n_channels,)
        ref_std    = StatsRef.std(data)
        ref_median = StatsRef.median(data)
    """

    @staticmethod
    def mean(data: np.ndarray) -> np.ndarray:
        """
        Среднее по столбцам (вдоль оси samples).

        Args:
            data: shape=(n_channels, n_samples) или (n_samples,)

        Returns: float32, shape=(n_channels,) или scalar
        """
        if data.ndim == 1:
            return np.float32(np.mean(np.abs(data)**2 if np.iscomplexobj(data) else data))
        return np.mean(
            np.abs(data)**2 if np.iscomplexobj(data) else data,
            axis=1
        ).astype(np.float32)

    @staticmethod
    def std(data: np.ndarray) -> np.ndarray:
        """Стандартное отклонение по каналам (Welford-совместимо)."""
        if data.ndim == 1:
            return np.float32(np.std(np.abs(data)**2 if np.iscomplexobj(data) else data))
        return np.std(
            np.abs(data)**2 if np.iscomplexobj(data) else data,
            axis=1
        ).astype(np.float32)

    @staticmethod
    def median(data: np.ndarray) -> np.ndarray:
        """Медиана по каналам."""
        if data.ndim == 1:
            arr = np.abs(data)**2 if np.iscomplexobj(data) else data
            return np.float32(np.median(arr))
        arr = np.abs(data)**2 if np.iscomplexobj(data) else data
        return np.median(arr, axis=1).astype(np.float32)

    @staticmethod
    def mean_std_median(data: np.ndarray) -> dict:
        """Все три метрики в одном вызове."""
        return {
            "mean":   StatisticsReferences.mean(data),
            "std":    StatisticsReferences.std(data),
            "median": StatisticsReferences.median(data),
        }
```

---

### 5. `common/references/fft_refs.py` — FftReferences

```python
"""NumPy FFT-эталоны. GRASP Information Expert."""

import numpy as np

class FftReferences:
    """
    NumPy-реализации GPU-FFT.

    Использование:
        from common.references import FftReferences

        spec     = FftReferences.fft(data)
        mag      = FftReferences.magnitude(data)
        peak_hz  = FftReferences.peak_freq(data, fs=12e6)
    """

    @staticmethod
    def fft(data: np.ndarray, n_fft: int | None = None) -> np.ndarray:
        """
        FFT через numpy.

        Returns: complex64, shape=(n_fft,) или (n_channels, n_fft)
        """
        if data.ndim == 1:
            return np.fft.fft(data, n=n_fft).astype(np.complex64)
        return np.fft.fft(data, n=n_fft, axis=-1).astype(np.complex64)

    @staticmethod
    def magnitude(data: np.ndarray, n_fft: int | None = None) -> np.ndarray:
        """
        |FFT| — амплитудный спектр.

        Returns: float32, shape=(n_fft,) или (n_channels, n_fft)
        """
        return np.abs(FftReferences.fft(data, n_fft)).astype(np.float32)

    @staticmethod
    def magnitude_db(data: np.ndarray, n_fft: int | None = None,
                     ref: float = 1.0) -> np.ndarray:
        """Амплитудный спектр в дБ."""
        mag = FftReferences.magnitude(data, n_fft)
        return (20 * np.log10(mag / ref + 1e-12)).astype(np.float32)

    @staticmethod
    def peak_freq(data: np.ndarray, fs: float,
                  n_fft: int | None = None) -> float:
        """
        Частота пика спектра (Гц).

        Returns: float — частота максимального бина
        """
        mag = FftReferences.magnitude(data, n_fft)
        n = mag.shape[-1]
        freqs = np.fft.fftfreq(n, d=1.0/fs)
        peak_idx = int(np.argmax(mag))
        return float(freqs[peak_idx])

    @staticmethod
    def freq_axis(n_fft: int, fs: float) -> np.ndarray:
        """Ось частот (Гц) для построения графиков."""
        return np.fft.fftfreq(n_fft, d=1.0/fs).astype(np.float32)
```

---

### 1. `common/references/__init__.py`

```python
"""
NumPy/SciPy эталоны — единое место для CPU-реализаций GPU-алгоритмов.
DRY: каждая формула живёт в одном месте.

Использование:
    from common.references import SignalReferences, FilterReferences
    from common.references import StatisticsReferences, FftReferences
"""

from .signal_refs import SignalReferences
from .filter_refs import FilterReferences
from .statistics_refs import StatisticsReferences
from .fft_refs import FftReferences

__all__ = [
    "SignalReferences",
    "FilterReferences",
    "StatisticsReferences",
    "FftReferences",
]
```

---

## 📋 Что обновить после создания файлов

### `common/__init__.py` — добавить строки:
```python
# Референсные реализации (NumPy/SciPy эталоны)
from .references import SignalReferences, FilterReferences
from .references import StatisticsReferences, FftReferences
```

### `signal_generators/conftest.py` — ЗАМЕНИТЬ:
```python
# БЫЛО (удалить):
def cw_numpy(fs, length, f0, amplitude=1.0, phase=0.0):
    ...
def lfm_numpy(fs, length, f_start, f_end, amplitude=1.0):
    ...

# СТАЛО (добавить):
from common.references import SignalReferences
# Псевдонимы для backward compat (пока все тесты не обновлены):
cw_numpy  = SignalReferences.cw
lfm_numpy = SignalReferences.lfm
```

### `heterodyne/conftest.py` — ЗАМЕНИТЬ:
```python
# БЫЛО:
def make_lfm_srx(params, delays_us) -> np.ndarray:
    ...

# СТАЛО:
from common.references import SignalReferences
def make_lfm_srx(params, delays_us):
    delays_s = np.array(delays_us) * 1e-6
    return SignalReferences.lfm_multi_antenna(
        params.fs, params.n_samples,
        params.f_start, params.f_end, delays_s
    )
```

---

## 📋 Маппинг полей: LfmParams → SignalConfig

```
СТАРОЕ (conftest.py)         НОВОЕ (SignalConfig)
──────────────────           ──────────────────
LfmParams.length       →    SignalConfig.n_samples
LfmParams.f_start      →    SignalConfig.f0_hz
LfmParams.f_end        →    SignalConfig.f0_hz + SignalConfig.fdev_hz
LfmParams.amplitude    →    SignalConfig.amplitude
LfmParams.phase        →    (нет в SignalConfig, default=0.0 в генераторе)
```

При реализации Arch-03: заменить `LfmParams` на `SignalConfig` в conftest.py, обновить вызовы.

---

## ✅ Критерии завершения

- [ ] Все 5 файлов созданы
- [ ] `from common.references import SignalReferences` работает
- [ ] `SignalReferences.cw(12e6, 4096, 2e6)` → ndarray complex64 shape=(4096,)
- [ ] `SignalReferences.lfm(12e6, 4096, 0, 2e6)` → ndarray complex64
- [ ] `SignalReferences.lfm_with_delay(12e6, 4096, 0, 2e6, 1e-4)` — нули до delay
- [ ] `SignalReferences.lfm_multi_antenna(...)` → shape=(n_ant, n_samples)
- [ ] `SignalReferences.form_signal(...)` → ndarray complex64 (FormSignal эталон)
- [ ] `FilterReferences.fir_lowpass(data, 50e3, 1e3)` → работает (scipy)
- [ ] `StatisticsReferences.mean_std_median(data)` → dict с 3 ключами
- [ ] `FftReferences.peak_freq(cw_signal, 12e6)` ≈ `2e6` (±freq_resolution)
- [ ] common/__init__.py обновлён

## 🧪 Быстрая проверка без GPU

```python
# python Python_test/common/references/test_references_smoke.py
import numpy as np
from common.runner import TestRunner
from common.result import TestResult, ValidationResult

class TestReferencesSmoke:
    """Smoke-тест references — работает без GPU (чистый NumPy)."""

    def test_signal_refs(self):
        from common.references import SignalReferences
        result = TestResult(test_name="signal_refs")

        cw = SignalReferences.cw(12e6, 4096, 2e6)
        result.add(ValidationResult(
            passed=cw.dtype == np.complex64 and cw.shape == (4096,),
            metric_name="cw_shape_dtype",
            actual_value=1.0 if cw.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={cw.dtype}, shape={cw.shape}"
        ))

        lfm = SignalReferences.lfm(12e6, 4096, 0.0, 2e6)
        result.add(ValidationResult(
            passed=lfm.dtype == np.complex64,
            metric_name="lfm_dtype",
            actual_value=1.0 if lfm.dtype == np.complex64 else 0.0,
            threshold=1.0
        ))

        form = SignalReferences.form_signal(12e6, 4096, 2e6, 1.0, 0.0, 1e6, 1.0)
        result.add(ValidationResult(
            passed=form.dtype == np.complex64 and form.shape == (4096,),
            metric_name="form_signal_shape",
            actual_value=1.0 if form.shape == (4096,) else 0.0,
            threshold=1.0
        ))
        return result

    def test_fft_refs(self):
        from common.references import SignalReferences, FftReferences
        result = TestResult(test_name="fft_refs")

        cw = SignalReferences.cw(12e6, 4096, 2e6)
        peak = FftReferences.peak_freq(cw, 12e6)
        err = abs(peak - 2e6)
        result.add(ValidationResult(
            passed=err < 12e6 / 4096,
            metric_name="peak_freq_hz",
            actual_value=err,
            threshold=12e6 / 4096,
            message=f"peak={peak:.1f} Hz, expected=2e6 Hz"
        ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestReferencesSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 1*
