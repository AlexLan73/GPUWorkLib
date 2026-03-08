# 📝 Python_test — Архитектурный план рефакторинга

> **Статус**: 🟡 ПЛАН (не реализован)
> **Приоритет**: P1 — после завершения текущих GPU модулей
> **Автор**: Alex + Кодо
> **Создано**: 2026-03-08
> **Паттерны**: OOP, SOLID, GRASP, GoF

---

## 🔍 Диагностика текущего состояния

### Метрики ДО (проблемы)

| Метрика | Значение | Статус |
|---------|----------|--------|
| Всего строк | 15 807 в 36 файлах | |
| Файлов > 600 строк | 6 | 🔴 |
| Функций > 100 строк | 5 | 🔴 |
| Дублирование кода | ~25% | 🔴 |
| Глобальных переменных | 50+ | 🔴 |
| Copy-paste паттернов | 12+ | 🔴 |
| Тесты без matplotlib | 0% | 🔴 |
| Единый путь к .so | нет | 🔴 |

### Файлы-монстры

| Файл | Строк | Главная проблема |
|------|-------|-----------------|
| `filters/test_ai_filter_pipeline.py` | 964 | 6 разных ответственностей в одном файле |
| `integration/test_gpuworklib.py` | 903 | 9 несвязанных тестов, функции >199 строк |
| `filters/plot_report_filters.py` | 821 | 3 функции по 190+ строк (God Functions) |
| `signal_generators/test_form_signal.py` | 722 | make_plots() = 369 строк |
| `heterodyne/test_heterodyne_step_by_step.py` | 680 | 4 "шага" слабо связаны |
| `strategies/pipeline_runner.py` | 626 | дублирование run_pipeline_a/b ~50% |

### Корневые причины

```
1. НАРУШЕНИЕ SRP
   test_ai_filter_pipeline.py делает:
     → парсинг LLM ответа (JSON + regex)
     → дизайн FIR/IIR фильтра (scipy)
     → генерацию сигнала на GPU
     → применение фильтра на GPU
     → валидацию GPU vs scipy
     → matplotlib визуализацию (4 панели)
     → pytest-тесты
   Это 7 разных ответственностей в одном файле!

2. ДУБЛИРОВАНИЕ (DRY нарушен)
   Паттерн повторяется 15+ раз:
     ctx = gpuworklib.GPUContext(0)
     gen = gpuworklib.SignalGenerator(ctx)
     signal = gen.generate_cw(...)
     result = filter.process(signal)
     # ... validate ...

3. НЕТ ЕДИНОГО ЗАГРУЗЧИКА .so
   Каждый файл ищет gpuworklib по-своему:
   - test_filters_stage1.py → build/python/Release|Debug
   - test_fir_filter_rocm.py → build/debian-radeon9070/python (хардкод!)
   - test_ai_filter_pipeline.py → 6 вариантов через цикл

4. I/O ПЕРЕМЕШАН С ЛОГИКОЙ
   - GROQ_API_KEY читается на уровне МОДУЛЯ при импорте
   - plt.show() прямо в тестах
   - os.makedirs() прямо в assert-функциях
```

---

## 🏗️ Новая архитектура

### Структура файлов (ПОСЛЕ)

```
Python_test/
│
├── conftest.py                          ← ROOT: GPUContextManager fixture
│
├── common/                              ← НОВЫЙ: общая инфраструктура
│   ├── __init__.py
│   ├── gpu_loader.py                    ← GPULoader (Singleton)
│   ├── gpu_context.py                   ← GPUContextManager (Singleton)
│   ├── test_base.py                     ← TestBase (Template Method)
│   ├── configs.py                       ← SignalConfig, FilterConfig... (dataclasses)
│   ├── validators.py                    ← IValidator + 3 реализации (Strategy)
│   ├── reporters.py                     ← IReporter + 3 реализации (Observer)
│   ├── result.py                        ← TestResult, ValidationResult (Value Objects)
│   └── plotting/
│       ├── __init__.py
│       ├── plotter_base.py              ← IPlotter (ABC)
│       ├── spectrum_plotter.py          ← SpectrumPlotter
│       ├── time_plotter.py              ← TimeDomainPlotter
│       └── comparison_plotter.py       ← ComparisonPlotter
│
├── filters/                             ← рефакторинг: 10 файлов → структура
│   ├── conftest.py                      ← filter fixtures: ctx, signal_factory
│   ├── configs.py                       ← FilterConfig, FilterTestParams
│   ├── filter_test_base.py              ← FilterTestBase(TestBase)
│   ├── test_fir.py                      ← только FIR тесты (< 150 строк)
│   ├── test_iir.py                      ← только IIR тесты (< 150 строк)
│   ├── test_moving_average.py           ← SMA/EMA/DEMA (< 200 строк)
│   ├── test_kalman.py                   ← Kalman ROCm (< 150 строк)
│   ├── test_kaufman.py                  ← Kaufman ROCm (< 150 строк)
│   ├── ai_pipeline/                     ← разбитый test_ai_filter_pipeline.py
│   │   ├── llm_parser.py                ← LLMParser (Strategy ABC)
│   │   ├── filter_designer.py           ← FilterDesigner
│   │   ├── filter_validator.py          ← FilterValidator(IValidator)
│   │   ├── test_ai_pipeline.py          ← только pytest (< 150 строк)
│   │   └── demo_ai_pipeline.py          ← демо с графиками (отдельно)
│   └── visualization/                   ← разбитый plot_report_filters.py
│       ├── filter_plotter.py            ← FilterPlotter (< 200 строк)
│       └── report_plotter.py            ← ReportPlotter (< 200 строк)
│
├── signal_generators/                   ← рефакторинг монолитов
│   ├── conftest.py
│   ├── configs.py                       ← SignalGeneratorConfig
│   ├── signal_test_base.py              ← SignalTestBase(TestBase)
│   ├── test_cw.py                       ← только CW (< 150 строк)
│   ├── test_lfm.py                      ← только LFM (< 150 строк)
│   ├── test_form_signal.py              ← только form signal тесты (< 200 строк)
│   └── visualization/
│       └── signal_plotter.py            ← SignalPlotter (< 200 строк)
│
├── heterodyne/                          ← разбивка step_by_step (680 строк)
│   ├── conftest.py
│   ├── configs.py                       ← HeterodyneConfig
│   ├── test_dechirp.py                  ← тесты дечирпа (< 200 строк)
│   ├── test_comparison.py               ← сравнение методов (< 200 строк)
│   ├── test_rocm.py                     ← ROCm тесты (< 200 строк)
│   └── visualization/
│       └── heterodyne_plotter.py
│
├── strategies/                          ← уже хорошо! минимальные правки
│   ├── conftest.py                      ← НОВЫЙ: pytest fixtures
│   ├── scenario_builder.py              ← OK, без изменений
│   ├── farrow_delay.py                  ← OK, без изменений
│   ├── pipeline_runner.py               ← рефакторинг → PipelineBase
│   ├── test_scenario_builder.py         ← OK
│   └── test_farrow_pipeline.py          ← OK
│
├── integration/                         ← разбивка test_gpuworklib.py (903 строки)
│   ├── conftest.py
│   ├── test_fft_integration.py          ← тесты 1-3 (< 250 строк)
│   ├── test_signal_gen_integration.py   ← тесты 4-7 (< 250 строк)
│   └── test_script_generator.py         ← тесты 8-9 (< 250 строк)
│
├── fft_maxima/                          ← почти OK, только conftest
│   ├── conftest.py                      ← НОВЫЙ
│   └── ...                             ← остальные файлы без изменений
│
└── lch_farrow/ statistics/ vector_algebra/ hybrid/ zero_copy/ fm_correlator/
    └── conftest.py (НОВЫЙ в каждом)    ← только добавить conftest
```

### Целевые метрики (ПОСЛЕ)

| Метрика | ДО | ПОСЛЕ |
|---------|-----|-------|
| Макс. строк в файле | 964 | ≤ 300 |
| Дублирование кода | ~25% | < 5% |
| Функций > 100 строк | 5 | 0 |
| Функций > 50 строк | 40+ | < 10 |
| Глобальных переменных | 50+ | 0 |
| Файлов без conftest | 90% | 0% |
| Тесты без matplotlib | 0% | 100% |
| Путей к .so | N файлов | 1 (GPULoader) |

---

## 🔧 Ключевые классы и интерфейсы

### 1. GPULoader — Singleton (GoF) + Protected Variations (GRASP)

**Проблема**: каждый из 36 файлов ищет `.so` по-своему, разными путями.
**Решение**: один класс, один список путей.

```python
# common/gpu_loader.py
import sys
from pathlib import Path
from typing import Optional

class GPULoader:
    """Singleton — находит gpuworklib.so один раз для всей pytest сессии.

    GoF: Singleton
    GRASP: Protected Variations (скрывает детали поиска .so)
    """
    _instance: Optional['GPULoader'] = None
    _gpuworklib = None

    @classmethod
    def get(cls):
        """Вернуть загруженный модуль gpuworklib."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._gpuworklib

    def __init__(self):
        search_paths = [
            "build/python/Release",
            "build/python/Debug",
            "build/debian-radeon9070/python",
            "build/Release",
        ]
        root = Path(__file__).parents[2]   # e:\C++\GPUWorkLib
        for rel in search_paths:
            candidate = root / rel
            if candidate.exists():
                sys.path.insert(0, str(candidate))
                break
        try:
            import gpuworklib as _gw
            GPULoader._gpuworklib = _gw
        except ImportError as e:
            raise RuntimeError(
                f"gpuworklib не найден. Проверь сборку. Искали в: "
                f"{[str(root/p) for p in search_paths]}"
            ) from e
```

### 2. GPUContextManager — Singleton (GoF)

```python
# common/gpu_context.py

class GPUContextManager:
    """Singleton — один GPU контекст на pytest сессию.

    GoF: Singleton
    GRASP: Creator (создаёт ctx один раз)
    """
    _ctx = None

    @classmethod
    def get(cls, device_id: int = 0):
        if cls._ctx is None:
            gw = GPULoader.get()
            cls._ctx = gw.GPUContext(device_id)
        return cls._ctx

    @classmethod
    def reset(cls):
        """Для тестов — сбросить контекст."""
        cls._ctx = None
```

### 3. Config Dataclasses — Information Expert (GRASP)

**Проблема**: `fs = 12e6`, `n_samples = 4096` разбросаны как магические числа.
**Решение**: конфиг-классы, которые сами умеют вычислять производные.

```python
# common/configs.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SignalConfig:
    """Параметры входного сигнала.

    GRASP Information Expert — сам вычисляет производные (duration, freq_res).
    """
    fs: float = 12e6
    n_samples: int = 4096
    f0_hz: float = 2e6
    fdev_hz: float = 0.0       # 0 = CW
    amplitude: float = 1.0
    noise_sigma: float = 0.0

    def duration_ms(self) -> float:
        return self.n_samples / self.fs * 1e3

    def freq_resolution_hz(self, nfft: Optional[int] = None) -> float:
        n = nfft or self.n_samples
        return self.fs / n

    def nyquist_hz(self) -> float:
        return self.fs / 2.0


@dataclass
class FilterConfig:
    """Параметры фильтра.

    GRASP Information Expert.
    """
    filter_type: str             # "fir" | "iir" | "kalman" | "kaufman" | "moving_avg"
    cutoff_hz: float
    fs: float
    order: int = 4
    backend: str = "rocm"        # "rocm" | "opencl"
    tolerance: float = 0.01      # для валидации GPU vs scipy

    def cutoff_normalized(self) -> float:
        """Нормированная частота среза (0..1) для scipy."""
        return self.cutoff_hz / (self.fs / 2.0)


@dataclass
class HeterodyneConfig:
    fs: float = 50e6
    f_start_hz: float = 10e6
    f_end_hz: float = 20e6
    n_samples: int = 8192
    n_channels: int = 1
```

### 4. TestResult и ValidationResult — Value Objects

```python
# common/result.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ValidationResult:
    """Результат одной проверки."""
    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    message: str = ""

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.metric_name}]: {self.actual_value:.6f} (порог: {self.threshold})"


@dataclass
class TestResult:
    """Агрегированный результат теста."""
    test_name: str
    validations: List[ValidationResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.validations)

    def add(self, v: ValidationResult) -> 'TestResult':
        self.validations.append(v)
        return self

    def summary(self) -> str:
        n_pass = sum(1 for v in self.validations if v.passed)
        return f"{self.test_name}: {n_pass}/{len(self.validations)} passed"
```

### 5. IValidator — Strategy (GoF) + Polymorphism (GRASP)

**Проблема**: валидация разбросана — где-то `max_err < 0.1`, где-то `ratio > 0.8`.
**Решение**: полиморфный интерфейс, разные стратегии проверки.

```python
# common/validators.py
from abc import ABC, abstractmethod
import numpy as np

class IValidator(ABC):
    """Strategy интерфейс для валидации.

    GoF: Strategy
    GRASP: Polymorphism (нет if/elif по типу проверки)
    """
    @abstractmethod
    def validate(self, actual: np.ndarray,
                 reference: np.ndarray) -> ValidationResult: ...


class NumericValidator(IValidator):
    """Проверяет: max |actual - ref| / max|ref| < tolerance."""

    def __init__(self, tolerance: float = 0.01, metric_name: str = "numeric_error"):
        self.tolerance = tolerance
        self.metric_name = metric_name

    def validate(self, actual, reference) -> ValidationResult:
        max_ref = np.max(np.abs(reference))
        if max_ref < 1e-12:
            max_ref = 1.0
        err = float(np.max(np.abs(actual - reference))) / max_ref
        return ValidationResult(
            passed=err < self.tolerance,
            metric_name=self.metric_name,
            actual_value=err,
            threshold=self.tolerance,
            message=f"relative_error={err:.4f}"
        )


class SpectralValidator(IValidator):
    """Проверяет пик в спектре: частота ±freq_tol, магнитуда ±mag_tol_db."""

    def __init__(self, target_freq_hz: float, fs: float,
                 freq_tol_hz: float = None, mag_tol_db: float = 3.0):
        self.target_freq = target_freq_hz
        self.fs = fs
        self.freq_tol = freq_tol_hz  # None = 2 * freq_resolution
        self.mag_tol_db = mag_tol_db

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray = None) -> ValidationResult:
        nfft = len(actual)
        freq_axis = np.fft.fftfreq(nfft, 1.0 / self.fs)
        half = nfft // 2
        mags = np.abs(actual[:half])
        freqs = freq_axis[:half]

        peak_bin = int(np.argmax(mags))
        peak_freq = float(freqs[peak_bin])

        freq_res = self.fs / nfft
        tol = self.freq_tol or (2 * freq_res)
        diff = abs(peak_freq - self.target_freq)

        return ValidationResult(
            passed=diff < tol,
            metric_name="spectral_peak_freq",
            actual_value=peak_freq,
            threshold=tol,
            message=f"peak={peak_freq/1e6:.3f}MHz, target={self.target_freq/1e6:.3f}MHz"
        )


class EnergyValidator(IValidator):
    """Проверяет: энергия в полосе >= min_ratio * total_energy."""

    def __init__(self, band_hz: tuple, fs: float, min_ratio: float = 0.5):
        self.band_hz = band_hz    # (f_low, f_high)
        self.fs = fs
        self.min_ratio = min_ratio

    def validate(self, actual: np.ndarray,
                 reference: np.ndarray = None) -> ValidationResult:
        nfft = len(actual)
        freq_res = self.fs / nfft
        mags_sq = np.abs(actual[:nfft//2]) ** 2
        freqs = np.fft.fftfreq(nfft, 1.0 / self.fs)[:nfft//2]

        mask = (freqs >= self.band_hz[0]) & (freqs <= self.band_hz[1])
        band_energy = float(np.sum(mags_sq[mask]))
        total_energy = float(np.sum(mags_sq))
        ratio = band_energy / max(total_energy, 1e-12)

        return ValidationResult(
            passed=ratio >= self.min_ratio,
            metric_name="energy_in_band",
            actual_value=ratio,
            threshold=self.min_ratio,
            message=f"band=[{self.band_hz[0]/1e6:.1f}..{self.band_hz[1]/1e6:.1f}]MHz ratio={ratio:.3f}"
        )
```

### 6. IReporter — Observer (GoF)

**Проблема**: `print()` разбросан по 36 файлам (30+ вызовов), нет структуры.
**Решение**: Observer — подписка на события теста.

```python
# common/reporters.py
from abc import ABC, abstractmethod
import json
from datetime import datetime

class IReporter(ABC):
    """Observer интерфейс для репортинга тестов."""

    @abstractmethod
    def on_test_started(self, test_name: str): ...

    @abstractmethod
    def on_passed(self, test_name: str, result: TestResult): ...

    @abstractmethod
    def on_failed(self, test_name: str, error: Exception): ...


class ConsoleReporter(IReporter):
    """Структурированный вывод в консоль."""

    def on_test_started(self, test_name: str):
        print(f"\n▶ {test_name}")

    def on_passed(self, test_name: str, result: TestResult):
        print(f"  ✅ PASS — {result.summary()}")
        for v in result.validations:
            print(f"     {v}")

    def on_failed(self, test_name: str, error: Exception):
        print(f"  ❌ FAIL — {error}")


class JSONReporter(IReporter):
    """Сохраняет результаты в Results/JSON/."""

    def __init__(self, output_dir: str = "Results/JSON"):
        self.output_dir = output_dir
        self._results = []

    def on_test_started(self, test_name: str):
        pass

    def on_passed(self, test_name: str, result: TestResult):
        self._results.append({
            "test": test_name,
            "status": "pass",
            "timestamp": datetime.now().isoformat(),
            "validations": [{"metric": v.metric_name,
                              "value": v.actual_value,
                              "threshold": v.threshold}
                            for v in result.validations]
        })

    def on_failed(self, test_name: str, error: Exception):
        self._results.append({
            "test": test_name,
            "status": "fail",
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })

    def save(self, filename: str = None):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        fname = filename or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.output_dir, fname)
        with open(path, 'w') as f:
            json.dump(self._results, f, indent=2)
```

### 7. TestBase — Template Method (GoF)

**Проблема**: структура теста копируется в каждом файле.
**Решение**: Template Method — неизменный скелет, переопределяемые hooks.

```python
# common/test_base.py
from abc import ABC, abstractmethod
import numpy as np

class TestBase(ABC):
    """Template Method — скелет GPU теста.

    GoF: Template Method
    GRASP: Low Coupling (через абстракции, не конкретные классы)

    Подкласс реализует hooks:
        get_params()    → что тестируем
        generate_data() → входные данные
        process()       → GPU обработка
        validate()      → проверка результата
    """

    def __init__(self, reporters: list = None):
        self._reporters = reporters or [ConsoleReporter()]

    def run(self, test_name: str = None) -> TestResult:
        """Неизменный Template Method — переопределяй только hooks."""
        name = test_name or self.__class__.__name__
        self._notify_started(name)

        try:
            ctx = GPUContextManager.get()
            params = self.get_params()
            data = self.generate_data(params)
            raw_result = self.process(data, ctx)
            test_result = self.validate(raw_result, params)
            self._notify_passed(name, test_result)
            return test_result

        except Exception as e:
            self._notify_failed(name, e)
            raise

    # ---- Hooks (переопределяй в подклассах) ----

    @abstractmethod
    def get_params(self):
        """Возвращает конфиг (dataclass)."""
        ...

    @abstractmethod
    def generate_data(self, params) -> np.ndarray:
        """Генерирует входные данные."""
        ...

    @abstractmethod
    def process(self, data: np.ndarray, ctx) -> np.ndarray:
        """GPU обработка → выходные данные."""
        ...

    @abstractmethod
    def validate(self, result: np.ndarray, params) -> TestResult:
        """Проверка результата → TestResult."""
        ...

    # ---- Observer уведомления ----

    def _notify_started(self, name: str):
        for r in self._reporters:
            r.on_test_started(name)

    def _notify_passed(self, name: str, result: TestResult):
        for r in self._reporters:
            r.on_passed(name, result)

    def _notify_failed(self, name: str, error: Exception):
        for r in self._reporters:
            r.on_failed(name, error)
```

### 8. FilterTestBase — конкретный Template Method для фильтров

```python
# filters/filter_test_base.py
class FilterTestBase(TestBase, ABC):
    """Базовый класс для тестов фильтров.

    Template Method для общей логики: generate_reference() — numpy/scipy эталон.
    """

    def validate(self, result: np.ndarray, params: FilterConfig) -> TestResult:
        """Общая валидация: GPU vs scipy reference."""
        reference = self.generate_reference(params)

        test_result = TestResult(test_name=self.__class__.__name__)
        validator = NumericValidator(tolerance=params.tolerance)
        test_result.add(validator.validate(result, reference))

        # Спектральная проверка (если задана)
        if hasattr(params, 'f0_hz') and params.f0_hz > 0:
            spec_validator = SpectralValidator(
                target_freq_hz=params.f0_hz, fs=params.fs
            )
            # validate spectrum of result
            import numpy.fft as fft
            spectrum = fft.fft(result)
            test_result.add(spec_validator.validate(spectrum))

        return test_result

    @abstractmethod
    def generate_reference(self, params: FilterConfig) -> np.ndarray:
        """Scipy/numpy эталон для сравнения."""
        ...
```

### 9. LLMParser — Strategy (GoF) для AI Pipeline

**Проблема**: разбор ответов Groq/Ollama/Mock в одном файле, if/elif по MODE.
**Решение**: Strategy — каждый парсер независимый класс.

```python
# filters/ai_pipeline/llm_parser.py
from abc import ABC, abstractmethod

class LLMParser(ABC):
    """Strategy — парсит ответ LLM в FilterSpec.

    GoF: Strategy
    Реализации: GroqParser, OllamaParser, MockParser
    """
    @abstractmethod
    def parse(self, prompt: str) -> dict:
        """Отправляет запрос LLM, возвращает parsed FilterSpec."""
        ...

class GroqParser(LLMParser):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        ...
    def parse(self, prompt: str) -> dict: ...

class OllamaParser(LLMParser):
    def __init__(self, host: str = "localhost:11434", model: str = "llama3"):
        ...
    def parse(self, prompt: str) -> dict: ...

class MockParser(LLMParser):
    """Для тестов без API — возвращает предустановленный ответ."""
    def __init__(self, preset_response: dict):
        self.response = preset_response
    def parse(self, prompt: str) -> dict:
        return self.response

def create_parser(mode: str = "mock", **kwargs) -> LLMParser:
    """Factory function — создаёт нужный парсер."""
    parsers = {
        "groq": GroqParser,
        "ollama": OllamaParser,
        "mock": MockParser,
    }
    return parsers[mode](**kwargs)
```

### 10. PipelineBase — Template Method для strategies/

**Проблема**: `run_pipeline_a` и `run_pipeline_b` дублируют ~50% кода.
**Решение**: Abstract Base с общим pipeline и hook `_preprocess()`.

```python
# strategies/pipeline_runner.py (рефакторинг)
from abc import ABC, abstractmethod

class PipelineBase(ABC):
    """Abstract pipeline — вынесен общий код FFT + stats + peaks.

    GoF: Template Method
    Hook: _preprocess() — разная логика у A (phase) и B (Farrow)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    def run(self, scenario: dict, steer_theta: float,
            config: PipelineConfig = None) -> PipelineResult:
        """Общий pipeline: preprocess → GEMM → FFT → peaks."""
        cfg = config or PipelineConfig()
        S = scenario['S']
        array = scenario['array']

        result = PipelineResult(pipeline_name=self.name, S_raw=S)
        result.stats_input = compute_matrix_stats(S)

        # Hook — разная логика для A и B
        W, X = self._preprocess(S, array, steer_theta, scenario['fs'])
        result.W = W
        result.X_gemm = X
        result.stats_gemm = compute_matrix_stats(X)

        # Общий: FFT + peaks (не дублируется!)
        result = self._run_fft_and_peaks(result, scenario['fs'])
        return result

    @abstractmethod
    def _preprocess(self, S, array, steer_theta, fs):
        """Hook: вернуть (W, X_gemm)."""
        ...

    def _run_fft_and_peaks(self, result: PipelineResult, fs: float) -> PipelineResult:
        """Общий код: Window + FFT + find_peaks."""
        # ... одна реализация вместо двух copy-paste ...


class PipelineA(PipelineBase):
    """Pipeline A: фазовая коррекция через W_phase."""
    name = "A"

    def _preprocess(self, S, array, steer_theta, fs):
        # Phase weight matrix
        delays = array.compute_delays(steer_theta)
        W = _make_phase_weight_matrix(delays, self._steer_freq, array.n_ant)
        return W, W @ S

    def run(self, scenario, steer_theta, steer_freq=None, config=None):
        self._steer_freq = steer_freq or scenario['targets'][0].f0_hz
        return super().run(scenario, steer_theta, config)


class PipelineB(PipelineBase):
    """Pipeline B: временно́е выравнивание через Farrow."""
    name = "B"

    def _preprocess(self, S, array, steer_theta, fs):
        delays_s = array.compute_delays(steer_theta)
        farrow = FarrowDelay()
        S_aligned = farrow.compensate_seconds(S, delays_s, fs)
        n = array.n_ant
        W = np.full((n, n), 1.0 / np.sqrt(n), dtype=np.complex64)
        return W, W @ S_aligned


class PipelineRunner:
    """Façade + Factory — публичный API без изменений."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir

    def run_pipeline_a(self, scenario, steer_theta, steer_freq,
                       config=None) -> PipelineResult:
        return PipelineA().run(scenario, steer_theta, steer_freq, config)

    def run_pipeline_b(self, scenario, steer_theta,
                       config=None) -> PipelineResult:
        return PipelineB().run(scenario, steer_theta, config)
```

---

## 📋 Применение паттернов

### SOLID

| Принцип | Нарушение (ДО) | Решение (ПОСЛЕ) |
|---------|----------------|-----------------|
| **S** — SRP | `test_ai_filter_pipeline.py` делает 7 вещей | 4 класса в `ai_pipeline/` |
| **S** — SRP | `plot_report_filters.py` = данные + matplotlib | `visualization/` отдельно |
| **O** — OCP | Новый фильтр = copy-paste целого файла | `FilterTestBase` + новый подкласс |
| **L** — LSP | Нет иерархий | `IValidator`, `LLMParser` подтипы взаимозаменяемы |
| **I** — ISP | `dict` как "всё в одном" | мелкие `SignalConfig`, `FilterConfig` |
| **D** — DIP | `import gpuworklib` прямо в тестах | `GPULoader.get()` через абстракцию |

### GRASP

| Паттерн | Применение |
|---------|-----------|
| **Information Expert** | `SignalConfig.freq_resolution_hz()` — конфиг сам вычисляет |
| **Creator** | `conftest.py` создаёт `ctx`, `gen` — не тесты |
| **Controller** | `PipelineRunner` координирует, `TestBase.run()` управляет потоком |
| **Low Coupling** | `IValidator` отделяет валидацию от GPU обработки |
| **High Cohesion** | Каждый файл — одна тема (только FIR, только IIR) |
| **Polymorphism** | `IValidator.validate()` — нет `if/elif` по типу проверки |
| **Protected Variations** | `GPULoader` скрывает детали поиска `.so` |

### GoF

| Паттерн | Класс | Файл |
|---------|-------|------|
| **Singleton** | `GPULoader` | `common/gpu_loader.py` |
| **Singleton** | `GPUContextManager` | `common/gpu_context.py` |
| **Template Method** | `TestBase.run()` | `common/test_base.py` |
| **Template Method** | `FilterTestBase.validate()` | `filters/filter_test_base.py` |
| **Template Method** | `PipelineBase.run()` | `strategies/pipeline_runner.py` |
| **Strategy** | `IValidator` (Numeric/Spectral/Energy) | `common/validators.py` |
| **Strategy** | `IPlotter` (Spectrum/Time/Comparison) | `common/plotting/` |
| **Strategy** | `LLMParser` (Groq/Ollama/Mock) | `filters/ai_pipeline/llm_parser.py` |
| **Observer** | `IReporter` (Console/JSON/Plot) | `common/reporters.py` |
| **Builder** | `ScenarioBuilder` | `strategies/scenario_builder.py` ← УЖЕ ЕСТЬ |
| **Façade** | `PipelineRunner` | `strategies/pipeline_runner.py` |
| **Factory Method** | `create_parser()` | `filters/ai_pipeline/llm_parser.py` |

---

## 📐 Root conftest.py

```python
# Python_test/conftest.py
import pytest
from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager

@pytest.fixture(scope="session")
def gpuworklib():
    """Загрузить gpuworklib один раз на всю сессию."""
    return GPULoader.get()

@pytest.fixture(scope="session")
def gpu_ctx(gpuworklib):
    """GPU контекст — один на сессию."""
    return GPUContextManager.get()

@pytest.fixture
def signal_factory(gpuworklib, gpu_ctx):
    """Фабрика сигналов для тестов."""
    return gpuworklib.SignalGenerator(gpu_ctx)
```

---

## 🗺️ Шаги миграции

> Поэтапно — каждый шаг не ломает работающие тесты.

### Этап 0: Инфраструктура (1-2 дня) ← не ломает ничего

```
Создать:
  Python_test/common/__init__.py
  Python_test/common/gpu_loader.py          ← GPULoader (Singleton)
  Python_test/common/gpu_context.py         ← GPUContextManager
  Python_test/common/configs.py             ← SignalConfig, FilterConfig
  Python_test/common/validators.py          ← IValidator + 3 реализации
  Python_test/common/reporters.py           ← IReporter + ConsoleReporter
  Python_test/common/result.py              ← TestResult, ValidationResult
  Python_test/common/test_base.py           ← TestBase (Template Method)
  Python_test/common/plotting/__init__.py
  Python_test/common/plotting/plotter_base.py  ← IPlotter (ABC)
  Python_test/conftest.py                   ← root fixtures

Проверка: pytest --collect-only (все старые тесты видны)
```

### Этап 1: filters/ (2-3 дня) ← крупнейший выигрыш

```
1. test_ai_filter_pipeline.py (964 строки) → ai_pipeline/ (4 файла)
   - llm_parser.py (LLMParser ABC + 3 реализации)
   - filter_designer.py (FilterDesigner)
   - filter_validator.py (FilterValidator)
   - test_ai_pipeline.py (pytest тесты, < 150 строк)
   - demo_ai_pipeline.py (демо с графиками)

2. plot_report_filters.py (821 строки) → visualization/
   - filter_plotter.py
   - report_plotter.py

3. conftest.py (filter fixtures)
4. filter_test_base.py (FilterTestBase)
5. Рефакторинг test_kalman/kaufman/moving_average_rocm.py

Проверка: pytest filters/ -v (все тесты pass)
```

### Этап 2: signal_generators/ (1-2 дня)

```
1. test_form_signal.py (722 строки) →
   - test_form_signal.py (только тесты, < 200 строк)
   - visualization/signal_plotter.py

2. conftest.py + signal_test_base.py

Проверка: pytest signal_generators/ -v
```

### Этап 3: heterodyne/ (1 день)

```
1. test_heterodyne_step_by_step.py (680 строк) →
   - test_dechirp.py (< 200 строк)
   - test_comparison.py (< 200 строк)

2. conftest.py

Проверка: pytest heterodyne/ -v
```

### Этап 4: integration/ (1 день)

```
1. test_gpuworklib.py (903 строки) →
   - test_fft_integration.py (тесты 1-3)
   - test_signal_gen_integration.py (тесты 4-7)
   - test_script_generator.py (тесты 8-9)

2. conftest.py

Проверка: pytest integration/ -v
```

### Этап 5: strategies/ (0.5 дня) ← мелкие правки

```
1. pipeline_runner.py: добавить PipelineBase + PipelineA/B
   (PipelineRunner — Façade, публичный API без изменений)
2. conftest.py (pytest fixtures)

Проверка: pytest strategies/ -v (все 19+ тестов pass)
```

### Этап 6: остальные модули (0.5 дня)

```
Добавить conftest.py в:
  fft_maxima/, lch_farrow/, statistics/, vector_algebra/,
  hybrid/, zero_copy/, fm_correlator/

Проверка: pytest Python_test/ -v --tb=short
```

---

## 📁 Файлы

### Новые файлы (создать)

| Файл | Паттерн | Строк |
|------|---------|-------|
| `common/gpu_loader.py` | Singleton | ~50 |
| `common/gpu_context.py` | Singleton | ~30 |
| `common/configs.py` | dataclass (Info Expert) | ~80 |
| `common/validators.py` | Strategy | ~120 |
| `common/reporters.py` | Observer | ~80 |
| `common/result.py` | Value Object | ~50 |
| `common/test_base.py` | Template Method | ~60 |
| `common/plotting/plotter_base.py` | Strategy ABC | ~30 |
| `conftest.py` (root) | Factory Method | ~30 |
| `filters/ai_pipeline/llm_parser.py` | Strategy | ~100 |
| `filters/ai_pipeline/filter_designer.py` | — | ~80 |
| `filters/ai_pipeline/filter_validator.py` | Strategy | ~60 |

### Файлы для разбивки

| Было | Строк | Станет | Строк каждый |
|------|-------|--------|-------------|
| `test_ai_filter_pipeline.py` | 964 | 4 файла в `ai_pipeline/` | < 200 |
| `test_gpuworklib.py` | 903 | 3 файла в `integration/` | < 300 |
| `plot_report_filters.py` | 821 | 2 файла в `visualization/` | < 250 |
| `test_form_signal.py` | 722 | test + signal_plotter.py | < 250 |
| `test_heterodyne_step_by_step.py` | 680 | 2 файла | < 250 |
| `pipeline_runner.py` | 626 | рефакторинг на месте | < 300 |

### Файлы без изменений (хорошие уже!)

- `strategies/scenario_builder.py` — Builder паттерн, ОК
- `strategies/farrow_delay.py` — маленький, чистый
- `strategies/test_farrow_pipeline.py` — структурированный
- `strategies/test_scenario_builder.py` — ОК
- `fft_maxima/*.py` — небольшие, читаемые
- `lch_farrow/*.py` — небольшие

---

## ✅ Критерии завершения

- [ ] `pytest Python_test/ -v` — все старые тесты проходят
- [ ] `pytest Python_test/ -v --ignore=filters/visualization` — тесты без matplotlib
- [ ] Нет файла > 300 строк
- [ ] Нет функции > 50 строк
- [ ] `GPULoader` используется во всех модулях
- [ ] `common/` покрыт юнит-тестами на `MockParser`, `MockValidator`

---

## 🔗 Связи с проектом

- Паттерн `ScenarioBuilder` (Builder) уже в `strategies/` — расширить
- `PipelineResult`, `ChannelStats` — хорошие dataclass'ы — поднять в `common/`
- `GPUProfiler` из DrvGPU — подключить через `PlotReporter`

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-08 | Alex + Кодо | Создание спецификации (анализ + архитектурный план) |
