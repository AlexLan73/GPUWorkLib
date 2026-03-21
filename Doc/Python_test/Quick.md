# Python_test — Краткий справочник

> Тестовая инфраструктура Python для всех GPU-модулей GPUWorkLib
> **Рефакторинг**: OOP/SOLID/GRASP/GoF (2026-03-08)

---

## Структура

```
Python_test/
├── conftest.py              ← ROOT: session fixtures (gw, gpu_ctx, rng, plot_dir)
│
├── common/                  ← НОВЫЙ: общая инфраструктура (GoF + SOLID)
│   ├── gpu_loader.py        ← GPULoader (Singleton) — находит .so один раз
│   ├── gpu_context.py       ← GPUContextManager (Singleton)
│   ├── test_base.py         ← TestBase (Template Method)
│   ├── configs.py           ← SignalConfig, FilterConfig (dataclasses)
│   ├── validators.py        ← IValidator + Numeric/Spectral/Energy
│   ├── reporters.py         ← IReporter + Console/JSON/Multi
│   ├── result.py            ← TestResult, ValidationResult
│   └── plotting/
│       └── plotter_base.py  ← IPlotter (Strategy ABC)
│
├── filters/                 # 9 файлов — FIR, IIR, Moving Avg, AI pipeline
│   ├── conftest.py          ← fixtures: fir_coeffs, iir_coeffs, signals
│   ├── filter_test_base.py  ← FilterTestBase(TestBase)
│   └── ai_pipeline/         ← НОВЫЙ: разбивка монолита (964 строки → 4 файла)
│       ├── llm_parser.py    ← LLMParser + Mock/Groq/OllamaParser (Strategy)
│       ├── filter_designer.py  ← FilterDesigner (scipy FIR/IIR)
│       └── test_ai_pipeline.py ← чистые тесты (без matplotlib)
│
├── signal_generators/       # 4 файла + инфраструктура
│   ├── conftest.py          ← fixtures: sig_gen, fft_proc, lfm_params
│   └── signal_test_base.py  ← SignalTestBase(TestBase)
│
├── heterodyne/              # 4 файла + инфраструктура
│   ├── conftest.py          ← fixtures: DechirpParams, het_proc, lfm_srx
│   └── heterodyne_test_base.py ← HeterodyneTestBase + validate_beat_frequency()
│
├── integration/             # 3 файла (было 1 монолит 903 строки)
│   ├── conftest.py
│   ├── test_fft_integration.py       ← тесты 1-3: CW/LFM/Noise → FFT
│   └── test_signal_gen_integration.py ← тесты 4-7: multichannel, pipeline
│
├── strategies/              # 5 файлов + инфраструктура
│   ├── conftest.py          ← fixtures: scenario_8ant, pipeline_runner
│   ├── pipeline_runner.py   ← + PipelineBase/PipelineA/PipelineB (рефакторинг)
│   ├── scenario_builder.py  ← ScenarioBuilder (Builder pattern, без изменений)
│   └── farrow_delay.py      ← FarrowDelay numpy (без изменений)
│
├── fft_maxima/   + conftest.py
├── lch_farrow/   + conftest.py
├── statistics/   + conftest.py
└── vector_algebra/ + conftest.py
```

**Итого: 10 модулей + common/, 57 файлов (было 36)**

---

## Быстрый старт

### Сборка

```bash
cmake -B build -DBUILD_PYTHON=ON && cmake --build build --config Release
```

### Запуск (TestRunner)

```bash
# Все тесты (GPULoader найдёт .so автоматически)
python run_tests.py -v

# Модуль целиком
python run_tests.py -m filters
python run_tests.py -m integration

# Только тесты без GPU (не принимают gpu_ctx fixture)
python run_tests.py -m filters/ai_pipeline  # MockParser, FilterDesigner

# Один тест
python run_tests.py -m filters/ai_pipeline/test_ai_pipeline.py::TestMockParser -v

# ROCm тесты (Linux)
bash Python_test/run_all_rocm_tests.sh
```

### Запуск (standalone, с графиками)

```bash
# Старые тесты — работают как раньше
python Python_test/signal_generators/test_form_signal.py
python Python_test/heterodyne/test_heterodyne_step_by_step.py
```

---

## GPULoader — решение проблемы путей

**Было** (хардкод в каждом файле):
```python
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'debian-radeon9070', 'python'))
import gpuworklib
```

**Стало** (один Singleton для всей сессии):
```python
from common.gpu_loader import GPULoader
gw = GPULoader.get()   # None если не найден
```

**Или через хелпер-функция** (автоматически):
```python
def test_something(gpu_ctx):    # fixture из root conftest.py
    f = gpuworklib.FirFilterROCm(gpu_ctx, coeffs)
    ...
```

Поиск в 6 стандартных путях: `build/python/Release`, `build/python/Debug`, `build/debian-radeon9070/python`, ...

---

## Ключевые абстракции common/

### TestBase — Template Method

```python
class MyFilterTest(FilterTestBase):
    def get_params(self): return FilterConfig(cutoff_hz=1e3, fs=50e3)

    def process(self, data, ctx):
        return gpuworklib.FirFilterROCm(ctx, coeffs).process(data)

    def compute_reference(self, data, params):
        return scipy.signal.lfilter(coeffs, [1.0], data)

    def validate(self, result, params):
        return self._validate_with_scipy(result, params, tolerance=1e-4)

# Запуск
result = MyFilterTest("fir_lowpass").run()
print(result.summary())  # PASS / FAIL + детали
```

### IValidator — Strategy

```python
from common.validators import NumericValidator, SpectralValidator, EnergyValidator

v = NumericValidator(tolerance=1e-4)
vr = v.validate(gpu_output, scipy_reference)
# vr.passed → True/False
# vr.actual_value → относительная ошибка

v = SpectralValidator(fs=12e6, freq_tol_hz=500.0)
vr = v.validate(gpu_signal, reference_signal)

v = EnergyValidator(fs=50e3, band_hz=(200, 1000), min_ratio=0.8)
vr = v.validate(filtered_signal, _)
```

### LLMParser — ai_pipeline (Strategy)

```python
from filters.ai_pipeline.llm_parser import MockParser, GroqParser, create_parser
from filters.ai_pipeline.filter_designer import FilterDesigner

# В тестах — без сети:
parser = MockParser()
spec = parser.parse("FIR lowpass 1kHz", fs=50_000)
# spec.filter_class = "fir", spec.f_cutoff = 1000.0

design = FilterDesigner().design(spec)
# design.coeffs_b = [scipy firwin коэффициенты]
ref_output = design.apply_scipy(signal)

# В production — через AI:
parser = create_parser("groq")   # читает GROQ_API_KEY / api_keys.json
```

### PipelineBase — strategies/ (Template Method)

```python
# Старый API PipelineRunner работает без изменений:
runner = PipelineRunner(output_dir="Results/strategies/test_01")
result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6)
result_b = runner.run_pipeline_b(scenario, steer_theta=30)

# Новый: напрямую через классы (меньше параметров):
from pipeline_runner import PipelineA, PipelineB
result = PipelineA().run(scenario, steer_theta=30, steer_freq=2e6)
result = PipelineB().run(scenario, steer_theta=30)
```

---

## Покрытие модулей

| Модуль | Файлов | OpenCL | ROCm | Эталон | conftest |
|--------|--------|--------|------|--------|----------|
| signal_generators | 4+2 | ✅ | ✅ | NumPy getX | ✅ |
| filters | 5+3+4 | ✅ | ✅ | SciPy lfilter/sosfilt | ✅ |
| heterodyne | 4+2 | ✅ | ✅ | NumPy FFT | ✅ |
| fft_maxima | 3+1 | ✅ | ✅* | SciPy find_peaks | ✅ |
| lch_farrow | 2+1 | ✅ | ✅ | CPU Lagrange | ✅ |
| statistics | 1+1 | — | ✅ | NumPy mean/median | ✅ |
| vector_algebra | 2+1 | — | ✅ | NumPy linalg + CSV | ✅ |
| integration | 3+1 | ✅ | — | Combined | ✅ |
| strategies | 5+1 | — | — | numpy | ✅ |
| common/ | 10 | — | — | (инфраструктура) | — |

\* fft_maxima ROCm: indirect через HeterodyneDechirp

---

## Типичные пороги

| Операция | Порог | Причина |
|----------|-------|---------|
| Identity (delay=0) | 1e-4 — 1e-6 | Только copy |
| CW/LFM генерация | 1e-3 | float32 vs float64 |
| FIR фильтрация | 1e-3 | Линейная свёртка |
| IIR фильтрация | 1e-2 | Рекурсивная ошибка |
| Farrow задержка | 1e-2 — 1e-3 | Lagrange approx |

---

## Слои тестирования (разделение matplotlib)

```
Слой 1: тесты (test_*.py)
  → только assert, numpy, gpu_ctx fixture
  → НЕ импортируют matplotlib напрямую
  → запускаются в CI без дисплея

Слой 2: демо-скрипты (demo_*.py, example_*.py)
  → запускаются руками: python demo_ai_pipeline.py
  → рисуют графики, сохраняют в Results/Plots/
  → используют IPlotter из common/plotting/
```

---

## Графики

Все графики → `Results/Plots/{module}/`:
- `signal_generators/FormSignal/`, `DelayedFormSignal/`, `LfmAnalyticalDelay/`
- `filters/`, `heterodyne/`, `fft_maxima/`, `lch_farrow/`, `integration/`, `strategies/`

---

## Зависимости

| Пакет | Обязательный | Назначение |
|-------|-------------|------------|
| numpy | ✅ | CPU reference |
| scipy | ⚠️ рекомендуется | Фильтры, find_peaks, firwin/butter |
| matplotlib | ⚠️ опционально | Графики (только в demo-скриптах) |
| ~~pytest~~ | ❌ Не используется | Заменён на TestRunner из common/runner.py |
| groq / ollama | ❌ опционально | AI-pipeline (MockParser работает без них) |

---

## Python API (pybind11)

```python
# Контексты (создаётся автоматически через GPUContextManager)
ctx = gpuworklib.GPUContext(0)          # OpenCL
ctx = gpuworklib.ROCmGPUContext(0)      # ROCm

# Генераторы
gen = gpuworklib.FormSignalGenerator(ctx)
gen = gpuworklib.DelayedFormSignalGenerator(ctx)

# FFT & Spectrum
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# Фильтры
fir = gpuworklib.FirFilterROCm(ctx, coeffs)
iir = gpuworklib.IirFilterROCm(ctx, b, a)

# Signal Processing
farrow = gpuworklib.LchFarrowROCm(ctx)
het = gpuworklib.HeterodyneDechirp(ctx, fs, f_start, f_end, n, n_ant)
stats = gpuworklib.StatisticsProcessor(ctx)
```

Подробнее: `Doc/Python/*.md`

---

## Ссылки

- [Full.md](Full.md) — полное описание, все тесты, архитектура, binding'ы
- [Doc/Python/](../Python/) — Python API документация по модулям
- [MemoryBank/specs/python_test_refactoring.md](../../MemoryBank/specs/python_test_refactoring.md) — план рефакторинга

---

*Обновлено: 2026-03-08*
