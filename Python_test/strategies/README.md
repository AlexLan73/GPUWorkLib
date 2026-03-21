# Python_test/strategies — Тесты антенной обработки сигналов

Тесты **ULA beamforming pipeline** с Farrow-задержками и GPU-ускорением (ROCm).

Покрывают: физическую модель антенной решётки → генерацию сигналов →
NumPy-эталон → Pipeline A (фазовая коррекция) vs Pipeline B (Farrow) →
пошаговую GPU-валидацию через pybind11.

---

## Быстрый старт

```bash
# Из корня проекта (GPUWorkLib/)

# Тесты без GPU (NumPy-only):
python Python_test/strategies/test_scenario_builder.py
python Python_test/strategies/test_base_pipeline.py
python Python_test/strategies/test_debug_steps.py
python Python_test/strategies/test_farrow_pipeline.py
python Python_test/strategies/test_timing_analysis.py

# Тесты с GPU (нужен ROCm + gpuworklib.so):
python Python_test/strategies/test_strategies_pipeline.py
python Python_test/strategies/test_strategies_step_by_step.py

# Визуализация:
python Python_test/strategies/plot_strategies_results.py

# PyCharm-отладчик (breakpoints):
python Python_test/strategies/debug_pipeline_steps.py
```

---

## Параметры тестов

| Параметр | Значение | Описание |
|----------|----------|----------|
| `N_ANT` | 5 | Число антенн (small) / 2500 (full_spec) |
| `N_SAMPLES` | 8000 | Отсчётов на антенну |
| `FS` | 12 МГц | Частота дискретизации |
| `F0` | 2 МГц | Несущая частота сигнала |
| `TAU_STEP` | 100 мкс | Шаг задержки между антеннами |
| `nFFT` | 16384 | next_pow2(8000) × 2 |

---

## Архитектура Pipeline

```
ULAGeometry                          Физическая модель антенной решётки ULA
    ↓ compute_delays(theta_deg)
ScenarioBuilder                      Генерация сигналов (CW, LFM, шум, цели + помехи)
    ↓ build() → {S, W, targets, ...}

          ┌──────────────────┬──────────────────────┐
          ▼                  ▼                      ▼
   Pipeline A            Pipeline B           NumPy Reference
   (фазовая             (Farrow задержка)     (CPU-эталон)
   коррекция)
          │                  │
          │  FarrowDelay.apply(S, delays)      ← subpixel delay
          │       ↓
          └──┬───┘
             ▼
         W @ S              GEMM (beamforming)
             ↓
         Hamming × X        Оконное взвешивание
             ↓
         FFT(zero-pad)      nFFT = 16384
             ↓
         OneMax / AllMaxima / GlobalMinMax     Поиск пиков
             ↓
         PipelineResult     Метрики, пики, спектры
```

**Pipeline A** — компенсирует задержки через фазу несущей `exp(-j·2π·f0·τ)`.
Хорошо работает для CW (узкополосный). Для ЛЧМ (широкополосный) появляется
временное размытие (temporal smearing).

**Pipeline B** — сначала выравнивает сигналы во времени через Farrow-интерполяцию
(дробные задержки), затем суммирует. Для ЛЧМ более точен: нет smearing.

---

## Файлы — вспомогательные модули

### `scenario_builder.py`
**Физическая модель антенной решётки (ULA)**

```python
from scenario_builder import ULAGeometry, ScenarioBuilder, make_single_target

# ULA: 8 антенн, шаг 5 см
array = ULAGeometry(n_ant=8, d_ant_m=0.05)
delays = array.compute_delays(theta_deg=30.0)   # [n_ant] секунды

# Генерация сценария
builder = ScenarioBuilder(array, fs=12e6, n_samples=8000)
builder.add_target(theta_deg=30, f0_hz=2e6, fdev_hz=1e6)
builder.add_jammer(theta_deg=-20, f0_hz=3e6, amplitude=0.5)
builder.set_noise(sigma=0.1, seed=42)
scenario = builder.build()  # dict: S, W, targets, jammers, array, fs

# Фабричные сценарии
scenario = make_single_target(n_ant=8, theta_deg=30, fdev_hz=1e6)
scenario = make_target_and_jammer(n_ant=8)
scenario = make_multi_target(n_ant=8, thetas=[20,45], f0s=[1e6,2e6], ...)

# Матрица весов
W = builder.generate_weight_matrix(steer_theta_deg=30)         # [n_ant, n_ant]
W = builder.generate_scan_weight_matrix([-30, 0, 30])          # [n_beams, n_ant]
```

**Классы:** `ULAGeometry`, `EmitterSignal`, `ScenarioBuilder`

---

### `farrow_delay.py`
**Дробные задержки через Farrow-интерполяцию (NumPy)**

```python
from farrow_delay import FarrowDelay

farrow = FarrowDelay()

# Применить задержки к каждой антенне
delays_samples = np.array([0.0, 1.5, 3.0, 4.5])   # в отсчётах
S_delayed = farrow.apply(S, delays_samples)          # [n_ant, n_samples]

# Компенсировать (обратная задержка)
S_restored = farrow.compensate(S_delayed, delays_samples)
```

Лагранжевая интерполяция 4-го порядка. Точна для целых задержек,
для дробных — характеристика Lagrange (Runge oscillation для больших frac).

---

### `pipeline_runner.py`
**PipelineRunner — два варианта pipeline (A и B)**

```python
from pipeline_runner import PipelineRunner, PipelineConfig, PipelineResult

runner = PipelineRunner(output_dir="Results/")
config = PipelineConfig(save_input=True, save_aligned=True, save_stats=True)

# Pipeline A: фазовая коррекция W
result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6, config=config)

# Pipeline B: Farrow временное выравнивание
result_b = runner.run_pipeline_b(scenario, steer_theta=30, config=config)

# Доступные данные в PipelineResult
result_b.S_raw        # [n_ant, n_samples] — входной сигнал
result_b.S_aligned    # [n_ant, n_samples] — после Farrow (только Pipeline B)
result_b.X_gemm       # [n_ant, n_samples] — после W @ S
result_b.spectrum     # [n_ant, nFFT] complex64
result_b.magnitudes   # [n_ant, nFFT] float32
result_b.peaks        # list[list[PeakInfo]] — найденные пики
result_b.stats_input  # list[ChannelStats] — статистика по шагам

# Сравнение
comp = runner.compare(result_a, result_b)   # dict: magnitude_ratio_b_over_a, freq_diff_hz, ...
runner.print_comparison(result_a, result_b)

# Строки
result_b.peak_summary()   # текст о пиках
result_b.stats_summary()  # текст о статистике
```

**Вспомогательные функции:** `compute_channel_stats()`, `find_peaks_per_beam()`

---

### `pipeline_step_validator.py`
**PipelineStepValidator — валидация GPU-шагов через pybind11**

```python
from pipeline_step_validator import PipelineStepValidator

# proc — объект gpuworklib.AntennaProcessorTest (создаётся снаружи)
psv = PipelineStepValidator(proc=proc, ref=numpy_ref, save_to_disk=False)

result = psv.run_all(
    d_S=data.d_S,          # GPU-буфер сигнала
    d_W=data.d_W,          # GPU-буфер матрицы весов
    is_identity_w=False,
    variant_name="V3_phase",
)  # → TestResult
```

Вызывает `proc.step_1_debug_input()`, `step_2_gemm()`, `step_3_debug_post_gemm()`,
`step_4_window_fft()`, `step_6_1_one_max_parabola()`, `step_6_2_all_maxima()`,
`step_6_3_global_minmax()` и сравнивает с NumPy-эталоном.

---

### `numpy_reference.py`
**NumpyReference — CPU-эталон для сравнения с GPU**

```python
from numpy_reference import NumpyReference

ref = NumpyReference(S=S_np, W=W_np, fs=12e6, f0=2e6, n_fft=8192)
ref.run()   # вычислить X_gemm, spectrum, peaks
ref.compare_gemm(X_gpu)       # → dict расхождений
ref.compare_spectrum(spec_gpu)
ref.compare_peaks(peaks_gpu)
```

---

### `signal_factory.py`
**SignalVariant + SignalConfig + SignalSourceFactory**

```python
from signal_factory import SignalVariant, SignalConfig, SignalSourceFactory

# Варианты сигнала (для GPU тестов)
class SignalVariant(Enum):
    V1_CW_CLEAN       # CW без шума, W = Identity
    V2_CW_NOISE       # CW + AWGN (SNR=20 дБ)
    V3_CW_PHASE_DELAY # CW с фазовой задержкой, W = delay_and_sum
    V4_CW_PHASE_NOISE # V3 + AWGN
    V5_FROM_FILE      # Из файла (заглушка → SkipTest)

cfg = SignalConfig(n_ant=5, n_samples=8000, fs=12e6, f0=2e6, tau_step=100e-6)
source = SignalSourceFactory.create(SignalVariant.V3_CW_PHASE_DELAY)
data = source.generate(cfg)   # SignalData: S_ref, W_ref, d_S, d_W
```

---

### `signal_generators_strategy.py`
**ISignalStrategy + SignalStrategyFactory (для NumPy тестов)**

```python
from signal_generators_strategy import SignalStrategyFactory, SignalVariant

# Варианты сигнала (для NumPy тестов)
class SignalVariant(Enum):
    SIN            # Синусоида (CW)
    LFM_NO_DELAY   # ЛЧМ без задержек
    LFM_WITH_DELAY # ЛЧМ с целочисленными задержками
    LFM_FARROW     # ЛЧМ с Farrow дробными задержками

strategy = SignalStrategyFactory.create(SignalVariant.LFM_FARROW)
S = strategy.generate(params)   # [n_ant, n_samples] complex64
```

---

### `strategy_test_base.py`
**StrategyTestBase — Template Method для NumPy-тестов**

Абстрактный базовый класс. Подклассы реализуют `process()` и `validate()`.
Автоматически: `generate_data()` → `process()` → `validate()` → `TestResult`.

Хелперы: `_make_weight_matrix()`, `_apply_gemm()`, `_apply_window_fft()`,
`_find_peak_freq()`, `_run_numpy_pipeline()`.

---

### `test_params.py`
**AntennaTestParams — dataclass с параметрами тестов** *(не тест!)*

```python
from test_params import AntennaTestParams, SignalVariant

params = AntennaTestParams.small()       # 100 антенн (быстро)
params = AntennaTestParams.full_spec()   # 2500 антенн × 5000 отсчётов
params = AntennaTestParams.debug()       # small + запись в файлы

params.bin_hz            # ширина FFT бина [Гц]
params.expected_peak_bin # ожидаемый бин пика
params.check_peak_freq   # True для SIN/CW, False для ЛЧМ без дечирпа
```

---

### `conftest.py`
**Фабричные функции (вспомогательные хелперы)**

```python
from conftest import make_farrow, make_scenario_8ant, make_scenario_multi, strategy_plot_dir

farrow    = make_farrow()          # FarrowDelay
scenario  = make_scenario_8ant()   # 8 антенн, 1 цель @ 30°, fs=12 МГц
scenario  = make_scenario_multi()  # 8 антенн, 3 цели @ 15°/30°/45°
plot_dir  = strategy_plot_dir      # путь Results/Plots/strategies/
```

---

## Файлы — тесты

### `test_scenario_builder.py` — физическая модель ULA *(без GPU)*

Проверяет что физическая модель математически корректна.

| Класс | Что проверяет |
|-------|---------------|
| `TestULAGeometry` | theta=0→задержки=0, theta=90→максимальные, theta<0→отрицательные, lambda/2 |
| `TestSignalGeneration` | CW→пик на f0, ЛЧМ→полоса≈fdev, fdev=0≡CW, амплитуда, задержка между антеннами |
| `TestMultiSource` | 2 цели→2 пика, цель+помеха оба видны |
| `TestNoise` | AWGN mean≈0, std≈sigma, воспроизводимость по seed |
| `TestWeightMatrix` | shape, unit_norm, scan_W, beamforming не обнуляет |
| `TestFactoryScenarios` | make_single_target, make_target_and_jammer, make_multi_target |

---

### `test_base_pipeline.py` — NumPy math validation (T1) *(без GPU)*

Проверяет что алгоритм математически правилен до запуска на GPU.
Если этот тест падает — сломана математика, а не GPU.

**Тесты:** `test_sin_full_pipeline`, `test_lfm_no_delay_pipeline`,
`test_lfm_delay_pipeline`, `test_lfm_farrow_pipeline`, `test_all_variants`

**Класс:** `NumpyPipelineTest` — `process()` запускает `_run_numpy_pipeline()`.
**Критерии:** peak_freq≈f0 (±2 бина), dynamic_range>20 дБ, GEMM gain≈1/sqrt(n_ant).

---

### `test_debug_steps.py` — пошаговая NumPy-валидация (T2) *(без GPU)*

Проверяет каждый шаг в отдельности. Если T1 упал — запустить этот тест
чтобы понять на каком шаге ошибка.

| Тест | Шаг | Что проверяет |
|------|-----|---------------|
| `test_gemm_shape_and_gain` | GEMM | shape=(n_ant,n_samples), gain≈1/sqrt(n_ant) |
| `test_fft_peak_location` | FFT | peak_bin ≈ expected_bin (±2 бина) |
| `test_one_max_accuracy` | OneMax | параболическая интерполяция, refined_freq≈f0 |
| `test_minmax_dynamic_range_loop` | MinMax | max≥min, DR>20 дБ |

---

### `test_farrow_pipeline.py` — Farrow Pipeline A vs B *(без GPU)*

Ключевой вопрос: когда ЛЧМ сигнал, Pipeline B (Farrow) точнее Pipeline A?

| Класс | Что проверяет |
|-------|---------------|
| `TestFarrowDelay` | delay=0→identity, integer delay→точный сдвиг, compensate, per-antenna |
| `TestPipelineBasic` | CW Pipeline A пик на f0, CW Pipeline B пик на f0, ЛЧМ A/B ненулевые |
| `TestPipelineComparison` | CW: A≈B; ЛЧМ: energy_B/energy_A≥0.8 (KEY TEST) |
| `TestComplexScenarios` | 2 цели, цель+помеха, SNR gain от beamforming |
| `TestStatsAndCheckpoints` | Статистика по шагам, checkpoint на диск (.npy/.json), compare() |

---

### `test_strategies_step_by_step.py` — GPU vs NumPy пошагово *(ROCm или без)*

**Часть 1 (без GPU)** — `TestNumpyReference`:
- weight_matrix: shape и unit-norm
- hamming: форма окна (0.08 на краях, 1.0 в центре)
- gemm: W@S форма и ненулевой выход
- fft_peak: пик ≈ f0=2 МГц
- nfft: next_pow2(8000)×2 = 16384

**Часть 2 (ROCm GPU)** — `TestGPUvsNumPy`:
- `test_gpu_gemm_vs_numpy`: X_gpu ≈ X_ref (rtol=1e-3)
- `test_gpu_fft_shape`: spectrum.shape=(N_ANT, nFFT)
- `test_gpu_fft_peak`: GPU пик ≈ f0
- `test_gpu_one_max_frequency`: OneMax refined_freq≈f0
- `test_gpu_all_maxima`: ≥1 пик на луч
- `test_gpu_global_minmax`: max≥min, DR>0 дБ
- `test_gpu_full_pipeline`: process_full() → timing>0

---

### `test_strategies_pipeline.py` — полный GPU pipeline (5 вариантов) *(ROCm)*

Требует: ROCm GPU + `gpuworklib.AntennaProcessorTest` (pybind11).
Если GPU недоступен — все тесты SKIP.

**Класс:** `TestStrategiesPipeline` — один `proc` на все тесты (создаётся в `setUp()`).

| Тест | Вариант | W | Описание |
|------|---------|---|----------|
| `test_v1_cw_clean` | CW без шума | Identity | GEMM тривиален: X=I@S=S |
| `test_v2_cw_noise` | CW + AWGN | Identity | Шум не разрушает pipeline |
| `test_v3_cw_phase_delay` | CW + фазовая задержка | delay_and_sum | Нетривиальный GEMM |
| `test_v4_cw_phase_noise` | V3 + AWGN | delay_and_sum | Полный реальный сценарий |
| `test_v5_from_file` | Из файла | — | Заглушка → SKIP |

---

### `test_timing_analysis.py` — анализ timing JSON от C++ *(без GPU)*

Берёт `Results/strategies/timing_*.json` от C++ `TimingPerStepTest` и строит:
- Таблицу GPU ms / wall ms по шагам
- Bar chart → `Results/Plots/strategies/timing_{signal}.png`

**Без C++ JSON-файлов** — все тесты `SkipTest`.

| Тест | Что делает |
|------|-----------|
| `test_timing_files_exist` | Проверяет наличие JSON файлов |
| `test_timing_json_valid` | Структура JSON (signal, steps, gpu_ms, wall_ms) |
| `test_timing_sanity` | FullProcess < 1000 мс, все gpu_ms ≥ 0 |
| `test_plot_timing_bars` | Строит bar chart |

---

## Файлы — скрипты (не pytest)

### `debug_pipeline_steps.py` — PyCharm breakpoint-отладчик

```python
python Python_test/strategies/debug_pipeline_steps.py
```

Пошаговый NumPy pipeline для отладки в PyCharm (Shift+F9).
Ставь breakpoint на `← BREAKPOINT HERE`, нажимай F9 между шагами,
инспектируй переменные (S_raw, W, X_gemm, spectrum, magnitudes, one_max, all_maxima, minmax).

| Шаг | Переменная | Форма |
|-----|-----------|-------|
| STEP 0 | `S_raw` | [5, 8000] complex64 — входной сигнал |
| STEP 1 | `W` | [5, 5] complex64 — матрица весов |
| STEP 2 | `X_gemm` | [5, 8000] complex64 — X = W @ S |
| STEP 3 | `X_windowed` | [5, 8000] complex64 — после Hamming |
| STEP 4 | `spectrum`, `magnitudes` | [5, 16384] — после FFT |
| STEP 5_1 | `one_max` | list[dict] — 1 пик + парабола на луч |
| STEP 5_2 | `all_maxima` | list[list[dict]] — все пики |
| STEP 5_3 | `minmax` | list[dict] — MIN+MAX+DR |

Опциональные графики: `PLOT = True` → сохраняет в `Results/Plots/strategies/debug_pipeline_steps.png`.

---

### `plot_strategies_results.py` — визуализация Pipeline A vs B

```python
python Python_test/strategies/plot_strategies_results.py
```

Строит 4 графика в `Results/Plots/strategies/`:

| Файл | Описание |
|------|----------|
| `spectra_pipeline_a_vs_b.png` | Спектры Pipeline A vs B (луч 0, дБ) |
| `checkpoints_2_1_2_2_2_3.png` | S_raw / X_gemm / spectrum (антенна 0) |
| `peak_comparison_a_vs_b.png` | Найденные частоты по лучам (bar chart) |
| `farrow_raw_vs_aligned.png` | S_raw vs S_aligned (Farrow эффект) |

---

## Зависимости

### Без GPU (NumPy-only тесты)
```
numpy, scipy, matplotlib (опц.)
```
Работают: `test_scenario_builder`, `test_base_pipeline`, `test_debug_steps`,
`test_farrow_pipeline`, `test_timing_analysis` (после C++ run).

### С GPU (ROCm)
```
ROCm 7.2+ / HIP
gpuworklib.so (собрать: build/debian-radeon9070/python/)
```
Работают: `test_strategies_pipeline`, `test_strategies_step_by_step` (часть 2).

---

## Граф зависимостей между файлами

```
scenario_builder ←── test_scenario_builder
                 ←── test_farrow_pipeline
                 ←── plot_strategies_results

farrow_delay ←── pipeline_runner
             ←── test_farrow_pipeline
             ←── plot_strategies_results

pipeline_runner ←── test_farrow_pipeline
                ←── plot_strategies_results

signal_generators_strategy ←── strategy_test_base
                           ←── test_base_pipeline
                           ←── test_debug_steps

test_params ←── test_base_pipeline
            ←── test_debug_steps
            ←── strategy_test_base

strategy_test_base ←── test_base_pipeline

signal_factory ←── test_strategies_pipeline
numpy_reference ←── test_strategies_pipeline
pipeline_step_validator ←── test_strategies_pipeline

conftest ←── (фабричные хелперы для создания тестовых данных)
```

---

*Создан: 2026-03-21 | Author: Кодо (AI Assistant)*
