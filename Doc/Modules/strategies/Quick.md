# strategies — Краткий справочник

> GPU-обработка антенной матрицы: GEMM + Hamming + FFT + анализ спектра (ROCm / AMD GPU)

**Namespace**: `strategies` | **Каталог**: `modules/strategies/`

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Принять сигнал с N антенн, сформировать луч (GEMM), получить спектр (FFT) и найти пики — всё на GPU за ~35 мс (256 антенн × 1.2M отсчётов). Это высокоуровневый pipeline антенного цифрового формирования луча (DBF).

**Аналогия**: GEMM — "смешиваем" антенны по нужному направлению. FFT — смотрим спектр смеси. Остаток — ищем что интересное в спектре.

---

### AntennaProcessor / AntennaProcessor_v1

**Что делает**: принимает d_S (сигнал на GPU) + d_W (матрица весов на GPU), запускает полный pipeline и возвращает `AntennaResult`.

**Когда брать**: всегда в production. `AntennaProcessor` — абстрактный базовый класс. `AntennaProcessor_v1` — конкретная ROCm-реализация с 4 HIP-потоками.

**Ограничение**: только ROCm/AMD GPU (`ENABLE_ROCM=1`). На Windows без ROCm — бросает исключение.

---

### AntennaProcessorTest

**Что делает**: наследник `AntennaProcessor_v1`, открывает защищённые шаги для пошаговой отладки и тестирования.

**Когда брать**: в C++ тестах, когда нужно проверить каждый шаг pipeline по отдельности (step_0..step_6).

**Не брать** в production — только для тестов!

---

### WeightGenerator

**Что делает**: статический класс. Два метода:
1. `generate_delay_and_sum()` — вычислить матрицу весов W на CPU (формула delay-and-sum)
2. `upload_to_gpu()` — загрузить W на GPU

**Когда брать**: перед вызовом `process()`, нужно подготовить d_W.

---

### PostFftScenarioMode — что искать в спектре

| Режим | Что считает | Когда использовать |
|-------|-------------|-------------------|
| `ALL_REQUIRED` | Все три сценария | Production |
| `ONE_MAX_PARABOLA` | Один максимум + парабола (без фазы) | Debug Step2.1 |
| `ALL_MAXIMA` | Все локальные максимумы (limit=1000) | Debug Step2.2 |
| `GLOBAL_MINMAX` | Глобальный MIN + MAX + dynamic_range_dB | Debug Step2.3 |

---

### Checkpoint (ICheckpointSave / NullCheckpointSave)

**Что делает**: сохранение промежуточных данных (d_S, d_X, спектр, результаты) в бинарные файлы для отладки.

**Когда брать**: `NullCheckpointSave` — production (нулевой overhead, по умолчанию). `CheckpointSave` — включить через `save_cfg` в конфиге, если нужна диагностика данных.

---

## Pipeline (7 шагов)

```
[GPU] d_S + d_W (уже в VRAM)
      │
      ├─ Stream debug1 ─── Statistics(d_S) ──────────────── pre_input_stats
      │
      ├─ Stream main ──── hipBLAS Cgemm (X = W×S) ~13мс ── d_X
      │                         │
      │          ┌──────── debug2.2: Statistics(d_X) ──────── post_gemm_stats
      │          │
      │         Hamming window + zero-pad + hipFFT ~20мс ── d_spectrum
      │                         │
      │          ┌──────── debug2.3: Statistics(|spectrum|) ─ post_fft_stats
      │          │
      │   ┌──────┴──────────────────────────────────┐
      │   │ Step2.1: OneMax + 3-point Parabola      │ → one_max[]
      │   │ Step2.2: AllMaxima (limit=1000)         │ → all_maxima[]
      │   │ Step2.3: GlobalMinMax + dynamic_dB      │ → minmax[]
      │   └─────────────────────────────────────────┘
      │
      └─ AntennaResult { pre_stats, post_gemm_stats, post_fft_stats,
                         one_max, all_maxima, minmax, perf }
```

---

## 🔍 Пошаговая отладка в PyCharm

**Файл**: [`Python_test/strategies/debug_pipeline_steps.py`](../../Python_test/strategies/debug_pipeline_steps.py)

Запустить в PyCharm → поставить breakpoints на строки `← BREAKPOINT HERE` → F9 между шагами.

| Шаг | Переменная | Форма | Dtype | Что смотреть |
|-----|-----------|-------|-------|-------------|
| **STEP 0** | `S_raw` | `(5, 8000)` | complex64 | `np.abs(S_raw)` ≈ 1.0 кроме краёв; антенны задержаны |
| **STEP 1** | `W` | `(5, 5)` | complex64 | `np.linalg.norm(W, axis=1)` ≈ `[1.0, 1.0, ...]`; `np.abs(W)` ≈ 0.4472 |
| **STEP 2** | `X_gemm` | `(5, 8000)` | complex64 | Амплитуда выросла в √5≈2.24× (когерентное усиление) |
| **STEP 3** | `X_windowed` | `(5, 8000)` | complex64 | Края ≈ 0.08, центр ≈ 1.0 (Hamming taper) |
| **STEP 4** | `spectrum` | `(5, 16384)` | complex64 | — |
| | `magnitudes` | `(5, 16384)` | float32 | `peak_bin_beam0` → бин; `peak_freq_hz` ≈ 2 МГц |
| **STEP 5_1** | `one_max[b]` | `list[dict]` | — | `refined_freq_hz` ≈ 2.0 МГц; `freq_offset` ∈ [-0.5, +0.5] |
| **STEP 5_2** | `all_maxima[b]` | `list[list]` | — | Пики в порядке убывания; главный ≈ f₀ |
| **STEP 5_3** | `minmax[b]` | `list[dict]` | — | `dynamic_range_dB` > 40 дБ при хорошем сигнале |

**Ключевые переменные на финальном breakpoint:**
```python
one_max[0]['refined_freq_hz']   # уточнённая частота луча 0 (Гц)
one_max[0]['freq_offset']       # суббиновое смещение [-0.5..+0.5]
all_maxima[0][0]['freq_hz']     # первый пик из all_maxima
minmax[0]['dynamic_range_dB']   # динамический диапазон луча 0
```

**Включить графики**: установить `PLOT = True` в конце файла → сохраняет `Results/Plots/strategies/debug_pipeline_steps.png`

---

## Быстрый старт — C++

```cpp
#include "antenna_processor_v1.hpp"
#include "weight_generator.hpp"

// 1. Конфиг
strategies::AntennaProcessorConfig cfg;
cfg.n_ant          = 5;
cfg.n_samples      = 8000;
cfg.sample_rate    = 12.0e6f;
cfg.scenario_mode  = strategies::PostFftScenarioMode::ALL_REQUIRED;

// 2. Матрица весов
strategies::WeightParams wp;
wp.n_ant = 5;  wp.f0 = 2e6;  wp.tau_step = 100e-6;
auto W_cpu = strategies::WeightGenerator::generate_delay_and_sum(wp);
void* d_W = strategies::WeightGenerator::upload_to_gpu(backend, W_cpu);

// 3. Запуск
strategies::AntennaProcessor_v1 proc(backend, cfg);
strategies::AntennaResult r = proc.process(d_S, d_W);

// 4. Результаты
float f_peak = r.one_max[0].refined_freq_hz;   // Гц
float dyn_db = r.minmax[0].dynamic_range_dB;   // дБ
float t_ms   = r.perf.total_ms;                // мс
```

---

## Быстрый старт — Python

### Отладка по шагам (рекомендуется для знакомства)

Файл [`debug_pipeline_steps.py`](../../Python_test/strategies/debug_pipeline_steps.py) — NumPy pipeline с breakpoints для PyCharm:
```
python Python_test/strategies/debug_pipeline_steps.py
```

### Pipeline A и B (PipelineRunner)

```python
# cd GPUWorkLib && python
import sys; sys.path.insert(0, "Python_test/strategies")
from pipeline_runner import PipelineRunner, PipelineConfig
from scenario_builder import make_single_target

scenario = make_single_target(n_ant=5, theta_deg=30, f0_hz=2e6, fdev_hz=1e6)
runner   = PipelineRunner(output_dir="Results/strategies/test_01")
cfg      = PipelineConfig(save_input=True, save_spectrum=True)

result_a = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6, config=cfg)
result_b = runner.run_pipeline_b(scenario, steer_theta=30, config=cfg)
runner.print_comparison(result_a, result_b)

# Доступные данные после запуска:
result_a.S_raw        # [n_ant, n_samples] complex64 — входной сигнал
result_a.X_gemm       # [n_ant, n_samples] — после GEMM
result_a.spectrum     # [n_ant, nFFT] complex64 — спектр
result_a.magnitudes   # [n_ant, nFFT] float32 — |FFT|
result_a.peaks        # list[list[PeakInfo]] — пики по лучам
result_a.stats_input  # list[ChannelStats] — статистика до GEMM
result_a.stats_gemm   # list[ChannelStats] — статистика после GEMM
result_a.stats_spectrum  # list[ChannelStats] — статистика спектра
```

### GPU тест (AntennaProcessorTest, требует ROCm)

```python
import gpuworklib
ctx  = gpuworklib.ROCmGPUContext(0)
proc = gpuworklib.AntennaProcessorTest(ctx, n_ant=5, n_samples=8000,
                                       sample_rate=12e6, signal_frequency_hz=2e6,
                                       debug_mode=True)
proc.step_0_prepare_input(S_numpy, W_numpy)  # загрузка на GPU
X_gpu      = proc.step_2_gemm()              # [n_ant, n_samples] → CPU
spectrum   = proc.step_4_window_fft()        # [n_ant, nFFT] → CPU
one_max    = proc.step_6_1_one_max_parabola()   # list[dict]
all_maxima = proc.step_6_2_all_maxima()         # list[dict]
minmax     = proc.step_6_3_global_minmax()      # list[dict]
```

---

## Ключевые параметры AntennaProcessorConfig

| Параметр | Default | Описание |
|----------|---------|----------|
| `n_ant` | 5 | Число антенн |
| `n_samples` | 8000 | Отсчётов на антенну |
| `sample_rate` | 12e6 | Частота дискретизации, Гц |
| `scenario_mode` | ALL_REQUIRED | Что искать в спектре |
| `maxima_limit` | 1000 | Макс. кол-во пиков для Step2.2 |
| `signal_frequency_hz` | 2e6 | Целевая частота (для валидации) |
| `pre/post/fft_stats` | P61_ALL | Статистика на 3 точках отладки |
| `save_cfg` | nullptr | `nullptr` = NullCheckpointSave (zero overhead) |
| `debug_mode` | false | Включить D2H memcpy в отладочных точках |

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| ⚠️ | Только ROCm! На `ENABLE_ROCM=0` — бросает `std::runtime_error` |
| ⚠️ | d_S и d_W должны быть **уже на GPU** до вызова `process()` |
| ⚠️ | WeightGenerator::upload_to_gpu() выделяет память — нужно освободить вручную |
| ⚠️ | `AntennaProcessorTest` — только для тестов, не для production |
| ⚠️ | `debug_mode=true` → D2H memcpy в каждом шаге — медленно! |
| ⚠️ | Profiling: обязательно `SetGPUInfo()` перед `profiler.Start()` |

---

## Связи с другими модулями

- **fft_maxima** — Step2.2 (AllMaxima), переиспользует `AllMaximaBeamResult`
- **statistics** — PRE/POST статистика, переиспользует `StatisticsResult`
- **signal_generators** — FormSignalGeneratorROCm для тестов
- **lch_farrow** — PipelineB использует FarrowDelay для субсэмпловой задержки

---

## Ссылки

- [Full.md](Full.md) — математика, C4 диаграммы, таблица тестов
- [API.md](API.md) — все сигнатуры с цепочками вызовов
- [Doc/Modules/fft_maxima/Full.md](../fft_maxima/Full.md) — AllMaxima, SpectrumMaximaFinder
- [Doc/Modules/signal_generators/Full.md](../signal_generators/Full.md) — FormSignalGeneratorROCm
- [Doc/DrvGPU/Architecture.md](../../DrvGPU/Architecture.md) — IBackend, GPUProfiler

---

*Обновлено: 2026-03-09*
