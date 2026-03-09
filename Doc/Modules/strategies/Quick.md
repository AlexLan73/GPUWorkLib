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

## Pipeline pipeline (7 шагов)

```
[GPU] d_S + d_W (уже в VRAM)
      │
      ├─ Stream debug1 ─── Statistics(d_S) ──────────────── pre_input_stats
      │
      ├─ Stream main ──── hipBLAS Cgemm (X = W×S) ~13мс ── d_X
      │                         │
      │                   Hamming + FFT ~20мс ────────────── d_spectrum
      │                         │
      │   ┌─────────────────────┴──────────────────────┐
      │   │ Step2.1: OneMax + 3-point Parabola         │ → one_max[]
      │   │ Step2.2: AllMaxima (limit=1000)            │ → all_maxima[]
      │   │ Step2.3: GlobalMinMax                      │ → minmax[]
      │   └────────────────────────────────────────────┘
      │
      └─ AntennaResult { pre_stats, post_stats, one_max, all_maxima, minmax, perf }
```

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

## Быстрый старт — Python (pipeline_runner.py)

```python
from Python_test.strategies.pipeline_runner import PipelineRunner, PipelineConfig
from Python_test.strategies.scenario_builder import make_single_target

scenario = make_single_target(n_ant=8, theta_deg=30, f0_hz=2e6)
runner   = PipelineRunner(output_dir="Results/strategies/test_01")
cfg      = PipelineConfig(save_input=True, save_spectrum=True)

result   = runner.run_pipeline_a(scenario, steer_theta=30, steer_freq=2e6, config=cfg)
comp     = runner.run_pipeline_b(scenario, steer_theta=30, config=cfg)
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
