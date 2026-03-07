# strategies — Полная документация модуля
# GPUWorkLib — Antenna Array Processor

> **Module path**: `modules/strategies/`
> **Date**: 2026-03-06
> **Status**: 📐 Architecture designed, implementation pending
> **Архитектура**: C4 Model (Simon Brown) — полный комплект документов

---

## ⚡ Быстрая навигация

| Что нужно | Файл |
|-----------|------|
| 🗺️ Место модуля в системе, акторы | [AP_C1_SystemContext.md](AP_C1_SystemContext.md) |
| 🐳 GPU streams, VRAM буферы, события | [AP_C2_Container.md](AP_C2_Container.md) |
| 🧩 SOLID, GRASP, GoF паттерны, файловая структура | [AP_C3_Component.md](AP_C3_Component.md) |
| 💻 Классы, интерфейсы, сигнатуры, конфиг | [AP_C4_Code.md](AP_C4_Code.md) |
| ⏱️ Диаграммы последовательности, тайминги | [AP_Seq.md](AP_Seq.md) |

---

## 1. Что делает модуль

**strategies** — GPU-модуль обработки данных антенной матрицы.

```
S[N_ant × N_samples]  ──┐
W[N_ant × N_ant]      ──┘
        │
        ▼
  ┌────────────────────────────────────────────────────────────┐
  │                  AntennaProcessor_v1                       │
  │                                                            │
  │  1. Input already on GPU: d_S + metadata                   │
  │  2. Debug 2.1: stats/save/python по d_S                    │
  │  3. GEMM: X = W × S (hipBLAS Cgemm, ~13 мс)               │
  │  4. Debug 2.2: stats/save/python по d_X                    │
  │  5. Base block: Window + FFT                               │
  │  6. Debug 2.3: stats/save/python по |spectrum|             │
  │  7. Post-FFT сценарии (все обязательны по ТЗ):             │
  │     ├─ Step2.1: One MAX + 3-point Parabola (no phase)      │
  │     ├─ Step2.2: ALL maxima (limit=1000)                    │
  │     └─ Step2.3: Global MAX/MIN (limit=1000)                │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
  AntennaResult { pre_stats, post_stats, minmax/peaks/all_maxima, perf }
```

---

## 2. Классы модуля (C4 уровень)

### Иерархия

```
AntennaProcessor          ← abstract base (antenna_processor.hpp)
└── AntennaProcessor_v1   ← concrete impl (antenna_processor_v1.hpp)

IPostFftScenario          ← interface (i_post_fft_scenario.hpp)
├── OneMaxParabolaScenario ← Step2.1: one MAX + parabola (no phase)
├── AllMaximaScenario      ← Step2.2: all maxima
└── GlobalMinMaxScenario   ← Step2.3: global MAX/MIN

ICheckpointSave           ← interface (i_checkpoint_save.hpp)
├── NullCheckpointSave    ← production, no-op (zero overhead)
└── CheckpointSave        ← debug, binary files

StrategyFactory           ← создаёт AntennaProcessor_v1 + strategy + checkpoint
```

### Быстрый старт (usage)

```cpp
// Создание
AntennaProcessorConfig cfg;
cfg.n_ant          = 256;
cfg.n_samples      = 1'200'000;
cfg.scenario_mode      = PostFftScenarioMode::ALL_REQUIRED;
cfg.pre_input_stats    = StatPreset::P61_ALL;
cfg.post_gemm_stats    = StatPreset::P61_ALL;
cfg.post_fft_stats     = StatPreset::P61_ALL;

auto proc = StrategyFactory::create(ctx, cfg);  // ctx = DrvGPU

// Запуск
AntennaResult r = proc->process(S_data, W_data);
// S_data: hipFloatComplex[N_ant × N_samples]
// W_data: hipFloatComplex[N_ant × N_ant]

// Доступ к результатам
r.one_max[beam].refined_frequency_hz  // найденная частота (parabola, no phase)
r.pre_input_stats[beam].median        // медиана сигнала до GEMM
r.perf.total_ms              // время выполнения
```

---

## 3. Файловая структура

```
modules/strategies/
├── include/
│   ├── antenna_processor.hpp              # AntennaProcessor (abstract base)
│   ├── antenna_processor_v1.hpp           # AntennaProcessor_v1 (concrete, v1)
│   ├── interfaces/
│   │   ├── i_branch_strategy.hpp          # IBranchStrategy
│   │   └── i_checkpoint_save.hpp          # ICheckpointSave
│   ├── branch_strategies/
│   │   ├── one_max_parabola_scenario.hpp  # Step2.1
│   │   ├── all_maxima_scenario.hpp        # Step2.2
│   │   └── global_minmax_scenario.hpp     # Step2.3
│   ├── checkpoint/
│   │   ├── null_checkpoint_save.hpp       # no-op (production)
│   │   └── checkpoint_save.hpp            # binary saves (debug)
│   ├── strategy_factory.hpp               # StrategyFactory
│   └── config/
│       ├── antenna_processor_config.hpp   # AntennaProcessorConfig
│       ├── statistics_set.hpp             # StatisticsSet bitmask
│       └── post_fft_scenario_mode.hpp     # PostFftScenarioMode enum
├── src/
│   ├── antenna_processor_v1.cpp
│   ├── branch_strategies/
│   │   ├── one_max_parabola_scenario.cpp
│   │   ├── all_maxima_scenario.cpp
│   │   └── global_minmax_scenario.cpp
│   └── strategy_factory.cpp
├── kernels/
│   ├── gemm_wrapper.hpp                   # GemmWrapper (hipBLAS)
│   └── window_fft.hpp                     # общий блок Window + FFT
└── tests/
    ├── all_test.hpp
    ├── README.md
    ├── test_gemm_correctness.hpp
    ├── test_one_max_parabola_no_phase.hpp
    ├── test_all_maxima.hpp
    ├── test_global_minmax.hpp
    ├── test_statistics_pre_post_fft.hpp
    ├── test_checkpoint_save.hpp
    └── test_fft_mirror_fold.hpp
```

---

## 4. Конфигурация

```cpp
struct AntennaProcessorConfig {
    // Размеры
    uint32_t  n_ant          = 256;
    uint32_t  n_samples      = 1'200'000;
    float     sample_rate    = 12.0e6f;

    // Алгоритм
    PostFftScenarioMode scenario_mode = PostFftScenarioMode::ALL_REQUIRED;
    uint32_t  maxima_limit            = 1000;
    bool      run_all_required_post_fft = true; // по ТЗ считаем все 3 post-FFT сценария

    // Статистика (bitmask пресеты, можно = StatPreset::NONE)
    StatisticsSet pre_input_stats = StatPreset::P61_ALL;      // 2.1 по d_S
    StatisticsSet post_gemm_stats = StatPreset::P61_ALL;      // 2.2 по d_X
    StatisticsSet post_fft_stats  = StatPreset::P61_ALL;      // 2.3 по |spectrum|

    // Checkpoint (nullptr = NullCheckpointSave, zero overhead)
    const CheckpointSaveConfig* save_cfg = nullptr;
};

// Статистические пресеты
namespace StatPreset {
    NONE         = 0;                  // отключить
    P61_ALL      = MEAN|MED|STD|VAR|MIN|MAX;  // полный набор
    P62_MEAN_MED = MEAN|MEDIAN;        // среднее + медиана
    P63_MED_MM   = MEAN|MED|MIN|MAX;  // среднее + медиана + min/max
    P64_STD_VAR  = STD|VAR;           // дисперсия
}
```

---

## 5. Результаты

```cpp
struct AntennaResult {
    std::vector<StatisticsResult>    pre_input_stats;   // 2.1: по d_S
    std::vector<StatisticsResult>    post_gemm_stats;   // 2.2: по d_X
    std::vector<StatisticsResult>    post_fft_stats;    // 2.3: по |spectrum|
    std::vector<OneMaxParabolaLite>  one_max;           // Step2.1
    std::vector<AllMaximaBeamResult> all_maxima;        // Step2.2
    std::vector<MinMaxResult>        minmax;            // Step2.3
    PostFftScenarioMode              scenario_mode;
    PerfMetrics                      perf;        // timing per step, ms
};

// MaxValue — переиспользован из modules/fft_maxima (32 байта):
// struct MaxValue { frequency_hz, magnitude, bin_index, ... };

// MinMaxResult — Step2.3 (32 байта):
// struct MinMaxResult { beam_id, min_mag, min_bin, min_frequency_hz,
//                                max_mag, max_bin, max_frequency_hz, dynamic_range_dB };
```

---

## 6. GPU Потоки и тайминги

```
Stream 0 (DMA)  ──────────────► event_data_ready
                                       │
                    ┌──────────────────┴─────────────────┐
                    ▼                                      ▼
Stream 1 (Stats)   welford(d_S) + sort                Stream 2 (Main)
                   ► event_stats_done                   hipblasCgemm  ► event_gemm_done
                                                        │                      │
                                                        │              Stream 3 (SPost)
                                                        apply_hamming  welford(d_X) ► event_spost_done
                                                        hipFFT ► event_fft_done
                                                        Branch strategy

Синхронизация (CPU):
  hipEventSynchronize(event_stats_done)
  hipEventSynchronize(event_spost_done)
  hipEventSynchronize(event_fft_done)
```

| Шаг | Время (256×1.2M, 9070) | Ограничение |
|-----|------------------------|-------------|
| DMA Host→GPU | 78 мс (PCIe 4.0) / 2.6 мс (VRAM→VRAM) | PCIe BW |
| Stats PRE-GEMM | ~2.6 мс (параллельно!) | BW-bound |
| **GEMM** | **~13 мс** | Compute-bound |
| Stats POST-GEMM | ~2.6 мс (параллельно!) | BW-bound |
| Hamming | ~2.6 мс | BW-bound |
| **FFT batch** | **~20 мс** (TBD, нужен бенчмарк) | Compute+BW |
| Step2.1 / Step2.3 | < 1 мс | Compute-light |
| Step2.2 | 2–5 мс | Compute+BW |
| **ИТОГО (из VRAM)** | **~35 мс** | GEMM + FFT |

---

## 7. GoF Паттерны

| Паттерн | Реализация | Зачем |
|---------|-----------|-------|
| **Strategy** | `IBranchStrategy` → Min/Max/Parabola/AllMaxima | Переключение ветки без перекомпиляции |
| **Factory Method** | `StrategyFactory::create()` | Создаёт AntennaProcessor_v1 + нужную Strategy + CheckpointSave |
| **Null Object** | `NullCheckpointSave` | Production: сохранение = no-op, нулевой overhead |
| **Template Method** | `AntennaProcessor_v1::process()` | Фиксированный порядок шагов + изменяемый Branch |

---

## 8. Checkpoint сохранение

| Точка | Что сохраняется | Размер | По умолчанию |
|-------|----------------|--------|--------------|
| C1 signal | `d_S[N_ant × N_samples]` | 2.5 ГБ | ❌ (дорого!) |
| C1 weights | `d_W[N_ant × N_ant]` | 512 КБ | ❌ |
| C2 data | `d_X` после GEMM | 2.5 ГБ | ❌ (дорого!) |
| C2 stats | PRE+POST stats | 28 КБ | ❌ |
| C3 result | `MinMaxResult[N_ant]` | 8 КБ | ✅ (дёшево) |
| C3 spectrum | `d_spectrum` полный | 4.9 ГБ | ❌ (огромный!) |
| C4 peak | `MaxValue[N_ant]` | 12 КБ | ✅ (дёшево) |

**Путь**: `Logs/GPU_{id}/antenna_processor/YYYY-MM-DD/HH-MM-SS/`

**Production**: `cfg.save_cfg = nullptr` → `NullCheckpointSave` (zero overhead)

---

## 9. Tasks (план реализации)

### Фаза 1 — Инфраструктура (2–3 дня)
- [ ] Создать `modules/strategies/` файловую структуру
- [ ] `AntennaProcessor`, `IBranchStrategy`, `ICheckpointSave` (интерфейсы)
- [ ] `AntennaProcessorConfig`, `StatisticsSet`, `BranchMode`
- [ ] `NullCheckpointSave`
- [ ] `StrategyFactory`

### Фаза 2 — Ядро pipeline (3–4 дня)
- [ ] `GemmWrapper` (hipBLAS Cgemm)
- [ ] `HammingProcessor` (apply_hamming.hip)
- [ ] `StatisticsProcessor` интеграция (PRE-GEMM, POST-GEMM)
- [ ] `hipFFT` batch plan (с кешированием)
- [ ] `fold_fft_mirror()`

### Фаза 3 — Стратегии (2–3 дня)
- [ ] `OneMax + Parabola` без фазы в `modules/fft_maxima/`
- [ ] `AllMaxima` с limit=`1000` в `modules/fft_maxima/`
- [ ] `GlobalMinMax` с limit=`1000` в `modules/fft_maxima/`
- [ ] `PostFFTStatistics(|spectrum|)` в `modules/statistics/`

### Фаза 4 — Checkpoint (1–2 дня)
- [ ] `CheckpointSave` binary format + JSON header
- [ ] `DataFormatRegistry` новые форматы

### Фаза 5 — Тесты (2–3 дня)
- [ ] `test_gemm_correctness.hpp` vs NumPy
- [ ] `test_one_max_parabola_no_phase.hpp`
- [ ] `test_all_maxima.hpp`, `test_global_minmax.hpp`
- [ ] `test_statistics_pre_post_fft.hpp`
- [ ] `test_checkpoint_save.hpp`
- [ ] `test_fft_mirror_fold.hpp`
- [ ] Python тесты в `Python_test/strategies/`

### Фаза 6 — Профилирование (1 день)
- [ ] `GPUProfiler::SetGPUInfo()` + Start/Stop
- [ ] Бенчмарк GEMM, Window+FFT, Step2.1/2.2/2.3
- [ ] `profiler.ExportMarkdown()` + `ExportJSON()`

---

## 10. VRAM Layout (256 антенн × 1.2M выборок)

| Буфер | Размер | Назначение |
|-------|--------|-----------|
| `d_S` | 2.45 ГБ | Входной сигнал (READ-ONLY) |
| `d_W` | 512 КБ | Матрица весов (в L2 кеше!) |
| `d_X` | 2.45 ГБ | GEMM output + Hamming in-place |
| `d_hamming` | 4.8 МБ | Окно Хемминга (в L2 кеше!) |
| `d_spectrum` | 4.92 ГБ | FFT output |
| **ИТОГО** | **≈ 10.3 ГБ** | Влезает в 16 ГБ AMD 9070 ✅ |

---

## 11. Детальные C4 документы

| Уровень | Документ | Содержание |
|---------|----------|-----------|
| C1 System Context | [AP_C1_SystemContext.md](AP_C1_SystemContext.md) | Акторы, внешние системы, PlantUML |
| C2 Container | [AP_C2_Container.md](AP_C2_Container.md) | GPU streams, HIP events, VRAM схема |
| C3 Component | [AP_C3_Component.md](AP_C3_Component.md) | SOLID, GRASP, GoF, файловая структура |
| C4 Code | [AP_C4_Code.md](AP_C4_Code.md) | Все интерфейсы и сигнатуры |
| Sequences | [AP_Seq.md](AP_Seq.md) | Pipeline timing, FFT fold, chunking |

---

*Created: 2026-03-06 | Module: `modules/strategies/` | Author: Кодо*
