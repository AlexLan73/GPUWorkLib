# C3 — Component Diagram: AntennaProcessor Module
# GPUWorkLib — Antenna Array Processor

> **Project**: GPUWorkLib / AntennaProcessor
> **Date**: 2026-03-06
> **Reference**: [c4model.com](https://c4model.com)
> **Level**: 3 (Component) — компоненты внутри модуля AntennaProcessor

> **Update 2026-03-07**: итоговая семантика модуля — `Window + FFT` как общий блок, затем `Step2.1/2.2/2.3`. В нескольких старых ASCII-фрагментах ниже могут встречаться legacy-имена `Branch 2/3/4`; их следует читать как:
> - `Branch 3` → `Step2.1 OneMax + Parabola`
> - `Branch 4` → `Step2.2 AllMaxima`
> - `Branch 2` → `Step2.3 GlobalMinMax`

---

## 1. Общая структура компонентов

```
┌──────────────────────────────────────────────────── strategies ─────────────────────────────────────────────────────┐
│                                                                                                                       │
│  ┌─────────────────────────── Interface Layer ─────────────────────────────────────────────────────────────────┐    │
│  │                                                                                                               │    │
│  │  ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐   │    │
│  │  │ AntennaProcessor                                 │   │ IBranchStrategy                              │   │    │
│  │  │ ──────────────────────────────────────────────── │   │ ────────────────────────────────────────────  │   │    │
│  │  │ + process(S, W, cfg) → AntennaResult             │   │ + execute(spectrum, params) → BranchResult   │   │    │
│  │  │ + set_statistics(StatisticsSet pre, post)        │   │ + name() → string                            │   │    │
│  │  │ + set_branch(BranchMode mode)                    │   │                                               │   │    │
│  │  └──────────────────────────────────────────────────┘   └──────────────────────────────────────────────┘   │    │
│  │             ▲ implements                                              ▲ implements                             │    │
│  └─────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────┘    │
│                │                                                          │                                           │
│  ┌─────────────┼──────────────── Main Orchestrator ────────────┐         │                                           │
│  │             ▼                                                 │         │                                           │
│  │  ┌───────────────────────────────────────────────────────┐  │         ├────────────────────────────────────────┐  │
│  │  │  AntennaProcessor_v1                                │  │         │                                         │  │
│  │  │  ────────────────────────────────────────────────────│  │   ┌─────▼──────────────┐  ┌──────────────────┐   │  │
│  │  │  - gemm_ : GemmWrapper                               │  │   │ MinMaxStrategy     │  │ ParabolaStrategy │   │  │
│  │  │  - hamming_ : HammingProcessor                       │  │   │ (Branch 2)         │  │ (Branch 3)       │   │  │
│  │  │  - fft_ : hipFFTProcessor                            │  │   │ ──────────────────  │  │ ──────────────── │   │  │
│  │  │  - stats_pre_  : StatisticsProcessor&                │  │   │ minmax_spectrum     │  │ post_kernel_     │   │  │
│  │  │  - stats_post_ : StatisticsProcessor&                │  │   │ (1 блок/луч, 256T) │  │ one_peak +       │   │  │
│  │  │  - strategy_ : IBranchStrategy*                      │  │   │ → MinMaxResult[N]  │  │ парабола 3-point │   │  │
│  │  │  - checkpoint_ : ICheckpointSave*                    │  │   │                     │  │ → MaxValue[N]    │   │  │
│  │  │                                                       │  │   └─────────────────────┘  └──────────────────┘   │  │
│  │  │  + process(S, W, cfg) → AntennaResult                │  │   ┌──────────────────────────────────────────┐    │  │
│  │  │    1. DMA load (Stream 0)                             │  │   │ AllMaximaStrategy                         │    │  │
│  │  │    2. C1 checkpoint [optional]                        │  │   │ (legacy label, now Step2.2 AllMaxima)     │    │  │
│  │  │    3. Stats PRE-GEMM (Stream 1, parallel)            │  │   │ ──────────────────────────────────────────  │    │  │
│  │  │    4. GEMM: X = W × S (Stream 2)                     │  │   │ detect_all_maxima + stream compaction      │    │  │
│  │  │    5. C2 checkpoint [optional]                        │  │   │ → AllMaximaBeamResult[N]                   │    │  │
│  │  │    6. Stats POST-GEMM (Stream 3, parallel with 7)    │  │   └──────────────────────────────────────────┘    │  │
│  │  │    7. Hamming: X[n] *= w[n]                          │  └─────────────────────────────────────────────────────┘  │
│  │  │    8. FFT batch (hipFFT, N_ant × nFFT)               │                                                           │
│  │  │    9. FFT fold: bins k>nFFT/2 → negative freq        │                                                           │
│  │  │   10. strategy_->execute(spectrum, params)            │                                                           │
│  │  │   11. C3/C4 checkpoint [optional]                     │                                                           │
│  │  └───────────────────────────────────────────────────────┘                                                           │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                       │
│  ┌────────────────────── Support Components ────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                                                │    │
│  │  ┌────────────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────────────────┐    │    │
│  │  │ GemmWrapper                │    │ HammingProcessor          │    │ CheckpointSave / NullCheckpointSave │    │    │
│  │  │ ──────────────────────────  │    │ ──────────────────────────│    │ ────────────────────────────────────│    │    │
│  │  │ hipBLAS Cgemm              │    │ apply_hamming.hip         │    │ ICheckpointSave (abstract)          │    │    │
│  │  │ W[N×N] × S[N×M] = X[N×M]  │    │ w[n] = 0.54-0.46cos(2πn) │    │ + save_c1(), save_c2()             │    │    │
│  │  │                             │    │ reuses d_hamming (L2)     │    │ + save_c3(), save_c4()             │    │    │
│  │  │ caches plan when:          │    │ lazy init: only if        │    │                                     │    │    │
│  │  │   N_ant, N_samples same    │    │   N_samples changed       │    │ NullCheckpointSave: no-op (prod)   │    │    │
│  │  └────────────────────────────┘    └──────────────────────────┘    │ CheckpointSave: binary to           │    │    │
│  │                                                                      │   Logs/GPU_XX/antenna_processor/   │    │    │
│  │  ┌────────────────────────────┐    ┌──────────────────────────┐    └────────────────────────────────────┘    │    │
│  │  │ StrategyFactory    │    │ AntennaProcessorConfig   │                                               │    │
│  │  │ ──────────────────────────  │    │ ──────────────────────────│                                               │    │
│  │  │ GoF: Factory Method        │    │ N_ant, N_samples          │                                               │    │
│  │  │ + create(ctx, cfg)         │    │ sample_rate               │                                               │    │
│  │  │ → AntennaProcessor*       │    │ branch_mode : BranchMode  │                                               │    │
│  │  │                             │    │ pre_stats  : StatisticsSet│                                               │    │
│  │  │ Выбирает реализацию        │    │ post_stats : StatisticsSet│                                               │    │
│  │  │ по backend (ROCm/OpenCL)   │    │ search_range (nFFT/2)     │                                               │    │
│  │  └────────────────────────────┘    └──────────────────────────┘                                               │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                       │
│  ┌────────────────────── Reused Modules (external) ────────────────────────────────────────────────────────────┐    │
│  │  StatisticsProcessor   (modules/statistics/)        — welford_fused, radix_sort, medians                    │    │
│  │  FFTProcessor          (modules/fft_processor/)     — hipFFT batch plan, ColllectOrRelease pattern          │    │
│  │  MaxValue / SpectrumResult (modules/fft_maxima/)    — spectrum_result_types.hpp (REUSED as-is!)             │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. SOLID анализ

### S — Single Responsibility

| Класс | Одна ответственность |
|-------|---------------------|
| `AntennaProcessor_v1` | Оркестрирует pipeline: input GPU→debug 2.1→GEMM→debug 2.2→Window+FFT→debug 2.3→post-FFT scenarios |
| `GemmWrapper` | ТОЛЬКО GEMM: W × S = X (hipBLAS Cgemm) |
| `WindowFFTBlock` | ТОЛЬКО окно + FFT, общий reusable этап |
| `GlobalMinMaxScenario` | ТОЛЬКО поиск глобального max/min по спектру |
| `OneMaxParabolaScenario` | ТОЛЬКО поиск одного максимума + parabola interpolation без фазы |
| `AllMaximaScenario` | ТОЛЬКО поиск всех локальных пиков (CFAR) |
| `CheckpointSave` | ТОЛЬКО сохранение checkpoint-данных в файлы |
| `NullCheckpointSave` | ТОЛЬКО no-op заглушка (Null Object GoF) |

✅ **Каждый класс делает одно дело**

### O — Open/Closed

```
Для добавления Branch 5 (например, matched filter):

ДОБАВИТЬ:   class MatchedFilterStrategy : public IBranchStrategy { ... }
НЕ ТРОГАТЬ: AntennaProcessor_v1, MinMaxStrategy, ParabolaStrategy, ...

Для нового формата сохранения:

ДОБАВИТЬ:   class BinaryZipCheckpointSave : public ICheckpointSave { ... }
НЕ ТРОГАТЬ: AntennaProcessor_v1, CheckpointSave (binary), ...
```

✅ **Расширяется через новые реализации интерфейса, не модификацию существующего**

### L — Liskov Substitution

```cpp
AntennaProcessor* proc = factory.create(ctx, cfg);
// Работает с любой реализацией: AntennaProcessor_v1, MockAntennaProcessor, ...

proc->set_scenario_mode(PostFftScenarioMode::ALL_REQUIRED);
proc->process(S, W, cfg);                    // гарантированно возвращает AntennaResult

proc->set_scenario_mode(PostFftScenarioMode::ONE_MAX_PARABOLA);
proc->process(S, W, cfg);                    // тот же контракт
```

✅ **Все стратегии взаимозаменяемы через IBranchStrategy**

### I — Interface Segregation

```
AntennaProcessor        → только process() + set_branch() + set_statistics()
IBranchStrategy          → только execute() + name()
ICheckpointSave          → только save_c1(), save_c2(), save_c3(), save_c4()

НЕТ "fat interfaces" — каждый интерфейс минимален
```

✅ **Клиент не зависит от методов, которые не использует**

### D — Dependency Inversion

```
AntennaProcessor_v1 зависит от АБСТРАКЦИЙ:
  - IBranchStrategy*      (не MinMaxBranchStrategy напрямую)
  - ICheckpointSave*      (не CheckpointSave напрямую)
  - StatisticsProcessor&  (через DrvGPU context)

StrategyFactory создаёт КОНКРЕТНЫЕ реализации:
  - new MinMaxBranchStrategy()
  - new NullCheckpointSave() или new CheckpointSave(path)
```

✅ **Зависимость направлена вверх (к абстракциям)**

---

## 3. GRASP анализ

| Паттерн GRASP | Реализация |
|---------------|-----------|
| **Controller** | `AntennaProcessor_v1::process()` — управляет всем pipeline, координирует stream'ы и события |
| **Creator** | `StrategyFactory` — создаёт AntennaProcessor_v1 + нужные Strategy + CheckpointSave |
| **Information Expert** | `GemmWrapper` знает о GEMM; `HammingProcessor` знает о cosine window; `MinMaxBranchStrategy` знает о tree reduction |
| **High Cohesion** | Каждый класс сфокусирован: GemmWrapper = чистый GEMM, CheckpointSave = чистый I/O |
| **Low Coupling** | Стратегии подключаются через `IBranchStrategy*`; Stats через `StatisticsProcessor&` (не создаётся внутри) |
| **Polymorphism** | `strategy_->execute()` — runtime polymorphism через vtable; выбор ветки без switch в основном потоке |
| **Pure Fabrication** | `CheckpointSave` — не domain объект, но нужен для debug; `NullCheckpointSave` — исчезает в production |
| **Indirection** | `ICheckpointSave*` — развязывает main pipeline от I/O; pipeline не знает сохраняются данные или нет |
| **Protected Variations** | `IBranchStrategy` защищает pipeline от изменений в алгоритмах ветвления |

---

## 4. GoF Design Patterns

### 4.1 Strategy (Поведенческий)

```
Задача: POST-FFT обработка бывает 3 обязательных видов (Step2.1/2.2/2.3),
        и все они должны работать от одного общего `d_spectrum`

┌─────────────────────────────────────────────────────────────┐
│                     <<interface>>                            │
│                    IBranchStrategy                           │
│  ─────────────────────────────────────────────────────────  │
│  + execute(d_spectrum, nFFT, N_ant, params) → BranchResult  │
│  + name() → string                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ implements
          ┌────────────────┼────────────────────────┐
          ▼                ▼                         ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
  │GlobalMinMax  │  │OneMaxParabola│  │AllMaximaScenario         │
  │ (Step2.3)    │  │ (Step2.1)    │  │ (Step2.2)                │
  │ ────────────  │  │ ────────────  │  │ ────────────────────────  │
  │ global_minmax │  │ one_peak     │  │ detect_all_maxima        │
  │ spectrum      │  │ one_peak     │  │ + Blelloch prefix scan   │
  │ kernel        │  │ kernel       │  │ + stream compaction      │
  └──────────────┘  └──────────────┘  └──────────────────────────┘

Context:
  AntennaProcessor_v1 {
    vector<IPostFftScenario*> scenarios_;  // Step2.1 / 2.2 / 2.3
    ...
    // в process():
    run_window_fft_once(d_X, d_spectrum);
    for (auto* s : scenarios_) { s->execute(d_spectrum, ...); }
  }
```

### 4.2 Factory Method (Порождающий)

```cpp
// Abstract creator
class StrategyFactory {
public:
    virtual AntennaProcessor* create(DrvGPU& ctx,
                                       const AntennaProcessorConfig& cfg) = 0;
};

// Concrete creator (для ROCm backend)
class StrategyFactory_ROCm : public StrategyFactory {
public:
    AntennaProcessor* create(DrvGPU& ctx,
                               const AntennaProcessorConfig& cfg) override {
        auto* strategy = make_strategy(cfg.branch_mode);
        auto* checkpoint = make_checkpoint(cfg.save_cfg);
        return new AntennaProcessor_v1(ctx, strategy, checkpoint, cfg);
    }
private:
    IBranchStrategy* make_strategy(BranchMode mode) {
        switch (mode) {
            case BranchMode::MINMAX:      return new MinMaxBranchStrategy();
            case BranchMode::PARABOLA:    return new ParabolaBranchStrategy();
            case BranchMode::ALL_MAXIMA:  return new AllMaximaBranchStrategy();
        }
    }
    ICheckpointSave* make_checkpoint(const CheckpointSaveConfig* cfg) {
        if (cfg == nullptr) return new NullCheckpointSave();
        return new CheckpointSave(*cfg);
    }
};
```

### 4.3 Null Object (Структурный/Поведенческий)

```cpp
// Абстрактный интерфейс
class ICheckpointSave {
public:
    virtual ~ICheckpointSave() = default;
    virtual void save_c1_signal(const float2_t* d_data, ...) = 0;
    virtual void save_c2_data(const float2_t* d_X, ...) = 0;
    virtual void save_c3_minmax(const MinMaxResult* results, ...) = 0;
    virtual void save_c4_peak(const MaxValue* results, ...) = 0;
};

// PRODUCTION: Null Object — полностью бесплатный (inline, no-op)
class NullCheckpointSave final : public ICheckpointSave {
public:
    void save_c1_signal(...)  override {}  // ← compiler eliminates this
    void save_c2_data(...)    override {}
    void save_c3_minmax(...)  override {}
    void save_c4_peak(...)    override {}
};

// DEBUG: реальное сохранение
class CheckpointSave final : public ICheckpointSave {
    CheckpointSaveConfig cfg_;
    void save_c1_signal(const float2_t* d_data, size_t n_ant, size_t n_samples) override;
    void save_c2_data(const float2_t* d_X, size_t n_ant, size_t n_samples) override;
    void save_c3_minmax(const MinMaxResult* results, size_t n_ant) override;
    void save_c4_peak(const MaxValue* results, size_t n_ant) override;
};
```

### 4.4 Template Method (Поведенческий) — в процессе

```
AntennaProcessor_v1::process() — фиксирует порядок шагов:
  1. load_data()          ← фиксирован
  2. checkpoint_c1()      ← фиксирован (но NullCheckpointSave = nop)
  3. run_stats_pre()      ← фиксирован
  4. run_gemm()           ← фиксирован
  5. checkpoint_c2()      ← фиксирован
  6. run_stats_post()     ← фиксирован
  7. run_hamming()        ← фиксирован
  8. run_fft()            ← фиксирован
  9. fold_fft_mirror()    ← фиксирован
  10. strategy_->execute() ← ИЗМЕНЯЕМЫЙ ШАГ (через Strategy)
  11. checkpoint_c3c4()   ← фиксирован

Изменяемый шаг = Strategy.execute()
Фиксированная структура = Template Method
```

---

## 5. FFT Mirror Folding (Примечание #2)

```
Проблема: hipFFT C2C output для комплексного сигнала:

  Бин    Частота           Значение
  ─────  ─────────────    ────────────────────────────────────────
  0      0 Гц (DC)         spectrum[0]
  1..N/2 +fs/N .. +fs/2    положительные частоты
  N/2    +fs/2 (Nyquist)   spectrum[N/2]
  N/2+1  -fs/2+fs/N        spectrum[N/2+1] → соответствует -(N-N/2-1)*fs/N
  ...
  N-1    -fs/N             spectrum[N-1]  → соответствует -fs/N

Если цель: только положительные частоты, search_range = nFFT/2
           Если пик найден в [N/2+1, N-1] → он в зоне отрицательных частот

Алгоритм fold_fft_mirror():
  for each peak at bin k:
    if (k > nFFT / 2):
        // Перекладываем в отрицательную частоту (fold left)
        peak.frequency_hz = (k - nFFT) * sample_rate / nFFT;  // отрицательная!
        peak.bin_folded = k - nFFT;                             // [-N/2, -1]
    else:
        peak.frequency_hz = k * sample_rate / nFFT;            // положительная
        peak.bin_folded = k;

ПРИМЕЧАНИЕ: для beamforming сигнал обычно IQ (аналитический),
           поэтому спектр НЕ симметричен — оба полупространства несут информацию!
           Но peak search по умолчанию ищет в [0, nFFT/2) (positive side).
           Для full-spectrum search: search_range = nFFT (флаг full_spectrum_search).
```

---

## 6. Статистика: два прохода

```
Конфигурация через AntennaProcessorConfig:

  struct AntennaProcessorConfig {
      StatisticsSet pre_gemm_stats  = StatPreset::P61_ALL;    // на сырой S
      StatisticsSet post_gemm_stats = StatPreset::P62_MEAN_MED; // на beamformed X
      // ...
  };

  Если pre_gemm_stats == StatField::NONE:
      Stream 1 (Statistics) НЕ запускается → экономия GPU-времени

  Если post_gemm_stats == StatField::NONE:
      POST-GEMM stats pass НЕ запускается → только GEMM → Hamming → FFT
```

---

## 7. Структура файлов модуля

```
modules/strategies/
├── include/
│   ├── antenna_processor.hpp              # AntennaProcessor (abstract base class, без 'I')
│   ├── antenna_processor_v1.hpp           # AntennaProcessor_v1 (concrete, индекс версии)
│   ├── branch_strategies/
│   │   ├── i_branch_strategy.hpp          # IBranchStrategy (Strategy GoF)
│   │   ├── minmax_branch_strategy.hpp     # Branch 2: Global Min+Max
│   │   ├── parabola_branch_strategy.hpp   # Branch 3: One Max + Parabola
│   │   └── all_maxima_branch_strategy.hpp # Branch 4: All Maxima (INTERNAL ONLY)
│   ├── checkpoint/
│   │   ├── i_checkpoint_save.hpp          # ICheckpointSave (Null Object GoF)
│   │   ├── null_checkpoint_save.hpp       # Null Object (production, zero cost)
│   │   └── checkpoint_save.hpp            # Real save (debug/test)
│   ├── strategy_factory.hpp               # StrategyFactory (Factory Method GoF)
│   └── config/
│       ├── strategy_config.hpp            # AntennaProcessorConfig struct
│       ├── statistics_set.hpp             # StatisticsSet bitmask + presets
│       └── branch_mode.hpp                # BranchMode enum
├── src/
│   ├── antenna_processor_v1.cpp           # AntennaProcessor_v1 implementation
│   ├── branch_strategies/
│   │   ├── minmax_branch_strategy.cpp
│   │   ├── parabola_branch_strategy.cpp
│   │   └── all_maxima_branch_strategy.cpp
│   ├── checkpoint/
│   │   ├── checkpoint_save.cpp
│   │   └── null_checkpoint_save.cpp       # (может быть header-only)
│   └── strategy_factory.cpp
├── kernels/
│   ├── gemm_wrapper.hpp                   # hipBLAS Cgemm thin wrapper
│   ├── hamming_processor.hpp              # apply_hamming kernel launcher
│   ├── hamming.hip                        # HIP: apply_hamming kernel
│   ├── minmax_spectrum.hip                # HIP: Branch 2 kernel
│   └── all_maxima.hip                     # HIP: Branch 4 kernel
└── tests/
    ├── all_test.hpp
    ├── test_gemm_correctness.hpp          # X = W×S vs reference (NumPy)
    ├── test_minmax_branch.hpp             # Branch 2 vs naive CPU search
    ├── test_parabola_branch.hpp           # Branch 3 vs fft_maxima reference
    ├── test_statistics_pre_post.hpp       # PRE/POST GEMM stats vs reference
    ├── test_checkpoint_save.hpp           # Проверка записи/чтения C1-C4
    ├── test_fft_mirror_fold.hpp           # Проверка fold_fft_mirror()
    └── README.md

config/data_formats/
├── antenna_processor_cf32.json
└── antenna_processor_stats.json

Logs/GPU_XX/
└── antenna_processor/
    └── 2026-03-06/
        └── 14-32-05/
            ├── meta.json
            ├── C1_signal.bin   [optional]
            ├── C1_weights.bin  [optional]
            ├── C2_data.bin     [optional]
            ├── C2_stats_pre.bin   [optional]
            ├── C2_stats_post.bin  [optional]
            ├── C3_minmax.bin   [if Branch 2]
            └── C4_peak.bin     [if Branch 3]
```

---

## 8. PlantUML: Component Diagram

```plantuml
@startuml AP_C3_Component
!include <C4/C4_Component>
LAYOUT_WITH_LEGEND()
title AntennaProcessor — C3: Component Diagram

Container_Boundary(ap, "modules/strategies/") {

    Component(iap, "AntennaProcessor", "Abstract Base Class (без I-префикса)",
              "process(S, W, cfg) → AntennaResult\nset_branch(mode)\nset_statistics(pre, post)")

    Component(aap, "AntennaProcessor_v1", "Concrete Impl (индекс v1)",
              "Controls GPU pipeline:\nDMA → Stats → GEMM → Hamming → FFT → Branch")

    Component(gw, "GemmWrapper", "hipBLAS wrapper",
              "W[N×N] × S[N×M] = X[N×M]\nhipBLAS Cgemm, plan cached")

    Component(hp, "HammingProcessor", "HIP kernel",
              "apply_hamming.hip\nX[n]*=w[n], lazy init on N_samples change")

    Component(ibs, "IBranchStrategy", "Strategy Interface",
              "execute(spectrum, params) → BranchResult")

    Component(mm, "MinMaxBranchStrategy", "Branch 2",
              "minmax_spectrum.hip\n1 block/beam, tree reduction\n→ MinMaxResult[N_ant]")

    Component(pb, "ParabolaBranchStrategy", "Branch 3",
              "post_kernel_one_peak (reuses fft_maxima)\nParabolic interpolation\n→ MaxValue[N_ant]")

    Component(am, "AllMaximaBranchStrategy", "Branch 4 (internal test)",
              "detect_all_maxima + Blelloch scan\nCFAR threshold = median*alpha\n→ AllMaximaBeamResult[N_ant]")

    Component(ics, "ICheckpointSave", "Null Object Interface",
              "save_c1/c2/c3/c4()")

    Component(ncs, "NullCheckpointSave", "Production no-op",
              "All methods empty (inline)\nZero overhead in production")

    Component(cs, "CheckpointSave", "Debug save",
              "Binary files to Logs/GPU_XX/\nantenna_processor/YYYY-MM-DD/")

    Component(apf, "StrategyFactory", "Factory Method GoF",
              "create(ctx, cfg) → AntennaProcessor*\nSelects Strategy + CheckpointSave")
}

Container_Boundary(reused, "Reused modules") {
    Component(sp, "StatisticsProcessor", "modules/statistics/",
              "welford_fused, radix_sort\nmedians, min/max")
    Component(fft, "hipFFTProcessor", "modules/fft_processor/",
              "hipFFT batch plan\nCollectOrRelease pattern")
    Component(mv, "MaxValue / SpectrumResult", "modules/fft_maxima/",
              "spectrum_result_types.hpp\nReused as-is (no copy)")
}

Rel(aap, iap, "implements")
Rel(aap, gw, "uses")
Rel(aap, hp, "uses")
Rel(aap, ibs, "uses via ptr")
Rel(aap, ics, "uses via ptr")
Rel(aap, sp, "uses (pre+post stats)")
Rel(aap, fft, "uses")

Rel(mm, ibs, "implements")
Rel(pb, ibs, "implements")
Rel(am, ibs, "implements")

Rel(ncs, ics, "implements (Null Object)")
Rel(cs, ics, "implements")

Rel(apf, aap, "creates")
Rel(apf, mm, "creates on MINMAX mode")
Rel(apf, pb, "creates on PARABOLA mode")
Rel(apf, am, "creates on ALL_MAXIMA mode")
Rel(apf, ncs, "creates (production)")
Rel(apf, cs, "creates (debug/test)")

Rel(pb, mv, "uses MaxValue struct")
Rel(am, mv, "uses MaxValue struct")

SHOW_LEGEND()
@enduml
```

---

*Создано: 2026-03-06*
*Следующий уровень: [C4 — Code Diagram](AP_C4_Code.md)*
*Предыдущий уровень: [C2 — Container Diagram](AP_C2_Container.md)*
