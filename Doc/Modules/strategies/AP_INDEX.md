# AntennaProcessor — Architecture Documentation Index

> **Module**: `modules/strategies/`
> **Date**: 2026-03-06
> **Author**: Кодо (AI Assistant)
> **Notation**: C4 Model + UML Sequence Diagrams

---

## Документы

| # | Документ | Уровень | Описание |
|---|----------|---------|----------|
| 1 | [C1 — System Context](AP_C1_SystemContext.md) | Контекст | Акторы, внешние системы, место в GPUWorkLib |
| 2 | [C2 — Container Diagram](AP_C2_Container.md) | Контейнеры | GPU streams, компоненты, зависимости |
| 3 | [C3 — Component Diagram](AP_C3_Component.md) | Компоненты | SOLID, GRASP, GoF patterns (Strategy, Factory, Null Object) |
| 4 | [C4 — Code Diagram](AP_C4_Code.md) | Код | Интерфейсы, классы, сигнатуры, конфигурации |
| 5 | [Seq — Sequence Diagrams](AP_Seq.md) | Сценарии | Pipeline, timing, chunking, FFT mirror folding |

---

## Краткое описание модуля

**AntennaProcessor** — обрабатывает матрицу антенных данных через GPU pipeline:

```
S[N_ant × N_samples]  ──┐
W[N_ant × N_ant]      ──┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │               AntennaProcessor_v1                     │
  │                                                         │
  │  1. DMA load (Stream 0)                                 │
  │  2. Stats PRE-GEMM (Stream 1) ◄── параллельно с GEMM   │
  │  3. GEMM: X = W × S (hipBLAS Cgemm)                    │
  │  4. Stats POST-GEMM (Stream 3) ◄── параллельно с FFT   │
  │  5. Hamming window: X[n] *= w[n]                        │
  │  6. FFT batch: N_ant × hipFFT(nFFT)                     │
  │  7. FFT fold: bins k>nFFT/2 → negative frequency        │
  │  8. Branch Strategy (переключаемая):                    │
  │     ├─ Branch 2: Global MIN + Global MAX (minmax)       │
  │     ├─ Branch 3: One MAX + Parabola (precision freq)    │
  │     └─ Branch 4: ALL peaks (CFAR, internal test only)   │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  AntennaResult { pre_stats, post_stats, peaks/minmax, perf }
```

---

## GoF Patterns в модуле

| Паттерн | Реализация |
|---------|-----------|
| **Strategy** | `IBranchStrategy` → ветки 2/3/4 взаимозаменяемы без изменения pipeline |
| **Factory Method** | `StrategyFactory::create()` → создаёт нужную Strategy + CheckpointSave |
| **Null Object** | `NullCheckpointSave` → production-режим без оверхеда (no-op методы) |
| **Template Method** | `AntennaProcessor_v1::process()` → фиксированный порядок шагов + изменяемый Branch |

---

## Ключевые решения (из обсуждения)

| # | Решение | Обоснование |
|---|---------|-------------|
| 1 | W — квадратная [N_ant × N_ant] | "иначе потеряем лучи"; максимум 256×256 = 512 КБ |
| 2 | GEMM, не element-wise | "Стандартное умножение матриц!" — гетеродинирование лучей |
| 3 | 3 ветки (2/3/4) | Заказчик хотел 3 варианта; ветка 4 = внутреннее тестирование |
| 4 | MaxValue переиспользуется | Уже есть в `fft_maxima` — 32 байта, идеально подходит |
| 5 | FFT fold (mirror) | Пик в бине > nFFT/2 = отрицательная частота, нужно перевести |
| 6 | Logs/GPU_XX/... | Соответствует стандарту проекта (per-GPU логи) |
| 7 | NullCheckpointSave | Как в fft_processor: no save by default = zero overhead |
| 8 | StatisticsSet bitmask | Гибкие пресеты 6.1-6.4; PRE и POST GEMM независимо |
| 9 | Hamming ПОСЛЕ GEMM | DSP правило: окно перед FFT, но checkpoint C2 ДО окна |
| 10 | W в L2 cache | 512 КБ << 32 МБ (9070 L2) → GEMM compute-bound, не BW-bound |

---

## Tasks (для реализации)

### Фаза 1 — Инфраструктура (2-3 дня)
- [ ] Создать файловую структуру модуля (`modules/strategies/`)
- [ ] Написать `AntennaProcessor`, `IBranchStrategy`, `ICheckpointSave`
- [ ] Написать `AntennaProcessorConfig`, `StatisticsSet`, `BranchMode`
- [ ] Написать `NullCheckpointSave` (production заглушка)
- [ ] Написать `StrategyFactory`

### Фаза 2 — Ядро pipeline (3-4 дня)
- [ ] Реализовать `GemmWrapper` (hipBLAS Cgemm)
- [ ] Реализовать `HammingProcessor` (apply_hamming.hip)
- [ ] Интегрировать `StatisticsProcessor` (PRE-GEMM, POST-GEMM)
- [ ] Интегрировать `hipFFT` batch plan (с кешированием)
- [ ] Реализовать `fold_fft_mirror()` (Note #2)

### Фаза 3 — Стратегии (2-3 дня)
- [ ] `MinMaxBranchStrategy` + `minmax_spectrum.hip` (Branch 2)
- [ ] `ParabolaBranchStrategy` (адаптация из fft_maxima, Branch 3)
- [ ] `AllMaximaBranchStrategy` + `all_maxima.hip` (Branch 4, internal)

### Фаза 4 — Checkpoint сохранение (1-2 дня)
- [ ] `CheckpointSave` с binary format + JSON header option
- [ ] Именование: `Logs/GPU_XX/antenna_processor/YYYY-MM-DD/HH-MM-SS/`
- [ ] Обновить `DataFormatRegistry` с новыми форматами

### Фаза 5 — Тесты (2-3 дня)
- [ ] `test_gemm_correctness.hpp` — X vs NumPy
- [ ] `test_minmax_branch.hpp` — Branch 2 vs наивный CPU
- [ ] `test_parabola_branch.hpp` — Branch 3 vs fft_maxima
- [ ] `test_statistics_pre_post.hpp` — PRE/POST stats vs SciPy
- [ ] `test_checkpoint_save.hpp` — запись/чтение C1-C4
- [ ] `test_fft_mirror_fold.hpp` — fold для отрицательных частот
- [ ] Python тесты в `Python_test/antenna_processor/`

### Фаза 6 — Профилирование (1 день)
- [ ] Встроить `GPUProfiler::SetGPUInfo()` + Start/Stop
- [ ] Бенчмарк GEMM, FFT, Branch 2/3/4
- [ ] Экспорт отчётов: `profiler.ExportMarkdown()`, `ExportJSON()`

---

## Время исполнения (оценка, 256 × 1.2M, 9070)

| Шаг | Время | Тип ограничения |
|-----|-------|----------------|
| DMA (CPU→GPU) | 78 мс* | PCIe 4.0 |
| DMA (GPU→GPU, если из VRAM) | 2.6 мс | BW-bound |
| Stats PRE-GEMM | 2.6 мс | BW-bound (параллельно с GEMM) |
| GEMM | **13 мс** | Compute-bound |
| Stats POST-GEMM | 2.6 мс | BW-bound (параллельно с Hamming+FFT) |
| Hamming | 2.6 мс | BW-bound (параллельно с Stats POST) |
| FFT batch | **~20 мс** | TBD (бенчмарк) |
| Branch 2/3 | < 1 мс | Compute-light |
| Branch 4 | 2-5 мс | Compute+BW |
| **ИТОГО (VRAM input)** | **~35 мс** | GEMM + FFT bottleneck |

> *PCIe 4.0 x16: 32 GB/s теоретически. Для 2.5 ГБ: 78 мс. Если данные приходят стримом (NIC→GPU P2P DMA) — накладные расходы будут другими.

---

*Maintained by: Кодо (AI Assistant) | Created: 2026-03-06*
