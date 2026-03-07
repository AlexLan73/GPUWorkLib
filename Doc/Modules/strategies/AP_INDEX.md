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
  │  1. Input already on GPU: d_S + metadata                │
  │  2. Debug 2.1: stats/save/python по d_S                 │
  │  3. GEMM: X = W × S (hipBLAS Cgemm)                     │
  │  4. Debug 2.2: stats/save/python по d_X                 │
  │  5. Base block: Window + FFT                            │
  │  6. Debug 2.3: stats/save/python по |spectrum|          │
  │  7. Обязательные post-FFT сценарии:                     │
  │     ├─ Step2.1: One MAX + 3-point Parabola (no phase)   │
  │     ├─ Step2.2: All Maxima (limit=1000)                 │
  │     └─ Step2.3: Global MAX/MIN (limit=1000)             │
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
| 3 | `Window + FFT` — единый базовый блок | FFT считается один раз, затем несколько consumers читают один `d_spectrum` |
| 4 | 3 обязательных post-FFT сценария | По ТЗ считаются все: `OneMax+Parabola`, `AllMaxima`, `GlobalMinMax` |
| 5 | FFT fold (mirror) | Пик в бине > nFFT/2 = отрицательная частота, нужно перевести |
| 6 | Logs/GPU_XX/... | Соответствует стандарту проекта (per-GPU логи) |
| 7 | NullCheckpointSave | Как в fft_processor: no save by default = zero overhead |
| 8 | Статистика в 3 debug-точках | `2.1` по `d_S`, `2.2` по `d_X`, `2.3` по `|spectrum|` |
| 9 | Hamming ПОСЛЕ GEMM и ПЕРЕД FFT | DSP правило: окно перед FFT, но после beamforming |
| 10 | Матрица W — Delay-and-sum | Генерируется автоматически из параметров сигнала/решётки, плюс внешний ввод C++/Python |

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
- [ ] Реализовать общий блок `Window + FFT` на ROCm
- [ ] Интегрировать `StatisticsProcessor` для `2.1 / 2.2 / 2.3`
- [ ] Интегрировать `hipFFT` batch plan (с кешированием)
- [ ] Реализовать `fold_fft_mirror()` (Note #2)

### Фаза 3 — Стратегии (2-3 дня)
- [ ] `Step2.1`: `OneMax + Parabola` без фазы в `modules/fft_maxima/`
- [ ] `Step2.2`: `AllMaxima` с limit=`1000` в `modules/fft_maxima/`
- [ ] `Step2.3`: `GlobalMinMax` с limit=`1000` в `modules/fft_maxima/`
- [ ] `Post-FFT Statistics(|spectrum|)` в `modules/statistics/`

### Фаза 4 — Checkpoint сохранение (1-2 дня)
- [ ] `CheckpointSave` с binary format + JSON header option
- [ ] Именование: `Logs/GPU_XX/antenna_processor/YYYY-MM-DD/HH-MM-SS/`
- [ ] Обновить `DataFormatRegistry` с новыми форматами

### Фаза 5 — Тесты (2-3 дня)
- [ ] `test_gemm_correctness.hpp` — X vs NumPy
- [ ] `test_one_max_parabola_no_phase.hpp` — Step2.1 vs CPU reference
- [ ] `test_all_maxima.hpp` — Step2.2 vs CPU reference
- [ ] `test_global_minmax.hpp` — Step2.3 vs CPU reference
- [ ] `test_statistics_pre_post_fft.hpp` — `2.1 / 2.2 / 2.3` stats vs NumPy/SciPy
- [ ] `test_checkpoint_save.hpp` — запись/чтение C1-C4
- [ ] `test_fft_mirror_fold.hpp` — fold для отрицательных частот
- [ ] Python тесты в `Python_test/strategies/`

### Фаза 6 — Профилирование (1 день)
- [ ] Встроить `GPUProfiler::SetGPUInfo()` + Start/Stop
- [ ] Бенчмарк GEMM, Window+FFT, Step2.1/2.2/2.3
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
| Step2.1 / Step2.3 | < 1 мс | Compute-light |
| Step2.2 | 2-5 мс | Compute+BW |
| **ИТОГО (VRAM input)** | **~35 мс** | GEMM + FFT bottleneck |

> *PCIe 4.0 x16: 32 GB/s теоретически. Для 2.5 ГБ: 78 мс. Если данные приходят стримом (NIC→GPU P2P DMA) — накладные расходы будут другими.

---

*Maintained by: Кодо (AI Assistant) | Created: 2026-03-06*
