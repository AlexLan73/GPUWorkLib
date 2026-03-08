# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-03-08
> **Фокус**: ScenarioBuilder + Farrow Pipeline для модуля strategies

---

## 1. ScenarioBuilder — генератор тестовых сценариев

**Цель**: Python-генератор физически корректных сигналов для AntennaProcessor

**Спецификация**: `MemoryBank/specs/scenario_builder.md`

### Сделано
- [x] Спецификация написана
- [x] `Python_test/strategies/scenario_builder.py` — основной код
  - ULAGeometry (физика ULA: d_ant → tau через sin(θ)/c)
  - EmitterSignal (описание источника: θ, f0, fdev, A)
  - ScenarioBuilder (fluent API: add_target/jammer, set_noise, build)
  - 3 фабрики: single_target, target+jammer, multi_target
  - generate_scan_weight_matrix (multi-beam)
- [x] `Python_test/strategies/test_scenario_builder.py` — 17 numpy-only тестов

### Осталось
- [ ] Прогнать тесты (pytest) — на машине с Python/numpy
- [ ] Интеграция с AntennaProcessorTest (pybind11) — после GPU доступа

---

## 2. Farrow Pipeline — две программные ветки beamforming

**Цель**: сравнить Pipeline A (фазовая коррекция) vs Pipeline B (Farrow + суммирование)

**Спецификация**: `MemoryBank/specs/farrow_pipeline.md`

### Архитектура
```
ScenarioBuilder → S_raw
    ├── Pipeline A: GEMM(W_phase) → FFT → peaks
    └── Pipeline B: FarrowDelay → S_aligned → GEMM(W_sum) → FFT → peaks
                                  ↑ [stats] [save]
```

### Сделано
- [x] Спецификация написана
- [x] `Python_test/strategies/farrow_delay.py` — numpy Farrow (Lagrange 48×5)
  - FarrowDelay: apply, compensate, apply_seconds
  - Загрузка матрицы из modules/lch_farrow/lagrange_matrix_48x5.json
- [x] `Python_test/strategies/pipeline_runner.py` — PipelineRunner
  - run_pipeline_a() / run_pipeline_b()
  - Статистика (ChannelStats) на каждом шаге
  - Checkpoint'ы: save .npy + .json на диск (опционально)
  - compare() + print_comparison()
  - PipelineResult: все промежуточные данные доступны для Python тестов
- [x] `Python_test/strategies/test_farrow_pipeline.py` — 17 тестов
  - FarrowDelay unit (4), Basic A/B (4), Comparison (3)
  - Complex scenarios (3), Stats & checkpoints (5)

### Осталось
- [ ] Прогнать тесты (pytest)
- [ ] Визуализация: спектры Pipeline A vs B (matplotlib)
- [ ] Интеграция с GPU LchFarrowROCm + AntennaProcessor

---

## Полный список файлов

| Файл | Описание |
|------|----------|
| `Python_test/strategies/scenario_builder.py` | Генератор сценариев с физикой ULA |
| `Python_test/strategies/test_scenario_builder.py` | 17 тестов ScenarioBuilder |
| `Python_test/strategies/farrow_delay.py` | Numpy Farrow (Lagrange 48×5) |
| `Python_test/strategies/pipeline_runner.py` | Pipeline A/B runner + stats + checkpoints |
| `Python_test/strategies/test_farrow_pipeline.py` | 17 тестов Pipeline A vs B |
| `MemoryBank/specs/scenario_builder.md` | Спецификация ScenarioBuilder |
| `MemoryBank/specs/farrow_pipeline.md` | Спецификация Farrow Pipeline |

*Последнее обновление: 2026-03-08*
