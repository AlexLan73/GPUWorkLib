# Python Test Refactoring — Индекс тасков

**Цель**: Убрать pytest, добавить DataValidator, написать pipeline-тесты для strategies
**Стиль**: ООП, SOLID, GRASP, GoF — везде!
**Создан**: 2026-03-19

---

## 📋 Порядок выполнения

```
TASK_Python_01 → TASK_Python_02 → TASK_Python_03 → TASK_Python_04
                                        ↓                  ↓
                                   TASK_Python_05 ←────────┘
                                        ↓
                                   TASK_Python_06
                                        ↓
                                   TASK_Python_07
                                        ↓
                                   TASK_Python_08
```

---

## 📁 Таски

| # | Файл | Что делает | Зависимости | Статус |
|---|------|-----------|-------------|--------|
| 01 | [TASK_Python_01_common_runner.md](TASK_Python_01_common_runner.md) | `TestRunner` + `SkipTest` (инфраструктура) | — | 🔲 TODO |
| 02 | [TASK_Python_02_data_validator.md](TASK_Python_02_data_validator.md) | `DataValidator` — один класс вместо 4 | — | 🔲 TODO |
| 03 | [TASK_Python_03_numpy_reference.md](TASK_Python_03_numpy_reference.md) | `NumpyReference` — CPU-эталон pipeline | — | 🔲 TODO |
| 04 | [TASK_Python_04_signal_factory.md](TASK_Python_04_signal_factory.md) | `ISignalSource` + 5 вариантов + Factory | 01 | 🔲 TODO |
| 05 | [TASK_Python_05_pipeline_step_validator.md](TASK_Python_05_pipeline_step_validator.md) | `PipelineStepValidator` — 9 шагов, 14 CHECK | 02, 03 | 🔲 TODO |
| 06 | [TASK_Python_06_test_strategies.md](TASK_Python_06_test_strategies.md) | `TestStrategiesPipeline` — 5 тестов (V1–V5) | 01–05 | 🔲 TODO |
| 07 | [TASK_Python_07_remove_pytest_strategies.md](TASK_Python_07_remove_pytest_strategies.md) | Убрать pytest из `strategies/` (9 файлов) | 01, 02 | 🔲 TODO |
| 08 | [TASK_Python_08_remove_pytest_all_modules.md](TASK_Python_08_remove_pytest_all_modules.md) | Убрать pytest из всех остальных модулей | 01, 02 | 🔲 TODO |

---

## 🎯 Новые файлы (результат выполнения)

```
Python_test/
├── common/
│   ├── runner.py                        ← TASK_01 (TestRunner + SkipTest)
│   └── validators.py                    ← TASK_02 (DataValidator, переписать)
└── strategies/
    ├── numpy_reference.py               ← TASK_03 (NumpyReference)
    ├── signal_factory.py                ← TASK_04 (ISignalSource + Factory)
    ├── pipeline_step_validator.py       ← TASK_05 (PipelineStepValidator)
    └── test_strategies_pipeline.py      ← TASK_06 (TestStrategiesPipeline)
```

---

## 🔑 Ключевые решения (не менять без обсуждения!)

1. **DataValidator — ОДИН класс** с метриками `"max_rel"`, `"abs"`, `"rmse"`
2. **AntennaProcessorTest — ОДИН объект**, не пересоздаётся между шагами
3. **SkipTest** — наш класс из `common/runner.py`, NOT `unittest.SkipTest`
4. **float32/complex64** — все данные, не float64
5. **5 вариантов сигналов** — V1/V2/V3/V4 через GPU SignalGenerator, V5 заглушка
6. **PLOT** после STEP 4 — `Results/Plots/strategies/spectrum_{variant}.png`

---

## 📊 CHECK-точки (14 штук)

| ID | Шаг | Метрика | Порог |
|----|-----|---------|-------|
| CHECK-0 | setup | == | точно |
| CHECK-1a..d | step_1 (stats d_S) | max_rel | 0.01 |
| CHECK-2 | step_2 (GEMM) | max_rel | 1e-3 |
| CHECK-3a..b | step_3 (stats d_X) | max_rel | 0.01 |
| CHECK-4a..b | step_4 (FFT + peak_bin) | max_rel / abs | 0.01 / 2 бина |
| CHECK-5 | step_5 (stats spectrum) | max_rel | 0.01 |
| CHECK-6.1 | step_6_1 (freq) | abs | 50 кГц |
| CHECK-6.2 | step_6_2 (count) | >= | 1 |
| CHECK-6.3a..b | step_6_3 (minmax) | < / > | — / 0 дБ |

---

## ✅ Финальная проверка (после всех тасков)

```bash
# Нет pytest:
grep -r "import pytest" Python_test/ --include="*.py"   # → пусто!

# Тесты strategies запускаются:
python Python_test/strategies/test_strategies_pipeline.py
# → [PASS] V1, [PASS] V2, [PASS] V3, [PASS] V4, [SKIP] V5

# Спектры построены:
ls Results/Plots/strategies/spectrum_V*.png
# → spectrum_V1_clean.png, spectrum_V2_noise.png, spectrum_V3_phase.png, spectrum_V4_phase_noise.png
```
