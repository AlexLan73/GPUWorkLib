# C++ test_utils — Модуль общей тестовой инфраструктуры (INDEX)

> ✅ **Инфраструктура COMPLETED** (TASK 01-05). Миграция 5/10 модулей DONE (2026-03-23).
> **Цель**: Создать `modules/test_utils/` — единую C++ тестовую инфраструктуру (зеркало Python `common/`)
> **Создан**: 2026-03-21
> **Исследование**: `MemoryBank/research/cpp_test_utils_full_plan.md`
> **Одобрено**: Alex, 2026-03-21

---

## 🎯 Главная идея

```
БЫЛО:                                СТАНЕТ:
──────────────────────                ──────────────────────────────────
MaxError() в 15+ файлах       →      modules/test_utils/validators/
CpuMean/Median/Var в 8+ файлах →     modules/test_utils/references/
GPU readback в 40+ местах     →      modules/test_utils/gpu_transfer.hpp
PASS/FAIL форматирование ×80  →      modules/test_utils/test_runner.hpp
Tolerance хардкод ×100        →      modules/test_utils/test_configs.hpp
```

---

## 📋 Цепочка выполнения

```
Фаза 0 — Инфраструктура (5 TASK'ов, 14 файлов):
  CppTest-01 (result + configs)  →  Value Objects + Tolerances + SignalParams
        ↓
  CppTest-02 (validators)        →  MaxRel, Abs, Rmse, Frequency, Composite
  CppTest-03 (references)        →  CPU-эталоны: signals, statistics, fft
  CppTest-04 (gpu_transfer)      →  ReadGpuBuffer (OpenCL + ROCm)
        ↓
  CppTest-05 (runner + base)     →  TestRunner + GpuTestBase + reporters + master

Фаза 1 — Эталонная миграция:
  CppTest-06 (statistics)        →  Перевести test_statistics_rocm.hpp

Фаза 2 — Массовая миграция (по одному модулю):
  CppTest-07..11                 →  signal_gen, filters, heterodyne, fft, остальные
```

---

## 📁 Таски

| # | Файл | Что делает | Фаза | Статус |
|---|------|-----------|------|--------|
| 01 | [TASK_CppTest_01](TASK_CppTest_01_result_configs.md) | `test_result.hpp` + `test_configs.hpp` | 0 | ✅ DONE |
| 02 | [TASK_CppTest_02](TASK_CppTest_02_validators.md) | `validators/` — numeric, signal, composite | 0 | ✅ DONE |
| 03 | [TASK_CppTest_03](TASK_CppTest_03_references.md) | `references/` — signal, statistics, fft | 0 | ✅ DONE |
| 04 | [TASK_CppTest_04](TASK_CppTest_04_gpu_transfer.md) | `gpu_transfer.hpp` — OpenCL + ROCm | 0 | ✅ DONE |
| 05 | [TASK_CppTest_05](TASK_CppTest_05_runner_base.md) | `test_runner.hpp` + `gpu_test_base.hpp` + `reporters.hpp` + `test_utils.hpp` | 0 | ✅ DONE |

---

## 🏗️ Целевая структура (13 файлов — обновлено по ревью 2026-03-21)

```
modules/test_utils/
├── test_utils.hpp               ← Master include (все файлы)
├── test_result.hpp              ← ValidationResult(double) + TestResult + add_all + first_failed
├── test_configs.hpp             ← Tolerances + SignalParams + FilterParams + DechirpParams(double c_light)
├── test_runner.hpp              ← TestRunner + timing(chrono) + JSON export
├── gpu_test_base.hpp            ← GpuTestBase (Template Method)
├── reporters.hpp                ← ConsoleTestReporter + ANSI цвета
├── gpu_transfer.hpp             ← ReadGpuBuffer + PeekGpuBuffer(offset) + hipMemcpyAsync+StreamSync
├── references/
│   ├── signal_refs.hpp          ← GenerateCw, GenerateLfm, FormSignal, Noise, MultiBeam
│   ├── statistics_refs.hpp      ← CpuMean, CpuMedian, CpuVariance, CpuStd
│   └── fft_refs.hpp             ← FindPeakBin, PeakFreqHz, CpuMagnitude
└── validators/
    ├── numeric.hpp              ← MaxRelError<T>, AbsError<T>, RmseError<T> + value_to_string
    └── signal.hpp               ← CheckPeakFreq, CheckPower
```

> ℹ️ composite.hpp объединён с test_result.hpp (ревью R4)

---

## ⚠️ Критические правила

1. **Header-only** — все `.hpp`, inline/template, без `.cpp`
2. **Namespace**: `gpu_test_utils` (подпространства: `refs`, `vals`)
3. **Зависимость**: modules/test_utils → DrvGPU ТОЛЬКО (IBackend, ConsoleOutput). НЕ от других modules.
4. **C++17**: structured bindings, `if constexpr`, `std::optional`, `std::string_view`
5. **Template**: validators шаблонные (`MaxRelValidator<float>`, `<std::complex<float>>`)
6. **Strict `<`**: в валидаторах `err < tolerance` (как Python DataValidator), НЕ `<=`
7. **float64 для вычислений**: ошибки считать в double, данные GPU = float32 (как Python R-05)
8. **ConsoleOutput**: только через ConsoleTestReporter, не напрямую в тестах
9. **pytest ЗАПРЕЩЁН** в Python-тестах, `assert` тоже — но в C++ `assert` допустим в debug
10. **Путь**: `modules/test_utils/` — НЕ `include/test_utils/`

---

## 🔑 Два стиля API

### Функциональный (90% — быстрая миграция)
```cpp
TestRunner runner(backend, "ModuleName");
runner.test("test_name", [&]() {
    // ... GPU compute + CPU reference ...
    return MaxRelError(gpu, cpu, n, 1e-3f, "metric");
});
runner.print_summary();
```

### Классовый (10% — сложные pipeline-тесты)
```cpp
class MyTest : public GpuTestBase { ... };
runner.run(MyTest(backend, params));
```

---

## 📊 Маппинг C++ ↔ Python

```
C++ modules/test_utils/          Python_test/common/
═══════════════════════          ═══════════════════
test_result.hpp                  result.py
test_configs.hpp                 configs.py
references/signal_refs.hpp       references/signal_refs.py
references/statistics_refs.hpp   references/statistics_refs.py
references/fft_refs.hpp          references/fft_refs.py
validators/numeric.hpp           validators/numeric.py
validators/signal.hpp            validators/signal.py
validators/composite.hpp         validators/composite.py
gpu_transfer.hpp                 (нет аналога)
test_runner.hpp                  runner.py
gpu_test_base.hpp                test_base.py
reporters.hpp                    reporters.py
```

---

*Создан: 2026-03-21 | Кодо*
