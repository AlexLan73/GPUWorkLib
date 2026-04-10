# TASK_CppTest_06 — Инструкция миграции тестов на test_utils

> **Статус**: statistics ✅ | signal_generators ✅ | filters ✅ | fft_func ✅ | heterodyne ✅ | lch_farrow ⬜ | vector_algebra ⬜ | fm_correlator ⬜ | capon ⬜ | range_angle ⬜ | strategies ⬜
> **Эталон**: `modules/statistics/tests/test_statistics_rocm.hpp` — полностью мигрирован
> **Дата**: 2026-03-21

---

## 🎯 Что сделано (эталон)

| Файл | Было | Стало | Экономия |
|------|------|-------|----------|
| `test_statistics_rocm.hpp` | 1091 строк | ~350 строк | -741 LOC (-68%) |
| `test_statistics_float_rocm.hpp` | 327 строк | ~220 строк | -107 LOC (-33%) |

**Удалены дублированные функции:**
- `GenerateSinusoid()` → `refs::GenerateSinusoid()`
- `GenerateConstant()` → `refs::GenerateConstant()`
- `GenerateMultiBeam()` → `refs::GenerateMultiBeam()`
- `CpuMean()` → `refs::CpuMean()`
- `CpuMeanMagnitude()` → `refs::CpuMeanMagnitude()`
- `CpuVarianceMagnitude()` → `refs::CpuVarianceMagnitude()`
- `CpuStdMagnitude()` → `refs::CpuStdMagnitude()`
- `CpuMedianMagnitude()` → `refs::CpuMedianMagnitude()`
- `print_result()` → TestRunner автоматически
- Ручной подсчёт passed/total → `runner.print_summary()`
- 15× try/catch + bool → 15× runner.test() с автоперехватом

---

## 📋 Рецепт миграции (для любого модуля)

### Шаг 1: Добавить include

```cpp
// УБРАТЬ:
#include "services/console_output.hpp"

// ДОБАВИТЬ:
#include "modules/test_utils/test_utils.hpp"
```

### Шаг 2: Добавить `using namespace`

```cpp
using namespace gpu_test_utils;
// refs:: GenerateCw, CpuMean, ...
// tolerance:: kComplex32, kStatistics, ...
// MaxRelError, ScalarAbsError, ScalarRelError, ...
```

### Шаг 3: Удалить дублированные функции

Удалить ВСЁ из секции "Utilities":
- `print_result()` → НЕ НУЖЕН (TestRunner делает сам)
- `GenerateSinusoid()` → `refs::GenerateSinusoid(freq, fs, n)`
- `GenerateCw()` → `refs::GenerateCw(fs, n, f0)`
- `CpuMean()` → `refs::CpuMean(data, n)`
- `CpuMeanMagnitude()` → `refs::CpuMeanMagnitude(data, n)`
- `CpuVarianceMagnitude()` → `refs::CpuVarianceMagnitude(data, n)`
- `CpuStdMagnitude()` → `refs::CpuStdMagnitude(data, n)`
- `CpuMedianMagnitude()` → `refs::CpuMedianMagnitude(data, n)`
- `MaxError()` → `MaxRelError(actual, ref, n, tol, "name")`
- `FindPeakBin()` → `refs::FindPeakBin(mag, n)`
- `RefMean()`, `RefStd()`, `RefMedian()` → `refs::CpuMeanFloat`, `refs::CpuStdFloat`, `refs::CpuMedianFloat`

### Шаг 4: Заменить run() на TestRunner

```cpp
// БЫЛО:
inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  int gpu_id = 0;
  // ... setup ...
  int passed = 0, total = N;
  if (test_xxx(con, gpu_id)) ++passed;
  if (test_yyy(con, gpu_id)) ++passed;
  // ...
  con.Print(gpu_id, "Module", "Results: " + to_string(passed) + "/" + to_string(total));
}

// СТАЛО:
inline void run() {
  int gpu_id = 0;
  ROCmBackend backend;
  backend.Initialize(gpu_id);
  // ... создать процессоры ...

  TestRunner runner(&backend, "Module", gpu_id);

  runner.test("test_xxx", [&]() { ... return ValidationResult; });
  runner.test("test_yyy", [&]() -> TestResult { ... });

  runner.print_summary();
  runner.export_json("Results/JSON/test_module.json");
}
```

### Шаг 5: Конвертировать каждый тест

```cpp
// БЫЛО:
inline bool test_xxx(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    SomeProcessor proc(&backend);
    // ... generate data ...
    // ... GPU compute ...
    // ... CPU reference ...
    float err = std::fabs(gpu_val - cpu_val);
    bool ok = (err < 1e-3f);
    print_result(con, gpu_id, "test_xxx", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Module", "[X] EXCEPTION: " + string(e.what()));
    return false;
  }
}

// СТАЛО (простой тест — одна проверка):
runner.test("test_xxx", [&]() {
  // ... generate data (используя refs::) ...
  // ... GPU compute (backend уже создан выше) ...
  // ... CPU reference (используя refs::) ...
  return ScalarAbsError(gpu_val, cpu_val, tolerance::kStatistics, "metric_name");
});

// СТАЛО (сложный тест — несколько проверок):
runner.test("test_yyy", [&]() -> TestResult {
  // ... data generation + GPU compute ...
  TestResult tr{"test_yyy"};
  tr.add(ScalarAbsError(val1, ref1, tol, "check1"));
  tr.add(ScalarAbsError(val2, ref2, tol, "check2"));
  return tr;
});
```

### Шаг 6: GPU readback (если есть clEnqueueReadBuffer)

```cpp
// БЫЛО:
cl_mem buf = proc.ProcessToGpu(data);
std::vector<std::complex<float>> result(n);
auto q = static_cast<cl_command_queue>(backend->GetNativeQueue());
clEnqueueReadBuffer(q, buf, CL_TRUE, 0, n * sizeof(std::complex<float>),
                    result.data(), 0, nullptr, nullptr);
clReleaseMemObject(buf);

// СТАЛО:
cl_mem buf = proc.ProcessToGpu(data);
auto result = ReadGpuBuffer<std::complex<float>>(&backend, buf, n);
```

---

## 📊 Чеклист tolerance при миграции abs → rel

> ⚠️ Старые тесты используют **абсолютный** допуск (`err < 1e-3f`).
> Новые валидаторы — **относительный** (`err/ref < tol`).

| Старый паттерн | Новый | Примечание |
|---------------|-------|------------|
| `fabs(a - b) < 1e-3f` | `ScalarAbsError(a, b, 1e-3, "name")` | Абсолютный — 1:1 замена |
| `fabs(a - b) < 1e-2f` | `ScalarAbsError(a, b, 1e-2, "name")` | Абсолютный |
| `MaxError(gpu, cpu, n) < 1e-3f` | `MaxRelError(gpu, cpu, n, 1e-3, "name")` | ⚠️ ОТНОСИТЕЛЬНЫЙ! |
| `err < tol * ref + 1e-5f` | `ScalarRelError(a, b, tol, "name")` | Уже относительный |

**Правило**: если старый тест использовал `fabs(a - b) < tol`, то `ScalarAbsError`.
Если `MaxError()` — проверить: было абсолютное, стало относительное. Для amplitude ≈ 1.0 — одинаково. Для amplitude > 1 — нужно пересчитать.

---

## 📁 Очередь миграции

| # | Модуль | Файлов | Приоритет | Примечание |
|---|--------|--------|-----------|------------|
| ✅ | statistics | 2 | DONE | Эталон |
| 1 | signal_generators | 6 | 🔴 | Много дублирования |
| 2 | filters | 6 | 🟡 | FIR/IIR/Kalman/Kaufman/MA |
| 3 | heterodyne | 3 | 🟡 | Dechirp pipeline |
| 4 | fft_func | 6 | 🟡 | FFT/Magnitude/Maxima |
| 5 | lch_farrow | 2 | 🟢 | Farrow delay |
| 6 | vector_algebra | 2 | 🟢 | Cholesky, cross-backend |
| 7 | capon | 1 | 🟢 | Beamformer |
| 8 | fm_correlator | 4 | 🟢 | Correlation |
| 9 | range_angle | 1 | 🟢 | Angle estimation |
| 10 | strategies | 3 | 🟢 | Pipeline tests |

**Не мигрировать (benchmark файлы):**
- `*_benchmark*.hpp` — используют GpuBenchmarkBase, отдельная система
- `*_profiling*.hpp` — профилирование через GPUProfiler

---

## 🔧 Проверка на Linux (в понедельник)

```bash
cd /home/alex/C++/GPUWorkLib

# 1. Собрать
cmake --build build/debian-radeon9070 -j8

# 2. Запустить statistics тесты
./build/debian-radeon9070/GPUWorkLib statistics

# 3. Проверить вывод — должно быть:
#   ──── mean_single_beam ────
#   [PASS] mean_re: 1.2345e-04 (tol=1.0000e-03)
#   [PASS] mean_im: 2.3456e-05 (tol=1.0000e-03)
#   [PASS] mean_single_beam (42 ms)
#   ...
#   ════════════════════════════════════════
#   SUMMARY: 15 passed, 0 failed
#   ════════════════════════════════════════

# 4. Проверить JSON
cat Results/JSON/test_statistics_rocm.json
```

**Если тест не компилируется:**
1. Проверить include путь: `#include "modules/test_utils/test_utils.hpp"` — от корня проекта
2. CMakeLists.txt может требовать добавить include directory
3. Если `refs::CpuMeanFloat` не найден — проверить namespace `gpu_test_utils::refs`

**Если тест проходил раньше и не проходит сейчас:**
1. Проверить tolerance — abs→rel переход может дать другие числа
2. Логика теста ИДЕНТИЧНА старой — только обёртка изменилась
3. Сравнить вывод старого `Results:` и нового `SUMMARY:`

---

*Создан: 2026-03-21 | Кодо | CppTest-06*
