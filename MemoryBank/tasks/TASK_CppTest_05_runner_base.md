# TASK_CppTest_05 — TestRunner + GpuTestBase + reporters + master include

> **Фаза**: 0 (инфраструктура, финальный)
> **Зависимости**: TASK_CppTest_01 (result), TASK_CppTest_02 (validators), TASK_CppTest_04 (gpu_transfer)
> **Статус**: ⬜ TODO
> **Оценка**: ~2 часа
> **Паттерны**: Template Method (GoF), Coordinator (GRASP), Observer (GoF)

---

## 🎯 Цель

Создать `TestRunner` (координатор тестов) + `GpuTestBase` (Template Method) + `ConsoleTestReporter` + master include.

Зеркало Python: `common/runner.py` + `common/test_base.py` + `common/reporters.py`

---

## 📁 Создаваемые файлы (4 штуки)

```
modules/test_utils/
├── reporters.hpp        ← 1. ConsoleTestReporter (через ConsoleOutput)
├── test_runner.hpp      ← 2. TestRunner (функциональный + классовый API)
├── gpu_test_base.hpp    ← 3. GpuTestBase (Template Method для сложных тестов)
└── test_utils.hpp       ← 4. Master include
```

---

## 📝 Детальное ТЗ

### 1. `modules/test_utils/reporters.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include "test_result.hpp"
#include "DrvGPU/services/console_output.hpp"

namespace gpu_test_utils {

/**
 * ConsoleTestReporter: форматированный вывод через ConsoleOutput.
 * Зеркало Python: common/reporters.py (ConsoleReporter).
 *
 * Единый стиль PASS/FAIL для всех модулей.
 * Заменяет 80+ мест с ручным con.Print() + форматированием.
 */
class ConsoleTestReporter {
    int gpu_id_;
    std::string module_;
    bool use_colors_;  // ANSI цвета (review #9)

    static constexpr const char* kGreen = "\033[92m";
    static constexpr const char* kRed   = "\033[91m";
    static constexpr const char* kYellow= "\033[93m";
    static constexpr const char* kBold  = "\033[1m";
    static constexpr const char* kReset = "\033[0m";

    std::string color(const std::string& text, const char* c) const {
        return use_colors_ ? (std::string(c) + text + kReset) : text;
    }

public:
    ConsoleTestReporter(int gpu_id, const std::string& module,
                        bool use_colors = true)
        : gpu_id_(gpu_id), module_(module), use_colors_(use_colors) {}

    /// Заголовок теста
    void print_header(const std::string& test_name) const {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        con.Print(gpu_id_, module_,
                  color("──── " + test_name + " ────", kBold).c_str());
    }

    /// Результат одной валидации
    void print_validation(const ValidationResult& vr) const {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        std::string line = "  " + vr.to_string();
        con.Print(gpu_id_, module_,
                  color(line, vr.passed ? kGreen : kRed).c_str());
    }

    /// Результат теста (PASS/FAIL + все валидации + elapsed_ms)
    void print_test_result(const TestResult& tr, double elapsed_ms = 0.0) const {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        if (tr.skipped) {
            con.Print(gpu_id_, module_,
                      color("[SKIP] " + tr.test_name + " (" + tr.skip_reason + ")",
                            kYellow).c_str());
            return;
        }
        for (const auto& vr : tr.validations)
            print_validation(vr);

        // Timing (review #8)
        std::string status = tr.passed() ? "[PASS]" : "[FAIL]";
        std::string timing = (elapsed_ms > 0.0)
            ? " (" + std::to_string(static_cast<int>(elapsed_ms)) + " ms)"
            : "";
        con.Print(gpu_id_, module_,
                  color(status + " " + tr.test_name + timing,
                        tr.passed() ? kGreen : kRed).c_str());

        if (!tr.error.empty())
            con.PrintError(gpu_id_, module_,
                           color("[ERROR] " + tr.test_name + ": " + tr.error, kRed).c_str());
    }

    /// Getter для имени модуля (используется в JSON export)
    const std::string& module() const { return module_; }

    /// Итоговая сводка
    void print_summary(const std::vector<TestResult>& results) const {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        int n_pass = 0, n_fail = 0, n_skip = 0;
        for (const auto& r : results) {
            if (r.skipped) ++n_skip;
            else if (r.passed()) ++n_pass;
            else ++n_fail;
        }
        con.Print(gpu_id_, module_,
                  "════════════════════════════════════════");
        std::string msg = "  SUMMARY: " + std::to_string(n_pass) + " passed, "
                        + std::to_string(n_fail) + " failed";
        if (n_skip > 0) msg += ", " + std::to_string(n_skip) + " skipped";
        con.Print(gpu_id_, module_,
                  color(msg, (n_fail == 0) ? kGreen : kRed).c_str());
        con.Print(gpu_id_, module_,
                  "════════════════════════════════════════");
    }
};

} // namespace gpu_test_utils
```

---

### 2. `modules/test_utils/test_runner.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include "test_result.hpp"
#include "reporters.hpp"
#include "DrvGPU/backends/i_backend.hpp"

namespace gpu_test_utils {

// forward
class GpuTestBase;

/**
 * TestRunner: координатор тестов.
 * Зеркало Python: common/runner.py (TestRunner).
 *
 * ДВА стиля:
 *
 * 1. Функциональный (быстрая миграция, 90% тестов):
 *    TestRunner runner(backend, "Statistics");
 *    runner.test("mean_single", [&]() -> ValidationResult { ... });
 *    runner.test("median_multi", [&]() -> TestResult { ... });
 *    runner.print_summary();
 *
 * 2. Классовый (сложные pipeline-тесты):
 *    runner.run(MyGpuTest(backend, params));
 */
class TestRunner {
    drv_gpu_lib::IBackend* backend_;
    ConsoleTestReporter    reporter_;
    std::vector<TestResult> results_;

public:
    /**
     * @param backend   GPU backend (IBackend*)
     * @param module    имя модуля для ConsoleOutput (например "Statistics")
     * @param gpu_id    индекс GPU (default 0)
     */
    TestRunner(drv_gpu_lib::IBackend* backend,
               const std::string& module, int gpu_id = 0)
        : backend_(backend)
        , reporter_(gpu_id, module)
    {}

    // ── Функциональный стиль ────────────────────────────────────

    /**
     * Запустить тест-лямбду, возвращающую ValidationResult.
     * Для простых тестов с одной проверкой.
     */
    void test(const std::string& name,
              std::function<ValidationResult()> test_fn)
    {
        reporter_.print_header(name);
        TestResult tr{name};
        auto t0 = std::chrono::high_resolution_clock::now();
        try {
            auto vr = test_fn();
            tr.add(vr);
        } catch (const SkipTest& e) {
            tr.skipped = true;
            tr.skip_reason = e.what();
        } catch (const std::exception& e) {
            tr.error = e.what();
        }
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        reporter_.print_test_result(tr, elapsed);
        results_.push_back(std::move(tr));
    }

    /**
     * Запустить тест-лямбду, возвращающую TestResult.
     * Для тестов с множественными проверками.
     */
    void test(const std::string& name,
              std::function<TestResult()> test_fn)
    {
        reporter_.print_header(name);
        TestResult tr{name};
        auto t0 = std::chrono::high_resolution_clock::now();
        try {
            tr = test_fn();
            tr.test_name = name;
        } catch (const SkipTest& e) {
            tr.skipped = true;
            tr.skip_reason = e.what();
        } catch (const std::exception& e) {
            tr.error = e.what();
        }
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        reporter_.print_test_result(tr, elapsed);
        results_.push_back(std::move(tr));
    }

    // ── Классовый стиль ─────────────────────────────────────────

    /**
     * Запустить GpuTestBase (Template Method).
     */
    void run(GpuTestBase& test_obj);  // определяется после GpuTestBase

    // ── Итоги ───────────────────────────────────────────────────

    void print_summary() const {
        reporter_.print_summary(results_);
    }

    const std::vector<TestResult>& results() const { return results_; }

    int count_passed() const {
        int n = 0;
        for (const auto& r : results_) if (r.passed()) ++n;
        return n;
    }

    int count_failed() const {
        int n = 0;
        for (const auto& r : results_)
            if (!r.passed() && !r.skipped) ++n;
        return n;
    }

    bool all_passed() const { return count_failed() == 0; }

    drv_gpu_lib::IBackend* backend() const { return backend_; }

    // ── JSON export (review #10) ────────────────────────────────
    // Дополняет GPUProfiler::ExportJSON (тот для kernel timings,
    // этот для PASS/FAIL результатов тестов).

    bool export_json(const std::string& file_path) const {
        std::ofstream f(file_path);
        if (!f.is_open()) return false;
        f << "{\n  \"module\": \"" << reporter_.module() << "\",\n";
        f << "  \"total\": " << results_.size() << ",\n";
        f << "  \"passed\": " << count_passed() << ",\n";
        f << "  \"failed\": " << count_failed() << ",\n";
        f << "  \"tests\": [\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            f << "    {\"name\": \"" << r.test_name << "\", "
              << "\"passed\": " << (r.passed() ? "true" : "false") << ", "
              << "\"skipped\": " << (r.skipped ? "true" : "false") << ", "
              << "\"checks\": " << r.validations.size() << "}";
            if (i + 1 < results_.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n}\n";
        return true;
    }
};

} // namespace gpu_test_utils
```

---

### 3. `modules/test_utils/gpu_test_base.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include "test_result.hpp"
#include "test_runner.hpp"
#include "DrvGPU/backends/i_backend.hpp"

namespace gpu_test_utils {

/**
 * GpuTestBase: Template Method для сложных GPU-тестов.
 * Зеркало Python: common/test_base.py (TestBase).
 *
 * Скелет: Setup → GenerateInput → RunGpu → ComputeReference → Validate → Teardown
 *
 * Подклассы переопределяют абстрактные hooks.
 *
 * Использование:
 *   class MyTest : public GpuTestBase {
 *   protected:
 *       std::string GetName() override { return "my_test"; }
 *       void GenerateInput() override { ... }
 *       void RunGpu() override { ... }
 *       void ComputeReference() override { ... }
 *       std::vector<ValidationResult> Validate() override { return {...}; }
 *   };
 *
 *   TestRunner runner(backend, "Module");
 *   runner.run(MyTest(backend));
 */
class GpuTestBase {
protected:
    drv_gpu_lib::IBackend* backend_;

public:
    explicit GpuTestBase(drv_gpu_lib::IBackend* backend)
        : backend_(backend) {}

    virtual ~GpuTestBase() = default;

    /**
     * Template Method: неизменный скелет теста.
     * Вызывается из TestRunner::run().
     */
    TestResult Run() {
        TestResult result{GetName()};
        try {
            Setup();
            GenerateInput();
            RunGpu();
            ComputeReference();
            auto validations = Validate();
            for (auto& vr : validations)
                result.add(std::move(vr));
        } catch (const SkipTest& e) {
            result.skipped = true;
            result.skip_reason = e.what();
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        try {
            Teardown();
        } catch (...) {
            // Teardown не должен маскировать ошибки теста
        }
        return result;
    }

protected:
    /// Имя теста (обязательно переопределить)
    virtual std::string GetName() = 0;

    /// Инициализация ресурсов (опционально)
    virtual void Setup() {}

    /// Создать входные данные (CPU)
    virtual void GenerateInput() = 0;

    /// Выполнить GPU вычисление
    virtual void RunGpu() = 0;

    /// Вычислить CPU-эталон
    virtual void ComputeReference() = 0;

    /// Сравнить GPU vs CPU → список ValidationResult
    virtual std::vector<ValidationResult> Validate() = 0;

    /// Очистка ресурсов (опционально, вызывается всегда)
    virtual void Teardown() {}
};

// ── TestRunner::run() реализация ────────────────────────────────

inline void TestRunner::run(GpuTestBase& test_obj) {
    auto tr = test_obj.Run();
    reporter_.print_header(tr.test_name);
    reporter_.print_test_result(tr);
    results_.push_back(std::move(tr));
}

} // namespace gpu_test_utils
```

---

### 4. `modules/test_utils/test_utils.hpp` — Master Include

```cpp
#pragma once

/**
 * test_utils — единая C++ тестовая инфраструктура GPUWorkLib.
 *
 * Зеркало Python: Python_test/common/
 *
 * Использование (функциональный стиль):
 *   #include "modules/test_utils/test_utils.hpp"
 *   using namespace gpu_test_utils;
 *
 *   void run(IBackend* backend) {
 *       TestRunner runner(backend, "MyModule");
 *       runner.test("test_name", [&]() {
 *           auto data = refs::GenerateCw(12e6f, 4096, 2e6f);
 *           // ... GPU compute ...
 *           return MaxRelError(gpu, cpu, n, tolerance::kComplex32, "signal");
 *       });
 *       runner.print_summary();
 *   }
 *
 * Использование (классовый стиль):
 *   class MyTest : public GpuTestBase { ... };
 *   runner.run(MyTest(backend));
 */

// Value Objects + Config
#include "test_result.hpp"
#include "test_configs.hpp"

// References (CPU эталоны)
#include "references/signal_refs.hpp"
#include "references/statistics_refs.hpp"
#include "references/fft_refs.hpp"

// Validators (composite объединён с test_result.hpp — review #6)
#include "validators/numeric.hpp"
#include "validators/signal.hpp"

// GPU Transfer
#include "gpu_transfer.hpp"

// Runner + Base + Reporters
#include "reporters.hpp"
#include "test_runner.hpp"
#include "gpu_test_base.hpp"
```

---

## ✅ Критерии завершения

- [ ] Все 4 файла созданы
- [ ] `#include "modules/test_utils/test_utils.hpp"` — компилируется
- [ ] TestRunner функциональный стиль работает:
  ```cpp
  TestRunner runner(backend, "Test");
  runner.test("name", [&]() { return PassResult("ok"); });
  runner.print_summary();  // → "SUMMARY: 1 passed, 0 failed"
  ```
- [ ] TestRunner классовый стиль: `runner.run(MyGpuTest(backend))`
- [ ] SkipTest перехватывается: `throw SkipTest("no GPU")` → SKIP в summary
- [ ] Exception перехватывается → ERROR в summary
- [ ] ConsoleTestReporter: вывод через ConsoleOutput (не std::cout!)
- [ ] GpuTestBase: Teardown вызывается ВСЕГДА (даже при exception)
- [ ] `runner.all_passed()` → bool
- [ ] `runner.count_failed()` → int

---

## 🧪 Пример использования (полный тест statistics)

```cpp
#include "modules/test_utils/test_utils.hpp"
using namespace gpu_test_utils;

namespace test_statistics_rocm {

inline void run(drv_gpu_lib::IBackend* backend) {
    StatisticsProcessorROCm proc(backend);
    TestRunner runner(backend, "Statistics");

    runner.test("mean_single_beam", [&]() {
        auto data = refs::GenerateSinusoid(100.f, 12e6f, 500000);
        auto stats = proc.ComputeAll(data.data(), data.size(), 1);
        float cpu = refs::CpuMeanMagnitude(data.data(), data.size());
        return ScalarRelError(stats.mean, cpu, tolerance::kStatistics, "mean");
    });

    runner.test("median_multi_beam", [&]() -> TestResult {
        TestResult tr("median_multi_beam");
        auto data = refs::GenerateMultiBeam(4, 500000, 12e6f, 100.f);
        // ... GPU compute → gpu_medians[4] ...
        for (int b = 0; b < 4; ++b) {
            float cpu_med = refs::CpuMedianMagnitude(
                data.data() + b * 500000, 500000);
            tr.add(ScalarRelError(gpu_medians[b], cpu_med,
                   tolerance::kStatistics, "median_beam" + std::to_string(b)));
        }
        return tr;
    });

    runner.print_summary();
}

} // namespace test_statistics_rocm
```

---

*Создан: 2026-03-21 | Кодо | Фаза 0*
