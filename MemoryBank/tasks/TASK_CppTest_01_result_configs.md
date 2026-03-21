# TASK_CppTest_01 — test_result.hpp + test_configs.hpp

> **Фаза**: 0 (инфраструктура)
> **Зависимости**: — (первый в цепочке)
> **Статус**: ⬜ TODO
> **Оценка**: ~1 час
> **Паттерны**: Value Object (GoF), Information Expert (GRASP)

---

## 🎯 Цель

Создать базовые Value Objects для тестовой инфраструктуры — `ValidationResult` и `TestResult`,
плюс централизованные tolerance'ы и параметры сигналов.

Зеркало Python: `common/result.py` + `common/configs.py`

---

## 📁 Создаваемые файлы (2 штуки)

```
modules/test_utils/
├── test_result.hpp      ← 1. ValidationResult + TestResult
└── test_configs.hpp     ← 2. Tolerances + SignalParams + FilterParams
```

---

## 📝 Детальное ТЗ

### 1. `modules/test_utils/test_result.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace gpu_test_utils {

/**
 * Результат одной валидации (аналог Python ValidationResult).
 * Value Object — неизменяемый после создания.
 */
struct ValidationResult {
    bool        passed;
    std::string metric_name;
    double      actual_value;   // double: вмещает float без потерь (Python float = 64 бит)
    double      threshold;      // double: точность для мелких ошибок (1e-7 и т.п.)
    std::string message;

    std::string to_string() const {
        std::ostringstream ss;
        ss << (passed ? "[PASS]" : "[FAIL]") << " " << metric_name
           << ": " << std::scientific << std::setprecision(4) << actual_value
           << " (tol=" << threshold << ")";
        if (!message.empty()) ss << " " << message;
        return ss.str();
    }
};

/**
 * Сводный результат одного теста (аналог Python TestResult).
 * Содержит имя теста + список ValidationResult + ошибка/skip.
 */
struct TestResult {
    std::string                  test_name;
    std::vector<ValidationResult> validations;
    std::string                  error;        // exception message (пусто если OK)
    bool                         skipped = false;
    std::string                  skip_reason;

    bool passed() const {
        if (!error.empty()) return false;
        if (skipped) return false;
        if (validations.empty()) return false;
        for (const auto& v : validations)
            if (!v.passed) return false;
        return true;
    }

    TestResult& add(ValidationResult vr) {
        validations.push_back(std::move(vr));
        return *this;
    }

    /// Добавить все проверки из initializer_list (Composite из review #6)
    TestResult& add_all(std::initializer_list<ValidationResult> checks) {
        for (const auto& vr : checks)
            validations.push_back(vr);
        return *this;
    }

    int count_passed() const {
        int n = 0;
        for (const auto& v : validations)
            if (v.passed) ++n;
        return n;
    }

    /// Первый FAIL (для отчёта), nullptr если все PASS
    const ValidationResult* first_failed() const {
        for (const auto& v : validations)
            if (!v.passed) return &v;
        return nullptr;
    }

    std::string summary() const {
        std::ostringstream ss;
        ss << (passed() ? "[PASS]" : "[FAIL]") << " " << test_name
           << " (" << count_passed() << "/" << validations.size() << " checks)";
        if (!error.empty()) ss << " ERROR: " << error;
        if (skipped) ss << " SKIP: " << skip_reason;
        return ss.str();
    }
};

/**
 * Исключение для пропуска теста (аналог Python SkipTest).
 * Бросается внутри теста → TestRunner помечает как SKIP.
 */
class SkipTest : public std::exception {
    std::string reason_;
public:
    explicit SkipTest(const std::string& reason) : reason_(reason) {}
    const char* what() const noexcept override { return reason_.c_str(); }
};

// ── Удобные фабричные функции ────────────────────────────────────

/**
 * Быстрое создание PASS ValidationResult.
 */
inline ValidationResult PassResult(const std::string& name,
                                    double value = 0.0,
                                    double threshold = 0.0,
                                    const std::string& msg = "") {
    return {true, name, value, threshold, msg};
}

/**
 * Быстрое создание FAIL ValidationResult.
 */
inline ValidationResult FailResult(const std::string& name,
                                    double value = 0.0,
                                    double threshold = 0.0,
                                    const std::string& msg = "") {
    return {false, name, value, threshold, msg};
}

} // namespace gpu_test_utils
```

---

### 2. `modules/test_utils/test_configs.hpp`

```cpp
#pragma once

#include <string>
#include <cstdint>
#include <cmath>

namespace gpu_test_utils {

// ── Централизованные Tolerance'ы ─────────────────────────────────
// Вместо хардкода 1e-3f в каждом тесте.

namespace tolerance {
    /// complex<float> GPU vs CPU (основной, 90% тестов)
    constexpr float kComplex32     = 1e-3f;

    /// float32 statistics (mean, std, variance)
    constexpr float kStatistics    = 1e-3f;

    /// float64 / double precision
    constexpr float kDouble        = 1e-5f;

    /// Частота пика FFT (Гц)
    constexpr float kFreqHz        = 5000.0f;

    /// Мощность сигнала (относительная)
    constexpr float kPower         = 0.05f;

    /// Строгое сравнение (для точных операций: copy, transpose)
    constexpr float kExact         = 1e-7f;

    /// FIR/IIR фильтры (transient допуск больше)
    constexpr float kFilter        = 5e-3f;
}

// ── Параметры сигнала (аналог Python SignalConfig) ───────────────

struct SignalParams {
    float    fs        = 12e6f;     ///< Частота дискретизации (Гц)
    size_t   n_samples = 4096;      ///< Число отсчётов
    float    f0_hz     = 2e6f;      ///< Несущая частота (Гц)
    float    fdev_hz   = 0.0f;      ///< Девиация (для ЛЧМ)
    float    amplitude = 1.0f;      ///< Амплитуда
    uint32_t seed      = 42;        ///< Seed для PRNG

    /// Длительность сигнала (с)
    float duration_s() const { return static_cast<float>(n_samples) / fs; }

    /// Разрешение по частоте (Гц/бин)
    float freq_resolution_hz(size_t nfft = 0) const {
        return fs / static_cast<float>(nfft > 0 ? nfft : n_samples);
    }

    /// Частота Найквиста
    float nyquist_hz() const { return fs / 2.0f; }
};

// ── Параметры фильтра (аналог Python FilterConfig) ──────────────

struct FilterParams {
    std::string filter_type = "fir";   ///< "fir" | "iir"
    float       cutoff_hz   = 1e3f;    ///< Частота среза
    float       fs          = 12e6f;   ///< Частота дискретизации
    int         order       = 4;       ///< Порядок
    int         n_taps      = 64;      ///< Число отводов FIR
    std::string window      = "hamming";

    float normalized_cutoff() const { return cutoff_hz / (fs / 2.0f); }
};

// ── Параметры дечирпа (аналог Python HeterodyneConfig) ──────────

struct DechirpParams {
    float    fs         = 12e6f;
    float    f_start    = 0.0f;
    float    f_end      = 2e6f;
    size_t   n_samples  = 8000;
    int      n_antennas = 5;

    /// Скорость света — точное значение, double для precision
    static constexpr double kSpeedOfLight = 299792458.0;
    double   c_light    = kSpeedOfLight;

    float bandwidth() const { return f_end - f_start; }
    float duration_s() const { return static_cast<float>(n_samples) / fs; }
    float chirp_rate() const { return bandwidth() / duration_s(); }
    double range_from_delay(double delay_s) const { return c_light * delay_s / 2.0; }
    float fbeat_from_delay(float delay_s) const { return chirp_rate() * delay_s; }
};

} // namespace gpu_test_utils
```

---

## ✅ Критерии завершения

- [ ] `modules/test_utils/test_result.hpp` создан
- [ ] `modules/test_utils/test_configs.hpp` создан
- [ ] `ValidationResult{true, "name", 0.001f, 0.01f, ""}.to_string()` → `"[PASS] name: 0.001 (tol=0.01)"`
- [ ] `TestResult{"test1"}.add(vr).passed()` → корректный bool
- [ ] `SkipTest("reason").what()` → `"reason"`
- [ ] `tolerance::kComplex32` → `1e-3f`
- [ ] `SignalParams{}.duration_s()` → корректно
- [ ] `DechirpParams{}.chirp_rate()` → `3e9` (2e6 / 6.67e-4)
- [ ] Компилируется: `g++ -std=c++17 -fsyntax-only test_result.hpp`
- [ ] Компилируется: `g++ -std=c++17 -fsyntax-only test_configs.hpp`

---

*Создан: 2026-03-21 | Кодо | Фаза 0*
