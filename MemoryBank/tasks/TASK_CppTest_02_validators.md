# TASK_CppTest_02 — validators/ (числовые + сигнальные + composite)

> **Фаза**: 0 (инфраструктура)
> **Зависимости**: TASK_CppTest_01 (test_result.hpp)
> **Статус**: ⬜ TODO
> **Оценка**: ~1.5 часа
> **Паттерны**: Strategy (GoF), Composite (GoF), Template (C++), SOLID SRP/OCP

---

## 🎯 Цель

Заменить 15+ копий `MaxError()` + `bool pass = (err < tol)` единой системой валидаторов.

Зеркало Python: `common/validators/` (numeric.py, signal.py, composite.py)

**Было** (4 строки × 80 мест):
```cpp
float err = MaxError(gpu, cpu, n);
bool pass = (err < 1e-3f);
con.Print(0, "Mod", pass ? "[+]" : "[X]");
```

**Станет** (1 строка):
```cpp
return MaxRelError(gpu, cpu, n, 1e-3f, "metric_name");
```

---

## 📁 Создаваемые файлы (2 штуки)

> ⚠️ composite.hpp объединён с test_result.hpp (TASK_CppTest_01) — review #6

```
modules/test_utils/validators/
├── numeric.hpp      ← 1. MaxRelError<T>, AbsError<T>, RmseError<T>, Scalar*
└── signal.hpp       ← 2. CheckPeakFreq, CheckPower
```

---

## 📝 Детальное ТЗ

### 1. `modules/test_utils/validators/numeric.hpp`

```cpp
#pragma once

#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "../test_result.hpp"

namespace gpu_test_utils {

// ── value_to_string: работает с float, double, complex (review #3) ──

namespace detail {
    template<typename T>
    inline std::string value_to_string(const T& v) { return std::to_string(v); }

    template<typename T>
    inline std::string value_to_string(const std::complex<T>& v) {
        return "(" + std::to_string(v.real()) + "," + std::to_string(v.imag()) + ")";
    }
}

// ── Шаблонные хелперы для |a - b| ──────────────────────────────

/// |a - b| для float / double
template<typename T>
inline double AbsDiff(const T& a, const T& b) {
    return static_cast<double>(std::abs(a - b));
}

/// |a - b| для complex<float> / complex<double>
template<typename T>
inline double AbsDiff(const std::complex<T>& a, const std::complex<T>& b) {
    return static_cast<double>(std::abs(a - b));
}

/// |a| для скаляров
template<typename T>
inline double AbsVal(const T& a) {
    return static_cast<double>(std::abs(a));
}

/// |a| для complex
template<typename T>
inline double AbsVal(const std::complex<T>& a) {
    return static_cast<double>(std::abs(a));
}

// ══════════════════════════════════════════════════════════════════
// FREE FUNCTIONS — основной API для быстрой миграции (1 строка)
// ══════════════════════════════════════════════════════════════════

/**
 * max|actual - ref| / max|ref| < tolerance
 *
 * Основная метрика для сравнения GPU vs CPU.
 * Вычисления в double для точности (данные = float32).
 * Strict `<` (не <=) для совместимости с Python DataValidator.
 *
 * @tparam T  float, double, complex<float>, complex<double>
 * @param actual    GPU результат
 * @param reference CPU эталон
 * @param count     количество элементов
 * @param tolerance допуск (default = tolerance::kComplex32)
 * @param name      имя метрики для отчёта
 * @return ValidationResult
 */
template<typename T>
inline ValidationResult MaxRelError(
    const T* actual, const T* reference, size_t count,
    float tolerance, const std::string& name = "max_rel")
{
    double max_diff = 0.0;
    double max_ref  = 0.0;
    for (size_t i = 0; i < count; ++i) {
        max_diff = std::max(max_diff, AbsDiff(actual[i], reference[i]));
        max_ref  = std::max(max_ref,  AbsVal(reference[i]));
    }
    // near-zero reference → абсолютный допуск
    if (max_ref < 1e-15) {
        return {max_diff < 1e-10, name, max_diff,
                1e-10, "(near-zero reference)"};
    }
    double err = max_diff / max_ref;
    return {err < static_cast<double>(tolerance), name, err,
            static_cast<double>(tolerance), ""};
}

/**
 * max|actual - ref| < tolerance
 *
 * Для абсолютных величин: частоты в Гц, индексы бинов.
 */
template<typename T>
inline ValidationResult AbsError(
    const T* actual, const T* reference, size_t count,
    float tolerance, const std::string& name = "abs")
{
    double max_diff = 0.0;
    for (size_t i = 0; i < count; ++i)
        max_diff = std::max(max_diff, AbsDiff(actual[i], reference[i]));
    double err = max_diff;
    return {err < static_cast<double>(tolerance), name, err,
            static_cast<double>(tolerance), ""};
}

/**
 * rms(|actual - ref|) / rms(|ref|) < tolerance
 *
 * Для шумных данных, фильтров.
 */
template<typename T>
inline ValidationResult RmseError(
    const T* actual, const T* reference, size_t count,
    float tolerance, const std::string& name = "rmse")
{
    double sum_sq_diff = 0.0;
    double sum_sq_ref  = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double d = AbsDiff(actual[i], reference[i]);
        double r = AbsVal(reference[i]);
        sum_sq_diff += d * d;
        sum_sq_ref  += r * r;
    }
    double rms_diff = std::sqrt(sum_sq_diff / static_cast<double>(count));
    double rms_ref  = std::sqrt(sum_sq_ref  / static_cast<double>(count));
    if (rms_ref < 1e-15) {
        return {rms_diff < 1e-10, name, rms_diff,
                1e-10, "(near-zero reference)"};
    }
    double err = rms_diff / rms_ref;
    return {err < static_cast<double>(tolerance), name, err,
            static_cast<double>(tolerance), ""};
}

/**
 * Проверка скалярного значения: |actual - expected| / |expected| < tolerance
 *
 * Удобно для одиночных метрик (mean, std, median).
 */
template<typename T>
inline ValidationResult ScalarRelError(
    T actual, T expected, float tolerance,
    const std::string& name = "scalar_rel")
{
    double diff = AbsDiff(actual, expected);
    double ref  = AbsVal(expected);
    if (ref < 1e-15) {
        return {diff < 1e-10, name, diff, 1e-10, ""};
    }
    double err = diff / ref;
    return {err < static_cast<double>(tolerance), name, err,
            static_cast<double>(tolerance),
            "actual=" + detail::value_to_string(actual) +
            " expected=" + detail::value_to_string(expected)};
}

/**
 * Проверка скалярного значения: |actual - expected| < tolerance (абсолютная)
 */
template<typename T>
inline ValidationResult ScalarAbsError(
    T actual, T expected, float tolerance,
    const std::string& name = "scalar_abs")
{
    double err = AbsDiff(actual, expected);
    return {err < static_cast<double>(tolerance), name, err,
            static_cast<double>(tolerance),
            "actual=" + detail::value_to_string(actual) +
            " expected=" + detail::value_to_string(expected)};
}

} // namespace gpu_test_utils
```

---

### 2. `modules/test_utils/validators/signal.hpp`

```cpp
#pragma once

#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
#include "../test_result.hpp"

namespace gpu_test_utils {

/**
 * Проверяет что пик FFT-спектра в нужном месте.
 *
 * @param magnitude  |FFT| данные (float)
 * @param n_bins     число бинов
 * @param fs         частота дискретизации
 * @param expected_hz ожидаемая частота пика
 * @param tolerance_hz допуск в Гц
 */
inline ValidationResult CheckPeakFreq(
    const float* magnitude, size_t n_bins, float fs,
    float expected_hz, float tolerance_hz,
    const std::string& name = "peak_freq_hz")
{
    // Найти пик в первой половине (положительные частоты)
    size_t half = n_bins / 2;
    size_t peak_bin = 0;
    float peak_val = 0.0f;
    for (size_t i = 0; i < half; ++i) {
        if (magnitude[i] > peak_val) {
            peak_val = magnitude[i];
            peak_bin = i;
        }
    }
    float actual_hz = static_cast<float>(peak_bin) * fs / static_cast<float>(n_bins);
    float err = std::abs(actual_hz - expected_hz);
    return {err < tolerance_hz, name, actual_hz, tolerance_hz,
            "expected=" + std::to_string(expected_hz) + "Hz err=" + std::to_string(err) + "Hz"};
}

/**
 * CheckPeakFreq из complex спектра (автоматический |FFT|).
 */
inline ValidationResult CheckPeakFreqComplex(
    const std::complex<float>* spectrum, size_t n_bins, float fs,
    float expected_hz, float tolerance_hz,
    const std::string& name = "peak_freq_hz")
{
    std::vector<float> mag(n_bins);
    for (size_t i = 0; i < n_bins; ++i)
        mag[i] = std::abs(spectrum[i]);
    return CheckPeakFreq(mag.data(), n_bins, fs, expected_hz, tolerance_hz, name);
}

/**
 * Проверяет мощность сигнала: |mean(|x|^2) - expected| / expected < tolerance
 */
inline ValidationResult CheckPower(
    const std::complex<float>* data, size_t n,
    float expected_power, float tolerance = 0.05f,
    const std::string& name = "power")
{
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float m = std::abs(data[i]);
        sum_sq += static_cast<double>(m) * m;
    }
    float actual = static_cast<float>(sum_sq / static_cast<double>(n));
    float rel_err = (expected_power > 1e-10f)
        ? std::abs(actual - expected_power) / expected_power
        : std::abs(actual);
    return {rel_err < tolerance, name, actual, tolerance,
            "expected=" + std::to_string(expected_power)};
}

} // namespace gpu_test_utils
```

---

> ℹ️ **composite.hpp** объединён с `test_result.hpp` (TASK_CppTest_01, review #6).
> `TestResult::add_all()` и `TestResult::first_failed()` заменяют отдельный файл.

---

## ✅ Критерии завершения

- [ ] 2 файла созданы в `modules/test_utils/validators/` (numeric.hpp, signal.hpp)
- [ ] `MaxRelError<float>(a, b, 100, 1e-3f, "test")` → `ValidationResult`
- [ ] `MaxRelError<complex<float>>(a, b, 100, 1e-3f)` → работает
- [ ] `AbsError<float>(...)` → `ValidationResult` с strict `<`
- [ ] `RmseError<float>(...)` → `ValidationResult`
- [ ] `ScalarRelError(0.5f, 0.501f, 0.01f)` → PASS
- [ ] `CheckPeakFreq(mag, 4096, 12e6f, 2e6f, 5e3f)` → PASS при корректных данных
- [ ] `CheckPower(data, n, 1.0f, 0.05f)` → PASS
- [ ] `CompositeCheck({vr1, vr2, vr3})` → vector с 3 элементами
- [ ] Все вычисления ошибок в `double` (не float32!)
- [ ] Strict `<` (не `<=`) во всех валидаторах
- [ ] Near-zero reference обработан (как Python DataValidator)
- [ ] Компилируется: `g++ -std=c++17 -fsyntax-only validators/numeric.hpp`

---

*Создан: 2026-03-21 | Кодо | Фаза 0*
