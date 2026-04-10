# TASK SNR_08: C++ тесты `test_snr_estimator_rocm.hpp` + helpers

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/tests/`
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_06](TASK_SNR_06_facade.md)** (ComputeSnrDb)
> **Ревьюер**: Кодо
>
> 📐 **План**: **Часть 4** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Написать C++ тесты для SNR-estimator + утилиты (helpers) в стиле `test_capon_rocm.hpp`. **Код пишем сегодня**, запуск на Debian в понедельник.

---

## 📁 Файлы (создать)

```
modules/statistics/tests/
├── snr_test_helpers.hpp           — утилиты (namespace snr_test_helpers)
└── test_snr_estimator_rocm.hpp    — 7 тестов (test_01..test_06b)

Обновить:
├── all_test.hpp                   — добавить include + закомментированные вызовы
└── README.md                      — описание 7 SNR тестов
```

---

## 📝 Часть 1 — `snr_test_helpers.hpp`

```cpp
#pragma once

/**
 * @file snr_test_helpers.hpp
 * @brief Test utilities for SNR-estimator (namespace snr_test_helpers)
 */

#if ENABLE_ROCM

#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>

#include <hip/hip_runtime.h>

namespace snr_test_helpers {

using cx = std::complex<float>;

/// Сгенерировать комплексный CW (тональный сигнал после дечирпа LFM)
/// @param n_samples   число отсчётов
/// @param freq_norm   нормированная частота ∈ (-0.5, 0.5) (f_d / f_s)
/// @param amplitude   амплитуда A
inline std::vector<cx> MakeDechirpedCW(
    uint32_t n_samples, float freq_norm, float amplitude)
{
  std::vector<cx> signal(n_samples);
  const float two_pi_f = 2.0f * 3.14159265358979323846f * freq_norm;
  for (uint32_t n = 0; n < n_samples; ++n) {
    float phase = two_pi_f * static_cast<float>(n);
    signal[n] = cx(amplitude * std::cos(phase),
                   amplitude * std::sin(phase));
  }
  return signal;
}

/// Сгенерировать комплексный AWGN через LCG + Box-Muller
/// @param n_samples   число отсчётов
/// @param noise_power σ² (дисперсия комплексного шума)
/// @param seed        LCG seed
inline std::vector<cx> MakeNoise(
    uint32_t n_samples, float noise_power, uint32_t seed = 42u)
{
  std::vector<cx> noise(n_samples);
  uint32_t state = seed;
  auto rng_uniform = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };

  const float sigma = std::sqrt(noise_power / 2.0f);
  for (uint32_t i = 0; i < n_samples; ++i) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-30f) u1 = 1e-30f;
    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    noise[i] = cx(sigma * r * std::cos(theta),
                  sigma * r * std::sin(theta));
  }
  return noise;
}

/// Добавить AWGN к сигналу (in-place)
inline void AddNoise(std::vector<cx>& signal,
                     float noise_power, uint32_t seed = 0u)
{
  auto noise = MakeNoise(static_cast<uint32_t>(signal.size()),
                         noise_power, seed);
  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] += noise[i];
  }
}

/// Скопировать CPU complex данные в hipMalloc буфер
inline void* CopyToGpu(const std::vector<cx>& data) {
  void* gpu_ptr = nullptr;
  size_t bytes = data.size() * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_ptr, bytes);
  if (err != hipSuccess || !gpu_ptr) {
    throw std::runtime_error("CopyToGpu: hipMalloc failed");
  }
  err = hipMemcpy(gpu_ptr, data.data(), bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(gpu_ptr);
    throw std::runtime_error("CopyToGpu: hipMemcpy H2D failed");
  }
  return gpu_ptr;
}

/// Освободить hipMalloc буфер
inline void FreeGpu(void* ptr) {
  if (ptr) hipFree(ptr);
}

/// Shared test ROCm backend (singleton, device 0)
inline drv_gpu_lib::IBackend* GetTestBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  if (!backend.IsInitialized()) backend.Initialize(0);
  return &backend;
}

}  // namespace snr_test_helpers

#endif  // ENABLE_ROCM
```

---

## 📝 Часть 2 — `test_snr_estimator_rocm.hpp` (7 тестов)

```cpp
#pragma once

#if ENABLE_ROCM

#include "snr_test_helpers.hpp"
#include "statistics_processor.hpp"
#include "branch_selector.hpp"
#include "services/console_output.hpp"

#include <cassert>
#include <cmath>

namespace test_snr_estimator_rocm {

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SNR", msg);
}

// ============================================================================
// test_01 — Только шум (нет сигнала). Проверяем артефакт CFAR ≈ 8-10 dB
// ============================================================================
inline void test_01_noise_only_artifact() {
  TestPrint("[test_01] Noise only — CFAR artifact");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  auto data = snr_test_helpers::MakeNoise(n_samp, 1.0f, /*seed=*/42);

  statistics::SnrEstimationConfig cfg;
  // auto target_n_fft = 2048

  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // H0 артефакт: ≈ ln(N_fft) + γ → ≈ 8-10 dB
  assert(result.snr_db_global > 5.0f && result.snr_db_global < 15.0f);
  assert(result.used_bins >= 1024 && result.used_bins <= 4096);

  statistics::BranchSelector selector;
  auto branch = selector.Select(result.snr_db_global, cfg.thresholds);
  // С откалиброванными порогами шум должен быть Low
  // (Но пороги могут быть ещё дефолтными — тут это hint'овая проверка)

  TestPrint("[test_01] PASS");
}

// ============================================================================
// test_02 — Сигнал + шум (базовый): SNR_in = 20 dB, 1 антенна
// ============================================================================
inline void test_02_basic_signal() {
  TestPrint("[test_02] CW + noise, SNR_in=20 dB");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  // SNR_in = 20 dB → A² / σ² = 100
  const float A = 10.0f, noise_power = 1.0f;
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, 0.15f, A);
  snr_test_helpers::AddNoise(signal, noise_power, /*seed=*/42);

  statistics::SnrEstimationConfig cfg;
  auto result = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg);

  // SNR_fft = SNR_in + 10*log10(N_actual) ≈ 20 + 32 = 52 dB
  // С учётом CFAR bias: 48-55
  assert(result.snr_db_global > 40.0f);

  TestPrint("[test_02] PASS");
}

// ============================================================================
// test_03 — Отрицательная частота + search_full_spectrum
// ============================================================================
inline void test_03_negative_freq() {
  TestPrint("[test_03] Negative freq + search_full_spectrum toggle");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, -0.2f, 10.0f);
  snr_test_helpers::AddNoise(signal, 1.0f, 42);

  // With full spectrum
  statistics::SnrEstimationConfig cfg_full;
  cfg_full.search_full_spectrum = true;
  auto r_full = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg_full);
  assert(r_full.snr_db_global > 30.0f);

  // With only [0..n/2] — пик в отрицательной части пропускается
  statistics::SnrEstimationConfig cfg_half;
  cfg_half.search_full_spectrum = false;
  auto r_half = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg_half);
  assert(r_half.snr_db_global < 15.0f);

  TestPrint("[test_03] PASS");
}

// ============================================================================
// test_04 — Сценарий A (2500 × 5000, auto)
// ============================================================================
inline void test_04_scenario_a() {
  TestPrint("[test_04] Scenario A: 2500 ant x 5000 samp");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 2500, n_samp = 5000;
  // Генерируем сигнал со случайными частотами в диапазоне 0.05..0.3
  std::vector<std::complex<float>> data(n_ant * n_samp);
  uint32_t state = 1337;
  auto rng = [&]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float freq_norm = 0.05f + 0.25f * rng();
    auto sig = snr_test_helpers::MakeDechirpedCW(n_samp, freq_norm, 5.6f);  // SNR_in ~ 15 dB
    snr_test_helpers::AddNoise(sig, 1.0f, /*seed=*/ant);
    std::copy(sig.begin(), sig.end(), data.begin() + (size_t)ant * n_samp);
  }

  statistics::SnrEstimationConfig cfg;  // auto
  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  assert(result.used_antennas == 50);  // 2500 / 50
  assert(result.snr_db_global > 38.0f && result.snr_db_global < 52.0f);

  TestPrint("[test_04] PASS");
}

// ============================================================================
// test_05 — Сценарий B (256 × 1.3M) — 2.66 GB
// ============================================================================
inline void test_05_scenario_b() {
  TestPrint("[test_05] Scenario B: 256 ant x 1.3M samp (2.66 GB)");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 256, n_samp = 1'300'000;

  // Для теста — данные на GPU через hipMalloc (чтобы не держать 2.66 GB в CPU)
  // Генерируем поблочно: сначала noise, потом добавляем сигнал в первую антенну
  // Для простоты теста — генерируем только одну антенну полностью и тиражируем
  // NB: реально тест нужен для проверки pipeline, не физики — так что OK

  std::vector<std::complex<float>> one_ant =
      snr_test_helpers::MakeDechirpedCW(n_samp, 0.1f, 3.2f);
  snr_test_helpers::AddNoise(one_ant, 1.0f, 0);

  // Upload в GPU hipMalloc: 256 копий
  void* gpu_data = nullptr;
  size_t total_bytes = (size_t)n_ant * n_samp * sizeof(std::complex<float>);
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  assert(err == hipSuccess);
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    hipMemcpy(
        static_cast<char*>(gpu_data) + (size_t)ant * n_samp * sizeof(std::complex<float>),
        one_ant.data(),
        n_samp * sizeof(std::complex<float>),
        hipMemcpyHostToDevice);
  }

  statistics::SnrEstimationConfig cfg;  // auto
  auto result = proc.ComputeSnrDb(gpu_data, n_ant, n_samp, cfg);

  hipFree(gpu_data);

  assert(result.used_antennas == 43);  // ceil(256/6)
  assert(result.snr_db_global > 30.0f);

  TestPrint("[test_05] PASS");
}

// ============================================================================
// test_06 — Только шум в сценарии B
// ============================================================================
inline void test_06_scenario_b_noise() {
  TestPrint("[test_06] Scenario B noise only — CFAR artifact stable");
  // Похожий на test_05, только MakeNoise (без сигнала)
  // Проверки: snr_db_global ≈ 8-10 dB, повторный запуск σ < 1 dB
  // ... (опустим полный код, аналогично test_05)
  TestPrint("[test_06] PASS");
}

// ============================================================================
// test_06b — Сценарий C (9000 × 10000)
// ============================================================================
inline void test_06b_scenario_c() {
  TestPrint("[test_06b] Scenario C: 9000 ant x 10000 samp");
  // n_ant = 9000, n_samp = 10000, SNR_in = 10 dB
  // ... (аналогично test_04, но с другими размерами)
  // Проверки: used_antennas == 50, snr_db_global > 38 dB
  TestPrint("[test_06b] PASS");
}

inline void run_all() {
  test_01_noise_only_artifact();
  test_02_basic_signal();
  test_03_negative_freq();
  test_04_scenario_a();
  test_05_scenario_b();
  test_06_scenario_b_noise();
  test_06b_scenario_c();
}

}  // namespace test_snr_estimator_rocm

#endif  // ENABLE_ROCM
```

---

## 📝 Часть 3 — обновить `all_test.hpp` и `README.md`

**`modules/statistics/tests/all_test.hpp`**:
```cpp
#include "test_snr_estimator_rocm.hpp"
// ...
// В main функции модуля:
// test_snr_estimator_rocm::run_all();   // ← закомментировано, раскомментить после отладки
```

**`modules/statistics/tests/README.md`** — добавить раздел «SNR Estimator» с описанием test_01..test_06b.

---

## ✅ Definition of Done

- [ ] `snr_test_helpers.hpp` создан с функциями `MakeDechirpedCW`, `MakeNoise`, `AddNoise`, `CopyToGpu`, `FreeGpu`, `GetTestBackend`
- [ ] `test_snr_estimator_rocm.hpp` создан с 7 тестами
- [ ] `all_test.hpp` обновлён (include + закомментированный вызов)
- [ ] `README.md` дополнен описанием SNR тестов
- [ ] **Код НЕ запускается сегодня** — только компиляционная проверка в понедельник на Debian
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ namespace `snr_test_helpers` для утилит, `test_snr_estimator_rocm` для тестов
- ✅ Вывод через `ConsoleOutput::GetInstance().Print(...)` — НЕ `std::cout`
- ✅ Тесты работают с новыми именами полей: `target_n_fft`, `search_full_spectrum`, `n_actual`, `used_bins`
- ✅ `SnrEstimationResult` не содержит `branch` — ветка через `BranchSelector`
- ✅ `test_01` и `test_06` проверяют артефакт CFAR ≈ 8-15 dB (широкий диапазон, пока пороги не откалиброваны)
- ✅ `test_02` проверяет `snr_db_global > 40 dB` (не точное число — CFAR bias)
- ✅ Все assert'ы используют **диапазоны**, а не точные значения (физика имеет разброс)
- ✅ `test_05` использует `hipMalloc` напрямую для GPU данных (2.66 GB в CPU vector — убьёт RAM)
- ✅ `all_test.hpp` вызов **закомментирован** (раскомментится в понедельник)

---

## 🚫 Запреты

- ❌ **НЕ запускать тесты** сегодня — нет AMD GPU под Windows на main ветке
- ❌ НЕ использовать `std::cout` — только `ConsoleOutput::GetInstance()`
- ❌ НЕ использовать `pytest` (это C++!)
- ❌ НЕ делать точные assert'ы: `snr_db_global == 52.2f` — только диапазоны

---

## 🔗 Связанные таски

- **Требует:** [SNR_06](TASK_SNR_06_facade.md) (ComputeSnrDb методы)
- **Идёт параллельно:** [SNR_09](TASK_SNR_09_benchmark.md) (бенчмарк)

---

*Created 2026-04-09 | Кодо*
