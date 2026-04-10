#pragma once

/**
 * @file test_snr_estimator_rocm.hpp
 * @brief C++ tests for SNR-estimator (SNR_08) — 7 тестов test_01..test_06b
 *
 * ⚠️ КОД ТОЛЬКО НАПИСАН — запуск в понедельник на Debian/AMD (нет GPU под Windows).
 *
 * Tests:
 *   test_01 — noise only → CFAR artifact ≈ 8-15 dB
 *   test_02 — CW + noise SNR_in=20 dB → snr_db_global > 40 dB
 *   test_03 — negative freq + search_full_spectrum toggle
 *   test_04 — Scenario A (2500 ant × 5000 samp)
 *   test_05 — Scenario B (256 × 1.3M) = 2.66 GB (GPU-only)
 *   test_06 — Scenario B noise only
 *   test_06b — Scenario C (9000 × 10000)
 *
 * Все assert'ы используют диапазоны (физика имеет разброс), НЕ точные значения.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "snr_test_helpers.hpp"
#include "statistics_processor.hpp"
#include "branch_selector.hpp"

#include "services/console_output.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <complex>

namespace test_snr_estimator_rocm {

using cx = std::complex<float>;

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SNR", msg);
}

// ============================================================================
// test_01 — Только шум (нет сигнала). Проверяем артефакт CFAR ≈ 8-15 dB
// ============================================================================
inline void test_01_noise_only_artifact() {
  TestPrint("[test_01] Noise only — CFAR artifact");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  auto data = snr_test_helpers::MakeNoise(n_samp, /*noise_power=*/1.0f, /*seed=*/42u);

  statistics::SnrEstimationConfig cfg;  // defaults: target_n_fft=0→2048, Hann, guard=5, ref=16

  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // H0 артефакт: E[SNR_fft_db | noise] ≈ 10·log10(ln(N_fft) + γ) ≈ 8-10 dB.
  // Широкий диапазон — CFAR имеет разброс по реализациям.
  assert(result.snr_db_global > 3.0f && result.snr_db_global < 18.0f);
  assert(result.used_bins >= 1024u && result.used_bins <= 4096u);

  // BranchSelector с откалиброванными порогами: шум должен быть Low.
  statistics::BranchSelector selector;
  auto branch = selector.Select(result.snr_db_global, cfg.thresholds);
  assert(branch == statistics::BranchType::Low);

  TestPrint("[test_01] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_02 — CW + шум, SNR_in = 20 dB, 1 антенна
// ============================================================================
inline void test_02_basic_signal() {
  TestPrint("[test_02] CW + noise, SNR_in=20 dB");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  // SNR_in = 20 dB → A² / σ² = 100 → A = 10 при σ² = 1
  const float A = 10.0f, noise_power = 1.0f;
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/0.15f, A);
  snr_test_helpers::AddNoise(signal, noise_power, /*seed=*/42u);

  statistics::SnrEstimationConfig cfg;
  auto result = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg);

  // SNR_fft = SNR_in + 10·log10(N_actual) ≈ 20 + 32 = 52 dB.
  // С учётом Hann processing loss ~1.76 dB и CFAR bias ~−5 dB: 40-55 dB диапазон.
  assert(result.snr_db_global > 38.0f);

  TestPrint("[test_02] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_03 — Отрицательная частота + search_full_spectrum toggle
// ============================================================================
inline void test_03_negative_freq() {
  TestPrint("[test_03] Negative freq + search_full_spectrum toggle");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 1, n_samp = 5000;
  // Отрицательная частота: −0.2 → f * n_samples даёт пик в [nFFT/2..nFFT)
  auto signal = snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/-0.2f, /*A=*/10.0f);
  snr_test_helpers::AddNoise(signal, /*noise_power=*/1.0f, /*seed=*/42u);

  // Full spectrum: peak находится в отрицательной части — SNR должен быть высокий.
  statistics::SnrEstimationConfig cfg_full;
  cfg_full.search_full_spectrum = true;
  auto r_full = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg_full);
  assert(r_full.snr_db_global > 30.0f);

  // Только [0..nFFT/2): пик в отрицательной части пропускается — CFAR видит только шум.
  statistics::SnrEstimationConfig cfg_half;
  cfg_half.search_full_spectrum = false;
  auto r_half = proc.ComputeSnrDb(signal, n_ant, n_samp, cfg_half);
  assert(r_half.snr_db_global < 18.0f);

  TestPrint("[test_03] PASS — full=" + std::to_string(r_full.snr_db_global) +
            " half=" + std::to_string(r_half.snr_db_global));
}

// ============================================================================
// test_04 — Сценарий A (2500 × 5000, auto params)
// ============================================================================
inline void test_04_scenario_a() {
  TestPrint("[test_04] Scenario A: 2500 ant x 5000 samp");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 2500, n_samp = 5000;
  // Генерируем сигнал со случайными частотами в диапазоне 0.05..0.3
  // SNR_in ~ 15 dB → A ≈ 5.6 при σ² = 1
  std::vector<cx> data(static_cast<size_t>(n_ant) * n_samp);
  uint32_t state = 1337u;
  auto rng = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float freq_norm = 0.05f + 0.25f * rng();
    auto sig = snr_test_helpers::MakeDechirpedCW(n_samp, freq_norm, /*A=*/5.6f);
    snr_test_helpers::AddNoise(sig, /*noise_power=*/1.0f, /*seed=*/ant + 1u);
    std::copy(sig.begin(), sig.end(), data.begin() + static_cast<size_t>(ant) * n_samp);
  }

  statistics::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(2500/50) = 50
  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // n_ant_out = ceil(2500 / 50) = 50
  assert(result.used_antennas == 50u);
  // SNR_in=15 dB + 10·log10(2500) ≈ 15 + 34 = 49 dB, с bias → 35-52 dB
  assert(result.snr_db_global > 30.0f && result.snr_db_global < 55.0f);

  TestPrint("[test_04] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// test_05 — Сценарий B (256 × 1.3M) — 2.66 GB, GPU-only путь
// ============================================================================
inline void test_05_scenario_b() {
  TestPrint("[test_05] Scenario B: 256 ant x 1.3M samp (2.66 GB)");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 256, n_samp = 1'300'000u;

  // Для экономии RAM: генерируем одну антенну (сигнал+шум) и тиражируем
  // в GPU через hipMemcpy. Тест проверяет pipeline, не физическую уникальность.
  std::vector<cx> one_ant =
      snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/0.1f, /*A=*/3.2f);
  snr_test_helpers::AddNoise(one_ant, /*noise_power=*/1.0f, /*seed=*/1u);

  void* gpu_data = nullptr;
  const size_t total_bytes = static_cast<size_t>(n_ant) * n_samp * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  if (err != hipSuccess) {
    TestPrint("[test_05] SKIP — hipMalloc failed (" +
              std::to_string(total_bytes / (1024u * 1024u)) + " MB)");
    return;
  }

  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    char* dst = static_cast<char*>(gpu_data) +
                static_cast<size_t>(ant) * n_samp * sizeof(cx);
    hipMemcpy(dst, one_ant.data(), n_samp * sizeof(cx), hipMemcpyHostToDevice);
  }

  statistics::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(256/50) = 6
  auto result = proc.ComputeSnrDb(gpu_data, n_ant, n_samp, cfg);

  snr_test_helpers::FreeGpu(gpu_data);

  // n_ant_out = ceil(256 / 6) = 43
  assert(result.used_antennas == 43u);
  // SNR_in = 10 dB (A=3.2, σ²=1 → A²/σ² ≈ 10) + 10·log10(~1666) ≈ 10 + 32 = 42 dB
  // После CFAR bias и Hann loss: > 25 dB
  assert(result.snr_db_global > 25.0f);

  TestPrint("[test_05] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// test_06 — Scenario B только шум — CFAR artifact стабилен
// ============================================================================
inline void test_06_scenario_b_noise() {
  TestPrint("[test_06] Scenario B noise only — CFAR artifact stable");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 256, n_samp = 1'300'000u;

  // Только шум, одна антенна тиражируется в GPU memory
  auto one_ant = snr_test_helpers::MakeNoise(n_samp, /*noise_power=*/1.0f, /*seed=*/7u);

  void* gpu_data = nullptr;
  const size_t total_bytes = static_cast<size_t>(n_ant) * n_samp * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  if (err != hipSuccess) {
    TestPrint("[test_06] SKIP — hipMalloc failed");
    return;
  }
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    char* dst = static_cast<char*>(gpu_data) +
                static_cast<size_t>(ant) * n_samp * sizeof(cx);
    hipMemcpy(dst, one_ant.data(), n_samp * sizeof(cx), hipMemcpyHostToDevice);
  }

  statistics::SnrEstimationConfig cfg;
  auto result = proc.ComputeSnrDb(gpu_data, n_ant, n_samp, cfg);
  snr_test_helpers::FreeGpu(gpu_data);

  // Только noise → CFAR artifact ≈ 8-15 dB
  assert(result.snr_db_global > 3.0f && result.snr_db_global < 18.0f);

  TestPrint("[test_06] PASS — snr_db=" + std::to_string(result.snr_db_global));
}

// ============================================================================
// test_06b — Сценарий C (9000 × 10000)
// ============================================================================
inline void test_06b_scenario_c() {
  TestPrint("[test_06b] Scenario C: 9000 ant x 10000 samp");
  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const uint32_t n_ant = 9000, n_samp = 10000;

  // 9000 × 10000 × 8 = 720 MB — помещается в CPU
  std::vector<cx> data(static_cast<size_t>(n_ant) * n_samp);
  uint32_t state = 2026u;
  auto rng = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };
  // SNR_in = 10 dB → A²/σ² = 10 → A ≈ 3.16
  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float freq_norm = 0.05f + 0.25f * rng();
    auto sig = snr_test_helpers::MakeDechirpedCW(n_samp, freq_norm, /*A=*/3.16f);
    snr_test_helpers::AddNoise(sig, /*noise_power=*/1.0f, /*seed=*/ant + 100u);
    std::copy(sig.begin(), sig.end(), data.begin() + static_cast<size_t>(ant) * n_samp);
  }

  statistics::SnrEstimationConfig cfg;  // auto: step_antennas → ceil(9000/50) = 180
  auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

  // n_ant_out = ceil(9000 / 180) = 50
  assert(result.used_antennas == 50u);
  // SNR_in=10 + 10·log10(10000) ≈ 10 + 40 = 50 dB, с bias → > 30
  assert(result.snr_db_global > 30.0f);

  TestPrint("[test_06b] PASS — snr_db=" + std::to_string(result.snr_db_global) +
            " used_ant=" + std::to_string(result.used_antennas));
}

// ============================================================================
// run_all — вызывает все 7 тестов (запуск в понедельник!)
// ============================================================================
inline void run_all() {
  TestPrint("=== SNR Estimator C++ Tests ===");

  // Проверка что есть ROCm device
  int devices = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (devices == 0) {
    TestPrint("[!] No ROCm devices — skipping SNR tests");
    return;
  }

  test_01_noise_only_artifact();
  test_02_basic_signal();
  test_03_negative_freq();
  test_04_scenario_a();
  test_05_scenario_b();
  test_06_scenario_b_noise();
  test_06b_scenario_c();

  TestPrint("=== SNR Estimator Tests DONE ===");
}

}  // namespace test_snr_estimator_rocm

#endif  // ENABLE_ROCM
