#pragma once

/**
 * @file test_heterodyne_basic.hpp
 * @brief Basic heterodyne dechirp tests — facade HeterodyneDechirp (ROCm)
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. single_antenna       — delay=100us → f_beat=300kHz
 *   2. five_antennas_linear — delays [100,200,300,400,500] us
 *   3. random_delays        — seed=42, 5 antennas
 *
 * Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
 * Tolerance: ±5 kHz
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21 (migrated 2026-03-23)
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"

#include "modules/test_utils/test_utils.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace heterodyne { namespace tests {

using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// Test parameters
constexpr float  FS         = 12e6f;
constexpr float  F_START    = 0.0f;
constexpr float  F_END      = 2e6f;
constexpr int    N          = 8000;
constexpr int    ANTENNAS   = 5;
constexpr float  BANDWIDTH  = F_END - F_START;
constexpr float  DURATION   = static_cast<float>(N) / FS;
constexpr float  MU         = BANDWIDTH / DURATION;

const std::vector<float> DELAYS_LINEAR_US = {100.f, 200.f, 300.f, 400.f, 500.f};

constexpr float F_BEAT_TOL_HZ = 5000.f;

// Helper: CPU delayed LFM generation
inline std::vector<std::complex<float>> GenerateRxFlat(
    const std::vector<float>& delays_us) {
  float mu = (F_END - F_START) / (static_cast<float>(N) / FS);
  size_t total = delays_us.size() * N;
  std::vector<std::complex<float>> flat(total);
  for (size_t ant = 0; ant < delays_us.size(); ++ant) {
    float tau = delays_us[ant] * 1e-6f;
    for (int n = 0; n < N; ++n) {
      float t = static_cast<float>(n) / FS;
      if (t < tau) { flat[ant * N + n] = {0, 0}; continue; }
      float tl = t - tau;
      float phase = static_cast<float>(M_PI) * mu * tl * tl
                  + 2.0f * static_cast<float>(M_PI) * F_START * tl;
      flat[ant * N + n] = {std::cos(phase), std::sin(phase)};
    }
  }
  return flat;
}

// Helper: run dechirp and validate f_beat for each antenna
inline TestResult ValidateDechirp(
    IBackend* backend, const std::string& name,
    const std::vector<float>& delays_us) {
  auto rx_flat = GenerateRxFlat(delays_us);
  uint32_t n_ant = static_cast<uint32_t>(delays_us.size());

  HeterodyneParams params;
  params.f_start = F_START;
  params.f_end = F_END;
  params.sample_rate = FS;
  params.num_samples = N;
  params.num_antennas = static_cast<int>(n_ant);

  HeterodyneDechirp het(backend, BackendType::ROCm);
  het.SetParams(params);
  auto result = het.Process(rx_flat);

  TestResult tr{name};
  if (!result.success)
    return tr.add(FailResult("process", 0, 1));

  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    float expected_f = MU * delays_us[ant] * 1e-6f;
    float actual_f = result.antennas[ant].f_beat_hz;
    tr.add(ScalarAbsError(
        static_cast<double>(actual_f), static_cast<double>(expected_f),
        F_BEAT_TOL_HZ, "ant" + std::to_string(ant)));
  }
  return tr;
}

inline void run_basic_tests() {
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "Heterodyne Basic", gpu_id);

  // ── Test 1: Single antenna ────────────────────────────────────
  runner.test("single_antenna", [&]() {
    return ValidateDechirp(&backend, "single_antenna", {100.f});
  });

  // ── Test 2: 5 antennas linear ─────────────────────────────────
  runner.test("five_antennas_linear", [&]() {
    return ValidateDechirp(&backend, "five_antennas_linear", DELAYS_LINEAR_US);
  });

  // ── Test 3: Random delays ─────────────────────────────────────
  runner.test("random_delays", [&]() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(10.0f, 500.0f);
    std::vector<float> delays(ANTENNAS);
    for (int i = 0; i < ANTENNAS; ++i) delays[i] = dist(rng);
    return ValidateDechirp(&backend, "random_delays", delays);
  });

  runner.print_summary();
}

// Backward-compatible individual functions (called from all_test.hpp)
inline void run_test_single_antenna()    { /* now in run_basic_tests */ }
inline void run_test_5_antennas_linear() { /* now in run_basic_tests */ }
inline void run_test_random_delays()     { /* now in run_basic_tests */ }

}} // namespace heterodyne::tests

#else  // !ENABLE_ROCM

namespace heterodyne { namespace tests {
inline void run_basic_tests()            {}
inline void run_test_single_antenna()    {}
inline void run_test_5_antennas_linear() {}
inline void run_test_random_delays()     {}
}} // namespace heterodyne::tests

#endif  // ENABLE_ROCM
