#pragma once

/**
 * @file test_heterodyne_basic.hpp
 * @brief Basic heterodyne dechirp tests — facade HeterodyneDechirp (ROCm backend)
 *
 * Test 1: Single antenna dechirp (delay=100us -> f_beat=300kHz)
 * Test 2: 5 antennas, linear delays [100,200,300,400,500] us
 * Test 6: Random delays (seed=42)
 *
 * Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
 * Tolerance: F_BEAT_TOL_HZ = 5 kHz
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21 (ROCm port 2026-03-16)
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"

#include "DrvGPU/services/console_output.hpp"

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <memory>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace heterodyne { namespace tests {

// ════════════════════════════════════════════════════════════════════════════
// Test parameters
// ════════════════════════════════════════════════════════════════════════════

constexpr float  FS         = 12e6f;
constexpr float  F_START    = 0.0f;
constexpr float  F_END      = 2e6f;
constexpr int    N          = 8000;
constexpr int    ANTENNAS   = 5;
constexpr float  BANDWIDTH  = F_END - F_START;  // 2 MHz
constexpr float  DURATION   = static_cast<float>(N) / FS;  // 666.67 us
constexpr float  MU         = BANDWIDTH / DURATION;  // 3e9 Hz/s

const std::vector<float> DELAYS_LINEAR_US = {100.f, 200.f, 300.f, 400.f, 500.f};

constexpr float F_BEAT_TOL_HZ = 5000.f;  // +/- 5 kHz tolerance

// ════════════════════════════════════════════════════════════════════════════
// Helper: CPU-only delayed LFM generation (no OpenCL)
//   if t < tau: 0
//   else:  exp(j*(pi*mu*(t-tau)^2 + 2*pi*f_start*(t-tau)))
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateRxFlat(
    const std::vector<float>& delays_us) {

  float duration = static_cast<float>(N) / FS;
  float mu = (F_END - F_START) / duration;

  size_t total = delays_us.size() * N;
  std::vector<std::complex<float>> flat(total);

  for (size_t ant = 0; ant < delays_us.size(); ++ant) {
    float tau = delays_us[ant] * 1e-6f;
    for (int n = 0; n < N; ++n) {
      float t = static_cast<float>(n) / FS;
      if (t < tau) {
        flat[ant * N + n] = {0.0f, 0.0f};
      } else {
        float t_local = t - tau;
        float phase = static_cast<float>(M_PI) * mu * t_local * t_local
                    + 2.0f * static_cast<float>(M_PI) * F_START * t_local;
        flat[ant * N + n] = {std::cos(phase), std::sin(phase)};
      }
    }
  }
  return flat;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Single antenna dechirp
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_single_antenna() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 1: Single antenna dechirp (delay=100us)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    std::vector<float> delay = {100.f};
    auto rx_flat = GenerateRxFlat(delay);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = 1;

    drv_gpu_lib::HeterodyneDechirp het(backend.get(), drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    float expected_f_beat = MU * 100e-6f;  // 300 kHz
    float actual_f_beat = result.antennas[0].f_beat_hz;
    float error = std::abs(actual_f_beat - expected_f_beat);

    con.Print(gpu_id, "Heterodyne", "    f_beat expected: "
        + std::to_string(expected_f_beat) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    f_beat actual:   "
        + std::to_string(actual_f_beat) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    error:           "
        + std::to_string(error) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    SNR:             "
        + std::to_string(result.antennas[0].peak_snr_db) + " dB");

    bool passed = (error < F_BEAT_TOL_HZ);
    con.Print(gpu_id, "Heterodyne",
        passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: 5 antennas, linear delays
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_5_antennas_linear() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 2: 5 antennas, linear delays");

  try {
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    auto rx_flat = GenerateRxFlat(DELAYS_LINEAR_US);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get(), drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Ant | Delay us | f_beat Hz   | R m       | SNR dB");
    con.Print(gpu_id, "Heterodyne",
        "    ----|----------|-------------|-----------|-------");

    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float delay_us = DELAYS_LINEAR_US[ant];
      float expected_f_beat = MU * delay_us * 1e-6f;
      float actual_f_beat = result.antennas[ant].f_beat_hz;
      float error = std::abs(actual_f_beat - expected_f_beat);

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    %3d | %7.0f  | %11.0f | %9.2f | %6.1f  %s",
          ant, delay_us, actual_f_beat,
          result.antennas[ant].range_m,
          result.antennas[ant].peak_snr_db,
          (error < F_BEAT_TOL_HZ) ? "OK" : "FAIL");
      con.Print(gpu_id, "Heterodyne", buf);

      if (error >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: ALL PASSED" : "    RESULT: SOME FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Random delays (seed=42), 5 antennas, delays [10..500] us
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_random_delays() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 6: Random delays (seed=42)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(10.0f, 500.0f);
    std::vector<float> delays_us(ANTENNAS);
    for (int i = 0; i < ANTENNAS; ++i) {
      delays_us[i] = dist(rng);
    }

    auto rx_flat = GenerateRxFlat(delays_us);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get(), drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Ant | Delay us | f_beat Hz   | Expected Hz | Error Hz | SNR dB");
    con.Print(gpu_id, "Heterodyne",
        "    ----|----------|-------------|-------------|----------|-------");

    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float expected_f = MU * delays_us[ant] * 1e-6f;
      float actual_f = result.antennas[ant].f_beat_hz;
      float f_err = std::abs(actual_f - expected_f);

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    %3d | %7.1f  | %11.0f | %11.0f | %8.0f | %6.1f  %s",
          ant, delays_us[ant], actual_f, expected_f, f_err,
          result.antennas[ant].peak_snr_db,
          (f_err < F_BEAT_TOL_HZ) ? "OK" : "FAIL");
      con.Print(gpu_id, "Heterodyne", buf);

      if (f_err >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: ALL PASSED" : "    RESULT: SOME FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

}} // namespace heterodyne::tests

#else  // !ENABLE_ROCM

namespace heterodyne { namespace tests {
inline void run_test_single_antenna()   {}
inline void run_test_5_antennas_linear() {}
inline void run_test_random_delays()    {}
}} // namespace heterodyne::tests

#endif  // ENABLE_ROCM
