#pragma once

/**
 * @file test_moving_average_rocm.hpp
 * @brief ROCm tests for MovingAverageFilterROCm (SMA, EMA, MMA, DEMA, TEMA)
 *
 * Tests:
 *   1. EMA(N=10) GPU vs CPU, 8ch x 4096pts
 *   2. SMA(N=8) GPU vs CPU, 8ch x 4096pts
 *   3. MMA(N=10) GPU vs CPU
 *   4. DEMA(N=10) GPU vs CPU
 *   5. TEMA(N=10) GPU vs CPU
 *   6. Step response demo (120pts, 5 filter types)
 *
 * NOT RUN on Windows - compile-only check.
 * Will be tested on Linux + AMD GPU (Radeon 9070 / MI100).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM
#include "filters/moving_average_filter_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#endif

namespace test_moving_average_rocm {

#if ENABLE_ROCM

using namespace filters;
using namespace drv_gpu_lib;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate random complex signal
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> generate_random_signal(
    uint32_t channels, uint32_t points, unsigned int seed = 42)
{
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> signal(total);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < total; ++i) {
    signal[i] = {dist(rng), dist(rng)};
  }
  return signal;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: compare GPU vs CPU result
// ═══════════════════════════════════════════════════════════════════════════

inline float compare_gpu_cpu(
    const std::vector<std::complex<float>>& gpu_data,
    const std::vector<std::complex<float>>& cpu_data,
    size_t total)
{
  float max_error = 0.0f;
  for (size_t i = 0; i < total; ++i) {
    float err_re = std::abs(gpu_data[i].real() - cpu_data[i].real());
    float err_im = std::abs(gpu_data[i].imag() - cpu_data[i].imag());
    max_error = std::max(max_error, std::max(err_re, err_im));
  }
  return max_error;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test helper: run single MA type GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_ma_type(ConsoleOutput& con, int gpu_id,
                          const char* test_name, MAType type,
                          uint32_t window_size, uint32_t channels,
                          uint32_t points, float tolerance)
{
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    MovingAverageFilterROCm filter(&backend);
    filter.SetParams(type, window_size);

    size_t total = static_cast<size_t>(channels) * points;
    auto signal = generate_random_signal(channels, points);

    // CPU reference
    auto cpu_result = filter.ProcessCpu(signal, channels, points);

    // GPU
    auto gpu_result = filter.ProcessFromCPU(signal, channels, points);

    // Readback
    std::vector<std::complex<float>> gpu_data(total);
    hipError_t err = hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                                     total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    if (err != hipSuccess) {
      con.Print(gpu_id, "MA[ROCm]",
                std::string("[X] ") + test_name + ": hipMemcpyDtoH failed");
      return false;
    }

    float max_error = compare_gpu_cpu(gpu_data, cpu_result, total);

    bool passed = (max_error < tolerance);
    std::ostringstream oss;
    oss << test_name << ": max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED");
    con.Print(gpu_id, "MA[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "MA[ROCm]",
              std::string("[X] ") + test_name + " EXCEPTION: " + e.what());
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: EMA(N=10), 8ch x 4096pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_ema(ConsoleOutput& con, int gpu_id) {
  return test_ma_type(con, gpu_id, "Test 1 (EMA N=10 8ch)",
                       MAType::EMA, 10, 8, 4096, 1e-4f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: SMA(N=8), 8ch x 4096pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_sma(ConsoleOutput& con, int gpu_id) {
  return test_ma_type(con, gpu_id, "Test 2 (SMA N=8 8ch)",
                       MAType::SMA, 8, 8, 4096, 1e-4f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: MMA(N=10), 8ch x 4096pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_mma(ConsoleOutput& con, int gpu_id) {
  return test_ma_type(con, gpu_id, "Test 3 (MMA N=10 8ch)",
                       MAType::MMA, 10, 8, 4096, 1e-4f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: DEMA(N=10), 8ch x 4096pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_dema(ConsoleOutput& con, int gpu_id) {
  return test_ma_type(con, gpu_id, "Test 4 (DEMA N=10 8ch)",
                       MAType::DEMA, 10, 8, 4096, 1e-4f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: TEMA(N=10), 8ch x 4096pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_tema(ConsoleOutput& con, int gpu_id) {
  return test_ma_type(con, gpu_id, "Test 5 (TEMA N=10 8ch)",
                       MAType::TEMA, 10, 8, 4096, 1e-4f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Step response demo — 120pts step signal
//   20 zeros | 50 ones | 50 zeros  (like Examples/MA_python.md)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_step_response(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    const uint32_t channels = 1;
    const uint32_t points   = 120;
    const uint32_t N        = 10;

    // Step signal: 20 zeros, 50 ones, 50 zeros
    std::vector<std::complex<float>> sig(points, {0.0f, 0.0f});
    for (uint32_t i = 20; i < 70; ++i)
      sig[i] = {1.0f, 0.0f};

    MovingAverageFilterROCm filter(&backend);

    // Compute all 5 filter types via CPU
    struct FilterResult {
      const char* name;
      MAType      type;
      std::vector<std::complex<float>> out;
    };

    std::vector<FilterResult> results = {
      {"SMA",  MAType::SMA,  {}},
      {"EMA",  MAType::EMA,  {}},
      {"MMA",  MAType::MMA,  {}},
      {"DEMA", MAType::DEMA, {}},
      {"TEMA", MAType::TEMA, {}},
    };

    for (auto& r : results) {
      filter.SetParams(r.type, N);
      r.out = filter.ProcessCpu(sig, channels, points);
    }

    // Print table header
    con.Print(gpu_id, "MA[ROCm]", "");
    con.Print(gpu_id, "MA[ROCm]",
        "--- [MA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---");
    con.Print(gpu_id, "MA[ROCm]",
        "  t  | input | SMA(10) | EMA(10) | MMA(10) | DEMA(10)| TEMA(10)");
    con.Print(gpu_id, "MA[ROCm]",
        " ----+-------+---------+---------+---------+---------+---------");

    // Print every 5th sample
    char buf[160];
    for (uint32_t t = 0; t < points; t += 5) {
      std::snprintf(buf, sizeof(buf),
          " %3u |  %.1f  |  %5.3f  |  %5.3f  |  %5.3f  |  %5.3f  |  %5.3f",
          t,
          sig[t].real(),
          results[0].out[t].real(),
          results[1].out[t].real(),
          results[2].out[t].real(),
          results[3].out[t].real(),
          results[4].out[t].real()
      );
      con.Print(gpu_id, "MA[ROCm]", std::string(buf));
    }

    // Verify: all filters reach ~1.0 at mid-plateau (t=55)
    bool passed = true;
    for (auto& r : results) {
      float val_mid = r.out[55].real();
      if (std::abs(val_mid - 1.0f) > 0.05f) {
        std::snprintf(buf, sizeof(buf),
            "[X] %s(10) mid-plateau t=55: %.4f (expected ~1.0)",
            r.name, val_mid);
        con.Print(gpu_id, "MA[ROCm]", std::string(buf));
        passed = false;
      }
    }

    // TEMA should react faster than EMA at the front
    if (results[4].out[23].real() <= results[1].out[23].real()) {
      con.Print(gpu_id, "MA[ROCm]",
                "[X] TEMA should lead EMA at t=23 (front edge)");
      passed = false;
    }

    // GPU vs CPU check (EMA)
    filter.SetParams(MAType::EMA, N);
    auto gpu_result = filter.ProcessFromCPU(sig, channels, points);
    std::vector<std::complex<float>> gpu_data(points);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    points * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_err = compare_gpu_cpu(gpu_data, results[1].out, points);
    if (max_err > 1e-4f) {
      std::snprintf(buf, sizeof(buf),
          "[X] EMA step GPU vs CPU max_err = %.2e (>1e-4)", max_err);
      con.Print(gpu_id, "MA[ROCm]", std::string(buf));
      passed = false;
    }

    con.Print(gpu_id, "MA[ROCm]",
              std::string("Test 6 (Step demo): ") +
              (passed ? "PASSED" : "FAILED"));
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "MA[ROCm]",
              "[X] Step demo EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "MA[ROCm]", "");
  con.Print(gpu_id, "MA[ROCm]", "============================================");
  con.Print(gpu_id, "MA[ROCm]", "  Moving Average Filter ROCm Tests (HIP)");
  con.Print(gpu_id, "MA[ROCm]", "  SMA / EMA / MMA / DEMA / TEMA");
  con.Print(gpu_id, "MA[ROCm]", "============================================");

  int passed = 0;
  int total = 6;

  if (test_ema(con, gpu_id))            ++passed;
  if (test_sma(con, gpu_id))            ++passed;
  if (test_mma(con, gpu_id))            ++passed;
  if (test_dema(con, gpu_id))           ++passed;
  if (test_tema(con, gpu_id))           ++passed;
  if (test_step_response(con, gpu_id))  ++passed;

  con.Print(gpu_id, "MA[ROCm]",
            "Results: " + std::to_string(passed) + "/" +
            std::to_string(total) + " passed");
  con.Print(gpu_id, "MA[ROCm]", "============================================");
  con.Print(gpu_id, "MA[ROCm]", "");
}

#else  // !ENABLE_ROCM

inline void run() {
  std::cout << "\n[test_moving_average_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_moving_average_rocm
