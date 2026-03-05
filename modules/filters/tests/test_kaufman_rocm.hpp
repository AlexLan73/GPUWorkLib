#pragma once

/**
 * @file test_kaufman_rocm.hpp
 * @brief ROCm tests for KaufmanFilterROCm (KAMA)
 *
 * Tests:
 *   1. GPU vs CPU (8ch x 4096pts, random signal)
 *   2. Trend signal — ER~1, fast tracking
 *   3. Noise signal — ER~0, slow/frozen KAMA
 *   4. Adaptive transition (trend -> noise -> step)
 *   5. Step + trend-noise-trend demo (120pts, table output)
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
#include "filters/kaufman_filter_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#endif

namespace test_kaufman_rocm {

#if ENABLE_ROCM

using namespace filters;
using namespace drv_gpu_lib;

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: GPU vs CPU — random complex signal
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_gpu_vs_cpu(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    KaufmanFilterROCm kauf(&backend);
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 8;
    const uint32_t points   = 4096;
    const size_t   total    = static_cast<size_t>(channels) * points;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> signal(total);
    for (size_t i = 0; i < total; ++i)
      signal[i] = {dist(rng), dist(rng)};

    auto cpu_result = kauf.ProcessCpu(signal, channels, points);
    auto gpu_result = kauf.ProcessFromCPU(signal, channels, points);

    std::vector<std::complex<float>> gpu_data(total);
    hipError_t err = hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                                     total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    if (err != hipSuccess) {
      con.Print(gpu_id, "KAMA[ROCm]", "[X] Test 1: hipMemcpyDtoH failed");
      return false;
    }

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-4f);
    std::ostringstream oss;
    oss << "Test 1 (GPU vs CPU 8ch): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED");
    con.Print(gpu_id, "KAMA[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "KAMA[ROCm]",
              "[X] Test 1 EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Trend signal — ER~1, fast tracking
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_trend(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    KaufmanFilterROCm kauf(&backend);
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 256;

    // Linear trend: x[n] = n * 0.1
    std::vector<std::complex<float>> signal(points);
    for (uint32_t n = 0; n < points; ++n)
      signal[n] = {static_cast<float>(n) * 0.1f, 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // After warmup (n > 20), KAMA should track trend closely
    float max_lag = 0.0f;
    for (uint32_t n = 20; n < points; ++n) {
      float lag = std::abs(result[n].real() - signal[n].real());
      max_lag = std::max(max_lag, lag);
    }

    bool passed = (max_lag < 2.0f);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "Test 2 (trend tracking): max_lag = %.4f %s",
        max_lag, passed ? "PASSED" : "FAILED (>2.0)");
    con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "KAMA[ROCm]",
              "[X] Test 2 EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Noise signal — ER~0, KAMA almost frozen
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_noise(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    KaufmanFilterROCm kauf(&backend);
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 512;

    std::mt19937 rng(99);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<std::complex<float>> signal(points);
    for (uint32_t n = 0; n < points; ++n)
      signal[n] = {noise(rng), 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // After warmup: std(KAMA) should be much smaller than std(signal)
    float mean_sig = 0.0f, mean_kama = 0.0f;
    uint32_t count = 0;
    for (uint32_t n = 100; n < points; ++n) {
      mean_sig  += signal[n].real();
      mean_kama += result[n].real();
      ++count;
    }
    mean_sig  /= count;
    mean_kama /= count;

    float var_sig = 0.0f, var_kama = 0.0f;
    for (uint32_t n = 100; n < points; ++n) {
      float ds = signal[n].real() - mean_sig;
      float dk = result[n].real() - mean_kama;
      var_sig  += ds * ds;
      var_kama += dk * dk;
    }
    float std_sig  = std::sqrt(var_sig / count);
    float std_kama = std::sqrt(var_kama / count);

    float ratio = std_kama / (std_sig + 1e-10f);
    bool passed = (ratio < 0.2f);

    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "Test 3 (noise suppression): std_sig=%.3f std_kama=%.3f ratio=%.3f %s",
        std_sig, std_kama, ratio,
        passed ? "PASSED" : "FAILED (ratio>0.2)");
    con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "KAMA[ROCm]",
              "[X] Test 3 EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Adaptive transition — trend -> noise -> trend
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_adaptive_transition(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    KaufmanFilterROCm kauf(&backend);
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 2048;

    std::mt19937 rng(555);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<std::complex<float>> signal(points);

    // Phase 1 [0..511]: linear trend (slope=0.01)
    for (uint32_t n = 0; n < 512; ++n)
      signal[n] = {static_cast<float>(n) * 0.01f, 0.0f};

    // Phase 2 [512..1023]: white noise
    for (uint32_t n = 512; n < 1024; ++n)
      signal[n] = {noise(rng), 0.0f};

    // Phase 3 [1024..2047]: linear trend continuing
    for (uint32_t n = 1024; n < 2048; ++n)
      signal[n] = {static_cast<float>(n - 1024) * 0.01f + 5.0f, 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // Check Phase 1 (end): KAMA tracks trend
    float val_500 = result[500].real();
    float exp_500 = 500.0f * 0.01f;
    bool track_ok = (std::abs(val_500 - exp_500) < 0.5f);

    // Check Phase 2: KAMA is stable (small variation during noise)
    float delta_noise = std::abs(result[1020].real() - result[520].real());
    bool noise_ok = (delta_noise < 3.0f);

    // Check Phase 3 (end): KAMA recovers tracking
    float val_2040 = result[2040].real();
    float exp_2040 = (2040.0f - 1024.0f) * 0.01f + 5.0f;
    bool recover_ok = (std::abs(val_2040 - exp_2040) < 1.0f);

    bool passed = track_ok && noise_ok && recover_ok;
    char buf[160];
    std::snprintf(buf, sizeof(buf),
        "Test 4 (adaptive): trend_err=%.3f noise_delta=%.3f recover_err=%.3f %s",
        std::abs(val_500 - exp_500), delta_noise,
        std::abs(val_2040 - exp_2040),
        passed ? "PASSED" : "FAILED");
    con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "KAMA[ROCm]",
              "[X] Test 4 EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Step + trend-noise-trend demo (120pts, table output)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_step_kama_demo(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    const uint32_t channels = 1;
    const uint32_t points   = 120;

    // ── Signal A: clean step (20 zeros / 50 ones / 50 zeros) ──
    std::vector<std::complex<float>> sig_step(points, {0.0f, 0.0f});
    for (uint32_t i = 20; i < 70; ++i)
      sig_step[i] = {1.0f, 0.0f};

    KaufmanFilterROCm kauf(&backend);
    kauf.SetParams(10, 2, 30);

    auto out_step = kauf.ProcessCpu(sig_step, channels, points);

    // ── Print step table ──
    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "--- [KAMA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  t  | input | KAMA(10) | note");
    con.Print(gpu_id, "KAMA[ROCm]",
        " ----+-------+----------+-----------------");

    char buf[128];
    uint32_t sample_t[] = {0,5,10,15,20,22,25,28,30,35,40,50,65,
                           70,72,75,80,90,110,119};
    for (uint32_t i = 0; i < sizeof(sample_t)/sizeof(uint32_t); ++i) {
      uint32_t t = sample_t[i];
      const char* note = "";
      if (t == 20) note = "<-- step up";
      if (t == 70) note = "<-- step down";
      std::snprintf(buf, sizeof(buf), " %3u |  %.1f  |  %6.4f  | %s",
          t, sig_step[t].real(), out_step[t].real(), note);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    }

    // ── Signal B: trend-noise-step (adaptivity demo) ──
    std::vector<std::complex<float>> sig_tnt(120, {0.0f, 0.0f});
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    // Phase 1 [0..39]: clean trend 0->1
    for (uint32_t n = 0; n < 40; ++n)
      sig_tnt[n] = {static_cast<float>(n) * 0.025f, 0.0f};

    // Phase 2 [40..79]: white noise
    for (uint32_t n = 40; n < 80; ++n)
      sig_tnt[n] = {noise(rng), 0.0f};

    // Phase 3 [80..119]: step = 1.0 + small noise
    for (uint32_t n = 80; n < 120; ++n)
      sig_tnt[n] = {1.0f + noise(rng) * 0.05f, 0.0f};

    auto out_tnt = kauf.ProcessCpu(sig_tnt, 1, 120);

    // ── Print adaptive table ──
    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "--- [KAMA Demo] Trend->Noise->Step (adaptivity) ---");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  Phase        | Signal   | ER    | KAMA behavior");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [0..39]  trend| 0->1     | ~1.0  | fast tracking");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [40..79] noise| sigma=0.2| ~0.0  | almost frozen");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [80..119] step| 1.0+eps  | ~1.0  | fast recovery");

    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  t  | input  | KAMA    | phase");
    con.Print(gpu_id, "KAMA[ROCm]",
        " ----+--------+---------+------");

    uint32_t demo_t[] = {0,10,20,30,39,40,50,60,70,79,80,85,90,100,115,119};
    for (uint32_t i = 0; i < sizeof(demo_t)/sizeof(uint32_t); ++i) {
      uint32_t t = demo_t[i];
      const char* phase = (t < 40) ? "trend" :
                           (t < 80) ? "noise" : "step ";
      std::snprintf(buf, sizeof(buf), " %3u | %6.3f | %6.3f  | %s",
          t, sig_tnt[t].real(), out_tnt[t].real(), phase);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    }

    // ── Checks ──
    bool passed = true;

    // Step plateau: KAMA should reach ~1.0 at t=55
    if (std::abs(out_step[55].real() - 1.0f) > 0.01f) {
      con.Print(gpu_id, "KAMA[ROCm]",
          "[X] step plateau t=55 not ~1.0");
      passed = false;
    }

    // Trend tracking at t=38
    if (std::abs(out_tnt[38].real() - 0.950f) > 0.1f) {
      con.Print(gpu_id, "KAMA[ROCm]",
          "[X] trend tracking at t=38 failed");
      passed = false;
    }

    // Noise stability: KAMA variation within noise phase (t=55..79) should be small
    float delta_noise = std::abs(out_tnt[79].real() - out_tnt[55].real());
    if (delta_noise > 0.5f) {
      std::snprintf(buf, sizeof(buf),
          "[X] noise stability: delta=%.3f (>0.5)", delta_noise);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
      passed = false;
    }

    // GPU vs CPU (step signal)
    auto gpu_result = kauf.ProcessFromCPU(sig_step, channels, points);
    std::vector<std::complex<float>> gpu_data(points);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    points * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_err = 0.0f;
    for (uint32_t i = 0; i < points; ++i) {
      float e = std::abs(gpu_data[i].real() - out_step[i].real());
      max_err = std::max(max_err, e);
    }
    if (max_err > 1e-4f) {
      std::snprintf(buf, sizeof(buf),
          "[X] GPU vs CPU step max_err=%.2e (>1e-4)", max_err);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
      passed = false;
    }

    con.Print(gpu_id, "KAMA[ROCm]",
        std::string("Test 5 (step+adaptive demo): ") +
        (passed ? "PASSED" : "FAILED"));
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "KAMA[ROCm]",
              "[X] Test 5 EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "KAMA[ROCm]", "");
  con.Print(gpu_id, "KAMA[ROCm]", "============================================");
  con.Print(gpu_id, "KAMA[ROCm]", "  Kaufman (KAMA) Filter ROCm Tests (HIP)");
  con.Print(gpu_id, "KAMA[ROCm]", "  Adaptive MA: ER->SC->alpha");
  con.Print(gpu_id, "KAMA[ROCm]", "============================================");

  int passed = 0;
  int total = 5;

  if (test_gpu_vs_cpu(con, gpu_id))              ++passed;
  if (test_trend(con, gpu_id))                   ++passed;
  if (test_noise(con, gpu_id))                   ++passed;
  if (test_adaptive_transition(con, gpu_id))     ++passed;
  if (test_step_kama_demo(con, gpu_id))          ++passed;

  con.Print(gpu_id, "KAMA[ROCm]",
            "Results: " + std::to_string(passed) + "/" +
            std::to_string(total) + " passed");
  con.Print(gpu_id, "KAMA[ROCm]", "============================================");
  con.Print(gpu_id, "KAMA[ROCm]", "");
}

#else  // !ENABLE_ROCM

inline void run() {
  std::cout << "\n[test_kaufman_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_kaufman_rocm
