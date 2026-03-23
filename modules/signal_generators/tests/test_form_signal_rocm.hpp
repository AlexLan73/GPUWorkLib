#pragma once

/**
 * @file test_form_signal_rocm.hpp
 * @brief ROCm tests for FormSignalGeneratorROCm
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. no_noise        — 1 channel, GPU vs CPU reference (getX)
 *   2. window          — zeros outside [0, ti-dt] with tau=-0.1
 *   3. multi_channel   — 8 antennas, TAU_STEP=0.01, GPU vs CPU
 *   4. noise           — mean ~0, variance ~1 (N=100k)
 *   5. chirp           — fdev=5000, GPU vs CPU
 *   6. gpu_ptr         — GenerateInputData() void* round-trip
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (migrated 2026-03-23)
 */

#if ENABLE_ROCM

#include "generators/form_signal_generator_rocm.hpp"
#include "params/form_params.hpp"
#include "backends/rocm/rocm_backend.hpp"

// test_utils — единая тестовая инфраструктура
#include "modules/test_utils/test_utils.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_form_signal_rocm {

using namespace signal_gen;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// =========================================================================
// run() — TestRunner (функциональный стиль)
// =========================================================================

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "FormSignal ROCm", gpu_id);

  // ── Test 1: No noise, 1 channel — GPU vs CPU reference ────────

  runner.test("no_noise", [&]() {
    FormParams p;
    p.fs = 12e6;
    p.antennas = 1;
    p.points = 4096;
    p.f0 = 1e6;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.phase = 0.3;
    p.fdev = 0.0;
    p.norm = 1.0 / std::sqrt(2.0);

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto gpu_data = gen.GenerateToCpu();

    auto cpu_ref = refs::GenerateFormSignal(
        static_cast<float>(p.fs), p.points, static_cast<float>(p.f0),
        static_cast<float>(p.amplitude), static_cast<float>(p.phase),
        static_cast<float>(p.fdev), static_cast<float>(p.norm), 0.0f);

    return AbsError(gpu_data[0].data(), cpu_ref.data(), p.points,
                    tolerance::kComplex32, "max_err");
  });

  // ── Test 2: Window — zeros outside [0, ti-dt] ─────────────────

  runner.test("window", [&]() -> TestResult {
    FormParams p;
    p.fs = 1000.0;
    p.antennas = 1;
    p.points = 1000;
    p.f0 = 100.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.tau_base = -0.1;

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto data = gen.GenerateToCpu();

    int zero_count = 0;
    for (int i = 0; i < 100; ++i)
      if (std::abs(data[0][i]) < 1e-6f) zero_count++;

    int nonzero_count = 0;
    for (int i = 110; i < 500; ++i)
      if (std::abs(data[0][i]) > 0.01f) nonzero_count++;

    TestResult tr{"window"};
    tr.add(ValidationResult{
        zero_count >= 99, "zeros_first_100",
        static_cast<double>(zero_count), 99.0,
        "zeros in [0..99]: " + std::to_string(zero_count) + "/100"});
    tr.add(ValidationResult{
        nonzero_count > 350, "nonzeros_mid",
        static_cast<double>(nonzero_count), 350.0,
        "nonzeros in [110..500]: " + std::to_string(nonzero_count) + "/390"});
    return tr;
  });

  // ── Test 3: Multi-channel (8 antennas, TAU_STEP) ──────────────

  runner.test("multi_channel", [&]() -> TestResult {
    FormParams p;
    p.fs = 10000.0;
    p.antennas = 8;
    p.points = 2048;
    p.f0 = 500.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.tau_base = 0.0;
    p.tau_step = 0.01;

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto data = gen.GenerateToCpu();

    TestResult tr{"multi_channel"};
    for (uint32_t a = 0; a < p.antennas; ++a) {
      double tau_d = p.tau_base + a * p.tau_step;
      float tau = static_cast<float>(tau_d);
      auto cpu_ref = refs::GenerateFormSignal(
          static_cast<float>(p.fs), p.points, static_cast<float>(p.f0),
          static_cast<float>(p.amplitude), static_cast<float>(p.phase),
          static_cast<float>(p.fdev), static_cast<float>(p.norm), tau);

      // Skip boundary samples where float32 vs double64 window disagrees
      float max_err = 0.0f;
      for (uint32_t i = 0; i < p.points; ++i) {
        // Skip if either side is zero (window boundary)
        if (std::abs(cpu_ref[i]) < 1e-6f && std::abs(data[a][i]) < 1e-6f)
          continue;
        if (std::abs(cpu_ref[i]) < 1e-6f || std::abs(data[a][i]) < 1e-6f) {
          // One is zero, other is not — boundary sample, skip
          continue;
        }
        float d = std::abs(data[a][i] - cpu_ref[i]);
        max_err = std::max(max_err, d);
      }
      tr.add(ValidationResult{
          max_err < static_cast<float>(tolerance::kComplex32),
          "ant" + std::to_string(a),
          static_cast<double>(max_err), tolerance::kComplex32,
          ""});
    }
    return tr;
  });

  // ── Test 4: Noise statistics (an=1.0, amplitude=0) ────────────

  runner.test("noise", [&]() -> TestResult {
    FormParams p;
    p.fs = 10000.0;
    p.antennas = 1;
    p.points = 100000;
    p.f0 = 0.0;
    p.amplitude = 0.0;
    p.noise_amplitude = 1.0;
    p.norm = 1.0;
    p.noise_seed = 42;

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto data = gen.GenerateToCpu();

    double sum_re = 0, sum_im = 0;
    double var_re = 0, var_im = 0;

    for (auto& d : data[0]) { sum_re += d.real(); sum_im += d.imag(); }
    double mean_re = sum_re / p.points;
    double mean_im = sum_im / p.points;

    for (auto& d : data[0]) {
      var_re += (d.real() - mean_re) * (d.real() - mean_re);
      var_im += (d.imag() - mean_im) * (d.imag() - mean_im);
    }
    var_re /= p.points;
    var_im /= p.points;

    TestResult tr{"noise"};
    tr.add(ScalarAbsError(mean_re, 0.0, 0.05, "mean_re"));
    tr.add(ScalarAbsError(mean_im, 0.0, 0.05, "mean_im"));
    tr.add(ScalarAbsError(var_re, 1.0, 0.1, "var_re"));
    tr.add(ScalarAbsError(var_im, 1.0, 0.1, "var_im"));
    return tr;
  });

  // ── Test 5: Chirp (fdev=5000) ─────────────────────────────────

  runner.test("chirp", [&]() {
    FormParams p;
    p.fs = 100000.0;
    p.antennas = 1;
    p.points = 4096;
    p.f0 = 1000.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.fdev = 5000.0;
    p.norm = 1.0 / std::sqrt(2.0);

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);
    auto gpu_data = gen.GenerateToCpu();

    auto cpu_ref = refs::GenerateFormSignal(
        static_cast<float>(p.fs), p.points, static_cast<float>(p.f0),
        static_cast<float>(p.amplitude), static_cast<float>(p.phase),
        static_cast<float>(p.fdev), static_cast<float>(p.norm), 0.0f);

    return AbsError(gpu_data[0].data(), cpu_ref.data(), p.points,
                    tolerance::kComplex32, "chirp_max_err");
  });

  // ── Test 6: GPU pointer round-trip (GenerateInputData) ────────

  runner.test("gpu_ptr", [&]() -> TestResult {
    FormParams p;
    p.fs = 10000.0;
    p.antennas = 4;
    p.points = 2048;
    p.f0 = 500.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.norm = 1.0 / std::sqrt(2.0);

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);

    auto result = gen.GenerateInputData();

    TestResult tr{"gpu_ptr"};
    tr.add(ValidationResult{
        result.antenna_count == p.antennas && result.n_point == p.points
            && result.data != nullptr,
        "metadata", 1.0, 1.0,
        "antennas=" + std::to_string(result.antenna_count)
        + " points=" + std::to_string(result.n_point)});

    size_t total = gen.GetTotalSamples();
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), result.data, total);

    float worst_err = 0.0f;
    for (uint32_t a = 0; a < p.antennas; ++a) {
      auto cpu_ref = refs::GenerateFormSignal(
          static_cast<float>(p.fs), p.points, static_cast<float>(p.f0),
          static_cast<float>(p.amplitude), static_cast<float>(p.phase),
          static_cast<float>(p.fdev), static_cast<float>(p.norm), 0.0f);

      size_t offset = static_cast<size_t>(a) * p.points;
      for (size_t i = 0; i < p.points; ++i) {
        float d = std::abs(gpu_data[offset + i] - cpu_ref[i]);
        worst_err = std::max(worst_err, d);
      }
    }

    tr.add(ValidationResult{
        worst_err < static_cast<float>(tolerance::kComplex32),
        "gpu_ptr_err",
        static_cast<double>(worst_err), tolerance::kComplex32,
        "worst_err=" + std::to_string(worst_err)});
    return tr;
  });

  runner.print_summary();
}

}  // namespace test_form_signal_rocm

#else  // !ENABLE_ROCM

namespace test_form_signal_rocm {
inline void run() {}
}

#endif  // ENABLE_ROCM
