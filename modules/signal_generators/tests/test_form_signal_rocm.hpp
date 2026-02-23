#pragma once

/**
 * @file test_form_signal_rocm.hpp
 * @brief ROCm tests for FormSignalGeneratorROCm
 *
 * Tests:
 *   1. No noise, 1 channel — GPU vs CPU reference (getX)
 *   2. Window test — zeros outside [0, ti-dt] (tau shift)
 *   3. Multi-channel (8 antennas, TAU_STEP) — GPU vs CPU
 *   4. Noise statistics (an=1.0) — mean ~0, variance ~1
 *   5. Chirp (fdev != 0) — GPU vs CPU
 *   6. ProcessFromCPU convenience (GenerateToCpu round-trip)
 *
 * NOT RUN on Windows — compile-only check.
 * Will be tested on Linux + AMD GPU (Radeon 9070 / MI100).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_form_signal_rocm {

#if ENABLE_ROCM

#include "generators/form_signal_generator_rocm.hpp"
#include "params/form_params.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

using namespace signal_gen;
using namespace drv_gpu_lib;

// ═══════════════════════════════════════════════════════════════════════════
// CPU reference (getX formula, no noise)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GetXReference(
    double fs, uint32_t points, double f0, double amplitude,
    double phase, double fdev, double norm_val, double tau) {

  double dt = 1.0 / fs;
  double ti = static_cast<double>(points) * dt;
  std::vector<std::complex<float>> out(points);

  for (uint32_t i = 0; i < points; ++i) {
    double t = static_cast<double>(i) * dt + tau;

    if (t < 0.0 || t > ti - dt) {
      out[i] = {0.0f, 0.0f};
      continue;
    }

    double t_centered = t - ti * 0.5;
    double ph = 2.0 * M_PI * f0 * t
              + M_PI * fdev / ti * (t_centered * t_centered)
              + phase;

    float re = static_cast<float>(amplitude * norm_val * std::cos(ph));
    float im = static_cast<float>(amplitude * norm_val * std::sin(ph));
    out[i] = {re, im};
  }

  return out;
}

inline float MaxError(const std::complex<float>* a,
                      const std::complex<float>* b, size_t n) {
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = std::abs(a[i] - b[i]);
    if (d > max_err) max_err = d;
  }
  return max_err;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: No noise, 1 channel — GPU vs CPU reference
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_no_noise(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

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

    auto cpu_ref = GetXReference(
        p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, 0.0);

    float max_err = MaxError(
        gpu_data[0].data(), cpu_ref.data(), p.points);

    bool passed = max_err < 1e-3f;
    std::ostringstream oss;
    oss << "Test 1 (No noise 1ch): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] No noise EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Window test — zeros outside [0, ti-dt]
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_window(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    FormParams p;
    p.fs = 1000.0;
    p.antennas = 1;
    p.points = 1000;
    p.f0 = 100.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.tau_base = -0.1;  // negative delay -> first 100 samples X=0

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);

    auto data = gen.GenerateToCpu();

    int zero_count = 0;
    for (int i = 0; i < 100; ++i) {
      if (std::abs(data[0][i]) < 1e-6f) zero_count++;
    }

    int nonzero_count = 0;
    for (int i = 110; i < 500; ++i) {
      if (std::abs(data[0][i]) > 0.01f) nonzero_count++;
    }

    bool passed = (zero_count >= 99) && (nonzero_count > 350);
    std::ostringstream oss;
    oss << "Test 2 (Window tau=-0.1): zeros=" << zero_count << "/100, "
        << "nonzeros=" << nonzero_count << "/390"
        << (passed ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] Window EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Multi-channel (8 antennas, TAU_STEP)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_multi_channel(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    FormParams p;
    p.fs = 10000.0;
    p.antennas = 8;
    p.points = 2048;
    p.f0 = 500.0;
    p.amplitude = 1.0;
    p.noise_amplitude = 0.0;
    p.tau_base = 0.0;
    p.tau_step = 0.01;  // 10 ms step

    FormSignalGeneratorROCm gen(&backend);
    gen.SetParams(p);

    auto data = gen.GenerateToCpu();

    bool all_pass = true;
    float worst_err = 0.0f;

    for (uint32_t a = 0; a < p.antennas; ++a) {
      double tau = p.tau_base + a * p.tau_step;
      auto cpu_ref = GetXReference(
          p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, tau);

      float err = MaxError(data[a].data(), cpu_ref.data(), p.points);
      worst_err = std::max(worst_err, err);
      if (err >= 1e-3f) all_pass = false;
    }

    std::ostringstream oss;
    oss << "Test 3 (8ch TAU_STEP): worst_err = " << std::scientific
        << std::setprecision(2) << worst_err
        << (all_pass ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return all_pass;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] MultiChannel EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Noise statistics (an=1.0, amplitude=0)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_noise(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

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

    // Mean
    double sum_re = 0, sum_im = 0;
    for (auto& d : data[0]) { sum_re += d.real(); sum_im += d.imag(); }
    double mean_re = sum_re / p.points;
    double mean_im = sum_im / p.points;

    // Variance
    double var_re = 0, var_im = 0;
    for (auto& d : data[0]) {
      var_re += (d.real() - mean_re) * (d.real() - mean_re);
      var_im += (d.imag() - mean_im) * (d.imag() - mean_im);
    }
    var_re /= p.points;
    var_im /= p.points;

    bool mean_ok = (std::abs(mean_re) < 0.05) && (std::abs(mean_im) < 0.05);
    bool var_ok = (std::abs(var_re - 1.0) < 0.1) && (std::abs(var_im - 1.0) < 0.1);
    bool passed = mean_ok && var_ok;

    std::ostringstream oss;
    oss << "Test 4 (Noise N=100k): mean=(" << std::fixed << std::setprecision(4)
        << mean_re << "," << mean_im << ") var=("
        << var_re << "," << var_im << ")"
        << (passed ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] Noise EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Chirp (fdev != 0)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_chirp(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

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

    auto cpu_ref = GetXReference(
        p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, 0.0);

    float max_err = MaxError(gpu_data[0].data(), cpu_ref.data(), p.points);
    bool passed = max_err < 1e-3f;

    std::ostringstream oss;
    oss << "Test 5 (Chirp fdev=5k): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] Chirp EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: GenerateInputData + readback (void* GPU pointer round-trip)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_gpu_ptr(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

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

    // Use GenerateInputData (returns void* on GPU)
    auto result = gen.GenerateInputData();

    // Verify metadata
    if (result.antenna_count != p.antennas ||
        result.n_point != p.points ||
        result.data == nullptr) {
      con.Print(gpu_id, "FormSignal[ROCm]",
                "[X] Test 6 (GPU ptr): metadata mismatch FAILED");
      if (result.data) hipFree(result.data);
      return false;
    }

    // Readback and compare with CPU
    size_t total = gen.GetTotalSamples();
    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), result.data,
                    total * sizeof(std::complex<float>));
    hipFree(result.data);

    // Compare each channel
    float worst_err = 0.0f;
    for (uint32_t a = 0; a < p.antennas; ++a) {
      auto cpu_ref = GetXReference(
          p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, 0.0);
      size_t offset = static_cast<size_t>(a) * p.points;
      float err = MaxError(&gpu_data[offset], cpu_ref.data(), p.points);
      worst_err = std::max(worst_err, err);
    }

    bool passed = worst_err < 1e-3f;
    std::ostringstream oss;
    oss << "Test 6 (GPU ptr 4ch): worst_err = " << std::scientific
        << std::setprecision(2) << worst_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "FormSignal[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "FormSignal[ROCm]",
              "[X] GPU ptr EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "FormSignal[ROCm]", "");
  con.Print(gpu_id, "FormSignal[ROCm]", "============================================");
  con.Print(gpu_id, "FormSignal[ROCm]", "  FormSignalGenerator ROCm Tests (HIP)");
  con.Print(gpu_id, "FormSignal[ROCm]", "============================================");

  int passed = 0;
  int total = 6;

  if (test_no_noise(con, gpu_id))       ++passed;
  if (test_window(con, gpu_id))         ++passed;
  if (test_multi_channel(con, gpu_id))  ++passed;
  if (test_noise(con, gpu_id))          ++passed;
  if (test_chirp(con, gpu_id))          ++passed;
  if (test_gpu_ptr(con, gpu_id))        ++passed;

  con.Print(gpu_id, "FormSignal[ROCm]",
            "Results: " + std::to_string(passed) + "/" +
            std::to_string(total) + " passed");
  con.Print(gpu_id, "FormSignal[ROCm]", "============================================");
  con.Print(gpu_id, "FormSignal[ROCm]", "");
}

#else  // !ENABLE_ROCM

inline void run() {
  std::cout << "\n[test_form_signal_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_form_signal_rocm
