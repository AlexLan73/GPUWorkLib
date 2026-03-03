#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_basic.hpp
 * @brief Functional tests for FM Correlator
 *
 * Tests:
 *   1. Autocorrelation:  ref vs ref, SNR > 10
 *   2. Basic pipeline:   N=1024, K=4, S=2, n_kg=100
 *   3. Full pipeline:    N=32768, K=32, S=5, n_kg=2000
 *   4. Shift pattern:    CPU circshift vs GPU generate_test_inputs
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include "fm_correlator.hpp"
#include "services/console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fm_correlator::tests {

// ════════════════════════════════════════════════════════════════════════════
// Helper: create ROCm backend for tests
// ════════════════════════════════════════════════════════════════════════════

inline drv_gpu_lib::IBackend* GetTestBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  if (!backend.IsInitialized()) {
    backend.Initialize(0);
  }
  return &backend;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Autocorrelation (most important — correctness verification)
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_autocorrelation() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "Test: Autocorrelation (SNR > 10)");

  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size = 4096;
  params.num_shifts = 1;
  params.num_signals = 1;
  params.num_output_points = 200;
  corr.SetParams(params);

  auto ref = corr.GenerateMSequence();
  corr.PrepareReference(ref);

  auto result = corr.Process(ref);

  float peak = result.at(0, 0, 0);
  float max_noise = 0.0f;
  for (int j = 1; j < params.num_output_points; ++j) {
    float v = result.at(0, 0, j);
    if (v > max_noise) max_noise = v;
  }

  float snr = (max_noise > 0.0f) ? (peak / max_noise) : 1e6f;

  char buf[128];
  std::snprintf(buf, sizeof(buf),
      "  peak=%.4f noise=%.6f SNR=%.1f", peak, max_noise, snr);
  con.Print(gpu_id, "FM_Corr", buf);

  if (snr < 10.0f) {
    throw std::runtime_error("Autocorrelation: SNR=" + std::to_string(snr) +
        " < 10, expected > 10");
  }

  con.Print(gpu_id, "FM_Corr", "Test Autocorrelation: PASSED ✅ (SNR=" +
      std::to_string(static_cast<int>(snr)) + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Basic pipeline (small data)
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_basic_pipeline() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "Test: Basic pipeline (N=1024, K=4, S=2)");

  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size = 1024;
  params.num_shifts = 4;
  params.num_signals = 2;
  params.num_output_points = 100;
  corr.SetParams(params);

  corr.PrepareReference();

  auto ref = corr.GenerateMSequence();
  std::vector<float> inp(params.num_signals * params.fft_size);
  for (int s = 0; s < params.num_signals; ++s) {
    for (size_t i = 0; i < params.fft_size; ++i) {
      inp[s * params.fft_size + i] = ref[(i + s * 2) % params.fft_size];
    }
  }

  auto result = corr.Process(inp);

  size_t expected_size = static_cast<size_t>(params.num_signals) *
                         params.num_shifts * params.num_output_points;
  if (result.peaks.size() != expected_size) {
    throw std::runtime_error("Basic pipeline: peaks size " +
        std::to_string(result.peaks.size()) + " != expected " +
        std::to_string(expected_size));
  }

  con.Print(gpu_id, "FM_Corr",
      "  Result: " + std::to_string(result.peaks.size()) + " floats ✅");
  con.Print(gpu_id, "FM_Corr", "Test Basic pipeline: PASSED ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Full pipeline (default params)
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_full_pipeline() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "Test: Full pipeline (N=32768, K=32, S=5)");

  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  // Default: fft_size=32768, num_shifts=32, num_signals=5, n_kg=2000
  corr.SetParams(params);

  corr.PrepareReference();
  auto result = corr.RunTestPattern(2);

  size_t expected_size = static_cast<size_t>(params.num_signals) *
                         params.num_shifts * params.num_output_points;
  if (result.peaks.size() != expected_size) {
    throw std::runtime_error("Full pipeline: peaks size " +
        std::to_string(result.peaks.size()) + " != expected " +
        std::to_string(expected_size));
  }

  con.Print(gpu_id, "FM_Corr",
      "  Result: " + std::to_string(result.peaks.size()) + " floats (" +
      std::to_string(params.num_signals) + "x" +
      std::to_string(params.num_shifts) + "x" +
      std::to_string(params.num_output_points) + ") ✅");
  con.Print(gpu_id, "FM_Corr", "Test Full pipeline: PASSED ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Shift pattern — deterministic peak position verification
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_shift_pattern() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "Test: Shift pattern (CPU vs GPU)");

  auto* backend = GetTestBackend();

  const int N = 4096;
  const int K = 10;
  const int S = 5;
  const int n_kg = 200;
  const int shift_step = 2;

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size = N;
  params.num_shifts = K;
  params.num_signals = S;
  params.num_output_points = n_kg;

  drv_gpu_lib::FMCorrelator corr(backend);
  corr.SetParams(params);

  auto ref = corr.GenerateMSequence();
  corr.PrepareReference(ref);

  // CPU-generated test inputs
  std::vector<float> inp(S * N);
  for (int s = 0; s < S; ++s) {
    for (int i = 0; i < N; ++i) {
      inp[s * N + i] = ref[(i + s * shift_step) % N];
    }
  }
  auto result_cpu = corr.Process(inp);

  // GPU-generated test inputs
  corr.PrepareReference(ref);
  auto result_gpu = corr.RunTestPattern(shift_step);

  // Verify peak positions
  int verified = 0;
  int skipped = 0;
  for (int s = 0; s < S; ++s) {
    for (int k = 0; k < K; ++k) {
      int expected_pos = ((k - s * shift_step) % N + N) % N;
      if (expected_pos < n_kg) {
        float max_val = 0.0f;
        int max_pos = 0;
        for (int j = 0; j < n_kg; ++j) {
          float v = result_gpu.at(s, k, j);
          if (v > max_val) { max_val = v; max_pos = j; }
        }
        if (max_pos != expected_pos) {
          throw std::runtime_error(
              "Shift pattern: s=" + std::to_string(s) + " k=" + std::to_string(k) +
              " expected peak at " + std::to_string(expected_pos) +
              " got " + std::to_string(max_pos));
        }
        ++verified;
      } else {
        ++skipped;
      }
    }
  }

  con.Print(gpu_id, "FM_Corr",
      "  Peak positions verified: " + std::to_string(verified) +
      " (skipped " + std::to_string(skipped) + " out-of-range)");

  // Compare CPU vs GPU results
  float max_diff = 0.0f;
  for (size_t i = 0; i < result_cpu.peaks.size(); ++i) {
    float d = std::fabs(result_cpu.peaks[i] - result_gpu.peaks[i]);
    if (d > max_diff) max_diff = d;
  }

  char buf[128];
  std::snprintf(buf, sizeof(buf), "  CPU vs GPU max diff = %.2e", max_diff);
  con.Print(gpu_id, "FM_Corr", buf);

  if (max_diff > 1e-4f) {
    throw std::runtime_error("Shift pattern: CPU vs GPU diff=" +
        std::to_string(max_diff) + " > 1e-4");
  }

  con.Print(gpu_id, "FM_Corr", "Test Shift pattern: PASSED ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
