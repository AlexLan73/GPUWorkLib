#pragma once

/**
 * @file test_statistics_rocm.hpp
 * @brief Tests for StatisticsProcessor -- mean, median, variance, std (ROCm)
 *
 * ✅ MIGRATED to test_utils (2026-03-21, CppTest-06 etalon)
 *
 * Tests:
 *  1. mean_single_beam        -- sinusoid, complex mean ≈ 0
 *  2. mean_multi_beam         -- 4 beams, per-beam means
 *  3. welford_statistics      -- mean_mag, variance, std vs CPU
 *  4. median_linear           -- sorted magnitudes, verify median
 *  5. gpu_input               -- ComputeStatistics(void*)
 *  6. mean_constant           -- constant signal (mean = constant)
 *  7. benchmark_median        -- GPU vs CPU sort timing
 *  8. histogram_median_basic  -- 200K points, histogram path
 *  9. histogram_median_multi  -- 4×500K, compare with CPU sort
 * 10. histogram_median_float  -- ComputeMedianFloat path
 * 11. histogram_vs_radix      -- timing benchmark
 * 12. compute_all_cpu         -- ComputeAll vs separate calls
 * 13. compute_all_gpu         -- ComputeAll(void*) path
 * 14. compute_all_float       -- ComputeAllFloat path
 * 15. compute_all_edge_cases  -- boundary conditions
 *
 * IMPORTANT: Compiles ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (migrated 2026-03-21)
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"

// test_utils — единая тестовая инфраструктура
#include "modules/test_utils/test_utils.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

namespace test_statistics_rocm {

using namespace statistics;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// =========================================================================
// run() — TestRunner (функциональный стиль)
// =========================================================================

inline void run() {
  int gpu_id = 0;

  // Check for ROCm devices
  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    auto& con = ConsoleOutput::GetInstance();
    con.Print(gpu_id, "Stats ROCm", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  StatisticsProcessor stats(&backend);

  TestRunner runner(&backend, "Stats ROCm", gpu_id);

  // ── Test 1: ComputeMean — single beam, sinusoid ──────────────────

  runner.test("mean_single_beam", [&]() -> TestResult {
    auto data = refs::GenerateSinusoid(100.f, 1000.f, 4096);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = 4096;

    auto results = stats.ComputeMean(data, params);
    if (results.empty()) return TestResult{"mean_single_beam"}.add(FailResult("size", 0, 1));

    auto cpu_mean = refs::CpuMean(data.data(), data.size());

    TestResult tr{"mean_single_beam"};
    tr.add(ScalarAbsError(results[0].mean.real(), cpu_mean.real(),
                           tolerance::kStatistics, "mean_re"));
    tr.add(ScalarAbsError(results[0].mean.imag(), cpu_mean.imag(),
                           tolerance::kStatistics, "mean_im"));
    return tr;
  });

  // ── Test 2: ComputeMean — multi-beam ─────────────────────────────

  runner.test("mean_multi_beam", [&]() -> TestResult {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 2048;

    auto data = refs::GenerateMultiBeam(beam_count, n_point, 1000.f, 50.f, 1.0f, 0.5f);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto results = stats.ComputeMean(data, params);

    TestResult tr{"mean_multi_beam"};
    for (uint32_t b = 0; b < beam_count; ++b) {
      auto cpu_mean = refs::CpuMean(data.data() + b * n_point, n_point);
      tr.add(ScalarAbsError(results[b].mean.real(), cpu_mean.real(),
                             tolerance::kStatistics, "beam" + std::to_string(b) + "_re"));
      tr.add(ScalarAbsError(results[b].mean.imag(), cpu_mean.imag(),
                             tolerance::kStatistics, "beam" + std::to_string(b) + "_im"));
    }
    return tr;
  });

  // ── Test 3: ComputeStatistics — Welford ──────────────────────────

  runner.test("welford_statistics", [&]() -> TestResult {
    auto data = refs::GenerateSinusoid(100.f, 1000.f, 4096, 2.0f);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = 4096;

    auto results = stats.ComputeStatistics(data, params);

    float cpu_mean_mag = refs::CpuMeanMagnitude(data.data(), data.size());
    float cpu_variance = refs::CpuVarianceMagnitude(data.data(), data.size());
    float cpu_std      = refs::CpuStdMagnitude(data.data(), data.size());

    TestResult tr{"welford_statistics"};
    tr.add(ScalarAbsError(results[0].mean_magnitude, cpu_mean_mag, 1e-2, "mean_mag"));
    tr.add(ScalarAbsError(results[0].variance, cpu_variance, 1e-2, "variance"));
    tr.add(ScalarAbsError(results[0].std_dev, cpu_std, 1e-2, "std_dev"));
    return tr;
  });

  // ── Test 4: ComputeMedian — linear magnitudes ────────────────────

  runner.test("median_linear", [&]() {
    const uint32_t n_point = 1024;
    std::vector<std::complex<float>> data(n_point);
    for (uint32_t i = 0; i < n_point; ++i)
      data[i] = std::complex<float>(static_cast<float>(i + 1), 0.0f);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeMedian(data, params);
    float cpu_median = refs::CpuMedianMagnitude(data.data(), n_point);

    return ScalarAbsError(results[0].median_magnitude, cpu_median, 1.0, "median");
  });

  // ── Test 5: ComputeStatistics — GPU input (void*) ────────────────

  runner.test("gpu_input", [&]() {
    auto data = refs::GenerateSinusoid(200.f, 1000.f, 2048);

    size_t data_size = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(data_size);
    backend.MemcpyHostToDevice(gpu_data, data.data(), data_size);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = 2048;

    auto results = stats.ComputeStatistics(gpu_data, params);
    backend.Free(gpu_data);

    float cpu_mean_mag = refs::CpuMeanMagnitude(data.data(), data.size());
    return ScalarAbsError(results[0].mean_magnitude, cpu_mean_mag, 1e-2, "mean_mag_gpu");
  });

  // ── Test 6: ComputeMean — constant signal ────────────────────────

  runner.test("mean_constant", [&]() -> TestResult {
    std::complex<float> constant_value(3.14f, -2.71f);
    auto data = refs::GenerateConstant(constant_value, 4096);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = 4096;

    auto results = stats.ComputeMean(data, params);

    TestResult tr{"mean_constant"};
    tr.add(ScalarAbsError(results[0].mean.real(), constant_value.real(), 1e-4, "const_re"));
    tr.add(ScalarAbsError(results[0].mean.imag(), constant_value.imag(), 1e-4, "const_im"));
    return tr;
  });

  // ── Test 7: Benchmark — ComputeMedian GPU vs CPU sort ────────────
  // Benchmark: оставляем ручной формат (timing + speedup вывод)

  runner.test("benchmark_median", [&]() {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), 0.0f);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i)
        mags[i] = std::abs(data[b * n_point + i]);
      std::sort(mags.begin(), mags.end());
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - cpu_start).count();

    // GPU warmup + timing
    {
      std::vector<std::complex<float>> warm(beam_count * 1024, {1.0f, 0.0f});
      StatisticsParams wp; wp.beam_count = beam_count; wp.n_point = 1024;
      stats.ComputeMedian(warm, wp);
    }
    auto gpu_start = std::chrono::high_resolution_clock::now();
    stats.ComputeMedian(data, params);
    double gpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - gpu_start).count();

    double speedup = cpu_ms / gpu_ms;
    return PassResult("speedup", speedup, 1.0,
                      "CPU=" + std::to_string(cpu_ms) + "ms GPU=" + std::to_string(gpu_ms) + "ms");
  });

  // ── Test 8: Histogram median — basic (200K, linear) ──────────────

  runner.test("histogram_median_basic", [&]() {
    const uint32_t n_point = 200000;
    std::vector<std::complex<float>> data(n_point);
    for (uint32_t i = 0; i < n_point; ++i)
      data[i] = std::complex<float>(static_cast<float>(i + 1), 0.0f);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeMedian(data, params);
    float expected = static_cast<float>(n_point / 2 + 1);
    return ScalarAbsError(results[0].median_magnitude, expected, 1.0, "hist_median");
  });

  // ── Test 9: Histogram median — multi-beam (4×500K) ───────────────

  runner.test("histogram_median_multi", [&]() -> TestResult {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.1f, 1000.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), 0.0f);

    // CPU reference
    std::vector<float> cpu_medians(beam_count);
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i)
        mags[i] = std::abs(data[b * n_point + i]);
      std::sort(mags.begin(), mags.end());
      cpu_medians[b] = mags[n_point / 2];
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto results = stats.ComputeMedian(data, params);

    TestResult tr{"histogram_median_multi"};
    for (uint32_t b = 0; b < beam_count; ++b)
      tr.add(ScalarAbsError(results[b].median_magnitude, cpu_medians[b],
                             0.01, "beam" + std::to_string(b)));
    return tr;
  });

  // ── Test 10: Histogram median float (2×200K) ─────────────────────

  runner.test("histogram_median_float", [&]() -> TestResult {
    const uint32_t beam_count = 2;
    const uint32_t n_point = 200000;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(0.0f, 500.0f);
    std::vector<float> magnitudes(beam_count * n_point);
    for (auto& v : magnitudes) v = dist(rng);

    // CPU reference
    std::vector<float> cpu_medians(beam_count);
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> sorted_mags(
          magnitudes.begin() + b * n_point,
          magnitudes.begin() + (b + 1) * n_point);
      std::sort(sorted_mags.begin(), sorted_mags.end());
      cpu_medians[b] = sorted_mags[n_point / 2];
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto results = stats.ComputeMedianFloat(magnitudes, params);

    TestResult tr{"histogram_median_float"};
    for (uint32_t b = 0; b < beam_count; ++b)
      tr.add(ScalarAbsError(results[b].median_magnitude, cpu_medians[b],
                             0.01, "float_beam" + std::to_string(b)));
    return tr;
  });

  // ── Test 11: Histogram vs Radix benchmark ────────────────────────

  runner.test("histogram_vs_radix_benchmark", [&]() {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), 0.0f);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    // Warmup
    {
      std::vector<std::complex<float>> warm(beam_count * 1024, {1.0f, 0.0f});
      StatisticsParams wp; wp.beam_count = beam_count; wp.n_point = 1024;
      stats.ComputeMedian(warm, wp);
    }

    auto start_hist = std::chrono::high_resolution_clock::now();
    stats.ComputeMedian(data, params);
    double hist_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start_hist).count();

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i)
        mags[i] = std::abs(data[b * n_point + i]);
      std::sort(mags.begin(), mags.end());
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start_cpu).count();

    double speedup = cpu_ms / hist_ms;
    return PassResult("hist_speedup", speedup, 1.0,
                      "CPU=" + std::to_string(cpu_ms) + "ms Hist=" + std::to_string(hist_ms) + "ms");
  });

  // ── Test 12: ComputeAll — CPU data, verify all fields ────────────

  runner.test("compute_all_cpu", [&]() -> TestResult {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 65536;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto ref_stats   = stats.ComputeStatistics(data, params);
    auto ref_medians = stats.ComputeMedian(data, params);
    auto full        = stats.ComputeAll(data, params);

    TestResult tr{"compute_all_cpu"};
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::string pfx = "b" + std::to_string(b) + "_";
      tr.add_all({
        ScalarAbsError(full[b].mean.real(),       ref_stats[b].mean.real(),       1e-5, pfx + "mean_re"),
        ScalarAbsError(full[b].mean.imag(),       ref_stats[b].mean.imag(),       1e-5, pfx + "mean_im"),
        ScalarAbsError(full[b].variance,          ref_stats[b].variance,          1e-5, pfx + "var"),
        ScalarAbsError(full[b].std_dev,           ref_stats[b].std_dev,           1e-5, pfx + "std"),
        ScalarAbsError(full[b].mean_magnitude,    ref_stats[b].mean_magnitude,    1e-5, pfx + "mean_mag"),
        ScalarAbsError(full[b].median_magnitude,  ref_medians[b].median_magnitude,1e-5, pfx + "median"),
      });
    }
    return tr;
  });

  // ── Test 13: ComputeAll — GPU data (void*) ───────────────────────

  runner.test("compute_all_gpu", [&]() -> TestResult {
    const uint32_t beam_count = 2;
    const uint32_t n_point = 32768;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto ref_stats   = stats.ComputeStatistics(data, params);
    auto ref_medians = stats.ComputeMedian(data, params);

    size_t bytes = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(bytes);
    backend.MemcpyHostToDevice(gpu_data, data.data(), bytes);

    auto full = stats.ComputeAll(gpu_data, params);
    backend.Free(gpu_data);

    TestResult tr{"compute_all_gpu"};
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::string pfx = "b" + std::to_string(b) + "_";
      tr.add_all({
        ScalarAbsError(full[b].mean.real(),       ref_stats[b].mean.real(),       1e-5, pfx + "mean_re"),
        ScalarAbsError(full[b].mean.imag(),       ref_stats[b].mean.imag(),       1e-5, pfx + "mean_im"),
        ScalarAbsError(full[b].variance,          ref_stats[b].variance,          1e-5, pfx + "var"),
        ScalarAbsError(full[b].mean_magnitude,    ref_stats[b].mean_magnitude,    1e-5, pfx + "mean_mag"),
        ScalarAbsError(full[b].median_magnitude,  ref_medians[b].median_magnitude,1e-5, pfx + "median"),
      });
    }
    return tr;
  });

  // ── Test 14: ComputeAllFloat ─────────────────────────────────────

  runner.test("compute_all_float", [&]() -> TestResult {
    const uint32_t beam_count = 2;
    const uint32_t n_point = 16384;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    std::vector<float> magnitudes(beam_count * n_point);
    for (auto& v : magnitudes) v = dist(rng);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto ref_stats   = stats.ComputeStatisticsFloat(magnitudes, params);
    auto ref_medians = stats.ComputeMedianFloat(magnitudes, params);
    auto full        = stats.ComputeAllFloat(magnitudes, params);

    TestResult tr{"compute_all_float"};
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::string pfx = "b" + std::to_string(b) + "_";
      // Float path: mean.real()==0, mean.imag()==0
      tr.add(ScalarAbsError(full[b].mean.real(), 0.0f, 1e-8, pfx + "mean_re_zero"));
      tr.add(ScalarAbsError(full[b].mean.imag(), 0.0f, 1e-8, pfx + "mean_im_zero"));
      tr.add(ScalarAbsError(full[b].mean_magnitude, ref_stats[b].mean_magnitude, 1e-5, pfx + "mean_mag"));
      tr.add(ScalarAbsError(full[b].variance, ref_stats[b].variance, 1e-5, pfx + "var"));
      tr.add(ScalarAbsError(full[b].median_magnitude, ref_medians[b].median_magnitude, 1e-5, pfx + "median"));
    }
    return tr;
  });

  // ── Test 15: ComputeAll — edge cases ─────────────────────────────

  runner.test("compute_all_edge_cases", [&]() -> TestResult {
    TestResult tr{"compute_all_edge_cases"};

    // Case A: beam_count=1, n_point=100 (tiny → radix sort path)
    {
      std::mt19937 rng(1);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<std::complex<float>> data(100);
      for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

      StatisticsParams p; p.beam_count = 1; p.n_point = 100;
      auto ref_s = stats.ComputeStatistics(data, p);
      auto ref_m = stats.ComputeMedian(data, p);
      auto full  = stats.ComputeAll(data, p);

      tr.add(ScalarAbsError(full[0].mean_magnitude, ref_s[0].mean_magnitude, 1e-5, "caseA_mean_mag"));
      tr.add(ScalarAbsError(full[0].median_magnitude, ref_m[0].median_magnitude, 1e-5, "caseA_median"));
    }

    // Case B: beam_count=4, n_point=100000 (threshold boundary)
    {
      std::mt19937 rng(2);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<std::complex<float>> data(4 * 100000);
      for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

      StatisticsParams p; p.beam_count = 4; p.n_point = 100000;
      auto ref_s = stats.ComputeStatistics(data, p);
      auto ref_m = stats.ComputeMedian(data, p);
      auto full  = stats.ComputeAll(data, p);

      for (uint32_t b = 0; b < 4; ++b) {
        std::string pfx = "caseB_b" + std::to_string(b) + "_";
        tr.add(ScalarAbsError(full[b].mean_magnitude, ref_s[b].mean_magnitude, 1e-5, pfx + "mean"));
        tr.add(ScalarAbsError(full[b].median_magnitude, ref_m[b].median_magnitude, 1e-5, pfx + "median"));
      }
    }
    return tr;
  });

  // ── Summary ──────────────────────────────────────────────────────

  runner.print_summary();

  // JSON export (Results/JSON/)
  runner.export_json("Results/JSON/test_statistics_rocm.json");
}

}  // namespace test_statistics_rocm

#endif  // ENABLE_ROCM
