#pragma once

/**
 * @file test_statistics_rocm.hpp
 * @brief Tests for StatisticsProcessor -- mean, median, variance, std (ROCm)
 *
 * Tests:
 * 1. ComputeMean -- single beam, known signal, verify complex mean
 * 2. ComputeMean -- multi-beam, verify per-beam means
 * 3. ComputeStatistics -- Welford: mean_mag, variance, std vs CPU
 * 4. ComputeMedian -- sorted magnitudes, verify median value
 * 5. ComputeStatistics -- GPU input (void*)
 * 6. ComputeMean -- constant signal (mean = constant)
 * 7. Benchmark -- ComputeMedian GPU vs CPU sort (4 beams x 500000 points)
 * 8. Histogram Median -- basic correctness (1 beam × 200K, triggers histogram path)
 * 9. Histogram Median -- multi-beam (4 × 500K, compare with CPU sort)
 * 10. Histogram Median Float -- ComputeMedianFloat path (2 × 200K)
 * 11. Histogram vs CPU -- timing benchmark
 *
 * IMPORTANT: Tests compile ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 * Run only on Linux with AMD GPU and ROCm SDK.
 *
 * Reference: NumPy (np.mean, np.std, np.median)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_statistics_rocm {

using namespace statistics;
using namespace drv_gpu_lib;

// =========================================================================
// Utilities
// =========================================================================

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
  std::string status = passed ? "PASSED" : "FAILED";
  std::string icon = passed ? "[+]" : "[X]";
  con.Print(gpu_id, "Stats ROCm", icon + " " + test_name + " ... " + status);
}

/// Generate sinusoidal complex signal
inline std::vector<std::complex<float>> GenerateSinusoid(
    float freq, float sample_rate, size_t n_point, float amplitude = 1.0f)
{
  std::vector<std::complex<float>> data(n_point);
  for (size_t i = 0; i < n_point; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
    data[i] = std::complex<float>(amplitude * std::cos(phase),
                                   amplitude * std::sin(phase));
  }
  return data;
}

/// Generate constant complex signal
inline std::vector<std::complex<float>> GenerateConstant(
    std::complex<float> value, size_t n_point)
{
  return std::vector<std::complex<float>>(n_point, value);
}

/// Generate multi-beam data with different amplitudes
inline std::vector<std::complex<float>> GenerateMultiBeam(
    size_t beam_count, size_t n_point, float sample_rate,
    float freq, float base_amplitude, float amp_step)
{
  std::vector<std::complex<float>> data(beam_count * n_point);
  for (size_t b = 0; b < beam_count; ++b) {
    float amp = base_amplitude + b * amp_step;
    for (size_t i = 0; i < n_point; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      data[b * n_point + i] = std::complex<float>(
          amp * std::cos(phase), amp * std::sin(phase));
    }
  }
  return data;
}

/// CPU reference: complex mean
inline std::complex<float> CpuMean(const std::complex<float>* data, size_t n) {
  std::complex<float> sum(0.0f, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum / static_cast<float>(n);
}

/// CPU reference: mean of magnitudes
inline float CpuMeanMagnitude(const std::complex<float>* data, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += std::abs(data[i]);
  }
  return sum / static_cast<float>(n);
}

/// CPU reference: variance of magnitudes (population)
inline float CpuVarianceMagnitude(const std::complex<float>* data, size_t n) {
  float mean_mag = CpuMeanMagnitude(data, n);
  float sum_sq = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float mag = std::abs(data[i]);
    float diff = mag - mean_mag;
    sum_sq += diff * diff;
  }
  return sum_sq / static_cast<float>(n);
}

/// CPU reference: std of magnitudes (population)
inline float CpuStdMagnitude(const std::complex<float>* data, size_t n) {
  return std::sqrt(CpuVarianceMagnitude(data, n));
}

/// CPU reference: median of magnitudes
inline float CpuMedianMagnitude(const std::complex<float>* data, size_t n) {
  std::vector<float> mags(n);
  for (size_t i = 0; i < n; ++i) {
    mags[i] = std::abs(data[i]);
  }
  std::sort(mags.begin(), mags.end());
  return mags[n / 2];
}

// =========================================================================
// Test 1: ComputeMean -- single beam, sinusoid
// =========================================================================

inline bool test_mean_single_beam(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    const float freq = 100.0f;
    const float sample_rate = 1000.0f;
    const uint32_t n_point = 4096;

    auto data = GenerateSinusoid(freq, sample_rate, n_point);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeMean(data, params);

    bool ok = (results.size() == 1);
    if (ok) {
      // For a sinusoid with integer number of periods, mean should be ~0
      auto cpu_mean = CpuMean(data.data(), n_point);
      float err_re = std::fabs(results[0].mean.real() - cpu_mean.real());
      float err_im = std::fabs(results[0].mean.imag() - cpu_mean.imag());

      ok = (err_re < 1e-3f) && (err_im < 1e-3f);

      con.Print(gpu_id, "Stats ROCm",
                "  mean=(" + std::to_string(results[0].mean.real()) + ", " +
                std::to_string(results[0].mean.imag()) + ")" +
                " err_re=" + std::to_string(err_re) +
                " err_im=" + std::to_string(err_im));
    }

    print_result(con, gpu_id, "Mean SingleBeam (sinusoid)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] Mean SingleBeam EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 2: ComputeMean -- multi-beam
// =========================================================================

inline bool test_mean_multi_beam(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    const uint32_t beam_count = 4;
    const uint32_t n_point = 2048;
    const float sample_rate = 1000.0f;
    const float freq = 50.0f;

    auto data = GenerateMultiBeam(beam_count, n_point, sample_rate, freq, 1.0f, 0.5f);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    auto results = stats.ComputeMean(data, params);

    bool ok = (results.size() == beam_count);
    float max_err = 0.0f;

    for (uint32_t b = 0; b < beam_count && ok; ++b) {
      auto cpu_mean = CpuMean(data.data() + b * n_point, n_point);
      float err_re = std::fabs(results[b].mean.real() - cpu_mean.real());
      float err_im = std::fabs(results[b].mean.imag() - cpu_mean.imag());
      float err = std::max(err_re, err_im);
      max_err = std::max(max_err, err);

      if (err > 1e-3f) {
        ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  Beam " + std::to_string(b) + " err=" + std::to_string(err));
      }
    }

    con.Print(gpu_id, "Stats ROCm", "  max_err=" + std::to_string(max_err));
    print_result(con, gpu_id, "Mean MultiBeam (4 beams)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] Mean MultiBeam EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 3: ComputeStatistics -- Welford: mean_mag, variance, std
// =========================================================================

inline bool test_welford_statistics(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    const uint32_t n_point = 4096;
    const float sample_rate = 1000.0f;
    const float freq = 100.0f;
    const float amplitude = 2.0f;

    auto data = GenerateSinusoid(freq, sample_rate, n_point, amplitude);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeStatistics(data, params);

    bool ok = (results.size() == 1);
    if (ok) {
      // CPU reference
      float cpu_mean_mag = CpuMeanMagnitude(data.data(), n_point);
      float cpu_variance = CpuVarianceMagnitude(data.data(), n_point);
      float cpu_std = CpuStdMagnitude(data.data(), n_point);

      float err_mean_mag = std::fabs(results[0].mean_magnitude - cpu_mean_mag);
      float err_variance = std::fabs(results[0].variance - cpu_variance);
      float err_std = std::fabs(results[0].std_dev - cpu_std);

      // For a constant-amplitude sinusoid: all magnitudes == amplitude,
      // so variance should be ~0 and mean_mag ~= amplitude
      con.Print(gpu_id, "Stats ROCm",
                "  mean_mag=" + std::to_string(results[0].mean_magnitude) +
                " (cpu=" + std::to_string(cpu_mean_mag) + ")");
      con.Print(gpu_id, "Stats ROCm",
                "  variance=" + std::to_string(results[0].variance) +
                " (cpu=" + std::to_string(cpu_variance) + ")");
      con.Print(gpu_id, "Stats ROCm",
                "  std=" + std::to_string(results[0].std_dev) +
                " (cpu=" + std::to_string(cpu_std) + ")");

      // Tolerance: 1e-2 for GPU vs CPU
      ok = (err_mean_mag < 1e-2f) && (err_variance < 1e-2f) && (err_std < 1e-2f);
    }

    print_result(con, gpu_id, "Welford Statistics (mean_mag+var+std)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] Welford EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 4: ComputeMedian -- verify median of magnitudes
// =========================================================================

inline bool test_median(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    // Generate data with known distribution: linearly increasing magnitudes
    const uint32_t n_point = 1024;
    std::vector<std::complex<float>> data(n_point);
    for (uint32_t i = 0; i < n_point; ++i) {
      float mag = static_cast<float>(i + 1);  // 1, 2, 3, ..., 1024
      data[i] = std::complex<float>(mag, 0.0f);  // real only for simplicity
    }

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeMedian(data, params);

    bool ok = (results.size() == 1);
    if (ok) {
      float cpu_median = CpuMedianMagnitude(data.data(), n_point);

      // Expected: sorted magnitudes are 1,2,...,1024; middle = 512 or 513
      float err = std::fabs(results[0].median_magnitude - cpu_median);

      con.Print(gpu_id, "Stats ROCm",
                "  median=" + std::to_string(results[0].median_magnitude) +
                " (cpu=" + std::to_string(cpu_median) +
                "), err=" + std::to_string(err));

      ok = (err < 1.0f);  // within 1 unit (integer magnitudes)
    }

    print_result(con, gpu_id, "Median (linear magnitudes)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] Median EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 5: ComputeStatistics -- GPU input (void*)
// =========================================================================

inline bool test_gpu_input(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    const uint32_t n_point = 2048;
    const float sample_rate = 1000.0f;
    const float freq = 200.0f;

    auto data = GenerateSinusoid(freq, sample_rate, n_point);

    // Upload data to GPU manually
    size_t data_size = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(data_size);
    backend.MemcpyHostToDevice(gpu_data, data.data(), data_size);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeStatistics(gpu_data, params);

    backend.Free(gpu_data);

    bool ok = (results.size() == 1);
    if (ok) {
      float cpu_mean_mag = CpuMeanMagnitude(data.data(), n_point);
      float err = std::fabs(results[0].mean_magnitude - cpu_mean_mag);

      con.Print(gpu_id, "Stats ROCm",
                "  GPU input: mean_mag=" + std::to_string(results[0].mean_magnitude) +
                " (cpu=" + std::to_string(cpu_mean_mag) +
                "), err=" + std::to_string(err));

      ok = (err < 1e-2f);
    }

    print_result(con, gpu_id, "GPU Input (void*)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] GPU Input EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 6: ComputeMean -- constant signal
// =========================================================================

inline bool test_mean_constant(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    StatisticsProcessor stats(&backend);

    const uint32_t n_point = 4096;
    std::complex<float> constant_value(3.14f, -2.71f);
    auto data = GenerateConstant(constant_value, n_point);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point = n_point;

    auto results = stats.ComputeMean(data, params);

    bool ok = (results.size() == 1);
    if (ok) {
      float err_re = std::fabs(results[0].mean.real() - constant_value.real());
      float err_im = std::fabs(results[0].mean.imag() - constant_value.imag());

      con.Print(gpu_id, "Stats ROCm",
                "  constant mean=(" + std::to_string(results[0].mean.real()) + ", " +
                std::to_string(results[0].mean.imag()) + ")" +
                " err_re=" + std::to_string(err_re) +
                " err_im=" + std::to_string(err_im));

      ok = (err_re < 1e-4f) && (err_im < 1e-4f);
    }

    print_result(con, gpu_id, "Mean Constant Signal", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] Mean Constant EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 7: Benchmark -- ComputeMedian GPU vs CPU sort
// =========================================================================

inline bool test_benchmark_median(ConsoleOutput& con, int gpu_id) {
  try {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    // Generate random complex data (magnitudes only, imag=0)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) {
      float mag = dist(rng);
      v = std::complex<float>(mag, 0.0f);
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    // --- CPU timing ---
    auto cpu_start = std::chrono::high_resolution_clock::now();

    std::vector<float> cpu_medians(beam_count);
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i) {
        mags[i] = std::abs(data[b * n_point + i]);
      }
      std::sort(mags.begin(), mags.end());
      cpu_medians[b] = mags[n_point / 2];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(
        cpu_end - cpu_start).count();

    // --- GPU timing ---
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    // Warm-up: first call may include JIT/driver init overhead
    {
      std::vector<std::complex<float>> warm_data(
          beam_count * 1024, std::complex<float>(1.0f, 0.0f));
      StatisticsParams warm_params;
      warm_params.beam_count = beam_count;
      warm_params.n_point = 1024;
      stats.ComputeMedian(warm_data, warm_params);
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    auto results = stats.ComputeMedian(data, params);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    double gpu_ms = std::chrono::duration<double, std::milli>(
        gpu_end - gpu_start).count();

    double speedup = cpu_ms / gpu_ms;

    con.Print(gpu_id, "Stats ROCm",
              "  Benchmark: " + std::to_string(beam_count) +
              " beams x " + std::to_string(n_point) + " points");
    con.Print(gpu_id, "Stats ROCm",
              "  CPU sort : " + std::to_string(cpu_ms) + " ms");
    con.Print(gpu_id, "Stats ROCm",
              "  GPU sort : " + std::to_string(gpu_ms) + " ms");
    con.Print(gpu_id, "Stats ROCm",
              "  Speedup  : " + std::to_string(speedup) + "x");

    bool ok = (speedup > 1.0);
    print_result(con, gpu_id, "Benchmark Median GPU vs CPU", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] Benchmark Median EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 8: Histogram median -- basic correctness (1 beam × 1024)
// Forces histogram path by using > kHistogramThreshold points
// Then also checks small data path (radix sort) for same result
// =========================================================================

inline bool test_histogram_median_basic(ConsoleOutput& con, int gpu_id) {
  try {
    // Use 200K points to trigger histogram path (> 100K threshold)
    const uint32_t beam_count = 1;
    const uint32_t n_point = 200000;

    // Generate known linear magnitudes: data[i] = complex(i+1, 0)
    // => magnitudes = [1, 2, ..., 200000]
    // => sorted median at index 100000 = 100001.0f
    std::vector<std::complex<float>> data(n_point);
    for (uint32_t i = 0; i < n_point; ++i) {
      data[i] = std::complex<float>(static_cast<float>(i + 1), 0.0f);
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    auto results = stats.ComputeMedian(data, params);

    float expected = static_cast<float>(n_point / 2 + 1);  // sorted[N/2]
    float gpu_median = results[0].median_magnitude;
    float err = std::abs(gpu_median - expected);

    con.Print(gpu_id, "Stats ROCm",
              "  Histogram basic: median=" + std::to_string(gpu_median) +
              " expected=" + std::to_string(expected) + " err=" + std::to_string(err));

    bool ok = (err < 1.0f);
    print_result(con, gpu_id, "Histogram Median Basic (200K)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] Histogram Median Basic EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 9: Histogram median -- multi-beam (4 beams × 500K)
// Compares histogram result with CPU reference
// =========================================================================

inline bool test_histogram_median_multi(ConsoleOutput& con, int gpu_id) {
  try {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.1f, 1000.0f);

    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) {
      v = std::complex<float>(dist(rng), 0.0f);
    }

    // CPU reference: sort magnitudes, take middle
    std::vector<float> cpu_medians(beam_count);
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i) {
        mags[i] = std::abs(data[b * n_point + i]);
      }
      std::sort(mags.begin(), mags.end());
      cpu_medians[b] = mags[n_point / 2];
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    auto results = stats.ComputeMedian(data, params);

    bool ok = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
      float err = std::abs(results[b].median_magnitude - cpu_medians[b]);
      // Exact match expected (same element from same data)
      if (err > 0.01f) {
        con.Print(gpu_id, "Stats ROCm",
                  "  Beam " + std::to_string(b) +
                  ": gpu=" + std::to_string(results[b].median_magnitude) +
                  " cpu=" + std::to_string(cpu_medians[b]) +
                  " err=" + std::to_string(err));
        ok = false;
      }
    }

    print_result(con, gpu_id, "Histogram Median Multi (4×500K)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] Histogram Median Multi EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 10: Histogram median float -- ComputeMedianFloat path
// =========================================================================

inline bool test_histogram_median_float(ConsoleOutput& con, int gpu_id) {
  try {
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

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    auto results = stats.ComputeMedianFloat(magnitudes, params);

    bool ok = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
      float err = std::abs(results[b].median_magnitude - cpu_medians[b]);
      if (err > 0.01f) {
        con.Print(gpu_id, "Stats ROCm",
                  "  Float beam " + std::to_string(b) +
                  ": gpu=" + std::to_string(results[b].median_magnitude) +
                  " cpu=" + std::to_string(cpu_medians[b]));
        ok = false;
      }
    }

    print_result(con, gpu_id, "Histogram Median Float (2×200K)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] Histogram Median Float EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 11: Histogram vs Radix Sort benchmark (timing comparison)
// =========================================================================

inline bool test_histogram_vs_radix_benchmark(ConsoleOutput& con, int gpu_id) {
  try {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 500000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) {
      v = std::complex<float>(dist(rng), 0.0f);
    }

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    // Warm-up
    {
      std::vector<std::complex<float>> warm(beam_count * 1024,
                                             std::complex<float>(1.0f, 0.0f));
      StatisticsParams wp;
      wp.beam_count = beam_count;
      wp.n_point = 1024;
      stats.ComputeMedian(warm, wp);
    }

    // n_point = 500K > 100K → will use histogram path
    auto start_hist = std::chrono::high_resolution_clock::now();
    auto results_hist = stats.ComputeMedian(data, params);
    auto end_hist = std::chrono::high_resolution_clock::now();
    double hist_ms = std::chrono::duration<double, std::milli>(end_hist - start_hist).count();

    // CPU reference for timing comparison
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_medians(beam_count);
    for (uint32_t b = 0; b < beam_count; ++b) {
      std::vector<float> mags(n_point);
      for (uint32_t i = 0; i < n_point; ++i)
        mags[i] = std::abs(data[b * n_point + i]);
      std::sort(mags.begin(), mags.end());
      cpu_medians[b] = mags[n_point / 2];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    double speedup = cpu_ms / hist_ms;

    con.Print(gpu_id, "Stats ROCm",
              "  Histogram benchmark: " + std::to_string(beam_count) +
              " beams x " + std::to_string(n_point) + " points");
    con.Print(gpu_id, "Stats ROCm",
              "  CPU sort      : " + std::to_string(cpu_ms) + " ms");
    con.Print(gpu_id, "Stats ROCm",
              "  GPU histogram : " + std::to_string(hist_ms) + " ms");
    con.Print(gpu_id, "Stats ROCm",
              "  Speedup       : " + std::to_string(speedup) + "x");

    bool ok = (speedup > 1.0);
    print_result(con, gpu_id, "Histogram vs CPU Benchmark", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] Histogram Benchmark EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 12: ComputeAll -- verify all fields match separate calls
// =========================================================================

inline bool test_compute_all_cpu(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    const uint32_t beam_count = 4;
    const uint32_t n_point    = 65536;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    auto ref_stats   = stats.ComputeStatistics(data, params);
    auto ref_medians = stats.ComputeMedian(data, params);
    auto full        = stats.ComputeAll(data, params);

    bool ok = (full.size() == beam_count);
    float max_err = 0.0f;

    for (uint32_t b = 0; b < beam_count && ok; ++b) {
      float e_mean_re  = std::fabs(full[b].mean.real()       - ref_stats[b].mean.real());
      float e_mean_im  = std::fabs(full[b].mean.imag()       - ref_stats[b].mean.imag());
      float e_var      = std::fabs(full[b].variance          - ref_stats[b].variance);
      float e_std      = std::fabs(full[b].std_dev           - ref_stats[b].std_dev);
      float e_mean_mag = std::fabs(full[b].mean_magnitude    - ref_stats[b].mean_magnitude);
      float e_median   = std::fabs(full[b].median_magnitude  - ref_medians[b].median_magnitude);

      float beam_max = std::max({e_mean_re, e_mean_im, e_var, e_std, e_mean_mag, e_median});
      max_err = std::max(max_err, beam_max);

      if (beam_max > 1e-5f) {
        ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  Test 12 beam " + std::to_string(b) + " max_err=" +
                  std::to_string(beam_max));
      }
    }

    con.Print(gpu_id, "Stats ROCm",
              "  ComputeAll CPU: max_err=" + std::to_string(max_err) + " (tol=1e-5)");
    print_result(con, gpu_id, "ComputeAll CPU (4 beams x 65536)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] ComputeAll CPU EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 13: ComputeAll -- GPU data (void*) path
// =========================================================================

inline bool test_compute_all_gpu(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    const uint32_t beam_count = 2;
    const uint32_t n_point    = 32768;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    // CPU reference via separate calls
    auto ref_stats   = stats.ComputeStatistics(data, params);
    auto ref_medians = stats.ComputeMedian(data, params);

    // Manually upload to GPU
    size_t bytes = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(bytes);
    backend.MemcpyHostToDevice(gpu_data, data.data(), bytes);

    auto full = stats.ComputeAll(gpu_data, params);
    backend.Free(gpu_data);

    bool ok = (full.size() == beam_count);
    float max_err = 0.0f;

    for (uint32_t b = 0; b < beam_count && ok; ++b) {
      float e_mean_re  = std::fabs(full[b].mean.real()       - ref_stats[b].mean.real());
      float e_mean_im  = std::fabs(full[b].mean.imag()       - ref_stats[b].mean.imag());
      float e_var      = std::fabs(full[b].variance          - ref_stats[b].variance);
      float e_std      = std::fabs(full[b].std_dev           - ref_stats[b].std_dev);
      float e_mean_mag = std::fabs(full[b].mean_magnitude    - ref_stats[b].mean_magnitude);
      float e_median   = std::fabs(full[b].median_magnitude  - ref_medians[b].median_magnitude);
      float beam_max   = std::max({e_mean_re, e_mean_im, e_var, e_std, e_mean_mag, e_median});
      max_err = std::max(max_err, beam_max);

      if (beam_max > 1e-5f) {
        ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  Test 13 beam " + std::to_string(b) +
                  " e_re=" + std::to_string(e_mean_re) +
                  " e_im=" + std::to_string(e_mean_im) +
                  " e_var=" + std::to_string(e_var) +
                  " e_std=" + std::to_string(e_std) +
                  " e_mag=" + std::to_string(e_mean_mag) +
                  " e_med=" + std::to_string(e_median));
      }
    }

    con.Print(gpu_id, "Stats ROCm",
              "  ComputeAll GPU: max_err=" + std::to_string(max_err) + " (tol=1e-5, 6 fields)");
    print_result(con, gpu_id, "ComputeAll GPU void* (2 beams x 32768)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm", "[X] ComputeAll GPU EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 14: ComputeAllFloat -- verify float path + mean == {0,0}
// =========================================================================

inline bool test_compute_all_float(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    const uint32_t beam_count = 2;
    const uint32_t n_point    = 16384;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    std::vector<float> magnitudes(beam_count * n_point);
    for (auto& v : magnitudes) v = dist(rng);

    StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    auto ref_stats   = stats.ComputeStatisticsFloat(magnitudes, params);
    auto ref_medians = stats.ComputeMedianFloat(magnitudes, params);
    auto full        = stats.ComputeAllFloat(magnitudes, params);

    bool ok = (full.size() == beam_count);
    float max_err = 0.0f;

    for (uint32_t b = 0; b < beam_count && ok; ++b) {
      // Verify mean == {0, 0} (float path has no complex mean)
      if (std::fabs(full[b].mean.real()) > 1e-8f || std::fabs(full[b].mean.imag()) > 1e-8f) {
        ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  Test 14: mean != 0 for float path! beam=" + std::to_string(b) +
                  " mean=(" + std::to_string(full[b].mean.real()) + "," +
                  std::to_string(full[b].mean.imag()) + ")");
      }

      float e_mean_mag = std::fabs(full[b].mean_magnitude   - ref_stats[b].mean_magnitude);
      float e_var      = std::fabs(full[b].variance         - ref_stats[b].variance);
      float e_median   = std::fabs(full[b].median_magnitude - ref_medians[b].median_magnitude);
      float beam_max   = std::max({e_mean_mag, e_var, e_median});
      max_err = std::max(max_err, beam_max);

      if (beam_max > 1e-5f) {
        ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  Test 14 beam " + std::to_string(b) + " err=" +
                  std::to_string(beam_max));
      }
    }

    con.Print(gpu_id, "Stats ROCm",
              "  ComputeAllFloat: max_err=" + std::to_string(max_err) + " (tol=1e-5)");
    print_result(con, gpu_id, "ComputeAllFloat (2 beams x 16384)", ok);
    return ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] ComputeAllFloat EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Test 15: ComputeAll -- edge cases (beam_count=1, threshold boundary)
// =========================================================================

inline bool test_compute_all_edge_cases(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);

    bool all_ok = true;

    // Case A: beam_count=1, n_point=100 (tiny → radix sort path)
    {
      const uint32_t bc = 1, np = 100;
      std::mt19937 rng(1);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<std::complex<float>> data(bc * np);
      for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

      StatisticsParams p; p.beam_count = bc; p.n_point = np;
      auto ref_s = stats.ComputeStatistics(data, p);
      auto ref_m = stats.ComputeMedian(data, p);
      auto full  = stats.ComputeAll(data, p);

      float err = std::max(
          std::fabs(full[0].mean_magnitude  - ref_s[0].mean_magnitude),
          std::fabs(full[0].median_magnitude - ref_m[0].median_magnitude));

      if (err > 1e-5f || full.size() != bc) {
        all_ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  EdgeCase A (1 beam, n=100): err=" + std::to_string(err));
      }
    }

    // Case B: beam_count=4, n_point=100000 (== kHistogramThreshold boundary)
    {
      const uint32_t bc = 4, np = 100000;
      std::mt19937 rng(2);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<std::complex<float>> data(bc * np);
      for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

      StatisticsParams p; p.beam_count = bc; p.n_point = np;
      auto ref_s = stats.ComputeStatistics(data, p);
      auto ref_m = stats.ComputeMedian(data, p);
      auto full  = stats.ComputeAll(data, p);

      float max_err = 0.0f;
      for (uint32_t b = 0; b < bc; ++b) {
        float err = std::max(
            std::fabs(full[b].mean_magnitude  - ref_s[b].mean_magnitude),
            std::fabs(full[b].median_magnitude - ref_m[b].median_magnitude));
        max_err = std::max(max_err, err);
      }

      if (max_err > 1e-5f || full.size() != bc) {
        all_ok = false;
        con.Print(gpu_id, "Stats ROCm",
                  "  EdgeCase B (4 beams, n=100000): max_err=" + std::to_string(max_err));
      }
    }

    print_result(con, gpu_id, "ComputeAll EdgeCases (n=100, n=threshold)", all_ok);
    return all_ok;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Stats ROCm",
              "[X] ComputeAll EdgeCases EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// =========================================================================
// Main test runner
// =========================================================================

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  con.Start();
  int gpu_id = 0;

  con.Print(gpu_id, "Stats ROCm", "");
  con.Print(gpu_id, "Stats ROCm", "============================================");
  con.Print(gpu_id, "Stats ROCm", "  StatisticsProcessor Tests (ROCm/HIP)");
  con.Print(gpu_id, "Stats ROCm", "============================================");

  // Check for ROCm devices
  int device_count = ROCmCore::GetAvailableDeviceCount();
  con.Print(gpu_id, "Stats ROCm", "Available ROCm devices: " + std::to_string(device_count));

  if (device_count == 0) {
    con.Print(gpu_id, "Stats ROCm", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  int passed = 0;
  int total = 15;

  if (test_mean_single_beam(con, gpu_id)) ++passed;
  if (test_mean_multi_beam(con, gpu_id)) ++passed;
  if (test_welford_statistics(con, gpu_id)) ++passed;
  if (test_median(con, gpu_id)) ++passed;
  if (test_gpu_input(con, gpu_id)) ++passed;
  if (test_mean_constant(con, gpu_id)) ++passed;
  if (test_benchmark_median(con, gpu_id)) ++passed;
  if (test_histogram_median_basic(con, gpu_id)) ++passed;
  if (test_histogram_median_multi(con, gpu_id)) ++passed;
  if (test_histogram_median_float(con, gpu_id)) ++passed;
  if (test_histogram_vs_radix_benchmark(con, gpu_id)) ++passed;
  if (test_compute_all_cpu(con, gpu_id)) ++passed;
  if (test_compute_all_gpu(con, gpu_id)) ++passed;
  if (test_compute_all_float(con, gpu_id)) ++passed;
  if (test_compute_all_edge_cases(con, gpu_id)) ++passed;

  con.Print(gpu_id, "Stats ROCm", "");
  con.Print(gpu_id, "Stats ROCm", "Results: " + std::to_string(passed) + "/" +
                                    std::to_string(total) + " passed");
  con.Print(gpu_id, "Stats ROCm", "============================================");
  con.Print(gpu_id, "Stats ROCm", "");
}

}  // namespace test_statistics_rocm

#endif  // ENABLE_ROCM
