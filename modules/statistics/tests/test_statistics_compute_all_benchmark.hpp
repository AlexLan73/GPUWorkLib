#pragma once

/**
 * @file test_statistics_compute_all_benchmark.hpp
 * @brief Test runner: StatisticsProcessor::ComputeAll — ROCm benchmark
 *
 * Запускает бенчмарк ComputeAll (4 beams × 65536 complex float, CPU path).
 * Выводит breakdown: Upload | Welford_Fused | Median через GPUProfiler.
 * Для сравнения дополнительно измеряет раздельные вызовы
 * ComputeStatistics + ComputeMedian через std::chrono.
 *
 * Если нет AMD GPU — выводит [SKIP] и не падает.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-20
 * @see statistics_compute_all_benchmark.hpp, TASK_statistics_compute_all.md
 */

#if ENABLE_ROCM

#include "statistics_compute_all_benchmark.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "services/console_output.hpp"

#include <complex>
#include <chrono>
#include <random>
#include <stdexcept>
#include <vector>

namespace test_statistics_compute_all_benchmark {

inline int run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(0, "Stats Bench", "");
  con.Print(0, "Stats Bench", "============================================================");
  con.Print(0, "Stats Bench", "  Statistics ComputeAll Benchmark — ROCm");
  con.Print(0, "Stats Bench", "  (4 beams x 65536 complex float, CPU path)");
  con.Print(0, "Stats Bench", "============================================================");

  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    con.Print(0, "Stats Bench", "  [SKIP] No AMD GPU available");
    return 0;
  }

  try {
    // ── Backend init ──────────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Test data ─────────────────────────────────────────────────────────
    const uint32_t beam_count = 4;
    const uint32_t n_point    = 65536;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (auto& v : data) v = std::complex<float>(dist(rng), dist(rng));

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    // ── StatisticsProcessor ───────────────────────────────────────────────
    statistics::StatisticsProcessor proc(backend.get());

    // ── Benchmark: ComputeAll ─────────────────────────────────────────────
    con.Print(0, "Stats Bench", "--- Benchmark: StatisticsProcessor::ComputeAll() ---");
    {
      ComputeAllBenchmarkROCm bench(
          backend.get(), proc, params, data,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Statistics"});

      bench.Run();
      bench.Report();
      con.Print(0, "Stats Bench", "  [OK] ComputeAll ROCm benchmark complete");
    }

    // ── Comparison: separate calls (std::chrono) ──────────────────────────
    con.Print(0, "Stats Bench", "--- Comparison: ComputeStatistics + ComputeMedian (separate) ---");
    {
      // Warmup
      proc.ComputeAll(data, params);

      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 20; ++i) {
        proc.ComputeStatistics(data, params);
        proc.ComputeMedian(data, params);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double sep_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;

      auto t2 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 20; ++i) {
        proc.ComputeAll(data, params);
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      double all_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / 20.0;

      con.Print(0, "Stats Bench", "  Separate   (avg 20 runs): " + std::to_string(sep_ms) + " ms");
      con.Print(0, "Stats Bench", "  ComputeAll (avg 20 runs): " + std::to_string(all_ms) + " ms");
      con.Print(0, "Stats Bench", "  Speedup: " + std::to_string(sep_ms / all_ms) + "x");
    }

    return 0;

  } catch (const std::exception& e) {
    con.Print(0, "Stats Bench", "  [SKIP] " + std::string(e.what()));
    return 0;
  }
}

}  // namespace test_statistics_compute_all_benchmark

#endif  // ENABLE_ROCM
