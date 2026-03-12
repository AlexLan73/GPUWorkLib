#pragma once

/**
 * @file test_strategies_benchmark_streams.hpp
 * @brief Benchmark: 1 stream vs 3 streams for post-FFT scenarios
 *
 * Compares execution time of do_run_post_fft_scenarios (sequential, 1 stream)
 * vs do_run_post_fft_parallel (parallel, 3 streams) for Step2.1 + Step2.2 + Step2.3.
 *
 * Test setup:
 *   - AntennaProcessorTest subclass to access protected step methods
 *   - n_ant = 16, n_samples = 512 (small, repeatable)
 *   - N_WARMUP = 3, N_RUNS = 20 — average time per run
 *
 * IMPORTANT: ROCm-only.
 *
 * @date 2026-03-12
 */

#if ENABLE_ROCM

#include "antenna_processor_test.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include <cstring>

namespace test_strategies_benchmark_streams {

using namespace drv_gpu_lib;

// ============================================================================
// Helper: benchmark accessor subclass
// ============================================================================

class BenchmarkProcessor : public strategies::AntennaProcessorTest {
public:
  using strategies::AntennaProcessorTest::AntennaProcessorTest;

  /// Prepare internal buffers: run GEMM + Window+FFT once (fills d_spectrum_, d_magnitudes_)
  void prepare_buffers(const void* d_S, const void* d_W) {
    do_gemm(d_S, d_W);
    do_window_fft();
    hipDeviceSynchronize();
  }

  /// Post-FFT scenarios: sequential 1-stream (timed part)
  strategies::AntennaResult run_sequential() {
    strategies::AntennaResult result;
    result.scenario_mode = config().scenario_mode;
    do_run_post_fft_scenarios(result);
    return result;
  }

  /// Post-FFT scenarios: parallel 3-stream (timed part)
  strategies::AntennaResult run_parallel() {
    strategies::AntennaResult result;
    result.scenario_mode = config().scenario_mode;
    do_run_post_fft_parallel(result);
    return result;
  }
};

// ============================================================================
// Benchmark runner
// ============================================================================

inline void run_benchmark_streams(drv_gpu_lib::IBackend* backend) {
  auto& con = ConsoleOutput::GetInstance();
  int gpu_id = backend->GetDeviceIndex();

  con.Print(gpu_id, "BenchStreams", "");
  con.Print(gpu_id, "BenchStreams", "=== Benchmark: 1 stream vs 3 streams (post-FFT) ===");

  constexpr uint32_t kNAnt     = 16;
  constexpr uint32_t kNSamples = 512;
  constexpr int      kNWarmup  = 3;
  constexpr int      kNRuns    = 20;

  strategies::AntennaProcessorConfig cfg;
  cfg.n_ant      = kNAnt;
  cfg.n_samples  = kNSamples;
  cfg.sample_rate = 1e6f;
  cfg.scenario_mode = strategies::PostFftScenarioMode::ALL_REQUIRED;

  BenchmarkProcessor proc(backend, cfg);

  // Allocate dummy d_S and d_W on GPU (zero data — benchmarks kernel scheduling, not math)
  const size_t cf_bytes = sizeof(float) * 2;
  const size_t s_bytes  = kNAnt * kNSamples * cf_bytes;
  const size_t w_bytes  = kNAnt * kNAnt     * cf_bytes;

  void* d_S = nullptr;
  void* d_W = nullptr;
  hipMalloc(&d_S, s_bytes);
  hipMalloc(&d_W, w_bytes);
  hipMemset(d_S, 0, s_bytes);
  hipMemset(d_W, 0, w_bytes);

  // Fill d_spectrum_ + d_magnitudes_ via GEMM + Window+FFT (outside timing loop)
  proc.prepare_buffers(d_S, d_W);

  // ── 1 stream ──────────────────────────────────────────────────────────────
  for (int i = 0; i < kNWarmup; ++i) proc.run_sequential();
  hipDeviceSynchronize();

  std::vector<float> times_seq(kNRuns);
  for (int i = 0; i < kNRuns; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    proc.run_sequential();
    auto t1 = std::chrono::high_resolution_clock::now();
    times_seq[i] = std::chrono::duration<float, std::milli>(t1 - t0).count();
  }
  float avg_seq = std::accumulate(times_seq.begin(), times_seq.end(), 0.0f) / kNRuns;

  // ── 3 streams ─────────────────────────────────────────────────────────────
  for (int i = 0; i < kNWarmup; ++i) proc.run_parallel();
  hipDeviceSynchronize();

  std::vector<float> times_par(kNRuns);
  for (int i = 0; i < kNRuns; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    proc.run_parallel();
    auto t1 = std::chrono::high_resolution_clock::now();
    times_par[i] = std::chrono::duration<float, std::milli>(t1 - t0).count();
  }
  float avg_par = std::accumulate(times_par.begin(), times_par.end(), 0.0f) / kNRuns;

  hipFree(d_S);
  hipFree(d_W);

  // ── Results ───────────────────────────────────────────────────────────────
  float speedup = (avg_par > 0.0f) ? (avg_seq / avg_par) : 0.0f;
  std::string winner = (avg_par < avg_seq) ? "3-stream FASTER" : "1-stream FASTER (or equal)";

  con.Print(gpu_id, "BenchStreams",
      "  Config: n_ant=" + std::to_string(kNAnt) +
      " n_samples=" + std::to_string(kNSamples) +
      " N_runs=" + std::to_string(kNRuns));
  con.Print(gpu_id, "BenchStreams",
      "  1-stream (sequential):  avg = " +
      std::to_string(avg_seq).substr(0, 6) + " ms");
  con.Print(gpu_id, "BenchStreams",
      "  3-stream (parallel):    avg = " +
      std::to_string(avg_par).substr(0, 6) + " ms");
  con.Print(gpu_id, "BenchStreams",
      "  Speedup (seq/par): " + std::to_string(speedup).substr(0, 5) +
      "x  →  " + winner);
  con.Print(gpu_id, "BenchStreams", "=== Done ===");
}

}  // namespace test_strategies_benchmark_streams

#endif  // ENABLE_ROCM
