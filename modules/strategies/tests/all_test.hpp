#pragma once

/**
 * @file all_test.hpp
 * @brief Entry point for strategies module tests
 *
 * Include this from src/main.cpp to run strategies tests.
 *
 * Usage in main.cpp:
 *   #include "modules/strategies/tests/all_test.hpp"
 *   strategies_all_test::run();
 *
 * @date 2026-03-07
 */

#include "test_strategies_pipeline.hpp"
#include "test_strategies_step_profiling.hpp"
#include "test_strategies_benchmark_streams.hpp"

// ── New test infrastructure (OOP/SOLID/GRASP/GoF) ───────────────────────────
#include "test_base_strategy.hpp"
#include "test_debug_steps.hpp"
// #include "strategies_profiling_benchmark.hpp"  // включить отдельно если нужно
// #include "timing_per_step_test.hpp"             // включить отдельно если нужно

#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#endif

namespace strategies_all_test {

#if ENABLE_ROCM

inline drv_gpu_lib::IBackend* GetTestBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  if (!backend.IsInitialized()) {
    backend.Initialize(0);
  }
  return &backend;
}

#endif  // ENABLE_ROCM

inline void run() {
#if ENABLE_ROCM
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();
  int gpu_id = 0;

  con.Print(gpu_id, "Strategies", "");
  con.Print(gpu_id, "Strategies", "════════════════════════════════════════════════════════════");
  con.Print(gpu_id, "Strategies", " Strategies Tests (ROCm)");
  con.Print(gpu_id, "Strategies", "════════════════════════════════════════════════════════════");

  auto* backend = GetTestBackend();
  test_strategies::test_full_pipeline(backend);
  test_strategies::test_external_weights(backend);
  // test_strategies_profiling::run_step_profiling(backend);
  test_strategies_benchmark_streams::run_benchmark_streams(backend);

  // ── New tests: OOP/SOLID/GRASP/GoF framework ────────────────────────────
  test_base_strategy::run_sin_only(backend);          // T1: быстрый smoke-тест
  // test_base_strategy::run_all_variants(backend);   // T1: все 4 сигнала
  // test_debug_steps::run_all(backend);              // T2: step-by-step debug

  con.Print(gpu_id, "Strategies", "════════════════════════════════════════════════════════════");
  con.Print(gpu_id, "Strategies", " All Strategies tests PASSED ✅");
  con.Print(gpu_id, "Strategies", "════════════════════════════════════════════════════════════");
#endif
}

}  // namespace strategies_all_test
