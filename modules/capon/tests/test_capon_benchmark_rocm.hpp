#pragma once

/**
 * @file test_capon_benchmark_rocm.hpp
 * @brief Test runner: CaponProcessor — ROCm benchmark (GpuBenchmarkBase)
 *
 * Запускает 2 бенчмарка:
 *  1. CaponProcessor::ComputeRelief()     → Results/Profiler/GPU_00_Capon_ROCm/
 *  2. CaponProcessor::AdaptiveBeamform()  → Results/Profiler/GPU_00_Capon_ROCm/
 *
 * Параметры: P=16 каналов, N=256 отсчётов, M=64 направления, mu=0.01
 * n_warmup=5, n_runs=20
 *
 * Если нет AMD GPU или профилирование отключено — выводит [SKIP].
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 * @see capon_benchmark.hpp
 */

#if ENABLE_ROCM

#include "capon_benchmark.hpp"
#include "capon_test_helpers.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "services/console_output.hpp"

#include <complex>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <string>

namespace test_capon_benchmark_rocm {

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "Capon[Bench]", msg);
}

inline void run() {
  TestPrint("============================================================");
  TestPrint("  Capon Benchmark (ComputeRelief / AdaptiveBeamform) — ROCm");
  TestPrint("============================================================");

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    TestPrint("  [SKIP] No AMD GPU available");
    return;
  }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────────
    auto* backend = &capon_test_helpers::GetROCmBackend();

    // ── Параметры Кейпона ─────────────────────────────────────────────────
    capon::CaponParams params;
    params.n_channels   = 16;   // P — число антенных каналов
    params.n_samples    = 256;  // N — число временных отсчётов
    params.n_directions = 64;   // M — число направлений сканирования
    params.mu           = 0.01f;

    // ── Тестовые данные ───────────────────────────────────────────────────
    const auto signal   = capon_test_helpers::MakeNoise(
        static_cast<size_t>(params.n_channels) * params.n_samples, 1.0f, 42u);
    const auto steering = capon_test_helpers::MakeSteeringMatrix(
        params.n_channels, params.n_directions,
        -static_cast<float>(M_PI) / 3.0f,
         static_cast<float>(M_PI) / 3.0f);

    // ── Создать процессор (компилирует HIP kernels один раз) ──────────────
    capon::CaponProcessor proc(backend);

    // ── Benchmark 1: ComputeRelief() ──────────────────────────────────────
    TestPrint("--- Benchmark 1: CaponProcessor::ComputeRelief() ---");
    {
      test_capon_rocm_bench::CaponReliefBenchmarkROCm bench(
          backend, proc, signal, steering, params,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"});

      bench.Run();
      bench.Report();
      TestPrint("  [OK] ComputeRelief ROCm benchmark complete");
    }

    // ── Benchmark 2: AdaptiveBeamform() ──────────────────────────────────
    TestPrint("--- Benchmark 2: CaponProcessor::AdaptiveBeamform() ---");
    {
      test_capon_rocm_bench::CaponBeamformBenchmarkROCm bench(
          backend, proc, signal, steering, params,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"});

      bench.Run();
      bench.Report();
      TestPrint("  [OK] AdaptiveBeamform ROCm benchmark complete");
    }

  } catch (const std::exception& e) {
    TestPrint(std::string("  [SKIP] ") + e.what());
  }
}

}  // namespace test_capon_benchmark_rocm

#endif  // ENABLE_ROCM
