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
#include "test_capon_rocm.hpp"          // MakeNoise, MakeSteeringMatrix
#include "DrvGPU/backends/rocm/rocm_backend.hpp"
#include "DrvGPU/backends/rocm/rocm_core.hpp"

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace test_capon_benchmark_rocm {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Capon Benchmark (ComputeRelief / AdaptiveBeamform) — ROCm\n"
            << "============================================================\n";

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    std::cout << "  [SKIP] No AMD GPU available\n";
    return 0;
  }

  try {
    // ── ROCm backend init ─────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Параметры Кейпона ─────────────────────────────────────────────────
    capon::CaponParams params;
    params.n_channels   = 16;   // P — число антенных каналов
    params.n_samples    = 256;  // N — число временных отсчётов
    params.n_directions = 64;   // M — число направлений сканирования
    params.mu           = 0.01f;

    // ── Тестовые данные ───────────────────────────────────────────────────
    const auto signal   = test_capon_rocm::MakeNoise(
        static_cast<size_t>(params.n_channels) * params.n_samples, 1.0f, 42u);
    const auto steering = test_capon_rocm::MakeSteeringMatrix(
        params.n_channels, params.n_directions,
        -static_cast<float>(M_PI) / 3.0f,
         static_cast<float>(M_PI) / 3.0f);

    // ── Создать процессор (компилирует HIP kernels один раз) ──────────────
    capon::CaponProcessor proc(backend.get());

    // ── Benchmark 1: ComputeRelief() ──────────────────────────────────────
    std::cout << "\n--- Benchmark 1: CaponProcessor::ComputeRelief() ---\n";
    {
      test_capon_rocm_bench::CaponReliefBenchmarkROCm bench(
          backend.get(), proc, signal, steering, params,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] ComputeRelief ROCm benchmark complete\n";
    }

    // ── Benchmark 2: AdaptiveBeamform() ──────────────────────────────────
    std::cout << "\n--- Benchmark 2: CaponProcessor::AdaptiveBeamform() ---\n";
    {
      test_capon_rocm_bench::CaponBeamformBenchmarkROCm bench(
          backend.get(), proc, signal, steering, params,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] AdaptiveBeamform ROCm benchmark complete\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cout << "  [SKIP] " << e.what() << "\n";
    return 0;
  }
}

}  // namespace test_capon_benchmark_rocm

#endif  // ENABLE_ROCM
