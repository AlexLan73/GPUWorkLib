#pragma once

/**
 * @file test_heterodyne_benchmark_rocm.hpp
 * @brief Test runner: HeterodyneProcessorROCm — ROCm benchmark (GpuBenchmarkBase)
 *
 * Запускает 2 бенчмарка:
 *  1. HeterodyneProcessorROCm::Dechirp()  → Results/Profiler/GPU_00_Heterodyne_ROCm/
 *  2. HeterodyneProcessorROCm::Correct()  → Results/Profiler/GPU_00_Heterodyne_ROCm/
 *
 * Параметры:
 *   num_antennas = 5,  num_samples = 4000,  sample_rate = 12 MHz
 *   n_warmup = 5,  n_runs = 20
 *
 * Если нет AMD GPU — выводит [SKIP] и не падает.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see heterodyne_benchmark_rocm.hpp, MemoryBank/tasks/TASK_heterodyne_profiling.md
 */

#if ENABLE_ROCM

#include "heterodyne_benchmark_rocm.hpp"
#include "DrvGPU/backends/rocm/rocm_backend.hpp"
#include "DrvGPU/backends/rocm/rocm_core.hpp"

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace test_heterodyne_benchmark_rocm {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Heterodyne Benchmark (Dechirp / Correct) — ROCm\n"
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

    // ── Параметры гетеродина ──────────────────────────────────────────────
    drv_gpu_lib::HeterodyneParams params;
    params.num_antennas = 5;
    params.num_samples  = 4000;
    params.sample_rate  = 12e6f;
    params.f_start      = 0.0f;
    params.f_end        = 1e6f;

    const int total = params.num_antennas * params.num_samples;

    // ── Тестовые данные (pure real = 1+0i, для бенчмарка данные не критичны)
    const std::vector<std::complex<float>> rx_data(total, {1.0f, 0.0f});
    const std::vector<std::complex<float>> ref_data(params.num_samples, {1.0f, 0.0f});
    const std::vector<std::complex<float>> dc_data(total, {1.0f, 0.0f});
    const std::vector<float> f_beat_hz = {300e3f, 600e3f, 900e3f, 1200e3f, 1500e3f};

    // ── Создать процессор (компилирует HIP kernels один раз) ──────────────
    drv_gpu_lib::HeterodyneProcessorROCm proc(backend.get());

    // ── Benchmark 1: Dechirp() ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 1: HeterodyneProcessorROCm::Dechirp() ---\n";
    {
      test_heterodyne_rocm::HeterodyneDechirpBenchmarkROCm bench(
          backend.get(), proc, params, rx_data, ref_data,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] Dechirp ROCm benchmark complete\n";
    }

    // ── Benchmark 2: Correct() ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 2: HeterodyneProcessorROCm::Correct() ---\n";
    {
      test_heterodyne_rocm::HeterodyneCorrectBenchmarkROCm bench(
          backend.get(), proc, params, dc_data, f_beat_hz,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] Correct ROCm benchmark complete\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cout << "  [SKIP] " << e.what() << "\n";
    return 0;
  }
}

}  // namespace test_heterodyne_benchmark_rocm

#endif  // ENABLE_ROCM
