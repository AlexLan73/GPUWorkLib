#pragma once

/**
 * @file test_signal_generators_benchmark_rocm.hpp
 * @brief Test runner: FormSignalGeneratorROCm benchmark (GpuBenchmarkBase)
 *
 * Запускает бенчмарк:
 *   FormSignalGeneratorROCm::GenerateInputData() → Results/Profiler/GPU_00_FormSignalROCm/
 *
 * 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если нет AMD GPU — выводит [SKIP] и не падает.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see signal_generators_benchmark_rocm.hpp, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#if ENABLE_ROCM

#include "signal_generators_benchmark_rocm.hpp"
#include "DrvGPU/backends/rocm/rocm_backend.hpp"
#include "DrvGPU/backends/rocm/rocm_core.hpp"

#include <iostream>
#include <stdexcept>

namespace test_signal_generators_benchmark_rocm {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  FormSignalGeneratorROCm Benchmark (GpuBenchmarkBase)\n"
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

    // ── Параметры генератора ───────────────────────────────────────────────
    signal_gen::FormParams params;
    params.fs              = 12e6;
    params.antennas        = 8;
    params.points          = 4096;
    params.f0              = 1e6;
    params.amplitude       = 1.0;
    params.noise_amplitude = 0.0;

    // ── Создать генератор ──────────────────────────────────────────────────
    signal_gen::FormSignalGeneratorROCm gen(backend.get());
    gen.SetParams(params);

    // ── Benchmark ─────────────────────────────────────────────────────────
    std::cout << "\n--- FormSignalGeneratorROCm::GenerateInputData() ---\n";
    {
      test_signal_generators_rocm::FormSignalGeneratorROCmBenchmark bench(
          backend.get(), gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_FormSignalROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] FormSignalGeneratorROCm benchmark complete\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cout << "  [SKIP] " << e.what() << "\n";
    return 0;
  }
}

}  // namespace test_signal_generators_benchmark_rocm

#endif  // ENABLE_ROCM
