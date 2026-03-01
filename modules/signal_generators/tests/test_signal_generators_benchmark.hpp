#pragma once

/**
 * @file test_signal_generators_benchmark.hpp
 * @brief Test runner: CwGenerator / LfmGenerator / LfmConjugateGenerator /
 *        NoiseGenerator — OpenCL benchmark (GpuBenchmarkBase)
 *
 * Запускает 4 бенчмарка:
 *  1. CwGenerator::GenerateToGpu()            → Results/Profiler/GPU_00_CwGenerator/
 *  2. LfmGenerator::GenerateToGpu()           → Results/Profiler/GPU_00_LfmGenerator/
 *  3. LfmConjugateGenerator::GenerateToGpu()  → Results/Profiler/GPU_00_LfmConjugateGenerator/
 *  4. NoiseGenerator::GenerateToGpu()         → Results/Profiler/GPU_00_NoiseGenerator/
 *
 * Каждый: 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если is_prof=false в configGPU.json — выводит [SKIP] и не падает.
 *
 * ⚠️ OpenCL queue создаётся с CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see signal_generators_benchmark.hpp, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#include "signal_generators_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <stdexcept>

namespace test_signal_generators_benchmark {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Signal Generators Benchmark (CW / LFM / LfmConj / Noise)\n"
            << "============================================================\n";

  try {
    // ── OpenCL инициализация ──────────────────────────────────────────────
    cl_int err;

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clGetPlatformIDs failed: " + std::to_string(err));

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clGetDeviceIDs failed: " + std::to_string(err));

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

    // ⚠️ CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
      clReleaseContext(context);
      throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));
    }

    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->InitializeFromExternalContext(context, device, queue);

    // ── Общие параметры дискретизации ─────────────────────────────────────
    signal_gen::SystemSampling system;
    system.fs     = 12e6;
    system.length = 4096;

    // ── Benchmark 1: CwGenerator ──────────────────────────────────────────
    std::cout << "\n--- Benchmark 1: CwGenerator::GenerateToGpu() ---\n";
    {
      signal_gen::CwParams cw_params;
      cw_params.f0        = 1e6;
      cw_params.amplitude = 1.0;

      signal_gen::CwGenerator cw_gen(backend.get(), cw_params);

      test_signal_generators::CwGeneratorBenchmark bench(
          backend.get(), cw_gen, system, /*beam_count=*/1,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_CwGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] CwGenerator benchmark complete\n";
      }
    }

    // ── Benchmark 2: LfmGenerator ─────────────────────────────────────────
    std::cout << "\n--- Benchmark 2: LfmGenerator::GenerateToGpu() ---\n";
    {
      signal_gen::LfmParams lfm_params;
      lfm_params.f_start   = 0.5e6;
      lfm_params.f_end     = 5.5e6;
      lfm_params.amplitude = 1.0;

      signal_gen::LfmGenerator lfm_gen(backend.get(), lfm_params);

      test_signal_generators::LfmGeneratorBenchmark bench(
          backend.get(), lfm_gen, system, /*beam_count=*/1,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_LfmGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] LfmGenerator benchmark complete\n";
      }
    }

    // ── Benchmark 3: LfmConjugateGenerator ───────────────────────────────
    std::cout << "\n--- Benchmark 3: LfmConjugateGenerator::GenerateToGpu() ---\n";
    {
      signal_gen::LfmParams conj_params;
      conj_params.f_start   = 0.5e6;
      conj_params.f_end     = 5.5e6;
      conj_params.amplitude = 1.0;

      signal_gen::LfmConjugateGenerator conj_gen(backend.get(), conj_params);
      conj_gen.SetSampling(system);

      test_signal_generators::LfmConjugateGeneratorBenchmark bench(
          backend.get(), conj_gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_LfmConjugateGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] LfmConjugateGenerator benchmark complete\n";
      }
    }

    // ── Benchmark 4: NoiseGenerator ───────────────────────────────────────
    std::cout << "\n--- Benchmark 4: NoiseGenerator::GenerateToGpu() ---\n";
    {
      signal_gen::NoiseParams noise_params;
      noise_params.power = 1.0;

      signal_gen::NoiseGenerator noise_gen(backend.get(), noise_params);

      test_signal_generators::NoiseGeneratorBenchmark bench(
          backend.get(), noise_gen, system, /*beam_count=*/1,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_NoiseGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] NoiseGenerator benchmark complete\n";
      }
    }

    // ── Cleanup ────────────────────────────────────────────────────────────
    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_signal_generators_benchmark
