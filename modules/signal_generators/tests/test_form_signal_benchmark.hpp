#pragma once

/**
 * @file test_form_signal_benchmark.hpp
 * @brief Test runner: FormSignalGenerator / DelayedFormSignalGenerator /
 *        LfmGeneratorAnalyticalDelay / FormScriptGenerator — OpenCL benchmark
 *
 * Запускает 4 бенчмарка:
 *  1. FormSignalGenerator::GenerateInputData()         → Results/Profiler/GPU_00_FormSignal/
 *  2. DelayedFormSignalGenerator::GenerateInputData()  → Results/Profiler/GPU_00_DelayedFormSignal/
 *  3. LfmGeneratorAnalyticalDelay::GenerateToGpu()     → Results/Profiler/GPU_00_LfmAnalyticalDelay/
 *  4. FormScriptGenerator::GenerateInputData()         → Results/Profiler/GPU_00_FormScriptGenerator/
 *
 * Каждый: 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если is_prof=false в configGPU.json — выводит [SKIP] и не падает.
 *
 * ⚠️ OpenCL queue создаётся с CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see form_signal_benchmark.hpp, MemoryBank/tasks/TASK_signal_generators_profiling.md
 */

#include "form_signal_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <stdexcept>

namespace test_form_signal_benchmark {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Form Signal Generators Benchmark\n"
            << "  (FormSignal / DelayedFormSignal / LfmAnalytDelay / FormScript)\n"
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

    // ── Общие FormParams (8 антенн, 4096 отсчётов, 1 МГц) ─────────────────
    signal_gen::FormParams form_params;
    form_params.fs               = 12e6;
    form_params.antennas         = 8;
    form_params.points           = 4096;
    form_params.f0               = 1e6;
    form_params.amplitude        = 1.0;
    form_params.noise_amplitude  = 0.0;

    // ── Benchmark 1: FormSignalGenerator ──────────────────────────────────
    std::cout << "\n--- Benchmark 1: FormSignalGenerator::GenerateInputData() ---\n";
    {
      signal_gen::FormSignalGenerator gen(backend.get());
      gen.SetParams(form_params);

      test_form_signal::FormSignalGeneratorBenchmark bench(
          backend.get(), gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_FormSignal"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] FormSignalGenerator benchmark complete\n";
      }
    }

    // ── Benchmark 2: DelayedFormSignalGenerator ───────────────────────────
    std::cout << "\n--- Benchmark 2: DelayedFormSignalGenerator::GenerateInputData() ---\n";
    {
      signal_gen::FormParams delayed_params = form_params;
      delayed_params.noise_amplitude = 0.1;  // шум добавляется после задержки

      signal_gen::DelayedFormSignalGenerator gen(backend.get());
      gen.SetParams(delayed_params);
      // Линейные задержки: 0, 1.5, 3.0, ..., 10.5 мкс
      gen.SetDelays({0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f, 10.5f});

      test_form_signal::DelayedFormSignalGeneratorBenchmark bench(
          backend.get(), gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_DelayedFormSignal"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] DelayedFormSignalGenerator benchmark complete\n";
      }
    }

    // ── Benchmark 3: LfmGeneratorAnalyticalDelay ─────────────────────────
    std::cout << "\n--- Benchmark 3: LfmGeneratorAnalyticalDelay::GenerateToGpu() ---\n";
    {
      signal_gen::LfmParams lfm_params;
      lfm_params.f_start   = 0.5e6;
      lfm_params.f_end     = 5.5e6;
      lfm_params.amplitude = 1.0;

      signal_gen::SystemSampling system;
      system.fs     = 12e6;
      system.length = 4096;

      signal_gen::LfmGeneratorAnalyticalDelay gen(backend.get(), lfm_params);
      gen.SetSampling(system);
      // 8 антенн с линейной задержкой
      gen.SetDelays({0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f, 10.5f});

      test_form_signal::LfmAnalyticalDelayBenchmark bench(
          backend.get(), gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_LfmAnalyticalDelay"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] LfmAnalyticalDelay benchmark complete\n";
      }
    }

    // ── Benchmark 4: FormScriptGenerator ──────────────────────────────────
    std::cout << "\n--- Benchmark 4: FormScriptGenerator::GenerateInputData() ---\n";
    {
      signal_gen::FormScriptGenerator gen(backend.get());
      gen.SetParams(form_params);
      gen.Compile();  // компиляция DSL-кернела (до бенчмарка)

      test_form_signal::FormScriptGeneratorBenchmark bench(
          backend.get(), gen,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_FormScriptGenerator"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] FormScriptGenerator benchmark complete\n";
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

}  // namespace test_form_signal_benchmark
