#pragma once

/**
 * @file test_heterodyne_benchmark.hpp
 * @brief Test runner: HeterodyneProcessorOpenCL — OpenCL benchmark (GpuBenchmarkBase)
 *
 * Запускает 2 бенчмарка:
 *  1. HeterodyneProcessorOpenCL::Dechirp()  → Results/Profiler/GPU_00_Heterodyne/
 *  2. HeterodyneProcessorOpenCL::Correct()  → Results/Profiler/GPU_00_Heterodyne/
 *
 * Параметры:
 *   num_antennas = 5,  num_samples = 4000,  sample_rate = 12 MHz
 *   n_warmup = 5,  n_runs = 20
 *
 * Каждый: 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если is_prof=false в configGPU.json — выводит [SKIP] и не падает.
 *
 * ⚠️ OpenCL queue создаётся с CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see heterodyne_benchmark.hpp, MemoryBank/tasks/TASK_heterodyne_profiling.md
 */

#include "heterodyne_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace test_heterodyne_benchmark {

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Heterodyne Benchmark (Dechirp / Correct) — OpenCL\n"
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

    // ── Параметры гетеродина ──────────────────────────────────────────────
    drv_gpu_lib::HeterodyneParams params;
    params.num_antennas = 5;
    params.num_samples  = 4000;
    params.sample_rate  = 12e6f;
    params.f_start      = 0.0f;
    params.f_end        = 1e6f;

    const int total = params.num_antennas * params.num_samples;

    // ── Генерация тестовых данных (pure real = 1+0i, для бенчмарка данные не критичны)
    const std::vector<std::complex<float>> rx_data(total, {1.0f, 0.0f});
    const std::vector<std::complex<float>> ref_data(params.num_samples, {1.0f, 0.0f});
    const std::vector<std::complex<float>> dc_data(total, {1.0f, 0.0f});
    const std::vector<float> f_beat_hz = {300e3f, 600e3f, 900e3f, 1200e3f, 1500e3f};

    // ── Создать процессор (компилирует kernels один раз) ──────────────────
    drv_gpu_lib::HeterodyneProcessorOpenCL proc(backend.get());

    // ── Benchmark 1: Dechirp() ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 1: HeterodyneProcessorOpenCL::Dechirp() ---\n";
    {
      test_heterodyne_opencl::HeterodyneDechirpBenchmark bench(
          backend.get(), proc, params, rx_data, ref_data,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Heterodyne"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] Dechirp benchmark complete\n";
      }
    }

    // ── Benchmark 2: Correct() ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 2: HeterodyneProcessorOpenCL::Correct() ---\n";
    {
      test_heterodyne_opencl::HeterodyneCorrectBenchmark bench(
          backend.get(), proc, params, dc_data, f_beat_hz,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_Heterodyne"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] Correct benchmark complete\n";
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

}  // namespace test_heterodyne_benchmark
