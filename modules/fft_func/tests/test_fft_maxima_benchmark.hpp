#pragma once

/**
 * @file test_fft_maxima_benchmark.hpp
 * @brief Test runner: SpectrumMaximaFinder OpenCL benchmark (GpuBenchmarkBase)
 *
 * Запускает два бенчмарка:
 *  Benchmark 1: Process (ONE_PEAK)  → Results/Profiler/GPU_00_SpectrumMaxima_Process/
 *  Benchmark 2: FindAllMaxima       → Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima/
 *
 * Каждый: 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если is_prof=false в configGPU.json — выводит [SKIP] и не падает.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see fft_maxima_benchmark.hpp, MemoryBank/tasks/TASK_fft_maxima_profiling_opencl.md
 */

#include "fft_maxima_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_maxima_benchmark {

// ─── Утилита — генерация мультилучевых данных ─────────────────────────────────

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t beam_count, uint32_t n_point, float sample_rate)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(beam_count) * n_point);
  for (uint32_t b = 0; b < beam_count; ++b) {
    float freq = 50.0f + b * 30.0f;
    for (uint32_t t = 0; t < n_point; ++t) {
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
      data[static_cast<size_t>(b) * n_point + t] = {std::cos(phase), std::sin(phase)};
    }
  }
  return data;
}

// ─── Точка входа ──────────────────────────────────────────────────────────────

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  SpectrumMaximaFinder Benchmark (GpuBenchmarkBase)\n"
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

    // ── Данные ────────────────────────────────────────────────────────────
    const uint32_t BEAM_COUNT  = 10;
    const uint32_t N_POINT     = 8192;
    const float    SAMPLE_RATE = 100000.0f;

    auto signal = GenerateSignal(BEAM_COUNT, N_POINT, SAMPLE_RATE);

    antenna_fft::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = BEAM_COUNT;
    input.n_point       = N_POINT;
    input.sample_rate   = SAMPLE_RATE;
    input.repeat_count  = 1;
    input.data          = signal;

    antenna_fft::SpectrumMaximaFinder proc(backend.get());

    // ── Benchmark 1: Process (ONE_PEAK) ───────────────────────────────────
    std::cout << "\n--- Benchmark 1: Process (ONE_PEAK) ---\n";
    {
      test_fft_maxima::SpectrumMaximaFinderBenchmark bench(
          backend.get(), proc, input,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_Process"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] Process benchmark complete\n";
      }
    }

    // ── Benchmark 2: FindAllMaxima (pipeline) ─────────────────────────────
    std::cout << "\n--- Benchmark 2: FindAllMaxima (pipeline) ---\n";
    {
      test_fft_maxima::SpectrumMaximaAllMaximaBenchmark bench(
          backend.get(), proc, input,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_AllMaxima"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] AllMaxima benchmark complete\n";
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

}  // namespace test_fft_maxima_benchmark
