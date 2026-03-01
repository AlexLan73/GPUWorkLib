#pragma once

/**
 * @file test_filters_benchmark.hpp
 * @brief Test runner: FirFilter + IirFilter OpenCL benchmark (GpuBenchmarkBase)
 *
 * Запускает два бенчмарка:
 *  Benchmark 1: FirFilter::Process(cl_mem)  → Results/Profiler/GPU_00_FirFilter/
 *  Benchmark 2: IirFilter::Process(cl_mem)  → Results/Profiler/GPU_00_IirFilter/
 *
 * Каждый: 5 прогревочных прогонов + 20 замерных → GPUProfiler (min/max/avg).
 * Если is_prof=false в configGPU.json — выводит [SKIP] и не падает.
 *
 * ⚠️ OpenCL queue создаётся с CL_QUEUE_PROFILING_ENABLE — обязательно для cl_event timing!
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see filters_benchmark.hpp, MemoryBank/tasks/TASK_filters_profiling.md
 */

#include "filters_benchmark.hpp"
#include "test_fir_basic.hpp"   // kTestFirCoeffs64
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

namespace test_filters_benchmark {

// ─── Утилита: генерация тестового сигнала ─────────────────────────────────────

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t channels, uint32_t points, float sample_rate)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(channels) * points);
  for (uint32_t ch = 0; ch < channels; ++ch) {
    float freq = 100.0f + ch * 500.0f;  // разные частоты на каждый канал
    for (uint32_t t = 0; t < points; ++t) {
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
      data[static_cast<size_t>(ch) * points + t] = {std::cos(phase), std::sin(phase)};
    }
  }
  return data;
}

// ─── Точка входа ──────────────────────────────────────────────────────────────

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Filters Benchmark (FirFilter + IirFilter, GpuBenchmarkBase)\n"
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

    // ── Параметры ─────────────────────────────────────────────────────────
    constexpr uint32_t CHANNELS    = 8;
    constexpr uint32_t POINTS      = 4096;
    constexpr float    SAMPLE_RATE = 50000.0f;

    // ── Входной буфер (cl_mem) ────────────────────────────────────────────
    auto signal = GenerateSignal(CHANNELS, POINTS, SAMPLE_RATE);
    size_t buf_size = signal.size() * sizeof(std::complex<float>);

    cl_mem input_buf = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        buf_size, signal.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      throw std::runtime_error("clCreateBuffer(input) failed: " + std::to_string(err));
    }

    // ── Benchmark 1: FirFilter ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 1: FirFilter::Process (64-tap LP FIR) ---\n";
    {
      filters::FirFilter fir(backend.get());
      fir.SetCoefficients(filters::tests::kTestFirCoeffs64);

      test_filters::FirFilterBenchmark bench(
          backend.get(), fir, input_buf, CHANNELS, POINTS,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_FirFilter"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] FirFilter benchmark complete\n";
      }
    }

    // ── Benchmark 2: IirFilter ────────────────────────────────────────────
    std::cout << "\n--- Benchmark 2: IirFilter::Process (Butterworth 2nd order LP) ---\n";
    {
      filters::IirFilter iir(backend.get());

      // Butterworth 2nd order LP, fc=0.1 (normalized)
      // scipy: butter(2, 0.1, output='sos')
      filters::BiquadSection sec0;
      sec0.b0 =  0.02008337f;
      sec0.b1 =  0.04016673f;
      sec0.b2 =  0.02008337f;
      sec0.a1 = -1.56101808f;
      sec0.a2 =  0.64135154f;

      iir.SetBiquadSections({sec0});

      test_filters::IirFilterBenchmark bench(
          backend.get(), iir, input_buf, CHANNELS, POINTS,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_IirFilter"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] IirFilter benchmark complete\n";
      }
    }

    // ── Cleanup ────────────────────────────────────────────────────────────
    clReleaseMemObject(input_buf);
    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_filters_benchmark
