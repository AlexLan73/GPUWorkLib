#pragma once

/**
 * @file test_fft_benchmark.hpp
 * @brief Бенчмарк FFTProcessor через GpuBenchmarkBase
 *
 * Запускает:
 *  - 5 прогревочных FFT (без замеров — данные warmup сбрасываются)
 *  - 20 замерных FFT → GPUProfiler (min/max/avg автоматически)
 *  - profiler.PrintReport() + ExportJSON + ExportMarkdown
 *
 * Результаты в: Results/Profiler/GPU_00_FFT/
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see MemoryBank/specs/Profil_GPU.md, MemoryBank/tasks/TASK_profiling_fftprocessor.md
 */

#include "fft_processor_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_benchmark {

// ═══════════════════════════════════════════════════════════════════════════
// Утилита — генерация мультилучевых данных
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateBenchmarkData(
    size_t beam_count, size_t n_point, float sample_rate,
    float base_freq, float freq_step)
{
  std::vector<std::complex<float>> data(beam_count * n_point);
  for (size_t b = 0; b < beam_count; ++b) {
    float freq = base_freq + b * freq_step;
    for (size_t i = 0; i < n_point; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      data[b * n_point + i] = std::complex<float>(
          std::cos(phase), std::sin(phase));
    }
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// Точка входа бенчмарка
// ═══════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  FFTProcessor Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  try {
    // ── OpenCL инициализация ──────────────────────────────────────────
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

    // Создаём DrvGPU backend
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->InitializeFromExternalContext(context, device, queue);

    // ── Параметры бенчмарка ───────────────────────────────────────────
    fft_processor::FFTProcessorParams params;
    params.beam_count  = 64;
    params.n_point     = 1024;
    params.sample_rate = 1e6f;
    params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

    auto input_data = GenerateBenchmarkData(
        params.beam_count, params.n_point,
        params.sample_rate, 100e3f, 10e3f);

    // ── Создать процессор и бенчмарк ──────────────────────────────────
    fft_processor::FFTProcessor proc(backend.get());

    test_fft_processor::FFTProcessorBenchmark bench(
        backend.get(), proc, params, input_data,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "Results/Profiler/GPU_00_FFT"});

    // ── Запуск ────────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
    } else {
      bench.Run();     // warmup(5) + measure(20) → GPUProfiler
      bench.Report();  // profiler.PrintReport() + ExportJSON + ExportMarkdown
      std::cout << "  [OK] Benchmark complete\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────
    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_fft_benchmark
