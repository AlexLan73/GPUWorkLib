#pragma once

/**
 * @file test_lch_farrow_benchmark.hpp
 * @brief Бенчмарк LchFarrow через GpuBenchmarkBase (OpenCL)
 *
 * Запускает:
 *  - 5 прогревочных итераций (без замеров — данные warmup сбрасываются)
 *  - 20 замерных итераций → GPUProfiler (min/max/avg автоматически)
 *  - bench.Report() → profiler.PrintReport() + ExportJSON + ExportMarkdown
 *
 * Параметры:
 *  - antennas  = 8, points = 4096, sample_rate = 1 MHz
 *  - delays    = {0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9} мкс
 *  - input_buf загружается ОДИН РАЗ (не входит в замер)
 *
 * Профилируемые стейджи: Upload_delay, Kernel
 * Результаты: Results/Profiler/GPU_00_LchFarrow/
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see lch_farrow_benchmark.hpp, GpuBenchmarkBase
 */

#include "lch_farrow_benchmark.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"

#include <CL/cl.h>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_lch_farrow_benchmark {

// ═══════════════════════════════════════════════════════════════════════════
// Утилита — генерация CW сигнала (flat: antennas × points)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateBenchmarkData(
    uint32_t antennas, uint32_t points, float fs, float freq)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(antennas) * points);
  for (uint32_t a = 0; a < antennas; ++a) {
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / fs;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      data[a * points + n] = std::complex<float>(
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
  std::cout << "  LchFarrow Benchmark (GpuBenchmarkBase, OpenCL)\n";
  std::cout << "============================================================\n";

  try {
    // ── OpenCL инициализация ─────────────────────────────────────────
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

    // ── SetGPUInfo для profiler ──────────────────────────────────────
    // Вызвать ДО bench.Run() — чтобы в отчёте был GPU name + драйвер
    {
      auto device_info = backend->GetDeviceInfo();
      drv_gpu_lib::GPUReportInfo gpu_info;
      gpu_info.gpu_name = device_info.name.empty() ? "Unknown" : device_info.name;
      gpu_info.backend_type = drv_gpu_lib::BackendType::OPENCL;
      gpu_info.global_mem_mb = static_cast<uint64_t>(
          device_info.global_memory_size / (1024 * 1024));
      std::map<std::string, std::string> driver_map;
      driver_map["driver_type"]    = "OpenCL";
      driver_map["version"]        = device_info.opencl_version;
      driver_map["driver_version"] = device_info.driver_version;
      driver_map["vendor"]         = device_info.vendor;
      gpu_info.drivers.push_back(driver_map);
      drv_gpu_lib::GPUProfiler::GetInstance().SetGPUInfo(0, gpu_info);
    }

    // ── Параметры бенчмарка ──────────────────────────────────────────
    const uint32_t antennas    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 1e6f;
    const float    freq        = 50e3f;

    std::vector<float> delays = {0.3f, 1.7f, 2.1f, 3.5f, 4.0f, 5.3f, 6.7f, 7.9f};

    auto cpu_data = GenerateBenchmarkData(antennas, points, sample_rate, freq);

    // ── Загрузить входные данные на GPU (один раз, вне замера) ───────
    size_t data_size = cpu_data.size() * sizeof(std::complex<float>);
    cl_mem input_buf = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        data_size,
        cpu_data.data(),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error(
          "clCreateBuffer(input) failed: " + std::to_string(err));

    // ── Создать LchFarrow ────────────────────────────────────────────
    lch_farrow::LchFarrow proc(backend.get());
    proc.SetDelays(delays);
    proc.SetSampleRate(sample_rate);

    // ── Создать бенчмарк ─────────────────────────────────────────────
    test_lch_farrow::LchFarrowBenchmark bench(
        backend.get(), proc, input_buf, antennas, points,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "Results/Profiler/GPU_00_LchFarrow"});

    // ── Запуск ───────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
    } else {
      bench.Run();     // warmup(5) + measure(20) → GPUProfiler
      bench.Report();  // profiler.PrintReport() + ExportJSON + ExportMarkdown
      std::cout << "  [OK] Benchmark complete\n";
    }

    // ── Cleanup ──────────────────────────────────────────────────────
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

}  // namespace test_lch_farrow_benchmark
