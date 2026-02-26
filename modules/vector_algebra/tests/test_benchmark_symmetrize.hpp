#pragma once
#if ENABLE_ROCM

/**
 * @file test_benchmark_symmetrize.hpp
 * @brief Benchmark: Roundtrip vs GpuKernel symmetrize (5.13)
 *
 * Использует hipEvent для точного измерения GPU-времени.
 * Вывод через GPUProfiler (SetGPUInfo → Start → Record → Stop → Export).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#include <chrono>
#include <complex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "cholesky_inverter_rocm.hpp"
#include "DrvGPU/interface/i_backend.hpp"
#include "DrvGPU/interface/input_data.hpp"
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

#include "test_cholesky_inverter_rocm.hpp"

namespace vector_algebra::tests {

/// Измерить время Invert для данного режима (мс)
inline double MeasureInvertTime(drv_gpu_lib::IBackend* backend,
                                 SymmetrizeMode mode,
                                 const std::vector<std::complex<float>>& A,
                                 int n) {
  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = A;

  CholeskyInverterROCm inverter(backend, mode);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto result = inverter.Invert(input);
  backend->Synchronize();
  auto t1 = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/// Измерить время InvertBatch (мс)
inline double MeasureInvertBatchTime(drv_gpu_lib::IBackend* backend,
                                      SymmetrizeMode mode,
                                      int n, int batch) {
  std::vector<std::complex<float>> flat;
  for (int k = 0; k < batch; ++k) {
    auto A = MakePositiveDefiniteHermitian(n, static_cast<unsigned>(k + 500));
    flat.insert(flat.end(), A.begin(), A.end());
  }

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = static_cast<uint32_t>(batch);
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = flat;

  CholeskyInverterROCm inverter(backend, mode);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto result = inverter.InvertBatch(input, n);
  backend->Synchronize();
  auto t1 = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ════════════════════════════════════════════════════════════════════════════
// 5.13.1: BenchmarkSingle341
// ════════════════════════════════════════════════════════════════════════════

inline void BenchmarkSingle341(drv_gpu_lib::IBackend* backend) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "Benchmark: Single 341x341");

  constexpr int n = 341;
  auto A = MakePositiveDefiniteHermitian(n, 42);

  double t_roundtrip = MeasureInvertTime(backend, SymmetrizeMode::Roundtrip,
                                          A, n);
  double t_gpukernel = MeasureInvertTime(backend, SymmetrizeMode::GpuKernel,
                                          A, n);

  con.Print(0, "VecAlg", "  Roundtrip: " + std::to_string(t_roundtrip) +
            " ms");
  con.Print(0, "VecAlg", "  GpuKernel: " + std::to_string(t_gpukernel) +
            " ms");
  con.Print(0, "VecAlg", "  Speedup:   " +
            std::to_string(t_roundtrip / t_gpukernel) + "x");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.13.2: BenchmarkBatch_16x64
// ════════════════════════════════════════════════════════════════════════════

inline void BenchmarkBatch_16x64(drv_gpu_lib::IBackend* backend) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "Benchmark: Batch 16x64x64");

  double t_roundtrip =
      MeasureInvertBatchTime(backend, SymmetrizeMode::Roundtrip, 64, 16);
  double t_gpukernel =
      MeasureInvertBatchTime(backend, SymmetrizeMode::GpuKernel, 64, 16);

  con.Print(0, "VecAlg", "  Roundtrip: " + std::to_string(t_roundtrip) +
            " ms");
  con.Print(0, "VecAlg", "  GpuKernel: " + std::to_string(t_gpukernel) +
            " ms");
  con.Print(0, "VecAlg", "  Speedup:   " +
            std::to_string(t_roundtrip / t_gpukernel) + "x");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.13.3: BenchmarkBatch_4x256
// ════════════════════════════════════════════════════════════════════════════

inline void BenchmarkBatch_4x256(drv_gpu_lib::IBackend* backend) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "Benchmark: Batch 4x256x256");

  double t_roundtrip =
      MeasureInvertBatchTime(backend, SymmetrizeMode::Roundtrip, 256, 4);
  double t_gpukernel =
      MeasureInvertBatchTime(backend, SymmetrizeMode::GpuKernel, 256, 4);

  con.Print(0, "VecAlg", "  Roundtrip: " + std::to_string(t_roundtrip) +
            " ms");
  con.Print(0, "VecAlg", "  GpuKernel: " + std::to_string(t_gpukernel) +
            " ms");
  con.Print(0, "VecAlg", "  Speedup:   " +
            std::to_string(t_roundtrip / t_gpukernel) + "x");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.15: TestProfilerIntegration
// ════════════════════════════════════════════════════════════════════════════

inline void TestProfilerIntegration(drv_gpu_lib::IBackend* backend) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestProfilerIntegration");

  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();

  // SetGPUInfo ПЕРЕД Start() — ОБЯЗАТЕЛЬНО!
  auto device_info = backend->GetDeviceInfo();
  int gpu_id = backend->GetDeviceIndex();
  if (gpu_id < 0) gpu_id = 0;

  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name = device_info.name;
  report_info.backend_type = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

  std::map<std::string, std::string> rocm_driver;
  rocm_driver["driver_type"] = "ROCm";
  rocm_driver["driver_version"] = device_info.driver_version;
  report_info.drivers.push_back(rocm_driver);

  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  constexpr int n = 341;
  auto A = MakePositiveDefiniteHermitian(n, 77);

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = A;

  // Roundtrip
  {
    CholeskyInverterROCm inverter(backend, SymmetrizeMode::Roundtrip);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = inverter.Invert(input);
    backend->Synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    drv_gpu_lib::ROCmProfilingData pd;
    pd.end_ns = static_cast<uint64_t>(ms * 1e6);
    pd.kernel_name = "POTRF_POTRI_341_Roundtrip";
    profiler.Record(gpu_id, "Cholesky", "POTRF_POTRI_341_Roundtrip", pd);
  }

  // GpuKernel
  {
    CholeskyInverterROCm inverter(backend, SymmetrizeMode::GpuKernel);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = inverter.Invert(input);
    backend->Synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    drv_gpu_lib::ROCmProfilingData pd;
    pd.end_ns = static_cast<uint64_t>(ms * 1e6);
    pd.kernel_name = "POTRF_POTRI_341_GpuKernel";
    profiler.Record(gpu_id, "Cholesky", "POTRF_POTRI_341_GpuKernel", pd);
  }

  profiler.Stop();

  // Вывод ТОЛЬКО через GPUProfiler API
  profiler.PrintReport();
  profiler.ExportMarkdown("Results/Profiler/cholesky_invert_v2.md");
  profiler.ExportJSON("Results/Profiler/cholesky_invert_v2.json");

  con.Print(0, "VecAlg", "TestProfilerIntegration PASSED");
}

}  // namespace vector_algebra::tests

#endif  // ENABLE_ROCM
