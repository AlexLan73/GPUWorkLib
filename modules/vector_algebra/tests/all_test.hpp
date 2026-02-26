#pragma once

/**
 * @file all_test.hpp
 * @brief Точка вызова всех тестов модуля vector_algebra (Task_11 v2)
 *
 * Запускает все тесты в обоих режимах: Roundtrip и GpuKernel.
 * Создаёт ROCm backend внутри, как statistics.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#if ENABLE_ROCM
#include "test_cholesky_inverter_rocm.hpp"
#include "test_cross_backend_conversion.hpp"
#include "test_benchmark_symmetrize.hpp"
#include "test_stage_profiling.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "backends/rocm/rocm_backend.hpp"
#endif

namespace vector_algebra_all_test {

inline void run() {
#if ENABLE_ROCM
  using namespace vector_algebra;
  using namespace vector_algebra::tests;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    con.Print(0, "VecAlg", "[!] No ROCm devices found -- skipping vector_algebra tests");
    return;
  }

  // Create ROCm backend on device 0
  drv_gpu_lib::ROCmBackend rocm;
  rocm.Initialize(0);
  drv_gpu_lib::IBackend* backend = &rocm;

  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
  con.Print(0, "VecAlg", "  vector_algebra tests (Task_11 v2)");
  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");

  // --- Утилиты (без GPU) ---
  TestResolveMatrixSize();

  // --- Core: Roundtrip mode ---
  con.Print(0, "VecAlg", "─── Roundtrip mode ───");
  TestCpuIdentity(backend, SymmetrizeMode::Roundtrip);
  TestCpu341(backend, SymmetrizeMode::Roundtrip);
  TestGpuVoidPtr341(backend, SymmetrizeMode::Roundtrip);
  TestZeroCopyClMem(backend, SymmetrizeMode::Roundtrip);
  TestBatchCpu_4x64(backend, SymmetrizeMode::Roundtrip);
  TestBatchGpu_4x64(backend, SymmetrizeMode::Roundtrip);
  TestBatchSizes(backend, SymmetrizeMode::Roundtrip);
  TestMatrixSizes(backend, SymmetrizeMode::Roundtrip);
  TestResultAccess(backend, SymmetrizeMode::Roundtrip);

  // --- Core: GpuKernel mode ---
  con.Print(0, "VecAlg", "─── GpuKernel mode ───");
  TestCpuIdentity(backend, SymmetrizeMode::GpuKernel);
  TestCpu341(backend, SymmetrizeMode::GpuKernel);
  TestGpuVoidPtr341(backend, SymmetrizeMode::GpuKernel);
  TestZeroCopyClMem(backend, SymmetrizeMode::GpuKernel);
  TestBatchCpu_4x64(backend, SymmetrizeMode::GpuKernel);
  TestBatchGpu_4x64(backend, SymmetrizeMode::GpuKernel);
  TestBatchSizes(backend, SymmetrizeMode::GpuKernel);
  TestMatrixSizes(backend, SymmetrizeMode::GpuKernel);
  TestResultAccess(backend, SymmetrizeMode::GpuKernel);

  // --- Cross-backend 85×85 ---
  con.Print(0, "VecAlg", "─── Cross-backend 85×85 ───");
  TestConvert_VectorInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_VectorInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_HipInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_HipInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_ClMemInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_ClMemInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_OutputFormats(backend, SymmetrizeMode::Roundtrip);
  TestConvert_OutputFormats(backend, SymmetrizeMode::GpuKernel);

  // --- Stage Profiling (Task_12) ---
  TestStageProfiling(backend);

  // --- Profiler ---
  TestProfilerIntegration(backend);

  // --- Benchmark (hipEvent GPU timing, MD report) ---
  RunComprehensiveBenchmark(backend);

  rocm.Cleanup();

  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
  con.Print(0, "VecAlg", "  vector_algebra: ALL TESTS PASSED");
  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
#endif  // ENABLE_ROCM
}

}  // namespace vector_algebra_all_test
