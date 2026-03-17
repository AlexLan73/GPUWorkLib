#pragma once

/**
 * @file capon_benchmark.hpp
 * @brief ROCm benchmark-классы для CaponProcessor (GpuBenchmarkBase)
 *
 * CaponReliefBenchmarkROCm    → ComputeRelief()
 * CaponBeamformBenchmarkROCm  → AdaptiveBeamform()
 *
 * Замер через hipEvent: hipEventRecord до/после вызова → длительность в ms
 * → ROCmProfilingData (start_ns/end_ns) → RecordROCmEvent → GPUProfiler.
 *
 * Компилируется только при ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * Использование:
 * @code
 *   capon::CaponProcessor proc(backend);
 *   test_capon_rocm_bench::CaponReliefBenchmarkROCm bench(backend, proc, signal, steering, params);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 * @see GpuBenchmarkBase, heterodyne_benchmark_rocm.hpp
 */

#if ENABLE_ROCM

#include "capon_processor.hpp"
#include "services/gpu_benchmark_base.hpp"

#include <hip/hip_runtime.h>
#include <complex>
#include <vector>
#include <string>

namespace test_capon_rocm_bench {

// ============================================================================
// Вспомогательная функция: hipEvent timing → ROCmProfilingData
// ============================================================================

/// Создать ROCmProfilingData из длительности hipEvent (в мс → нс).
/// Использует тот же формат что MakeOpenCLFromDurationMs, но для ROCm.
inline drv_gpu_lib::ROCmProfilingData MakeROCmFromHipEvents(
    hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0f;
  hipEventSynchronize(stop);
  hipEventElapsedTime(&ms, start, stop);

  drv_gpu_lib::ROCmProfilingData d{};
  d.start_ns   = 0;
  d.end_ns     = static_cast<uint64_t>(ms * 1e6f);
  d.queued_ns  = d.submit_ns = d.complete_ns = d.end_ns;
  d.kernel_name = "capon_pipeline";
  return d;
}

// ─── Benchmark 1: CaponProcessor::ComputeRelief() ─────────────────────────

class CaponReliefBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  CaponReliefBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      capon::CaponProcessor& proc,
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const capon::CaponParams& params,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"})
    : GpuBenchmarkBase(backend, "Capon_ComputeRelief_ROCm", cfg),
      proc_(proc), signal_(signal), steering_(steering), params_(params) {}

protected:
  /// Warmup — ComputeRelief без замера
  void ExecuteKernel() override {
    proc_.ComputeRelief(signal_, steering_, params_);
  }

  /// Замер — ComputeRelief с hipEvent timing → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    hipEvent_t t_start = nullptr, t_stop = nullptr;
    hipEventCreate(&t_start);
    hipEventCreate(&t_stop);

    hipEventRecord(t_start);
    proc_.ComputeRelief(signal_, steering_, params_);
    hipEventRecord(t_stop);

    RecordROCmEvent("ComputeRelief_Total",
                    MakeROCmFromHipEvents(t_start, t_stop));

    hipEventDestroy(t_start);
    hipEventDestroy(t_stop);
  }

private:
  capon::CaponProcessor&                   proc_;
  std::vector<std::complex<float>>         signal_;
  std::vector<std::complex<float>>         steering_;
  capon::CaponParams                       params_;
};

// ─── Benchmark 2: CaponProcessor::AdaptiveBeamform() ─────────────────────

class CaponBeamformBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  CaponBeamformBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      capon::CaponProcessor& proc,
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const capon::CaponParams& params,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Capon_ROCm"})
    : GpuBenchmarkBase(backend, "Capon_AdaptiveBeamform_ROCm", cfg),
      proc_(proc), signal_(signal), steering_(steering), params_(params) {}

protected:
  /// Warmup — AdaptiveBeamform без замера
  void ExecuteKernel() override {
    proc_.AdaptiveBeamform(signal_, steering_, params_);
  }

  /// Замер — AdaptiveBeamform с hipEvent timing
  void ExecuteKernelTimed() override {
    hipEvent_t t_start = nullptr, t_stop = nullptr;
    hipEventCreate(&t_start);
    hipEventCreate(&t_stop);

    hipEventRecord(t_start);
    proc_.AdaptiveBeamform(signal_, steering_, params_);
    hipEventRecord(t_stop);

    RecordROCmEvent("AdaptiveBeamform_Total",
                    MakeROCmFromHipEvents(t_start, t_stop));

    hipEventDestroy(t_start);
    hipEventDestroy(t_stop);
  }

private:
  capon::CaponProcessor&                   proc_;
  std::vector<std::complex<float>>         signal_;
  std::vector<std::complex<float>>         steering_;
  capon::CaponParams                       params_;
};

}  // namespace test_capon_rocm_bench

#endif  // ENABLE_ROCM
