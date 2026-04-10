#pragma once

/**
 * @file snr_estimator_benchmark.hpp
 * @brief SnrEstimatorBenchmark — benchmark class для SNR-estimator (SNR_09)
 *
 * Наследник GpuBenchmarkBase. Измеряет end-to-end время ComputeSnrDb()
 * через hipEvent пары → RecordROCmEvent → GPUProfiler.
 *
 * ⚠️ КОД НАПИСАН — запуск в понедельник на Debian/AMD.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "snr_test_helpers.hpp"
#include "services/gpu_benchmark_base.hpp"
#include "services/console_output.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>

#include <complex>
#include <string>
#include <vector>

namespace test_snr_estimator {

/**
 * @brief SnrEstimatorBenchmark — measures ComputeSnrDb() end-to-end.
 *
 * Pre-allocated GPU input per instance (так benchmark measures
 * чистую работу SnrEstimatorOp pipeline, без upload от каждой итерации).
 */
class SnrEstimatorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  SnrEstimatorBenchmark(
      drv_gpu_lib::IBackend* backend,
      statistics::StatisticsProcessor& proc,
      void* gpu_input,                                  // pre-allocated, owned by caller
      uint32_t n_antennas,
      uint32_t n_samples,
      const statistics::SnrEstimationConfig& cfg,
      std::string instance_name,
      GpuBenchmarkBase::Config bench_cfg = {
          .n_warmup   = 3,
          .n_runs     = 10,
          .output_dir = "Results/Profiler/GPU_00_SNR_Estimator_ROCm"})
    : GpuBenchmarkBase(backend, instance_name, bench_cfg),
      proc_(proc),
      gpu_input_(gpu_input),
      n_antennas_(n_antennas),
      n_samples_(n_samples),
      cfg_(cfg) {}

protected:
  /// Warmup — один вызов, без measurement
  void ExecuteKernel() override {
    (void)proc_.ComputeSnrDb(gpu_input_, n_antennas_, n_samples_, cfg_);
  }

  /// Timed — записываем e2e время через hipEvent пару + RecordROCmEvent
  void ExecuteKernelTimed() override {
    hipEvent_t ev_start = nullptr, ev_end = nullptr;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_end);

    hipEventRecord(ev_start, 0);  // stream 0 (default)
    (void)proc_.ComputeSnrDb(gpu_input_, n_antennas_, n_samples_, cfg_);
    hipEventRecord(ev_end, 0);

    hipEventSynchronize(ev_end);

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);

    // Вручную заполним ROCmProfilingData
    drv_gpu_lib::ROCmProfilingData data;
    data.execution_time_ms = elapsed_ms;
    data.transfer_time_ms  = 0.0f;
    data.description       = "ComputeSnrDb total (gather → FFT|X|² → CFAR → median)";

    RecordROCmEvent("ComputeSnrDb_total", data);

    hipEventDestroy(ev_start);
    hipEventDestroy(ev_end);
  }

private:
  statistics::StatisticsProcessor&    proc_;
  void*                                gpu_input_ = nullptr;
  uint32_t                             n_antennas_ = 0;
  uint32_t                             n_samples_  = 0;
  statistics::SnrEstimationConfig      cfg_;
};

}  // namespace test_snr_estimator

#endif  // ENABLE_ROCM
