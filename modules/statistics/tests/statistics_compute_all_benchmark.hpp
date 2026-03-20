#pragma once

/**
 * @file statistics_compute_all_benchmark.hpp
 * @brief ROCm benchmark-класс для StatisticsProcessor::ComputeAll (GpuBenchmarkBase)
 *
 * ComputeAllBenchmarkROCm — times ComputeAll() (CPU path: 4 beams × 65536 points)
 * Per-event breakdown: Upload | Welford_Fused | Median
 *
 * Компилируется только при ENABLE_ROCM=1.
 *
 * Использование:
 * @code
 *   statistics::StatisticsProcessor proc(backend);
 *   test_statistics_compute_all_benchmark::ComputeAllBenchmarkROCm bench(
 *       backend, proc, params, data);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-20
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_statistics_compute_all.md
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <complex>
#include <vector>

namespace test_statistics_compute_all_benchmark {

// ─── Benchmark: StatisticsProcessor::ComputeAll() ─────────────────────────

class ComputeAllBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  ComputeAllBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      statistics::StatisticsProcessor& proc,
      const statistics::StatisticsParams& params,
      const std::vector<std::complex<float>>& data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Statistics"})
    : GpuBenchmarkBase(backend, "Statistics_ComputeAll_ROCm", cfg),
      proc_(proc), params_(params), data_(data) {}

protected:
  /// Warmup — ComputeAll без profiling events
  void ExecuteKernel() override {
    proc_.ComputeAll(data_, params_);
  }

  /// Замер — ComputeAll с StatisticsROCmProfEvents → RecordROCmEvent
  void ExecuteKernelTimed() override {
    statistics::StatisticsROCmProfEvents events;
    proc_.ComputeAll(data_, params_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  statistics::StatisticsProcessor&         proc_;
  statistics::StatisticsParams             params_;
  std::vector<std::complex<float>>         data_;
};

}  // namespace test_statistics_compute_all_benchmark

#endif  // ENABLE_ROCM
