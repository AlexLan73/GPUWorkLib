#pragma once

/**
 * @file heterodyne_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для HeterodyneProcessorROCm (GpuBenchmarkBase)
 *
 * HeterodyneDechirpBenchmarkROCm → Dechirp():  Upload_Rx, Upload_Ref, Kernel_Multiply, Download
 * HeterodyneCorrectBenchmarkROCm → Correct():  Upload_DC, Upload_PhaseStep, Kernel_Correct, Download
 *
 * Компилируется только при ENABLE_ROCM=1 (Linux + AMD GPU).
 * На Windows без AMD GPU: compile-only, не выполняется.
 *
 * Использование:
 * @code
 *   drv_gpu_lib::HeterodyneProcessorROCm proc(backend);
 *   test_heterodyne_rocm::HeterodyneDechirpBenchmarkROCm bench(backend, proc, params, rx, ref);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_heterodyne_profiling.md
 */

#if ENABLE_ROCM

#include "processors/heterodyne_processor_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <hip/hip_runtime.h>
#include <complex>
#include <vector>

namespace test_heterodyne_rocm {

// ─── Benchmark 1: HeterodyneProcessorROCm::Dechirp() ──────────────────────

class HeterodyneDechirpBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneDechirpBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorROCm& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"})
    : GpuBenchmarkBase(backend, "Heterodyne_Dechirp_ROCm", cfg),
      proc_(proc), params_(params), rx_data_(rx_data), ref_data_(ref_data) {}

protected:
  /// Warmup — Dechirp без prof_events
  void ExecuteKernel() override {
    proc_.Dechirp(rx_data_, ref_data_, params_);
  }

  /// Замер — Dechirp с HeterodyneROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneROCmProfEvents events;
    proc_.Dechirp(rx_data_, ref_data_, params_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  drv_gpu_lib::HeterodyneProcessorROCm&    proc_;
  drv_gpu_lib::HeterodyneParams            params_;
  std::vector<std::complex<float>>         rx_data_;
  std::vector<std::complex<float>>         ref_data_;
};

// ─── Benchmark 2: HeterodyneProcessorROCm::Correct() ──────────────────────

class HeterodyneCorrectBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneCorrectBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorROCm& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Heterodyne_ROCm"})
    : GpuBenchmarkBase(backend, "Heterodyne_Correct_ROCm", cfg),
      proc_(proc), params_(params), dc_data_(dc_data), f_beat_hz_(f_beat_hz) {}

protected:
  /// Warmup — Correct без prof_events
  void ExecuteKernel() override {
    proc_.Correct(dc_data_, f_beat_hz_, params_);
  }

  /// Замер — Correct с HeterodyneROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneROCmProfEvents events;
    proc_.Correct(dc_data_, f_beat_hz_, params_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  drv_gpu_lib::HeterodyneProcessorROCm&    proc_;
  drv_gpu_lib::HeterodyneParams            params_;
  std::vector<std::complex<float>>         dc_data_;
  std::vector<float>                       f_beat_hz_;
};

}  // namespace test_heterodyne_rocm

#endif  // ENABLE_ROCM
