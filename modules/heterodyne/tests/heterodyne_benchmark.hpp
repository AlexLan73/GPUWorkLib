#pragma once

/**
 * @file heterodyne_benchmark.hpp
 * @brief OpenCL benchmark-классы для HeterodyneProcessorOpenCL (GpuBenchmarkBase)
 *
 * HeterodyneDechirpBenchmark  → Dechirp():  Upload_Rx, Upload_Ref, Kernel_Multiply, Download
 * HeterodyneCorrectBenchmark  → Correct():  Upload_DC, Upload_PhaseStep, Kernel_Correct, Download
 *
 * Использование:
 * @code
 *   drv_gpu_lib::HeterodyneProcessorOpenCL proc(backend);
 *   test_heterodyne_opencl::HeterodyneDechirpBenchmark bench(backend, proc, params, rx, ref);
 *   bench.Run();
 *   bench.Report();
 * @endcode
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_heterodyne_profiling.md
 */

#include "processors/heterodyne_processor_opencl.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

#include <complex>
#include <vector>

namespace test_heterodyne_opencl {

// ─── Benchmark 1: HeterodyneProcessorOpenCL::Dechirp() ────────────────────

class HeterodyneDechirpBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneDechirpBenchmark(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorOpenCL& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Heterodyne"})
    : GpuBenchmarkBase(backend, "Heterodyne_Dechirp", cfg),
      proc_(proc), params_(params), rx_data_(rx_data), ref_data_(ref_data) {}

protected:
  /// Warmup — Dechirp без prof_events
  void ExecuteKernel() override {
    proc_.Dechirp(rx_data_, ref_data_, params_);
  }

  /// Замер — Dechirp с HeterodyneOCLProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneOCLProfEvents events;
    proc_.Dechirp(rx_data_, ref_data_, params_, &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  drv_gpu_lib::HeterodyneProcessorOpenCL& proc_;
  drv_gpu_lib::HeterodyneParams           params_;
  std::vector<std::complex<float>>        rx_data_;
  std::vector<std::complex<float>>        ref_data_;
};

// ─── Benchmark 2: HeterodyneProcessorOpenCL::Correct() ────────────────────

class HeterodyneCorrectBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  HeterodyneCorrectBenchmark(
      drv_gpu_lib::IBackend* backend,
      drv_gpu_lib::HeterodyneProcessorOpenCL& proc,
      const drv_gpu_lib::HeterodyneParams& params,
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_Heterodyne"})
    : GpuBenchmarkBase(backend, "Heterodyne_Correct", cfg),
      proc_(proc), params_(params), dc_data_(dc_data), f_beat_hz_(f_beat_hz) {}

protected:
  /// Warmup — Correct без prof_events
  void ExecuteKernel() override {
    proc_.Correct(dc_data_, f_beat_hz_, params_);
  }

  /// Замер — Correct с HeterodyneOCLProfEvents → RecordEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    drv_gpu_lib::HeterodyneOCLProfEvents events;
    proc_.Correct(dc_data_, f_beat_hz_, params_, &events);
    for (auto& [name, ev] : events)
      RecordEvent(name, ev);
  }

private:
  drv_gpu_lib::HeterodyneProcessorOpenCL& proc_;
  drv_gpu_lib::HeterodyneParams           params_;
  std::vector<std::complex<float>>        dc_data_;
  std::vector<float>                      f_beat_hz_;
};

}  // namespace test_heterodyne_opencl
