#pragma once

/**
 * @file test_range_angle_benchmark.hpp
 * @brief Benchmark test for RangeAngleProcessor (pipeline latency via GPUProfiler)
 *
 * Measures end-to-end Process() latency for small (8×8×50K) configuration.
 * Output via GPUProfiler::PrintReport() only.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "range_angle_processor.hpp"
#include "interface/i_backend.hpp"
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

#include <complex>
#include <vector>
#include <string>
#include <map>
#include <chrono>

namespace range_angle_benchmark_test {

inline void run(drv_gpu_lib::IBackend* backend) {
  auto& con      = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  int   gpu_id   = backend ? backend->GetDeviceIndex() : 0;

  con.Print(gpu_id, "RangeAngle", "=== RangeAngle Benchmark (8x8x50K) ===");

  if (!backend) {
    con.Print(gpu_id, "RangeAngle", "SKIP: backend is null");
    return;
  }

  range_angle::RangeAngleParams p;
  p.n_ant_az    = 8;
  p.n_ant_el    = 8;
  p.n_samples   = 50000;
  p.f_start     = -5e6f;
  p.f_end       = +5e6f;
  p.sample_rate = 12e6f;
  p.carrier_freq    = 435e6f;
  p.antenna_spacing = 0.345f;

  range_angle::RangeAngleProcessor proc(backend);
  proc.SetParams(p);

  uint32_t n_ant = p.get_n_antennas();
  std::vector<std::complex<float>> data(
      static_cast<size_t>(n_ant) * p.n_samples, {1.f, 0.f});

  // Warmup: JIT compile + buffer alloc
  proc.Process(data, false);

  // Build GPU info for profiler report header
  // IMPORTANT: SetGPUInfo BEFORE Start()
  {
    auto dev_info = backend->GetDeviceInfo();
    drv_gpu_lib::GPUReportInfo report_info;
    report_info.gpu_name      = dev_info.name.empty() ? "Unknown" : dev_info.name;
    report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
    report_info.global_mem_mb = static_cast<size_t>(
        backend->GetGlobalMemorySize() / (1024u * 1024u));

    std::map<std::string, std::string> drv;
    drv["driver_type"]    = "ROCm";
    drv["driver_version"] = dev_info.driver_version;
    report_info.drivers.push_back(drv);

    profiler.SetGPUInfo(gpu_id, report_info);
  }

  profiler.Start();

  static constexpr int kRuns = 5;
  for (int i = 0; i < kRuns; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    proc.Process(data, false);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    drv_gpu_lib::ROCmProfilingData pd;
    pd.end_ns     = static_cast<uint64_t>(ms * 1e6);
    pd.kernel_name = "RangeAngle_Process_" + std::to_string(i);
    profiler.Record(gpu_id, "RangeAngle", "Process_" + std::to_string(i), pd);
  }

  profiler.Stop();
  profiler.PrintReport();

  con.Print(gpu_id, "RangeAngle", "=== RangeAngle Benchmark Done ===");
}

// Legacy run() without args
inline void run() {
  // NOTE: backend must be injected via run(backend) overload
}

}  // namespace range_angle_benchmark_test

#endif  // ENABLE_ROCM
