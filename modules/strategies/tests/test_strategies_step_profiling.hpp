#pragma once
#if ENABLE_ROCM

/**
 * @file test_strategies_step_profiling.hpp
 * @brief AntennaProcessor — GPU profiling per pipeline step
 *
 * Measures each pipeline step independently via hipEvent (GPU hardware timer):
 *   step1 = GEMM (hipBLAS)
 *   step2 = Window+FFT+Magnitudes (hamming_pad_fused + hipFFT + compute_magnitudes)
 *   step3 = OneMax+Parabola (one_max_no_phase kernel)
 *   step4 = GlobalMinMax (global_minmax kernel)
 *   step5 = Full process() — total pipeline wall-clock
 *
 * All timing via GPUProfiler.Record() → PrintReport().
 * Pattern: fm_correlator/test_fm_step_profiling.hpp
 *
 * @date 2026-03-07
 */

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "antenna_processor_test.hpp"
#include "weight_generator.hpp"
#include "generators/form_signal_generator_rocm.hpp"

#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"
#include "backends/rocm/rocm_backend.hpp"

namespace test_strategies_profiling {

// ── Test parameters ─────────────────────────────────────────────────────────
constexpr uint32_t kProfNAnt       = 10;
constexpr uint32_t kProfNSamples   = 16384;
constexpr float    kProfSampleRate = 12.0e6f;
constexpr float    kProfF0         = 2.0e6f;
constexpr int      kProfWarmup     = 10;
constexpr int      kProfRuns       = 20;
// ─────────────────────────────────────────────────────────────────────────────

// Helper: measure one step via hipEvent, record to GPUProfiler
// Returns average GPU time in ms
inline double MeasureStep(
    const std::string& step_name,
    std::function<void()> fn,
    int warmup, int runs,
    int gpu_id,
    drv_gpu_lib::GPUProfiler& profiler)
{
  // Warmup
  for (int w = 0; w < warmup; ++w) {
    fn();
    (void)hipDeviceSynchronize();
  }

  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  using SClock = std::chrono::steady_clock;
  using NS     = std::chrono::nanoseconds;

  for (int r = 0; r < runs; ++r) {
    auto tq = SClock::now();
    (void)hipDeviceSynchronize();
    auto ts = SClock::now();
    (void)hipEventRecord(ev_start, nullptr);
    auto tk = SClock::now();

    fn();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    auto tc = SClock::now();

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);

    uint64_t ns_queue    = static_cast<uint64_t>(
        std::chrono::duration_cast<NS>(ts - tq).count());
    uint64_t ns_submit   = static_cast<uint64_t>(
        std::chrono::duration_cast<NS>(tk - ts).count());
    uint64_t ns_exec     = static_cast<uint64_t>(ms * 1.0e6f);
    double   tc_tk_ns    = static_cast<double>(
        std::chrono::duration_cast<NS>(tc - tk).count());
    uint64_t ns_complete = static_cast<uint64_t>(
        std::max(0.0, tc_tk_ns - static_cast<double>(ns_exec)));

    drv_gpu_lib::ROCmProfilingData pd{};
    pd.queued_ns   = 0;
    pd.submit_ns   = ns_queue;
    pd.start_ns    = ns_queue + ns_submit;
    pd.end_ns      = ns_queue + ns_submit + ns_exec;
    pd.complete_ns = ns_queue + ns_submit + ns_exec + ns_complete;
    pd.kernel_name = step_name;
    profiler.Record(gpu_id, "Strategies", step_name, pd);
  }

  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // Return avg exec time (last run's ms is approximate, profiler has details)
  return 0.0;  // GPUProfiler has all data
}

inline void run_step_profiling(drv_gpu_lib::IBackend* backend) {
  auto& con      = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler  = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  char prof_buf[256];
  con.Print(gpu_id, "Strat_Prof", "════════════════════════════════════════════════");
  con.Print(gpu_id, "Strat_Prof", "  AntennaProcessor Step Profiling");
  std::snprintf(prof_buf, sizeof(prof_buf), "  n_ant=%u  n_samples=%u  warmup=%d  runs=%d",
                kProfNAnt, kProfNSamples, kProfWarmup, kProfRuns);
  con.Print(gpu_id, "Strat_Prof", prof_buf);
  con.Print(gpu_id, "Strat_Prof", "════════════════════════════════════════════════");

  // ── 1. Generate test signal ──────────────────────────────────────────────
  signal_gen::FormParams fp;
  fp.antennas        = kProfNAnt;
  fp.points          = kProfNSamples;
  fp.fs              = static_cast<double>(kProfSampleRate);
  fp.f0              = static_cast<double>(kProfF0);
  fp.amplitude       = 1.0;
  fp.noise_amplitude = 0.01;
  fp.tau_base        = 0.0;
  fp.tau_step        = 100e-6;

  signal_gen::FormSignalGeneratorROCm gen(backend);
  gen.SetParams(fp);
  auto input = gen.GenerateInputData();

  std::snprintf(prof_buf, sizeof(prof_buf), "  Signal: %u ant x %u pts", fp.antennas, fp.points);
  con.Print(gpu_id, "Strat_Prof", prof_buf);

  // ── 2. Generate W matrix ─────────────────────────────────────────────────
  strategies::WeightParams wp;
  wp.n_ant    = fp.antennas;
  wp.f0       = fp.f0;
  wp.tau_base = fp.tau_base;
  wp.tau_step = fp.tau_step;

  auto W_cpu = strategies::WeightGenerator::generate_delay_and_sum(wp);
  void* d_W = strategies::WeightGenerator::upload_to_gpu(backend, W_cpu);

  // ── 3. Create processor ──────────────────────────────────────────────────
  strategies::AntennaProcessorConfig cfg;
  cfg.n_ant               = fp.antennas;
  cfg.n_samples           = fp.points;
  cfg.sample_rate         = kProfSampleRate;
  cfg.signal_frequency_hz = kProfF0;
  cfg.scenario_mode       = strategies::PostFftScenarioMode::ALL_REQUIRED;
  cfg.debug_mode          = false;  // no debug stats during profiling

  strategies::AntennaProcessorTest proc(backend, cfg);
  proc.step_0_prepare_input(input.data, d_W);

  // ── 4. GPUProfiler setup ─────────────────────────────────────────────────
  auto dev = backend->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = dev.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = dev.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv_map;
  drv_map["driver_type"]    = "ROCm";
  drv_map["driver_version"] = dev.driver_version;
  report_info.drivers.push_back(drv_map);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Reset();
  profiler.Start();

  // ── 5. Measure individual steps ──────────────────────────────────────────

  // step1: GEMM only
  con.Print(gpu_id, "Strat_Prof", "  Measuring step1 (GEMM)...");
  MeasureStep("step1_GEMM", [&]() {
    proc.step_2_gemm();
  }, kProfWarmup, kProfRuns, gpu_id, profiler);

  // step2: Window + FFT + Magnitudes
  // Need GEMM output in d_X first
  proc.step_2_gemm();
  hipDeviceSynchronize();
  con.Print(gpu_id, "Strat_Prof", "  Measuring step2 (Window+FFT+Magnitudes)...");
  MeasureStep("step2_WindowFFT", [&]() {
    proc.step_4_window_fft();
  }, kProfWarmup, kProfRuns, gpu_id, profiler);

  // step3: OneMax+Parabola
  // Need magnitudes computed
  proc.step_4_window_fft();
  hipDeviceSynchronize();
  con.Print(gpu_id, "Strat_Prof", "  Measuring step3 (OneMax+Parabola)...");
  MeasureStep("step3_OneMax", [&]() {
    proc.step_6_1_one_max_parabola();
  }, kProfWarmup, kProfRuns, gpu_id, profiler);

  // step4: GlobalMinMax
  con.Print(gpu_id, "Strat_Prof", "  Measuring step4 (GlobalMinMax)...");
  MeasureStep("step4_MinMax", [&]() {
    proc.step_6_3_global_minmax();
  }, kProfWarmup, kProfRuns, gpu_id, profiler);

  // ── 6. Measure full pipeline ─────────────────────────────────────────────
  con.Print(gpu_id, "Strat_Prof", "  Measuring step5 (Full process)...");
  MeasureStep("step5_FullProcess", [&]() {
    proc.process_full();
  }, kProfWarmup, kProfRuns, gpu_id, profiler);

  // ── 7. Report (ONLY via GPUProfiler) ─────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  // ── 8. Cleanup ───────────────────────────────────────────────────────────
  backend->Free(d_W);
  hipFree(input.data);

  con.Print(gpu_id, "Strat_Prof", "  Step profiling complete");
}

}  // namespace test_strategies_profiling

#endif  // ENABLE_ROCM
