#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_combined.hpp
 * @brief FM Correlator — комбинированный тест (2.3)
 *
 * Один GPUProfiler прогон, 3 синтетических события:
 *   "all_time" = avg(PrepareReference + Process) за 20 замеров
 *   "step1"    = avg(PrepareReference) за 20 замеров
 *   "step2"    = avg(Process) за 20 замеров
 *
 * Внимание: тест 2.3 отлаживать только после принятия теста 2.2.
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "fm_correlator.hpp"
#include "test_fm_benchmark_rocm_all_time.hpp"   // ← FmGetDateForFilename(), GetTestBackend()
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

namespace fm_correlator::tests {

// ── Параметры теста 2.3 — изменить для другой конфигурации ─────────────────
constexpr size_t kCmbFftSize    = 32768;
constexpr int    kCmbNumShifts  = 32;
constexpr int    kCmbNumSignals = 5;
constexpr int    kCmbNumOutPts  = 2000;
constexpr int    kCmbWarmup     = 20;
constexpr int    kCmbRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_combined() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Cmb", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Cmb", "  FM Correlator Combined Test (2.3)");
  con.Print(gpu_id, "FM_Cmb", "  all_time + step1 + step2 в одном прогоне");
  con.Print(gpu_id, "FM_Cmb", "  warmup=" + std::to_string(kCmbWarmup) +
            "  runs=" + std::to_string(kCmbRuns));
  con.Print(gpu_id, "FM_Cmb", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kCmbFftSize;
  params.num_shifts        = kCmbNumShifts;
  params.num_signals       = kCmbNumSignals;
  params.num_output_points = kCmbNumOutPts;
  corr.SetParams(params);

  std::vector<float> inp(kCmbFftSize * static_cast<size_t>(kCmbNumSignals), 1.0f);
  corr.PrepareReference();

  // ── Warmup ────────────────────────────────────────────────────────────────
  con.Print(gpu_id, "FM_Cmb", "  Warmup...");
  for (int w = 0; w < kCmbWarmup; ++w) {
    corr.PrepareReference();
    (void)corr.Process(inp);
    (void)hipDeviceSynchronize();
  }

  // ── GPUProfiler setup ─────────────────────────────────────────────────────
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
  profiler.Start();

  // ── hipEvent ──────────────────────────────────────────────────────────────
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  // ── (A) Замер all_time: { PrepareReference + Process } ───────────────────
  std::vector<double> all_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring all_time...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();
    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    all_times[r] = static_cast<double>(ms);
  }
  const double avg_all_ms =
      std::accumulate(all_times.begin(), all_times.end(), 0.0) / kCmbRuns;

  // ── (B) Замер step1: PrepareReference ─────────────────────────────────────
  std::vector<double> step1_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step1_times[r] = static_cast<double>(ms);
  }
  const double avg_step1_ms =
      std::accumulate(step1_times.begin(), step1_times.end(), 0.0) / kCmbRuns;

  // ── (C) Замер step2: Process ───────────────────────────────────────────────
  std::vector<double> step2_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring step2 (Process)...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step2_times[r] = static_cast<double>(ms);
  }
  const double avg_step2_ms =
      std::accumulate(step2_times.begin(), step2_times.end(), 0.0) / kCmbRuns;

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Записать 3 синтетических события в GPUProfiler ────────────────────────
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_all_ms * 1.0e6);
    pd.kernel_name = "PrepareReference+Process";
    profiler.Record(gpu_id, "FM_Combined", "all_time", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_step1_ms * 1.0e6);
    pd.kernel_name = "PrepareReference_avg";
    profiler.Record(gpu_id, "FM_Combined", "step1", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_step2_ms * 1.0e6);
    pd.kernel_name = "Process_avg";
    profiler.Record(gpu_id, "FM_Combined", "step2", pd);
  }

  // ── Отчёт через GPUProfiler ───────────────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_combined_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  // ── Compact summary (con.Print — НЕ профилирующие данные) ─────────────────
  char buf[128];
  con.Print(gpu_id, "FM_Cmb", "  ┌─────────────────────────────────────────┐");
  std::snprintf(buf, sizeof(buf), "  │ all_time:            avg = %8.3f ms  │", avg_all_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  std::snprintf(buf, sizeof(buf), "  │ step1 (PrepareRef):  avg = %8.3f ms  │", avg_step1_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  std::snprintf(buf, sizeof(buf), "  │ step2 (Process):     avg = %8.3f ms  │", avg_step2_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  con.Print(gpu_id, "FM_Cmb", "  └─────────────────────────────────────────┘");
  con.Print(gpu_id, "FM_Cmb", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Cmb", "  Combined test complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
