#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_avg_summary.hpp
 * @brief FM Correlator — среднее время шагов для планирования (тест 2.2)
 *
 * step1 = PrepareReference(): 20 замеров → avg → 1 Record("step1")
 * step2 = Process(inp):       20 замеров → avg → 1 Record("step2")
 * Compact summary: con.Print() с двумя строками avg
 * Назначение: знать на какое время закладываться при планировании.
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

// ── Параметры теста 2.2 — изменить для другой конфигурации ─────────────────
constexpr size_t kAvgFftSize    = 32768;
constexpr int    kAvgNumShifts  = 32;
constexpr int    kAvgNumSignals = 5;
constexpr int    kAvgNumOutPts  = 2000;
constexpr int    kAvgWarmup     = 20;
constexpr int    kAvgRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_avg_summary() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Avg", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Avg", "  FM Correlator Avg Summary (2.2)");
  con.Print(gpu_id, "FM_Avg", "  Цель: среднее время step1 и step2 для планирования");
  con.Print(gpu_id, "FM_Avg", "  warmup=" + std::to_string(kAvgWarmup) +
            "  runs=" + std::to_string(kAvgRuns));
  con.Print(gpu_id, "FM_Avg", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kAvgFftSize;
  params.num_shifts        = kAvgNumShifts;
  params.num_signals       = kAvgNumSignals;
  params.num_output_points = kAvgNumOutPts;
  corr.SetParams(params);

  std::vector<float> inp(kAvgFftSize * static_cast<size_t>(kAvgNumSignals), 1.0f);
  corr.PrepareReference();

  // ── Warmup ────────────────────────────────────────────────────────────────
  con.Print(gpu_id, "FM_Avg", "  Warmup...");
  for (int w = 0; w < kAvgWarmup; ++w) {
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

  // ── Замер step1: собираем в вектор → вычисляем avg ────────────────────────
  std::vector<double> step1_times(kAvgRuns);
  con.Print(gpu_id, "FM_Avg", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kAvgRuns; ++r) {
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
      std::accumulate(step1_times.begin(), step1_times.end(), 0.0) / kAvgRuns;

  // ── Замер step2: собираем в вектор → вычисляем avg ────────────────────────
  std::vector<double> step2_times(kAvgRuns);
  con.Print(gpu_id, "FM_Avg", "  Measuring step2 (Process)...");
  for (int r = 0; r < kAvgRuns; ++r) {
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
      std::accumulate(step2_times.begin(), step2_times.end(), 0.0) / kAvgRuns;

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Записываем 1 синтетическое событие на шаг в GPUProfiler ──────────────
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(avg_step1_ms * 1.0e6);
    pd.kernel_name = "PrepareReference_avg";
    profiler.Record(gpu_id, "FM_Avg", "step1", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(avg_step2_ms * 1.0e6);
    pd.kernel_name = "Process_avg";
    profiler.Record(gpu_id, "FM_Avg", "step2", pd);
  }

  // ── Отчёт через GPUProfiler ───────────────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_avg_summary_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  // ── Compact summary для планирования (con.Print — НЕ профилирующие данные) ─
  char buf[128];
  con.Print(gpu_id, "FM_Avg", "  ┌─────────────────────────────────────────┐");
  std::snprintf(buf, sizeof(buf), "  │ step1 (PrepareRef):  avg = %8.3f ms  │", avg_step1_ms);
  con.Print(gpu_id, "FM_Avg", buf);
  std::snprintf(buf, sizeof(buf), "  │ step2 (Process):     avg = %8.3f ms  │", avg_step2_ms);
  con.Print(gpu_id, "FM_Avg", buf);
  con.Print(gpu_id, "FM_Avg", "  └─────────────────────────────────────────┘");
  con.Print(gpu_id, "FM_Avg", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Avg", "  Avg summary complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
