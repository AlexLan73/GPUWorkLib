#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_step_profiling.hpp
 * @brief FM Correlator — детальное профилирование по шагам (тест 2.1)
 *
 * step1 = PrepareReference(): 20 прогревов + 20 замеров → 20 Record("step1")
 * step2 = Process(inp):       20 прогревов + 20 замеров → 20 Record("step2")
 * Вывод: profiler.PrintReport() + ExportJSON + ExportMarkdown
 */

#include <algorithm>
#include <chrono>
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

// ── Параметры теста 2.1 — изменить для другой конфигурации ─────────────────
constexpr size_t kSpFftSize    = 32768;
constexpr int    kSpNumShifts  = 32;
constexpr int    kSpNumSignals = 5;
constexpr int    kSpNumOutPts  = 2000;
constexpr int    kSpWarmup     = 20;
constexpr int    kSpRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_step_profiling() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Step", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Step", "  FM Correlator Step Profiling (2.1)");
  con.Print(gpu_id, "FM_Step", "  step1=PrepareReference  step2=Process");
  con.Print(gpu_id, "FM_Step", "  warmup=" + std::to_string(kSpWarmup) +
            "  runs=" + std::to_string(kSpRuns));
  con.Print(gpu_id, "FM_Step", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kSpFftSize;
  params.num_shifts        = kSpNumShifts;
  params.num_signals       = kSpNumSignals;
  params.num_output_points = kSpNumOutPts;
  corr.SetParams(params);

  // Плоский вектор входного сигнала: num_signals × fft_size
  std::vector<float> inp(kSpFftSize * static_cast<size_t>(kSpNumSignals), 1.0f);

  // Начальный вызов PrepareReference для корректного состояния
  corr.PrepareReference();

  // ── Warmup: kSpWarmup итераций (прогрев GPU + hiprtc compile) ────────────
  con.Print(gpu_id, "FM_Step", "  Warmup...");
  for (int w = 0; w < kSpWarmup; ++w) {
    corr.PrepareReference();
    (void)corr.Process(inp);
    (void)hipDeviceSynchronize();
  }

  // ── GPUProfiler setup (SetGPUInfo перед Start — обязательно!) ────────────
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
  profiler.Reset();   // сброс накопленных событий от предыдущих тестов
  profiler.Start();

  // ── hipEvent (создаём один раз, переиспользуем) ───────────────────────────
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  using SClock = std::chrono::steady_clock;
  using NS     = std::chrono::nanoseconds;

  // ── Замер step1: PrepareReference — kSpRuns итераций ─────────────────────
  // Все 5 полей ROCmProfilingData:
  //   В очереди  = submit - queued  = hipDeviceSynchronize (ожидание свободного GPU)
  //   Запуск     = start  - submit  = hipEventRecord overhead (отправка маркера)
  //   Выполн.    = end    - start   = аппаратное время GPU (hipEventElapsedTime)
  //   Заверш.    = complete - end   = hipEventSynchronize overhead (CPU ждёт GPU)
  con.Print(gpu_id, "FM_Step", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kSpRuns; ++r) {
    auto tq = SClock::now();
    (void)hipDeviceSynchronize();
    auto ts = SClock::now();
    (void)hipEventRecord(ev_start, nullptr);
    auto tk = SClock::now();

    corr.PrepareReference();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    auto tc = SClock::now();

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);

    uint64_t ns_queue    = (uint64_t)std::chrono::duration_cast<NS>(ts - tq).count();
    uint64_t ns_submit   = (uint64_t)std::chrono::duration_cast<NS>(tk - ts).count();
    uint64_t ns_exec     = (uint64_t)(ms * 1.0e6f);
    double   tc_tk_ns    = (double)std::chrono::duration_cast<NS>(tc - tk).count();
    uint64_t ns_complete = (uint64_t)std::max(0.0, tc_tk_ns - (double)ns_exec);

    drv_gpu_lib::ROCmProfilingData pd{};
    pd.queued_ns   = 0;
    pd.submit_ns   = ns_queue;
    pd.start_ns    = ns_queue + ns_submit;
    pd.end_ns      = ns_queue + ns_submit + ns_exec;
    pd.complete_ns = ns_queue + ns_submit + ns_exec + ns_complete;
    pd.kernel_name = "PrepareReference";
    profiler.Record(gpu_id, "FM_Step", "step1", pd);
  }

  // ── Замер step2: Process — kSpRuns итераций ───────────────────────────────
  // Ссылка валидна после последнего PrepareReference в цикле step1.
  con.Print(gpu_id, "FM_Step", "  Measuring step2 (Process)...");
  for (int r = 0; r < kSpRuns; ++r) {
    auto tq = SClock::now();
    (void)hipDeviceSynchronize();
    auto ts = SClock::now();
    (void)hipEventRecord(ev_start, nullptr);
    auto tk = SClock::now();

    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    auto tc = SClock::now();

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);

    uint64_t ns_queue    = (uint64_t)std::chrono::duration_cast<NS>(ts - tq).count();
    uint64_t ns_submit   = (uint64_t)std::chrono::duration_cast<NS>(tk - ts).count();
    uint64_t ns_exec     = (uint64_t)(ms * 1.0e6f);
    double   tc_tk_ns    = (double)std::chrono::duration_cast<NS>(tc - tk).count();
    uint64_t ns_complete = (uint64_t)std::max(0.0, tc_tk_ns - (double)ns_exec);

    drv_gpu_lib::ROCmProfilingData pd{};
    pd.queued_ns   = 0;
    pd.submit_ns   = ns_queue;
    pd.start_ns    = ns_queue + ns_submit;
    pd.end_ns      = ns_queue + ns_submit + ns_exec;
    pd.complete_ns = ns_queue + ns_submit + ns_exec + ns_complete;
    pd.kernel_name = "Process";
    profiler.Record(gpu_id, "FM_Step", "step2", pd);
  }

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Отчёт (ТОЛЬКО через GPUProfiler) ─────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_step_profiling_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  con.Print(gpu_id, "FM_Step", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Step", "  Step profiling complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
