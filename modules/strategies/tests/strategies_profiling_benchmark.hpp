#pragma once

/**
 * @file strategies_profiling_benchmark.hpp
 * @brief StrategiesProfilingBenchmark — GPU profiling per Step (T3)
 *
 * Измеряет время каждого шага pipeline через hipEvent + GPUProfiler.
 * Реализует паттерн GpuBenchmarkBase (Template Method) для ROCm.
 *
 * Прогоны:
 *   n_warmup прогревочных итераций (без записи)
 *   n_runs   замерных итераций → GPUProfiler::Record()
 *
 * Вывод ТОЛЬКО через GPUProfiler:
 *   profiler.PrintReport()
 *   profiler.ExportMarkdown("Results/Profiler/strategies/...")
 *   profiler.ExportJSON("Results/Profiler/strategies/...")
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "strategy_test_base.hpp"
#include "antenna_processor_test.hpp"
#include "signal_strategy_factory.hpp"

#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>
#include <functional>
#include <chrono>
#include <algorithm>
#include <string>
#include <cstdio>

namespace test_strategies {

// ─────────────────────────────────────────────────────────────────────────────
// ProfilingConfig — параметры профилирования
// ─────────────────────────────────────────────────────────────────────────────

struct ProfilingConfig {
  int         n_warmup   = 10;
  int         n_runs     = 20;
  std::string output_dir = "Results/Profiler/strategies";
};

// ─────────────────────────────────────────────────────────────────────────────
// StrategiesProfilingBenchmark — профилирование per Step
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief GPU profiling каждого шага AntennaProcessor pipeline
 *
 * Использует тот же подход что и test_strategies_step_profiling.hpp,
 * но параметризован через StrategyTestBase (Template Method + Strategy).
 *
 * Шаги:
 *   "GEMM"       — hipBLAS Cgemm
 *   "WindowFFT"  — Hamming + hipFFT + magnitudes
 *   "OneMax"     — one_max_no_phase kernel
 *   "AllMaxima"  — AllMaximaPipelineROCm
 *   "MinMax"     — global_minmax kernel
 *   "FullProcess" — полный pipeline (суммарное время)
 */
class StrategiesProfilingBenchmark : public StrategyTestBase {
public:
  StrategiesProfilingBenchmark(drv_gpu_lib::IBackend* backend,
                                std::unique_ptr<ISignalStrategy> signal_strategy,
                                const AntennaTestParams& params,
                                ProfilingConfig prof_cfg = {})
    : StrategyTestBase(backend, std::move(signal_strategy), params),
      prof_cfg_(prof_cfg) {}

  std::string GetName() const override { return "StrategiesProfilingBenchmark"; }

protected:
  void Execute() override {
    auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const int g   = 0;

    // ── Конфиг: disable debug stats во время профилирования ──────────────
    cfg_.debug_mode = false;
    cfg_.pre_input_stats = strategies::StatPreset::P62_MEAN_MED;  // лёгкая статистика
    cfg_.post_gemm_stats = strategies::StatPreset::P62_MEAN_MED;
    cfg_.post_fft_stats  = strategies::StatPreset::P62_MEAN_MED;

    strategies::AntennaProcessorTest proc(backend_, cfg_);
    proc.step_0_prepare_input(d_S_, d_W_);

    // ── GPUProfiler setup ────────────────────────────────────────────────
    auto dev = backend_->GetDeviceInfo();
    drv_gpu_lib::GPUReportInfo ri;
    ri.gpu_name      = dev.name;
    ri.backend_type  = drv_gpu_lib::BackendType::ROCm;
    ri.global_mem_mb = dev.global_memory_size / (1024 * 1024);
    std::map<std::string, std::string> drv;
    drv["driver_type"]    = "ROCm";
    drv["driver_version"] = dev.driver_version;
    ri.drivers.push_back(drv);
    profiler.SetGPUInfo(g, ri);
    profiler.Reset();
    profiler.Start();

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  Profiling: n_ant=%u n_samples=%u warmup=%d runs=%d signal=%s",
        params_.n_ant, params_.n_samples,
        prof_cfg_.n_warmup, prof_cfg_.n_runs,
        signal_strategy_name_.c_str());
    con.Print(g, "ProfBench", buf);

    // ── Вспомогательная лямбда: прогрев + замер шага ─────────────────────
    auto measure = [&](const std::string& name,
                       std::function<void()> prepare,
                       std::function<void()> step) {
      // Подготовка (не входит в замер)
      if (prepare) { prepare(); hipDeviceSynchronize(); }

      // Прогрев
      for (int w = 0; w < prof_cfg_.n_warmup; ++w) {
        step();
        hipDeviceSynchronize();
      }

      // Замер
      hipEvent_t ev_start, ev_stop;
      hipEventCreate(&ev_start);
      hipEventCreate(&ev_stop);

      using SClock = std::chrono::steady_clock;
      using NS     = std::chrono::nanoseconds;

      for (int r = 0; r < prof_cfg_.n_runs; ++r) {
        hipDeviceSynchronize();
        auto t0 = SClock::now();
        hipEventRecord(ev_start, nullptr);
        step();
        hipEventRecord(ev_stop, nullptr);
        hipEventSynchronize(ev_stop);
        auto t1 = SClock::now();

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev_start, ev_stop);

        uint64_t ns_exec = static_cast<uint64_t>(ms * 1.0e6f);
        uint64_t ns_wall = static_cast<uint64_t>(
            std::chrono::duration_cast<NS>(t1 - t0).count());
        uint64_t ns_overhead = (ns_wall > ns_exec) ? ns_wall - ns_exec : 0;

        drv_gpu_lib::ROCmProfilingData pd{};
        pd.queued_ns   = 0;
        pd.submit_ns   = 0;
        pd.start_ns    = 0;
        pd.end_ns      = ns_exec;
        pd.complete_ns = ns_exec + ns_overhead;
        pd.kernel_name = name;
        profiler.Record(g, "Strategies", name, pd);
      }

      hipEventDestroy(ev_start);
      hipEventDestroy(ev_stop);
      con.Print(g, "ProfBench", ("  measured: " + name).c_str());
    };

    // ── Измеряем каждый шаг ───────────────────────────────────────────────

    measure("GEMM", nullptr,
        [&]() { proc.step_2_gemm(); });

    measure("WindowFFT",
        [&]() { proc.step_2_gemm(); },       // подготовка: нужен d_X
        [&]() { proc.step_4_window_fft(); });

    measure("OneMax",
        [&]() { proc.step_2_gemm(); proc.step_4_window_fft(); },
        [&]() { proc.step_6_1_one_max_parabola(); });

    measure("AllMaxima",
        [&]() { proc.step_2_gemm(); proc.step_4_window_fft(); },
        [&]() { proc.step_6_2_all_maxima(); });

    measure("MinMax",
        [&]() { proc.step_2_gemm(); proc.step_4_window_fft(); },
        [&]() { proc.step_6_3_global_minmax(); });

    measure("FullProcess", nullptr,
        [&]() { proc.process_full(); });

    // ── Отчёт ТОЛЬКО через GPUProfiler ──────────────────────────────────
    profiler.Stop();
    profiler.PrintReport();
    profiler.ExportMarkdown(prof_cfg_.output_dir);
    profiler.ExportJSON(prof_cfg_.output_dir);
  }

  void Validate() override {
    // Профилирование — нет assert'ов, только вывод отчёта
  }

private:
  ProfilingConfig prof_cfg_;
  std::string     signal_strategy_name_ = "unknown";
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
