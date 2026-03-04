#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_benchmark_rocm.hpp
 * @brief Benchmark for FM Correlator (hipEvent GPU timing)
 *
 * Methodology (same as vector_algebra benchmark):
 *   - Warmup: kWarmupRuns iterations (GPU warm + hiprtc compile)
 *   - Measurement: kBenchmarkRuns iterations with hipEvent (GPU hardware timer)
 *   - Metrics: avg_ms, min_ms, max_ms per config
 *   - GPUProfiler: one Record per config (event name = "N%d_K%d_S%d")
 *   - Custom Markdown report: system info + full parametric table
 *   - Parametric sweep: fft_size × num_shifts × num_signals
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <unistd.h>
#include <sys/utsname.h>

#include "fm_correlator.hpp"
#include "interface/i_backend.hpp"
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"
#include "backends/rocm/rocm_backend.hpp"

namespace fm_correlator::tests {

constexpr int kWarmupRuns    = 3;
constexpr int kBenchmarkRuns = 20;

// ════════════════════════════════════════════════════════════════════════════
// BenchStats
// ════════════════════════════════════════════════════════════════════════════

struct BenchStats {
  double avg_ms = 0.0;
  double min_ms = 1e9;
  double max_ms = 0.0;
};

struct BenchConfig {
  int fft_size;
  int num_shifts;
  int num_signals;
  int num_output_points;
  BenchStats stats;
  bool failed = false;
  std::string fail_reason;
};

// ════════════════════════════════════════════════════════════════════════════
// MeasureProcessTime — hipEvent timing (warmup + N measurements)
// ════════════════════════════════════════════════════════════════════════════

inline BenchStats MeasureProcessTime(
    drv_gpu_lib::FMCorrelator& corr, int shift_step,
    int warmup = kWarmupRuns, int runs = kBenchmarkRuns) {

  // Warmup: GPU warm-up + hiprtc kernel compile (first call triggers compile)
  for (int w = 0; w < warmup; ++w) {
    corr.RunTestPattern(shift_step);
    (void)hipDeviceSynchronize();
  }

  // Create events once
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  std::vector<double> times(runs);
  for (int r = 0; r < runs; ++r) {
    (void)hipDeviceSynchronize();          // clean GPU state
    (void)hipEventRecord(ev_start, nullptr);

    corr.RunTestPattern(shift_step);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    times[r] = static_cast<double>(ms);
  }

  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  BenchStats stats;
  stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / runs;
  stats.min_ms = *std::min_element(times.begin(), times.end());
  stats.max_ms = *std::max_element(times.begin(), times.end());
  return stats;
}

// ════════════════════════════════════════════════════════════════════════════
// System info helpers (same as vector_algebra)
// ════════════════════════════════════════════════════════════════════════════

inline std::string FmGetHostname() {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) == 0) return buf;
  return "unknown";
}

inline std::string FmGetOsDistro() {
  std::ifstream f("/etc/os-release");
  if (!f.is_open()) return "unknown";
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("PRETTY_NAME=", 0) == 0) {
      auto val = line.substr(13);
      if (!val.empty() && val.front() == '"') val.erase(0, 1);
      if (!val.empty() && val.back() == '"') val.pop_back();
      return val;
    }
  }
  return "unknown";
}

inline std::string FmGetOsKernel() {
  struct utsname uts;
  if (uname(&uts) == 0)
    return std::string(uts.sysname) + " " + uts.release + " (" + uts.machine + ")";
  return "unknown";
}

inline std::string FmGetRocmVersion() {
  std::ifstream f("/opt/rocm/.info/version");
  if (f.is_open()) {
    std::string ver;
    std::getline(f, ver);
    while (!ver.empty() && (ver.back() == '\n' || ver.back() == '\r' || ver.back() == ' '))
      ver.pop_back();
    if (!ver.empty()) return ver;
  }
  int ver = 0;
  if (hipDriverGetVersion(&ver) == hipSuccess && ver > 0)
    return std::to_string(ver / 10000000) + "." +
           std::to_string((ver / 100000) % 100) + "." +
           std::to_string((ver / 100) % 1000);
  return "unknown";
}

inline std::string FmGetHipRuntimeVersion() {
  int ver = 0;
  if (hipRuntimeGetVersion(&ver) == hipSuccess && ver > 0)
    return std::to_string(ver / 10000000) + "." +
           std::to_string((ver / 100000) % 100) + "." +
           std::to_string((ver / 100) % 1000);
  return "unknown";
}

inline std::string FmGetDatetime() {
  auto now = std::chrono::system_clock::now();
  auto tt  = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf;
  localtime_r(&tt, &tm_buf);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
  return buf;
}

inline std::string FmGetDateForFilename() {
  auto now = std::chrono::system_clock::now();
  auto tt  = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf;
  localtime_r(&tt, &tm_buf);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm_buf);
  return buf;
}

// Базовый путь для всех отчётов FM Correlator
inline std::string FmReportBase() {
  return "../Results/Profiler/fm_correlator/fm_correlator_" + FmGetDateForFilename();
}

// ════════════════════════════════════════════════════════════════════════════
// fmt_N — форматирование размера FFT: 1024→1K, 1048576→1M
// ════════════════════════════════════════════════════════════════════════════

inline std::string fmt_N(int n) {
  if (n >= 1048576) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%dM", n / 1048576);
    return buf;
  } else if (n >= 1024) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%dK", n / 1024);
    return buf;
  }
  return std::to_string(n);
}

// ════════════════════════════════════════════════════════════════════════════
// WriteMarkdownReport — custom Markdown с системной инфой + полной таблицей
// ════════════════════════════════════════════════════════════════════════════

inline void WriteMarkdownReport(
    drv_gpu_lib::IBackend* backend,
    const std::vector<BenchConfig>& results,
    const std::string& filepath) {

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto dev = backend->GetDeviceInfo();

  std::ofstream f(filepath, std::ios::out);
  if (!f.is_open()) {
    con.PrintError(0, "FM_Bench", "Cannot create report: " + filepath);
    return;
  }

  f << "# FM Correlator Benchmark Report\n\n";
  f << "> **Модуль**: fm_correlator / FMCorrelatorProcessorROCm\n";
  f << "> **Pipeline**: apply_cyclic_shifts → C2C FFT (ref) → R2C FFT (inp) → multiply_conj_fused → C2R IFFT → extract_magnitudes\n";
  f << "> **Замер времени**: `hipEvent` — аппаратный таймер GPU (не CPU chrono)\n";
  f << "> **Прогрев**: " << kWarmupRuns << " итераций, **Измерений**: " << kBenchmarkRuns << "\n\n";
  f << "---\n\n";

  // System info
  f << "## Информация о системе\n\n";
  f << "| Параметр | Значение |\n";
  f << "|----------|----------|\n";
  f << "| **Дата** | " << FmGetDatetime() << " |\n";
  f << "| **Хост** | " << FmGetHostname() << " |\n";
  f << "| **ОС** | " << FmGetOsDistro() << " |\n";
  f << "| **Ядро** | " << FmGetOsKernel() << " |\n";
  f << "| **GPU** | " << dev.name << " |\n";
  f << "| **Вендор** | " << dev.vendor << " |\n";
  f << "| **Память GPU** | " << (dev.global_memory_size / (1024 * 1024))
    << " MB (" << std::fixed << std::setprecision(1)
    << (dev.global_memory_size / (1024.0 * 1024.0 * 1024.0)) << " GB) |\n";
  f << "| **Compute Units** | " << dev.max_compute_units << " |\n";
  f << "| **Max Clock** | " << dev.max_clock_frequency << " MHz |\n";
  f << "| **GPU Arch** | " << dev.driver_version << " |\n";
  f << "| **ROCm** | " << FmGetRocmVersion() << " |\n";
  f << "| **HIP Runtime** | " << FmGetHipRuntimeVersion() << " |\n";
  f << "\n";

  // Legend
  f << "## Расшифровка столбцов\n\n";
  f << "| Столбец | Описание |\n";
  f << "|---------|----------|\n";
  f << "| **N** | Длина сигнала (точек FFT) |\n";
  f << "| **K** | Кол-во корреляций (циклических сдвигов опорного) |\n";
  f << "| **S** | Число антенн |\n";
  f << "| **n_kg** | Число выходных точек корреляции (первые n_kg из IFFT) |\n";
  f << "| **avg (ms)** | Среднее время одного прогона по " << kBenchmarkRuns << " замерам (hipEvent, GPU hardware timer) |\n";
  f << "\n";

  // Results table
  f << "## Результаты\n\n";
  f << "| N | K | S | n_kg | avg (ms) |\n";
  f << "|--:|--:|--:|-----:|---------:|\n";

  // Группировка: N показываем только в первой строке группы
  int prev_N = -1;
  int prev_K = -1;
  for (const auto& r : results) {
    bool new_N = (r.fft_size != prev_N);
    bool new_K = new_N || (r.num_shifts != prev_K);

    if (new_N && prev_N != -1) {
      f << "| | | | | |\n";
    }

    std::string n_str = new_N ? ("**" + fmt_N(r.fft_size) + "**") : "";
    std::string k_str = new_K ? std::to_string(r.num_shifts) : "";

    f << std::fixed << std::setprecision(3)
      << "| " << n_str
      << " | " << k_str
      << " | " << r.num_signals
      << " | " << r.num_output_points
      << " | " << r.stats.avg_ms << " |\n";

    prev_N = r.fft_size;
    prev_K = r.num_shifts;
  }

  f << "\n*avg — среднее по " << kBenchmarkRuns << " запускам (мс). Прогрев: " << kWarmupRuns << " итерации.*\n";
  f.close();
  con.Print(0, "FM_Bench", "  MD report: " + filepath);
}

// ════════════════════════════════════════════════════════════════════════════
// run_benchmark — базовый бенчмарк с параметрами в имени события
// ════════════════════════════════════════════════════════════════════════════

inline void run_benchmark() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Corr", "  FM Correlator Benchmark (hipEvent GPU timing)");
  con.Print(gpu_id, "FM_Corr", "  warmup=" + std::to_string(kWarmupRuns) +
            "  runs=" + std::to_string(kBenchmarkRuns));
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");

  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;  // defaults: N=32768, K=32, S=5, n_kg=2000
  corr.SetParams(params);
  corr.PrepareReference();

  auto stats = MeasureProcessTime(corr, 2);

  char buf[256];
  std::snprintf(buf, sizeof(buf),
      "  N=%zu K=%d S=%d n_kg=%d | avg=%.3f ms  min=%.3f  max=%.3f",
      params.fft_size, params.num_shifts, params.num_signals,
      params.num_output_points,
      stats.avg_ms, stats.min_ms, stats.max_ms);
  con.Print(gpu_id, "FM_Corr", buf);

  // GPUProfiler: имя события = параметры конфига
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  auto device_info = backend->GetDeviceInfo();

  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name = device_info.name;
  report_info.backend_type = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv;
  drv["driver_type"]    = "ROCm";
  drv["driver_version"] = device_info.driver_version;
  report_info.drivers.push_back(drv);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  // Имя события: "N32768_K32_S5" — сразу видно конфиг
  char event_name[64];
  std::snprintf(event_name, sizeof(event_name),
      "N%zu_K%d_S%d", params.fft_size, params.num_shifts, params.num_signals);

  drv_gpu_lib::ROCmProfilingData pd;
  pd.end_ns      = static_cast<uint64_t>(stats.avg_ms * 1e6);
  pd.kernel_name = "FM_Correlator_Process";
  profiler.Record(gpu_id, "FM_Correlator", event_name, pd);

  profiler.Stop();
  profiler.PrintReport();

  std::string base = FmReportBase();
  profiler.ExportMarkdown(base + ".md");
  profiler.ExportJSON(base + ".json");

  con.Print(gpu_id, "FM_Corr", "Benchmark complete ✅");
  con.Print(gpu_id, "FM_Corr", "  Report: " + base + ".md");
}

// ════════════════════════════════════════════════════════════════════════════
// run_parametric_benchmark — sweep fft_size × shifts × signals
//   каждый конфиг → отдельное событие в GPUProfiler
//   итог → custom Markdown + GPUProfiler ExportMarkdown/JSON
// ════════════════════════════════════════════════════════════════════════════

inline void run_parametric_benchmark() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Corr", "  FM Correlator Parametric Benchmark");
  con.Print(gpu_id, "FM_Corr", "  warmup=" + std::to_string(kWarmupRuns) +
            "  runs=" + std::to_string(kBenchmarkRuns));
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");

  auto* backend = GetTestBackend();

  const int fft_sizes[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                             131072, 262144, 524288, 1048576 };
  const int shifts[]    = { 5, 10, 20, 32, 50 };
  const int signals[]   = { 1, 5, 10, 20 };

  // GPUProfiler setup
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  auto device_info = backend->GetDeviceInfo();

  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = device_info.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv;
  drv["driver_type"]    = "ROCm";
  drv["driver_version"] = device_info.driver_version;
  report_info.drivers.push_back(drv);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  std::vector<BenchConfig> results;

  for (int N : fft_sizes) {
    for (int K : shifts) {
      for (int S : signals) {
        drv_gpu_lib::FMCorrelatorParams params;
        params.fft_size         = N;
        params.num_shifts       = K;
        params.num_signals      = S;
        params.num_output_points = std::min(2000, N);

        try {
          drv_gpu_lib::FMCorrelator corr(backend);
          corr.SetParams(params);
          corr.PrepareReference();

          auto stats = MeasureProcessTime(corr, 2);

          // Консольный вывод
          char buf[256];
          std::snprintf(buf, sizeof(buf),
              "  N=%5d K=%2d S=%2d n_kg=%4d | avg=%.3f  min=%.3f  max=%.3f ms",
              N, K, S, params.num_output_points,
              stats.avg_ms, stats.min_ms, stats.max_ms);
          con.Print(gpu_id, "FM_Bench", buf);

          // GPUProfiler: имя = "N4096_K10_S5"
          char event_name[64];
          std::snprintf(event_name, sizeof(event_name), "N%d_K%d_S%d", N, K, S);

          drv_gpu_lib::ROCmProfilingData pd;
          pd.end_ns      = static_cast<uint64_t>(stats.avg_ms * 1e6);
          pd.kernel_name = "FM_Correlator_Process";
          profiler.Record(gpu_id, "FM_Correlator", event_name, pd);

          BenchConfig bc;
          bc.fft_size         = N;
          bc.num_shifts       = K;
          bc.num_signals      = S;
          bc.num_output_points = params.num_output_points;
          bc.stats            = stats;
          results.push_back(bc);

        } catch (const std::exception& e) {
          con.PrintError(gpu_id, "FM_Bench",
              "  N=" + std::to_string(N) + " K=" + std::to_string(K) +
              " S=" + std::to_string(S) + " FAILED: " + e.what());
        }
      }
    }
  }

  profiler.Stop();
  profiler.PrintReport();

  std::string base = FmReportBase() + "_parametric";

  // GPUProfiler export (полный лог событий)
  profiler.ExportMarkdown(base + "_profiler.md");
  profiler.ExportJSON(base + "_profiler.json");

  // Custom Markdown report (сводная таблица N×K×S)
  WriteMarkdownReport(backend, results, base + ".md");

  con.Print(gpu_id, "FM_Corr", "Parametric benchmark complete ✅");
  con.Print(gpu_id, "FM_Corr", "  Summary: " + base + ".md");
  con.Print(gpu_id, "FM_Corr", "  Profiler: " + base + "_profiler.md");
}

// ════════════════════════════════════════════════════════════════════════════
// run_sweep_correlations — S=5 фиксировано
//   N: 10K → 1M (10240, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
//   K: 5 → 250 (5, 10, 20, 50, 100, 150, 200, 250)
// Ответ на вопрос: как растёт время при увеличении числа корреляций?
// ════════════════════════════════════════════════════════════════════════════

inline void run_sweep_correlations() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Corr", "  Sweep: S=5 fixed, N: 10K→512K, K: 5→250");
  con.Print(gpu_id, "FM_Corr", "  warmup=" + std::to_string(kWarmupRuns) +
            "  runs=" + std::to_string(kBenchmarkRuns));
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════");

  auto* backend = GetTestBackend();

  const int fixed_S = 5;
  const int fft_sizes[] = { 10240, 16384, 32768, 65536,
                             131072, 262144, 524288 };
  const int shifts[]    = { 5, 10, 20, 50, 100, 150, 200, 250 };

  // GPUProfiler setup
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  auto device_info = backend->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = device_info.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv;
  drv["driver_type"]    = "ROCm";
  drv["driver_version"] = device_info.driver_version;
  report_info.drivers.push_back(drv);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  std::vector<BenchConfig> results;

  for (int N : fft_sizes) {
    for (int K : shifts) {
      drv_gpu_lib::FMCorrelatorParams params;
      params.fft_size          = N;
      params.num_shifts        = K;
      params.num_signals       = fixed_S;
      params.num_output_points = std::min(2000, N);

      try {
        drv_gpu_lib::FMCorrelator corr(backend);
        corr.SetParams(params);
        corr.PrepareReference();

        auto stats = MeasureProcessTime(corr, 2);

        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  N=%6d K=%3d S=%d | avg=%.3f  min=%.3f  max=%.3f ms",
            N, K, fixed_S, stats.avg_ms, stats.min_ms, stats.max_ms);
        con.Print(gpu_id, "FM_Sweep", buf);

        char event_name[64];
        std::snprintf(event_name, sizeof(event_name), "N%d_K%d_S%d", N, K, fixed_S);
        drv_gpu_lib::ROCmProfilingData pd;
        pd.end_ns      = static_cast<uint64_t>(stats.avg_ms * 1e6);
        pd.kernel_name = "FM_Correlator_Process";
        profiler.Record(gpu_id, "FM_Correlator", event_name, pd);

        BenchConfig bc;
        bc.fft_size          = N;
        bc.num_shifts        = K;
        bc.num_signals       = fixed_S;
        bc.num_output_points = params.num_output_points;
        bc.stats             = stats;
        results.push_back(bc);

      } catch (const std::exception& e) {
        std::string reason = e.what();
        // Определяем причину: OOM (план не создать) или другое
        bool is_oom = (reason.find("plan_") != std::string::npos ||
                       reason.find("hipMalloc") != std::string::npos ||
                       reason.find("Alloc") != std::string::npos);
        con.PrintError(gpu_id, "FM_Sweep",
            "  N=" + std::to_string(N) + " K=" + std::to_string(K) +
            (is_oom ? " — OOM (нет памяти GPU)" : " FAILED: " + reason));
        BenchConfig bc;
        bc.fft_size    = N;
        bc.num_shifts  = K;
        bc.num_signals = fixed_S;
        bc.num_output_points = std::min(2000, N);
        bc.failed      = true;
        bc.fail_reason = is_oom ? "OOM" : "ERR";
        results.push_back(bc);
      }
    }
  }

  profiler.Stop();

  std::string base = FmReportBase() + "_sweep_S5";

  // GPUProfiler export
  profiler.ExportMarkdown(base + "_profiler.md");
  profiler.ExportJSON(base + "_profiler.json");

  // Custom Markdown: S фиксировано, поэтому убираем колонку S из таблицы
  auto& con2 = drv_gpu_lib::ConsoleOutput::GetInstance();
  std::ofstream f(base + ".md", std::ios::out);
  if (f.is_open()) {
    auto dev = backend->GetDeviceInfo();
    f << "# FM Correlator — Sweep: S=5 (антенны фикс.), N×K до 512K\n\n";
    f << "> **S = 5 антенн (фиксировано)**\n";
    f << "> **N**: 10K → 512K | **K (корреляций)**: 5 → 250\n";
    f << "> Замер: `hipEvent`, прогрев=" << kWarmupRuns
      << ", измерений=" << kBenchmarkRuns << "\n\n---\n\n";

    f << "## Система\n\n";
    f << "| | |\n|---|---|\n";
    f << "| **GPU** | " << dev.name << " |\n";
    f << "| **Arch** | " << dev.driver_version << " |\n";
    f << "| **Память** | " << (dev.global_memory_size/(1024*1024)) << " MB |\n";
    f << "| **CU** | " << dev.max_compute_units << " |\n";
    f << "| **Clock** | " << dev.max_clock_frequency << " MHz |\n";
    f << "| **ROCm** | " << FmGetRocmVersion() << " |\n";
    f << "| **Дата** | " << FmGetDatetime() << " |\n\n";

    // Шапка: N + колонки K
    f << "## avg (мс) по N и K\n\n";
    f << "| N |";
    for (int K : shifts) f << " K=" << K << " |";
    f << "\n|--:|";
    for (size_t ki = 0; ki < sizeof(shifts)/sizeof(shifts[0]); ++ki) f << "------:|";
    f << "\n";

    // Строки: одна на N, ячейки — avg для каждого K
    for (int N : fft_sizes) {
      f << "| **" << fmt_N(N) << "** |";
      for (int K : shifts) {
        std::string cell = " — |";
        for (const auto& r : results) {
          if (r.fft_size == N && r.num_shifts == K) {
            if (r.failed) {
              cell = " *" + r.fail_reason + "* |";
            } else {
              char buf[32];
              std::snprintf(buf, sizeof(buf), " %.3f |", r.stats.avg_ms);
              cell = buf;
            }
            break;
          }
        }
        f << cell;
      }
      f << "\n";
    }

    f << "\n*Время в мс. S=5 антенн фиксировано.*\n";
    f.close();
    con2.Print(gpu_id, "FM_Sweep", "  Report: " + base + ".md");
  }

  con.Print(gpu_id, "FM_Corr", "Sweep S=5 complete ✅");
  con.Print(gpu_id, "FM_Corr", "  Summary: " + base + ".md");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
