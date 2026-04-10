#pragma once

/**
 * @file test_snr_estimator_benchmark.hpp
 * @brief SNR-estimator benchmark runner (SNR_09)
 *
 * Запускает SnrEstimatorBenchmark на 4 сценариях:
 *   Py-Small   (5 ant × 1.3M samp)      — лёгкий Python сценарий
 *   Scenario A (2500 × 5000)
 *   Scenario B (256 × 1.3M, 2.66 GB)    — главный замер
 *   Scenario C (9000 × 10000)
 *
 * Данные для больших сценариев генерируются прямо на GPU (hipMalloc) —
 * CPU vector 2.66 GB убил бы RAM.
 *
 * ⚠️ КОД НАПИСАН — запуск в понедельник на Debian/AMD.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "snr_estimator_benchmark.hpp"
#include "snr_test_helpers.hpp"
#include "statistics_processor.hpp"

#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"

#include "backends/rocm/rocm_core.hpp"

#include <hip/hip_runtime.h>

#include <complex>
#include <string>
#include <vector>

namespace test_snr_estimator_benchmark {

struct SnrBenchScenario {
  std::string name;
  uint32_t    n_antennas;
  uint32_t    n_samples;
  uint32_t    target_n_fft;  // 0 → auto (2048)
};

/// Заполняем GPU буфер синтетическими данными (noise + CW в первой антенне).
/// Для больших сценариев: генерируем одну антенну и тиражируем через hipMemcpy.
inline bool FillGpuBufferWithNoisePlusCW(void* gpu_data,
                                         uint32_t n_ant, uint32_t n_samp)
{
  using cx = std::complex<float>;
  auto one = snr_test_helpers::MakeDechirpedCW(n_samp, /*freq_norm=*/0.12f, /*A=*/3.2f);
  snr_test_helpers::AddNoise(one, /*noise_power=*/1.0f, /*seed=*/2026u);

  for (uint32_t ant = 0; ant < n_ant; ++ant) {
    char* dst = static_cast<char*>(gpu_data) +
                static_cast<size_t>(ant) * n_samp * sizeof(cx);
    hipError_t err = hipMemcpy(dst, one.data(),
                                n_samp * sizeof(cx),
                                hipMemcpyHostToDevice);
    if (err != hipSuccess) return false;
  }
  return true;
}

/// Запустить SnrEstimatorBenchmark на одной конфигурации.
inline void RunOneScenario(drv_gpu_lib::IBackend* backend,
                           statistics::StatisticsProcessor& proc,
                           const SnrBenchScenario& scenario)
{
  using cx = std::complex<float>;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(0, "SnrBench",
            "Running " + scenario.name + " (" +
            std::to_string(scenario.n_antennas) + " x " +
            std::to_string(scenario.n_samples) + ")");

  // Allocate GPU buffer (hipMalloc — чтобы не держать 2.66 GB в CPU vector)
  const size_t total_bytes =
      static_cast<size_t>(scenario.n_antennas) * scenario.n_samples * sizeof(cx);

  void* gpu_data = nullptr;
  hipError_t err = hipMalloc(&gpu_data, total_bytes);
  if (err != hipSuccess) {
    con.Print(0, "SnrBench",
              "SKIP " + scenario.name + ": hipMalloc failed (" +
              std::to_string(total_bytes / (1024u * 1024u)) + " MB)");
    return;
  }

  if (!FillGpuBufferWithNoisePlusCW(gpu_data, scenario.n_antennas, scenario.n_samples)) {
    con.Print(0, "SnrBench", "SKIP " + scenario.name + ": fill failed");
    hipFree(gpu_data);
    return;
  }

  statistics::SnrEstimationConfig scfg;
  scfg.target_n_fft = scenario.target_n_fft;  // 0 → auto

  // Запуск benchmark: warmup + measurement
  test_snr_estimator::SnrEstimatorBenchmark bench(
      backend, proc, gpu_data,
      scenario.n_antennas, scenario.n_samples, scfg,
      "SNR_" + scenario.name);
  bench.Run();
  bench.Report();

  hipFree(gpu_data);
}

/// Главный runner — вызывается из all_test.hpp
inline void run_benchmark() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "SnrBench", "=== SNR Estimator Benchmark ===");

  // Проверка ROCm devices
  int devices = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (devices == 0) {
    con.Print(0, "SnrBench", "[!] No ROCm devices — skip benchmark");
    return;
  }

  auto* backend = snr_test_helpers::GetTestBackend();
  statistics::StatisticsProcessor proc(backend);

  const std::vector<SnrBenchScenario> scenarios = {
    {"Py-Small",        5u,    1'300'000u, 2048u},
    {"Scenario_A",      2500u, 5000u,      2048u},
    {"Scenario_B",      256u,  1'300'000u, 2048u},  // 2.66 GB — главный замер
    {"Scenario_C",      9000u, 10000u,     2048u},
    // Опционально — сравнение target_n_fft
    {"Scenario_B_1024", 256u,  1'300'000u, 1024u},
    {"Scenario_B_4096", 256u,  1'300'000u, 4096u},
  };

  for (const auto& sc : scenarios) {
    RunOneScenario(backend, proc, sc);
  }

  con.Print(0, "SnrBench", "=== SNR Estimator Benchmark DONE ===");
}

}  // namespace test_snr_estimator_benchmark

#endif  // ENABLE_ROCM
