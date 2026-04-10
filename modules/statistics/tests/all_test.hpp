#pragma once

/**
 * @file all_test.hpp
 * @brief Test index for statistics module
 *
 * main.cpp calls this file -- NOT individual tests directly.
 * Enable/disable tests here.
 *
 * NOTE: Statistics is ROCm-only. All tests are under #if ENABLE_ROCM.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM
#include "test_statistics_rocm.hpp"
#include "test_statistics_float_rocm.hpp"
#include "statistics_compute_all_benchmark.hpp"
#include "test_statistics_compute_all_benchmark.hpp"
// SNR-estimator (SNR_08, SNR_09) — запуск в понедельник на Debian/AMD
#include "test_snr_estimator_rocm.hpp"
#include "snr_estimator_benchmark.hpp"
#include "test_snr_estimator_benchmark.hpp"
#endif

namespace statistics_all_test {

inline void run() {
  // StatisticsProcessor: mean, median, variance, std (ROCm only)
#if ENABLE_ROCM
  test_statistics_rocm::run();
  // Statistics float API + ProcessMagnitude→Statistics pipeline
  test_statistics_float_rocm::run();
  // ComputeAll benchmark (раскомментировать для профилирования):
  // test_statistics_compute_all_benchmark::run();

  // SNR-estimator: написано 2026-04-09, запуск в понедельник на Debian/AMD
  // test_snr_estimator_rocm::run_all();
  // test_snr_estimator_benchmark::run_benchmark();
#endif
}

}  // namespace statistics_all_test
