#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for filters module
 *
 * Called from src/main.cpp:
 *   #include "modules/filters/tests/all_test.hpp"
 *   filters_all_test::run();
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "test_fir_basic.hpp"
#include "test_iir_basic.hpp"
#include "test_filters_rocm.hpp"
#include "test_filters_benchmark.hpp"
#if ENABLE_ROCM
#include "test_filters_benchmark_rocm.hpp"
#include "test_moving_average_rocm.hpp"
#include "test_kalman_rocm.hpp"
#include "test_kaufman_rocm.hpp"
#endif

namespace filters_all_test {

inline void run() {
  filters::tests::run_fir_basic();
  filters::tests::run_iir_basic();
  test_filters_rocm::run();  // ROCm — Linux only, uncomment on AMD GPU

  // BENCHMARK: FirFilter + IirFilter (OpenCL, GpuBenchmarkBase)
  // test_filters_benchmark::run();

  // BENCHMARK: FirFilterROCm + IirFilterROCm (ROCm, GpuBenchmarkBase)
#if ENABLE_ROCM
//  test_filters_benchmark_rocm::run();

  // ROCm — Linux + AMD GPU только:
  test_moving_average_rocm::run();   // Task_20: SMA/EMA/MMA/DEMA/TEMA
  test_kalman_rocm::run();           // Task_21: 1D Kalman filter
  test_kaufman_rocm::run();           // Task_22: KAMA (Kaufman adaptive MA)
#endif
}

}  // namespace filters_all_test
