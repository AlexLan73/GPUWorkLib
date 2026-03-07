#pragma once

/**
 * @file all_test.hpp
 * @brief Entry point for strategies module tests
 *
 * Include this from src/main.cpp to run strategies tests.
 *
 * @date 2026-03-07
 */

#include "test_strategies_pipeline.hpp"
#include "test_strategies_step_profiling.hpp"

namespace test_strategies_all {

inline void run_all(drv_gpu_lib::IBackend* backend) {
#if ENABLE_ROCM
  test_strategies::test_full_pipeline(backend);
  // test_strategies_profiling::run_step_profiling(backend);
#else
  (void)backend;
#endif
}

}  // namespace test_strategies_all
