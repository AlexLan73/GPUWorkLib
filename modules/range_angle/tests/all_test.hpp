#pragma once

/**
 * @file all_test.hpp
 * @brief Entry point for all range_angle module tests
 *
 * Included from src/main.cpp (see CLAUDE.md architecture rule).
 * Enable/disable individual tests here.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM
#include "test_range_angle_basic.hpp"
#include "test_range_angle_benchmark.hpp"
#endif

#include "interface/i_backend.hpp"

namespace range_angle_all_test {

inline void run(drv_gpu_lib::IBackend* backend = nullptr) {
#if ENABLE_ROCM
  range_angle_basic_test::run(backend);
  // Benchmark — uncomment when needed:
  // range_angle_benchmark_test::run(backend);
#else
  (void)backend;
#endif
}

}  // namespace range_angle_all_test
