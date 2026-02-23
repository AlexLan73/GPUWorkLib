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
#endif

namespace statistics_all_test {

inline void run() {
  // StatisticsProcessor: mean, median, variance, std (ROCm only)
#if ENABLE_ROCM
  test_statistics_rocm::run();
#endif
}

}  // namespace statistics_all_test
