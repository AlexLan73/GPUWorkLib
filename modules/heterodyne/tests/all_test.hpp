#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for heterodyne module (ROCm)
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21 (migrated 2026-03-23)
 */

#include "test_heterodyne_basic.hpp"
#include "test_heterodyne_pipeline.hpp"
#include "test_heterodyne_rocm.hpp"
#if ENABLE_ROCM
#include "test_heterodyne_benchmark_rocm.hpp"
#endif

namespace heterodyne_all_test {

inline void run() {
  // Basic tests (TestRunner: single_antenna, five_antennas, random)
  heterodyne::tests::run_basic_tests();

  // Pipeline integration (TestRunner: full_pipeline, process_external)
  heterodyne::tests::run_pipeline_tests();

  // HeterodyneProcessorROCm tests
  test_heterodyne_rocm::run();

  // Benchmark (uncomment when needed)
#if ENABLE_ROCM
  //  test_heterodyne_benchmark_rocm::run();
#endif
}

}  // namespace heterodyne_all_test
