#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for lch_farrow module
 *
 * main.cpp calls this file - NOT individual tests directly.
 *
 * Тесты:
 *  - test_lch_farrow        : функциональные тесты (OpenCL)
 *  - test_lch_farrow_rocm   : функциональные тесты (ROCm)
 *
 * Бенчмарки (раскомментировать для запуска):
 *  - test_lch_farrow_benchmark      : OpenCL (GpuBenchmarkBase)
 *  - test_lch_farrow_benchmark_rocm : ROCm   (GpuBenchmarkBase)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18, обновлено 2026-03-01
 */

#include "test_lch_farrow.hpp"
#include "test_lch_farrow_rocm.hpp"
#include "test_lch_farrow_benchmark.hpp"
#if ENABLE_ROCM
#include "test_lch_farrow_benchmark_rocm.hpp"
#endif

namespace lch_farrow_all_test {

inline void run() {
    test_lch_farrow::run();

    // ROCm: LchFarrowROCm tests (run on Linux + AMD GPU)
    test_lch_farrow_rocm::run();

    // LchFarrow Benchmark (OpenCL, GpuBenchmarkBase)
    // Requires: configGPU.json → "is_prof": true
    // test_lch_farrow_benchmark::run();

    // LchFarrowROCm Benchmark (hipEvent timing, GpuBenchmarkBase)
    // Requires: Linux + AMD GPU, ENABLE_ROCM, "is_prof": true
#if ENABLE_ROCM
    // test_lch_farrow_benchmark_rocm::run();
#endif
}

}  // namespace lch_farrow_all_test
