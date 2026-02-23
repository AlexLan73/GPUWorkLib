#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for lch_farrow module
 *
 * main.cpp calls this file - NOT individual tests directly.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "test_lch_farrow.hpp"
#include "test_lch_farrow_rocm.hpp"

namespace lch_farrow_all_test {

inline void run() {
    test_lch_farrow::run();

    // ROCm: LchFarrowROCm tests (run on Linux + AMD GPU)
    // test_lch_farrow_rocm::run();
}

}  // namespace lch_farrow_all_test
