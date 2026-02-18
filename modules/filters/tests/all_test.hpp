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

namespace filters_all_test {

inline void run() {
  filters::tests::run_fir_basic();
  filters::tests::run_iir_basic();
}

}  // namespace filters_all_test
