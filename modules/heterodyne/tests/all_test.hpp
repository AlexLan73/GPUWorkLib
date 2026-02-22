#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for heterodyne module
 *
 * Called from src/main.cpp:
 *   #include "modules/heterodyne/tests/all_test.hpp"
 *   heterodyne_all_test::run();
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "test_heterodyne_basic.hpp"
#include "test_heterodyne_pipeline.hpp"

#include "DrvGPU/services/console_output.hpp"

namespace heterodyne_all_test {

inline void run() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "════════════════════════════════════════════════════════════");
  con.Print(gpu_id, "Heterodyne", " Heterodyne LFM Dechirp Tests");
  con.Print(gpu_id, "Heterodyne", "════════════════════════════════════════════════════════════");

  // Basic kernel tests
  heterodyne::tests::run_test_single_antenna();      // Test 1
  heterodyne::tests::run_test_5_antennas_linear();   // Test 2
  heterodyne::tests::run_test_correction();           // Test 3

  // Pipeline integration tests
  heterodyne::tests::run_test_full_pipeline();        // Test 4
  heterodyne::tests::run_test_process_external();     // Test 5

  // Additional tests
  heterodyne::tests::run_test_random_delays();        // Test 6
  heterodyne::tests::run_test_all_maxima();           // Test 7

  con.Print(gpu_id, "Heterodyne", "════════════════════════════════════════════════════════════");
}

}  // namespace heterodyne_all_test
