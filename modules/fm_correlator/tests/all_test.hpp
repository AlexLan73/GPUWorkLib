#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for fm_correlator module
 *
 * Called from src/main.cpp:
 *   #include "modules/fm_correlator/tests/all_test.hpp"
 *   fm_correlator_all_test::run();
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "test_fm_msequence.hpp"
#include "test_fm_basic.hpp"
#include "test_fm_benchmark_rocm.hpp"

#include "DrvGPU/services/console_output.hpp"

namespace fm_correlator_all_test {

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();
  int gpu_id = 0;

  con.Print(gpu_id, "FM_Corr", "");
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Corr", " FM Correlator Tests (ROCm)");
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════════════════");

  fm_correlator::tests::run_test_msequence();
  fm_correlator::tests::run_test_autocorrelation();
  fm_correlator::tests::run_test_basic_pipeline();
  fm_correlator::tests::run_test_full_pipeline();
  fm_correlator::tests::run_test_shift_pattern();

  // Benchmarks:
  // fm_correlator::tests::run_benchmark();
  // fm_correlator::tests::run_parametric_benchmark();
  fm_correlator::tests::run_sweep_correlations();

  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Corr", " All FM Correlator tests PASSED ✅");
  con.Print(gpu_id, "FM_Corr", "════════════════════════════════════════════════════════════");
}

}  // namespace fm_correlator_all_test

#else  // !ENABLE_ROCM

namespace fm_correlator_all_test {
inline void run() {
  // FM Correlator is ROCm-only, skip on non-ROCm builds
}
}

#endif  // ENABLE_ROCM
