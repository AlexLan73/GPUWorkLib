#pragma once

/**
 * @file test_heterodyne_pipeline.hpp
 * @brief Integration tests for HeterodyneDechirp facade (ROCm backend)
 *
 * Test 4: Full pipeline via Process() — 5 antennas, linear delays
 * Test 5: ProcessExternal() — external HIP buffer (hipMalloc)
 *
 * Uses constants from test_heterodyne_basic.hpp (same namespace)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21 (ROCm port 2026-03-16)
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"

#include "DrvGPU/services/console_output.hpp"

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <memory>

namespace heterodyne { namespace tests {

// Constants FS, F_START, F_END, N, ANTENNAS, MU, DELAYS_LINEAR_US, F_BEAT_TOL_HZ
// are defined in test_heterodyne_basic.hpp (same namespace)
// GenerateRxFlat() is also defined there.

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Full pipeline via Process()
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_full_pipeline() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 4: Full pipeline Process()");

  try {
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    std::vector<float> delays_us = DELAYS_LINEAR_US;
    auto rx_flat = GenerateRxFlat(delays_us);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get(), drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Ant | Delay us | f_beat Hz   | Expected Hz | Error Hz | Range m");
    con.Print(gpu_id, "Heterodyne",
        "    ----|----------|-------------|-------------|----------|--------");

    bool all_passed = true;
    float max_range_err = 0.0f;

    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float delay_us = delays_us[ant];
      float expected_f = MU * delay_us * 1e-6f;
      float actual_f = result.antennas[ant].f_beat_hz;
      float f_err = std::abs(actual_f - expected_f);

      float T = static_cast<float>(N) / FS;
      float B = F_END - F_START;
      float expected_range = (3e8f * T * expected_f) / (2.0f * B);
      float range_err = std::abs(result.antennas[ant].range_m - expected_range);
      max_range_err = std::max(max_range_err, range_err);

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    %3d | %7.0f  | %11.0f | %11.0f | %8.0f | %7.2f",
          ant, delay_us, actual_f, expected_f, f_err,
          result.antennas[ant].range_m);
      con.Print(gpu_id, "Heterodyne", buf);

      if (f_err >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Max range error: " + std::to_string(max_range_err) + " m");
    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: ProcessExternal with external HIP buffer
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_process_external() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 5: ProcessExternal (external HIP buffer)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    std::vector<float> delays_us = DELAYS_LINEAR_US;
    auto rx_flat = GenerateRxFlat(delays_us);

    // Upload to GPU manually (simulating external program)
    size_t total = static_cast<size_t>(ANTENNAS) * N;
    size_t buf_size = total * sizeof(std::complex<float>);

    void* external_buf = nullptr;
    hipError_t err = hipMalloc(&external_buf, buf_size);
    if (err != hipSuccess) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: hipMalloc failed");
      return;
    }

    err = hipMemcpy(external_buf, rx_flat.data(), buf_size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      hipFree(external_buf);
      con.Print(gpu_id, "Heterodyne", "    FAIL: hipMemcpy failed");
      return;
    }

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get(), drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.ProcessExternal(external_buf, params);

    // External buffer must still be valid (not freed by HeterodyneDechirp)
    std::vector<std::complex<float>> verify(total);
    err = hipMemcpy(verify.data(), external_buf, buf_size, hipMemcpyDeviceToHost);
    bool buf_valid = (err == hipSuccess);

    // Release external buffer (we own it)
    hipFree(external_buf);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    bool all_passed = buf_valid;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float expected_f = MU * DELAYS_LINEAR_US[ant] * 1e-6f;
      float f_error = std::abs(result.antennas[ant].f_beat_hz - expected_f);
      if (f_error >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne", "    External buffer read: "
        + std::string(buf_valid ? "OK" : "FAIL"));
    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

}} // namespace heterodyne::tests

#else  // !ENABLE_ROCM

namespace heterodyne { namespace tests {
inline void run_test_full_pipeline()      {}
inline void run_test_process_external()   {}
}} // namespace heterodyne::tests

#endif  // ENABLE_ROCM
