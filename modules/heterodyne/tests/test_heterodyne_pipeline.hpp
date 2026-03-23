#pragma once

/**
 * @file test_heterodyne_pipeline.hpp
 * @brief Integration tests for HeterodyneDechirp facade (ROCm)
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   4. full_pipeline      — Process() 5 antennas with range validation
 *   5. process_external   — ProcessExternal() with external HIP buffer
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21 (migrated 2026-03-23)
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"

#include "modules/test_utils/test_utils.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cmath>

namespace heterodyne { namespace tests {

// Constants and GenerateRxFlat from test_heterodyne_basic.hpp (same namespace)

inline void run_pipeline_tests() {
  int gpu_id = 0;

  drv_gpu_lib::ROCmBackend backend;
  backend.Initialize(gpu_id);

  gpu_test_utils::TestRunner runner(&backend, "Heterodyne Pipeline", gpu_id);

  // ── Test 4: Full pipeline with range ──────────────────────────

  runner.test("full_pipeline", [&]() -> gpu_test_utils::TestResult {
    auto rx_flat = GenerateRxFlat(DELAYS_LINEAR_US);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(&backend, drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    gpu_test_utils::TestResult tr{"full_pipeline"};
    if (!result.success)
      return tr.add(gpu_test_utils::FailResult("process", 0, 1));

    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float expected_f = MU * DELAYS_LINEAR_US[ant] * 1e-6f;
      tr.add(gpu_test_utils::ScalarAbsError(
          static_cast<double>(result.antennas[ant].f_beat_hz),
          static_cast<double>(expected_f), F_BEAT_TOL_HZ,
          "ant" + std::to_string(ant)));
    }
    return tr;
  });

  // ── Test 5: ProcessExternal with HIP buffer ───────────────────

  runner.test("process_external", [&]() -> gpu_test_utils::TestResult {
    auto rx_flat = GenerateRxFlat(DELAYS_LINEAR_US);

    size_t total = static_cast<size_t>(ANTENNAS) * N;
    size_t buf_size = total * sizeof(std::complex<float>);

    void* ext_buf = nullptr;
    hipError_t err = hipMalloc(&ext_buf, buf_size);
    if (err != hipSuccess)
      throw std::runtime_error("hipMalloc failed");

    (void)hipMemcpy(ext_buf, rx_flat.data(), buf_size, hipMemcpyHostToDevice);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(&backend, drv_gpu_lib::BackendType::ROCm);
    het.SetParams(params);
    auto result = het.ProcessExternal(ext_buf, params);

    // Verify buffer still valid after processing
    std::vector<std::complex<float>> verify(total);
    bool buf_valid = (hipMemcpy(verify.data(), ext_buf, buf_size,
                                hipMemcpyDeviceToHost) == hipSuccess);
    (void)hipFree(ext_buf);

    gpu_test_utils::TestResult tr{"process_external"};
    tr.add(gpu_test_utils::ValidationResult{
        buf_valid, "buf_valid", buf_valid ? 1.0 : 0.0, 1.0, ""});

    if (result.success) {
      for (int ant = 0; ant < ANTENNAS; ++ant) {
        float expected_f = MU * DELAYS_LINEAR_US[ant] * 1e-6f;
        tr.add(gpu_test_utils::ScalarAbsError(
            static_cast<double>(result.antennas[ant].f_beat_hz),
            static_cast<double>(expected_f), F_BEAT_TOL_HZ,
            "ant" + std::to_string(ant)));
      }
    }
    return tr;
  });

  runner.print_summary();
}

// Backward-compatible
inline void run_test_full_pipeline()    { /* now in run_pipeline_tests */ }
inline void run_test_process_external() { /* now in run_pipeline_tests */ }

}} // namespace heterodyne::tests

#else  // !ENABLE_ROCM

namespace heterodyne { namespace tests {
inline void run_pipeline_tests()         {}
inline void run_test_full_pipeline()     {}
inline void run_test_process_external()  {}
}} // namespace heterodyne::tests

#endif  // ENABLE_ROCM
