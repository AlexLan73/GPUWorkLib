#pragma once

/**
 * @file test_heterodyne_pipeline.hpp
 * @brief Integration test: full HeterodyneDechirp::Process() pipeline
 *
 * Test 4: Full pipeline with Process() — all 5 antennas, linear delays
 * Test 5: ProcessExternal() — external cl_mem buffer
 *
 * Uses constants from test_heterodyne_basic.hpp (same namespace)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"
#include "processors/heterodyne_processor_opencl.hpp"
#include "generators/lfm_conjugate_generator.hpp"
#include "generators/lfm_generator_analytical_delay.hpp"
#include "params/signal_request.hpp"
#include "params/system_sampling.hpp"

#include "spectrum_maxima_finder.h"

#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/console_output.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <memory>

namespace heterodyne { namespace tests {

// Constants FS, F_START, F_END, N, ANTENNAS, MU, DELAYS_LINEAR_US
// are defined in test_heterodyne_basic.hpp (same namespace)

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Full pipeline via Process()
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_full_pipeline() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 4: Full pipeline Process()");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    std::vector<float> delays_us = DELAYS_LINEAR_US;

    // Generate rx
    auto rx_flat = GenerateRxFlat(backend.get(), delays_us);

    // Process
    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get());
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
// Test 5: ProcessExternal with external cl_mem
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_process_external() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 5: ProcessExternal (external cl_mem)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    std::vector<float> delays_us = DELAYS_LINEAR_US;

    // Generate rx data on CPU
    auto rx_flat = GenerateRxFlat(backend.get(), delays_us);

    // Upload to GPU manually (simulating external program)
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_command_queue queue = static_cast<cl_command_queue>(backend->GetNativeQueue());

    size_t total = static_cast<size_t>(ANTENNAS) * N;
    size_t buf_size = total * sizeof(std::complex<float>);

    cl_int err;
    cl_mem external_buf = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
        rx_flat.data(), &err);

    if (err != CL_SUCCESS) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: clCreateBuffer failed");
      return;
    }

    // Process with external buffer
    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.ProcessExternal(static_cast<void*>(external_buf), params);

    // External buffer must still be valid (not freed by HeterodyneDechirp)
    // Verify by reading from it
    std::vector<std::complex<float>> verify(total);
    err = clEnqueueReadBuffer(queue, external_buf, CL_TRUE, 0, buf_size,
                               verify.data(), 0, nullptr, nullptr);

    // Now release external buffer (WE own it, not HeterodyneDechirp)
    clReleaseMemObject(external_buf);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    // Check results
    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float expected_f = MU * DELAYS_LINEAR_US[ant] * 1e-6f;
      float f_error = std::abs(result.antennas[ant].f_beat_hz - expected_f);
      if (f_error >= 10000.f) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne", "    External buffer read: "
        + std::string(err == CL_SUCCESS ? "OK" : "FAIL"));
    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 7: AllMaxima mode — verify main peak matches Process() result
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_all_maxima() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 7: AllMaxima mode");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    std::vector<float> delays_us = DELAYS_LINEAR_US;
    auto rx_flat = GenerateRxFlat(backend.get(), delays_us);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    // Process() — get reference f_beat values
    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL (Process): " + result.error_message);
      return;
    }

    // Now manually dechirp and run AllMaxima
    signal_gen::LfmParams lfm_p;
    lfm_p.f_start = F_START;
    lfm_p.f_end = F_END;
    lfm_p.amplitude = 1.0;
    lfm_p.complex_iq = true;

    signal_gen::SystemSampling sys;
    sys.fs = FS;
    sys.length = N;

    signal_gen::LfmConjugateGenerator conj_gen(backend.get(), lfm_p);
    conj_gen.SetSampling(sys);
    auto ref = conj_gen.GenerateToCpu();

    drv_gpu_lib::HeterodyneProcessorOpenCL proc(backend.get());
    auto dc_data = proc.Dechirp(rx_flat, ref, params);

    // AllMaxima via SpectrumMaximaFinder
    antenna_fft::SpectrumMaximaFinder finder(backend.get());

    antenna_fft::InputData<std::vector<std::complex<float>>> smf_input;
    smf_input.antenna_count = ANTENNAS;
    smf_input.n_point = N;
    smf_input.data = dc_data;
    smf_input.repeat_count = 1;
    smf_input.sample_rate = FS;

    auto all_results = finder.FindAllMaxima(smf_input,
        antenna_fft::OutputDestination::CPU,
        antenna_fft::DriverType::OPENCL);

    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float ref_f_beat = result.antennas[ant].f_beat_hz;

      // Find the closest peak in AllMaxima results
      float min_err = 1e9f;
      float closest_freq = 0.0f;
      if (ant < static_cast<int>(all_results.beams.size())) {
        for (auto& m : all_results.beams[ant].maxima) {
          float err = std::abs(m.refined_frequency - ref_f_beat);
          if (err < min_err) {
            min_err = err;
            closest_freq = m.refined_frequency;
          }
        }
      }

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    Ant %d: OnePeak=%.0f Hz, AllMaxima closest=%.0f Hz, diff=%.0f Hz  %s",
          ant, ref_f_beat, closest_freq, min_err,
          (min_err < F_BEAT_TOL_HZ) ? "OK" : "FAIL");
      con.Print(gpu_id, "Heterodyne", buf);

      if (min_err >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

}} // namespace heterodyne::tests
