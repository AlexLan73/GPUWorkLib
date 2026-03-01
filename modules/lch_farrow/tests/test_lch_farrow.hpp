#pragma once

/**
 * @file test_lch_farrow.hpp
 * @brief Tests for LchFarrow fractional delay processor
 *
 * Tests:
 *   1. Zero delay: output == input
 *   2. Integer delay: output[n] == input[n - D] (GPU vs CPU)
 *   3. Fractional delay: GPU vs CPU reference
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "lch_farrow.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/console_output.hpp"

#include <CL/cl.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_lch_farrow {

// Generate simple CW signal for testing
inline std::vector<std::complex<float>> generate_cw(
    uint32_t points, float fs, float freq, float amplitude = 1.0f) {
  std::vector<std::complex<float>> signal(points);
  for (uint32_t n = 0; n < points; ++n) {
    float t = static_cast<float>(n) / fs;
    float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
    signal[n] = std::complex<float>(
        amplitude * std::cos(phase), amplitude * std::sin(phase));
  }
  return signal;
}

inline void run() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();

  con.Print(gpu_id, "LchFarrow", "Running tests...");

  // Initialize OpenCL backend
  auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
  backend->Initialize(0);

  lch_farrow::LchFarrow processor(backend.get());

  const float fs = 1e6f;
  const uint32_t points = 4096;
  const float freq = 50000.0f;
  const uint32_t antennas = 1;

  auto cw_signal = generate_cw(points, fs, freq);

  // Upload signal to GPU
  cl_context cl_ctx = static_cast<cl_context>(backend->GetNativeContext());
  cl_command_queue cl_queue = static_cast<cl_command_queue>(backend->GetNativeQueue());
  cl_int err;
  cl_mem input_buf = clCreateBuffer(
      cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      points * sizeof(std::complex<float>),
      cw_signal.data(), &err);
  if (err != CL_SUCCESS) {
    con.PrintError(gpu_id, "LchFarrow", "clCreateBuffer input failed: " + std::to_string(err));
    return;
  }

  // ── Test 1: Zero delay ──
  {
    processor.SetDelays({0.0f});
    processor.SetSampleRate(fs);
    auto result = processor.Process(input_buf, antennas, points);

    std::vector<std::complex<float>> output(points);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        points * sizeof(std::complex<float>),
                        output.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - cw_signal[n]);
      if (e > max_err) max_err = e;
    }

    std::ostringstream oss1;
    oss1 << "Test 1 (zero delay): max_err = " << std::scientific << std::setprecision(2) << max_err
         << (max_err < 1e-4f ? " PASSED" : " FAILED (expected < 1e-4)");
    con.Print(gpu_id, "LchFarrow", oss1.str());
  }

  // ── Test 2: Integer delay (5 samples) ──
  {
    float delay_us = 5.0f;  // 5 us at 1 MHz = 5 samples
    processor.SetDelays({delay_us});
    auto result = processor.Process(input_buf, antennas, points);

    std::vector<std::complex<float>> output(points);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        points * sizeof(std::complex<float>),
                        output.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    // CPU reference
    std::vector<std::vector<std::complex<float>>> input_2d = {cw_signal};
    processor.SetDelays({delay_us});
    auto cpu_ref = processor.ProcessCpu(input_2d, antennas, points);

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - cpu_ref[0][n]);
      if (e > max_err) max_err = e;
    }

    std::ostringstream oss2;
    oss2 << "Test 2 (integer delay 5): max_err = " << std::scientific << std::setprecision(2) << max_err
         << (max_err < 1e-2f ? " PASSED" : " FAILED (expected < 1e-2)");
    con.Print(gpu_id, "LchFarrow", oss2.str());
  }

  // ── Test 3: Fractional delay (2.7 samples) ──
  {
    float delay_us = 2.7f;  // 2.7 us at 1 MHz = 2.7 samples
    processor.SetDelays({delay_us});
    auto result = processor.Process(input_buf, antennas, points);

    std::vector<std::complex<float>> output(points);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        points * sizeof(std::complex<float>),
                        output.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    // CPU reference
    std::vector<std::vector<std::complex<float>>> input_2d = {cw_signal};
    processor.SetDelays({delay_us});
    auto cpu_ref = processor.ProcessCpu(input_2d, antennas, points);

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - cpu_ref[0][n]);
      if (e > max_err) max_err = e;
    }

    std::ostringstream oss3;
    oss3 << "Test 3 (fractional delay 2.7): max_err = " << std::scientific << std::setprecision(2) << max_err
         << (max_err < 1e-2f ? " PASSED" : " FAILED (expected < 1e-2)");
    con.Print(gpu_id, "LchFarrow", oss3.str());
  }

  clReleaseMemObject(input_buf);

  con.Print(gpu_id, "LchFarrow", "All tests completed.");
}

}  // namespace test_lch_farrow
