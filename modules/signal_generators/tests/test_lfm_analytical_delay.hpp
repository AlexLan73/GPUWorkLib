#pragma once

/**
 * @file test_lfm_analytical_delay.hpp
 * @brief Tests for LfmGeneratorAnalyticalDelay
 *
 * 1. Zero delay: GPU matches standard LfmGenerator
 * 2. Fractional delay: first non-zero at correct index
 * 3. GPU vs CPU reference comparison
 * 4. Multi-antenna: different delays
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "generators/lfm_generator_analytical_delay.hpp"
#include "generators/lfm_generator.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_lfm_analytical_delay {

inline void run() {
  std::cout << "\n[LfmAnalyticalDelay] Running tests...\n";

  auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
  backend->Initialize(0);

  signal_gen::LfmParams params;
  params.f_start = 1e6;
  params.f_end = 2e6;
  params.amplitude = 1.0;
  params.complex_iq = true;

  signal_gen::SystemSampling system;
  system.fs = 12e6;
  system.length = 4096;

  cl_context cl_ctx = static_cast<cl_context>(backend->GetNativeContext());
  cl_command_queue cl_queue = static_cast<cl_command_queue>(backend->GetNativeQueue());

  // ── Test 1: Zero delay vs standard LfmGenerator ──
  {
    signal_gen::LfmGeneratorAnalyticalDelay gen(backend.get(), params);
    gen.SetSampling(system);
    gen.SetDelays({0.0f});

    auto result = gen.GenerateToGpu();
    std::vector<std::complex<float>> gpu_data(system.length);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        system.length * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    // Standard LFM (no delay)
    signal_gen::LfmGenerator lfm_ref(backend.get(), params);
    cl_mem ref_buf = lfm_ref.GenerateToGpu(system, 1);
    std::vector<std::complex<float>> ref_data(system.length);
    clEnqueueReadBuffer(cl_queue, ref_buf, CL_TRUE, 0,
                        system.length * sizeof(std::complex<float>),
                        ref_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(ref_buf);

    float max_err = 0.0f;
    for (size_t n = 0; n < system.length; ++n) {
      float e = std::abs(gpu_data[n] - ref_data[n]);
      if (e > max_err) max_err = e;
    }

    std::cout << "  Test 1 (zero delay vs LfmGenerator): max_err = "
              << std::scientific << std::setprecision(2) << max_err;
    if (max_err < 1e-3f) {
      std::cout << " PASSED\n";
    } else {
      std::cout << " FAILED (expected < 1e-3)\n";
    }
  }

  // ── Test 2: Delay 3.24 samples → first non-zero at index 4 ──
  {
    // delay_us such that delay_samples = 3.24 at fs=12e6
    // delay_us = 3.24 / fs * 1e6 = 3.24 / 12e6 * 1e6 = 0.27 us
    float delay_us_val = 3.24f / static_cast<float>(system.fs) * 1e6f;

    signal_gen::LfmGeneratorAnalyticalDelay gen(backend.get(), params);
    gen.SetSampling(system);
    gen.SetDelays({delay_us_val});

    auto result = gen.GenerateToGpu();
    std::vector<std::complex<float>> gpu_data(system.length);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        system.length * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    // Check: indices 0..3 should be zero, index 4 should be non-zero
    bool zeros_ok = true;
    for (int n = 0; n < 4; ++n) {
      if (std::abs(gpu_data[n]) > 1e-6f) {
        zeros_ok = false;
        std::cout << "  WARNING: index " << n << " not zero: "
                  << std::abs(gpu_data[n]) << "\n";
      }
    }
    bool first_nonzero_ok = std::abs(gpu_data[4]) > 0.1f;

    std::cout << "  Test 2 (delay 3.24 samp, first nonzero at 4): "
              << "zeros[0..3]=" << (zeros_ok ? "OK" : "FAIL")
              << " nonzero[4]=" << std::abs(gpu_data[4]);
    if (zeros_ok && first_nonzero_ok) {
      std::cout << " PASSED\n";
    } else {
      std::cout << " FAILED\n";
    }
  }

  // ── Test 3: GPU vs CPU reference ──
  {
    float delay_us_val = 0.5f;  // 0.5 us

    signal_gen::LfmGeneratorAnalyticalDelay gen(backend.get(), params);
    gen.SetSampling(system);
    gen.SetDelays({delay_us_val});

    auto result = gen.GenerateToGpu();
    std::vector<std::complex<float>> gpu_data(system.length);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        system.length * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    auto cpu_data = gen.GenerateToCpu();

    float max_err = 0.0f;
    for (size_t n = 0; n < system.length; ++n) {
      float e = std::abs(gpu_data[n] - cpu_data[0][n]);
      if (e > max_err) max_err = e;
    }

    std::cout << "  Test 3 (GPU vs CPU, delay 0.5us): max_err = "
              << std::scientific << std::setprecision(2) << max_err;
    if (max_err < 1e-3f) {
      std::cout << " PASSED\n";
    } else {
      std::cout << " FAILED (expected < 1e-3)\n";
    }
  }

  // ── Test 4: Multi-antenna ──
  {
    std::vector<float> delays = {0.0f, 0.1f, 0.2f, 0.5f};
    uint32_t ant = static_cast<uint32_t>(delays.size());

    signal_gen::LfmGeneratorAnalyticalDelay gen(backend.get(), params);
    gen.SetSampling(system);
    gen.SetDelays(delays);

    auto result = gen.GenerateToGpu();
    size_t total = static_cast<size_t>(ant) * system.length;
    std::vector<std::complex<float>> gpu_flat(total);
    clEnqueueReadBuffer(cl_queue, result.data, CL_TRUE, 0,
                        total * sizeof(std::complex<float>),
                        gpu_flat.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    auto cpu_data = gen.GenerateToCpu();

    float max_err = 0.0f;
    for (uint32_t a = 0; a < ant; ++a) {
      for (size_t n = 0; n < system.length; ++n) {
        float e = std::abs(gpu_flat[a * system.length + n] - cpu_data[a][n]);
        if (e > max_err) max_err = e;
      }
    }

    std::cout << "  Test 4 (multi-antenna " << ant << " ch): max_err = "
              << std::scientific << std::setprecision(2) << max_err;
    if (max_err < 1e-3f) {
      std::cout << " PASSED\n";
    } else {
      std::cout << " FAILED (expected < 1e-3)\n";
    }
  }

  std::cout << "[LfmAnalyticalDelay] All tests completed.\n";
}

}  // namespace test_lfm_analytical_delay
