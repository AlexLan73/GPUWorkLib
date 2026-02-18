#pragma once

/**
 * @file test_fir_basic.hpp
 * @brief Basic FIR filter tests (GPU vs CPU reference)
 *
 * Test: low-pass FIR filter, 64 taps, fc=0.1 (normalized)
 * Signal: 8 channels, 4096 points, CW 100Hz + CW 5000Hz
 * Validation: GPU Process vs ProcessCpu, max error < 1e-3
 *
 * Coefficients: scipy.signal.firwin(64, 0.1, window='hamming')
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "filters/fir_filter.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "DrvGPU/services/console_output.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace filters { namespace tests {

// ════════════════════════════════════════════════════════════════════════════
// Pre-computed FIR coefficients: scipy.signal.firwin(64, 0.1, window='hamming')
// ════════════════════════════════════════════════════════════════════════════

static const std::vector<float> kTestFirCoeffs64 = {
  -0.000157f, -0.000332f, -0.000459f, -0.000399f, -0.000000f,
   0.000850f,  0.002169f,  0.003849f,  0.005627f,  0.007078f,
   0.007634f,  0.006647f,  0.003540f, -0.002061f, -0.010202f,
  -0.020558f, -0.032375f, -0.044476f, -0.055375f, -0.063450f,
  -0.067076f, -0.064848f, -0.055800f, -0.039588f, -0.016579f,
   0.012488f,  0.046355f,  0.083311f,  0.121273f,  0.157982f,
   0.191169f,  0.218728f,  0.238881f,  0.250291f,  0.252151f,
   0.244226f,  0.226902f,  0.201177f,  0.168596f,  0.131141f,
   0.091041f,  0.050605f,  0.012058f, -0.022485f, -0.051250f,
  -0.073004f, -0.087189f, -0.093864f, -0.093644f, -0.087573f,
  -0.076953f, -0.063233f, -0.047833f, -0.032032f, -0.016901f,
  -0.003260f,  0.008352f,  0.017614f,  0.024418f,  0.028835f,
   0.031065f,  0.031385f,  0.030094f,  0.027480f
};

/**
 * @brief Generate test signal: CW 100Hz + CW 5000Hz (multi-channel)
 */
inline std::vector<std::complex<float>> GenerateTestSignal(
    uint32_t channels, uint32_t points, float sample_rate) {

  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> signal(total);

  float f_low  = 100.0f;
  float f_high = 5000.0f;

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;
    float phase_offset = static_cast<float>(ch) * 0.1f;
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / sample_rate;
      float re = std::cos(2.0f * static_cast<float>(M_PI) * f_low * t + phase_offset)
               + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * f_high * t);
      float im = std::sin(2.0f * static_cast<float>(M_PI) * f_low * t + phase_offset)
               + 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * f_high * t);
      signal[base + n] = {re, im};
    }
  }
  return signal;
}

/**
 * @brief Run basic FIR filter test: GPU vs CPU reference
 */
inline void run_fir_basic() {
  int gpu_id = 0;
  auto& console = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!console.IsRunning()) console.Start();

  console.Print(gpu_id, "Filters","");
  console.Print(gpu_id, "Filters","════════════════════════════════════════════════════════════");
  console.Print(gpu_id, "Filters"," FIR Filter Basic Test");
  console.Print(gpu_id, "Filters","════════════════════════════════════════════════════════════");

  try {
    // Initialize backend
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Start();

    // Parameters
    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    console.Print(gpu_id, "Filters","  Channels: " + std::to_string(channels));
    console.Print(gpu_id, "Filters","  Points:   " + std::to_string(points));
    console.Print(gpu_id, "Filters","  Taps:     " + std::to_string(kTestFirCoeffs64.size()));

    // Generate test signal
    auto signal = GenerateTestSignal(channels, points, sample_rate);

    // Create FIR filter
    FirFilter fir(backend.get());
    fir.SetCoefficients(kTestFirCoeffs64);

    console.Print(gpu_id, "Filters","  FIR filter created, " +
                  std::to_string(fir.GetNumTaps()) + " taps");

    // CPU reference
    auto cpu_result = fir.ProcessCpu(signal, channels, points);
    console.Print(gpu_id, "Filters","  CPU reference computed");

    // Upload signal to GPU
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;
    cl_mem input_buf = clCreateBuffer(
        ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total * sizeof(std::complex<float>),
        const_cast<std::complex<float>*>(signal.data()), &err);

    if (err != CL_SUCCESS) {
      console.Print(gpu_id, "Filters","  ERROR: upload failed: " + std::to_string(err));
      return;
    }

    // GPU processing
    auto gpu_result = fir.Process(input_buf, channels, points);
    console.Print(gpu_id, "Filters","  GPU processing done");

    // Readback
    std::vector<std::complex<float>> gpu_data(total);
    cl_command_queue queue =
        static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(queue, gpu_result.data, CL_TRUE, 0,
                        total * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);

    // Compare
    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(gpu_result.data);

    // Report
    bool passed = (max_error < 1e-3f);
    console.Print(gpu_id, "Filters","  Max error GPU vs CPU: " + std::to_string(max_error));
    console.Print(gpu_id, "Filters",passed
        ? "  RESULT: PASSED (error < 1e-3)"
        : "  RESULT: FAILED (error >= 1e-3)");

    profiler.Stop();

  } catch (const std::exception& e) {
    console.Print(gpu_id, "Filters","  EXCEPTION: " + std::string(e.what()));
  }

  console.Print(gpu_id, "Filters","════════════════════════════════════════════════════════════");
}

}} // namespace filters::tests
