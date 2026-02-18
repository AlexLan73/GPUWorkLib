#pragma once

/**
 * @file test_iir_basic.hpp
 * @brief Basic IIR biquad cascade test (GPU vs CPU reference)
 *
 * Test: 2nd order Butterworth low-pass, fc=0.1 (normalized)
 * Signal: 8 channels, 4096 points, CW 100Hz + CW 5000Hz
 * Validation: GPU Process vs ProcessCpu, max error < 1e-3
 *
 * Coefficients: scipy.signal.butter(2, 0.1, output='sos')
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "filters/iir_filter.hpp"
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

/**
 * @brief Run basic IIR biquad cascade test
 *
 * Butterworth 2nd order LP (scipy):
 *   sos = butter(2, 0.1, output='sos')
 *   section 0: b=[0.02008337, 0.04016673, 0.02008337], a=[1, -1.56101808, 0.64135154]
 */
inline void run_iir_basic() {
  int gpu_id = 0;
  auto& console = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!console.IsRunning()) console.Start();

  console.Print(gpu_id, "Filters","");
  console.Print(gpu_id, "Filters","════════════════════════════════════════════════════════════");
  console.Print(gpu_id, "Filters"," IIR Biquad Cascade Basic Test");
  console.Print(gpu_id, "Filters","════════════════════════════════════════════════════════════");

  try {
    // Initialize backend
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Start();

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    console.Print(gpu_id, "Filters","  Channels: " + std::to_string(channels));
    console.Print(gpu_id, "Filters","  Points:   " + std::to_string(points));

    // Butterworth 2nd order LP, fc=0.1 (normalized)
    // scipy: butter(2, 0.1, output='sos')
    BiquadSection sec0;
    sec0.b0 = 0.02008337f;
    sec0.b1 = 0.04016673f;
    sec0.b2 = 0.02008337f;
    sec0.a1 = -1.56101808f;
    sec0.a2 = 0.64135154f;

    std::vector<BiquadSection> sections = { sec0 };
    console.Print(gpu_id, "Filters","  Sections: " + std::to_string(sections.size()));

    // Generate test signal (reuse function from test_fir_basic.hpp)
    auto signal = GenerateTestSignal(channels, points, sample_rate);

    // Create IIR filter
    IirFilter iir(backend.get());
    iir.SetBiquadSections(sections);

    console.Print(gpu_id, "Filters","  IIR filter created");

    // CPU reference
    auto cpu_result = iir.ProcessCpu(signal, channels, points);
    console.Print(gpu_id, "Filters","  CPU reference computed");

    // Upload signal
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
    auto gpu_result = iir.Process(input_buf, channels, points);
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
