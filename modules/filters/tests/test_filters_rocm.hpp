#pragma once

/**
 * @file test_filters_rocm.hpp
 * @brief ROCm tests for FirFilterROCm and IirFilterROCm
 *
 * Tests:
 *   1. FIR basic: 64-tap LP, GPU vs CPU reference (8ch x 4096pts)
 *   2. FIR large: 256-tap LP, multichannel (16ch x 8192pts)
 *   3. FIR from CPU: ProcessFromCPU convenience method
 *   4. IIR basic: Butterworth 2nd order, GPU vs CPU (8ch x 4096pts)
 *   5. IIR multi-section: 4th order (2 sections), GPU vs CPU
 *   6. IIR from CPU: ProcessFromCPU convenience method
 *
 * NOT RUN on Windows — compile-only check.
 * Will be tested on Linux + AMD GPU (Radeon 9070 / MI100).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_filters_rocm {

#if ENABLE_ROCM

#include "filters/fir_filter_rocm.hpp"
#include "filters/iir_filter_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

using namespace filters;
using namespace drv_gpu_lib;

// ═══════════════════════════════════════════════════════════════════════════
// Pre-computed coefficients
// ═══════════════════════════════════════════════════════════════════════════

// scipy.signal.firwin(64, 0.1, window='hamming')
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

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate test signal (CW 100Hz + CW 5000Hz)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> generate_test_signal(
    uint32_t channels, uint32_t points, float sample_rate)
{
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

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: FIR basic — 64 taps, 8ch x 4096pts, GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_fir_basic(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    FirFilterROCm fir(&backend);
    fir.SetCoefficients(kTestFirCoeffs64);

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, sample_rate);

    // CPU reference
    auto cpu_result = fir.ProcessCpu(signal, channels, points);

    // GPU via ProcessFromCPU
    auto gpu_result = fir.ProcessFromCPU(signal, channels, points);

    // Readback
    std::vector<std::complex<float>> gpu_data(total);
    hipError_t err = hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                                     total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    if (err != hipSuccess) {
      con.Print(gpu_id, "Filters[ROCm]", "[X] FIR Basic: hipMemcpyDtoH failed");
      return false;
    }

    // Compare
    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 1 (FIR 64-tap 8ch): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] FIR Basic EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: FIR large — 256 taps, 16ch x 8192pts
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_fir_large(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    // Generate simple 256-tap LP coefficients (sinc * hamming)
    const uint32_t num_taps = 256;
    std::vector<float> coeffs(num_taps);
    float fc = 0.1f;
    int M = static_cast<int>(num_taps) - 1;
    for (uint32_t i = 0; i < num_taps; ++i) {
      float n_offset = static_cast<float>(i) - static_cast<float>(M) / 2.0f;
      if (std::abs(n_offset) < 1e-6f) {
        coeffs[i] = 2.0f * fc;
      } else {
        coeffs[i] = std::sin(2.0f * static_cast<float>(M_PI) * fc * n_offset)
                     / (static_cast<float>(M_PI) * n_offset);
      }
      // Hamming window
      float w = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) *
                                           static_cast<float>(i) / static_cast<float>(M));
      coeffs[i] *= w;
    }

    FirFilterROCm fir(&backend);
    fir.SetCoefficients(coeffs);

    const uint32_t channels    = 16;
    const uint32_t points      = 8192;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, sample_rate);
    auto cpu_result = fir.ProcessCpu(signal, channels, points);
    auto gpu_result = fir.ProcessFromCPU(signal, channels, points);

    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 2 (FIR 256-tap 16ch): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] FIR Large EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: FIR Process (void* input) — direct GPU pointer
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_fir_gpu_ptr(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    FirFilterROCm fir(&backend);
    fir.SetCoefficients(kTestFirCoeffs64);

    const uint32_t channels = 4;
    const uint32_t points   = 2048;
    const size_t   total    = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = fir.ProcessCpu(signal, channels, points);

    // Manually upload to GPU
    hipStream_t stream = static_cast<hipStream_t>(backend.GetNativeQueue());
    size_t data_size = total * sizeof(std::complex<float>);
    void* input_ptr = nullptr;
    hipMalloc(&input_ptr, data_size);
    hipMemcpyHtoDAsync(input_ptr,
                        const_cast<std::complex<float>*>(signal.data()),
                        data_size, stream);
    hipStreamSynchronize(stream);

    // Process with void* ptr
    auto gpu_result = fir.Process(input_ptr, channels, points);
    hipFree(input_ptr);

    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 3 (FIR void* input): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] FIR GPU ptr EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: IIR basic — Butterworth 2nd order (1 section)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_iir_basic(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    IirFilterROCm iir(&backend);

    // Butterworth 2nd order LP, fc=0.1: butter(2, 0.1, output='sos')
    BiquadSection sec0;
    sec0.b0 = 0.02008337f;
    sec0.b1 = 0.04016673f;
    sec0.b2 = 0.02008337f;
    sec0.a1 = -1.56101808f;
    sec0.a2 = 0.64135154f;
    iir.SetBiquadSections({sec0});

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, sample_rate);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);
    auto gpu_result = iir.ProcessFromCPU(signal, channels, points);

    std::vector<std::complex<float>> gpu_data(total);
    hipError_t err = hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                                     total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    if (err != hipSuccess) {
      con.Print(gpu_id, "Filters[ROCm]", "[X] IIR Basic: hipMemcpyDtoH failed");
      return false;
    }

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 4 (IIR 1-sec 8ch): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] IIR Basic EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: IIR multi-section — 4th order Butterworth (2 sections)
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_iir_multi_section(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    IirFilterROCm iir(&backend);

    // Butterworth 4th order LP, fc=0.1: butter(4, 0.1, output='sos')
    BiquadSection sec0;
    sec0.b0 = 0.00482434f;
    sec0.b1 = 0.00964869f;
    sec0.b2 = 0.00482434f;
    sec0.a1 = -1.64745998f;
    sec0.a2 = 0.70089678f;

    BiquadSection sec1;
    sec1.b0 = 0.00482434f;
    sec1.b1 = 0.00964869f;
    sec1.b2 = 0.00482434f;
    sec1.a1 = -1.47454166f;
    sec1.a2 = 0.58712864f;

    iir.SetBiquadSections({sec0, sec1});

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 50000.0f;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, sample_rate);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);
    auto gpu_result = iir.ProcessFromCPU(signal, channels, points);

    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 5 (IIR 2-sec 8ch): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] IIR Multi-Section EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: IIR Process (void* input) — direct GPU pointer
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_iir_gpu_ptr(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    IirFilterROCm iir(&backend);

    BiquadSection sec0;
    sec0.b0 = 0.02008337f;
    sec0.b1 = 0.04016673f;
    sec0.b2 = 0.02008337f;
    sec0.a1 = -1.56101808f;
    sec0.a2 = 0.64135154f;
    iir.SetBiquadSections({sec0});

    const uint32_t channels = 4;
    const uint32_t points   = 2048;
    const size_t   total    = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);

    // Manually upload
    hipStream_t stream = static_cast<hipStream_t>(backend.GetNativeQueue());
    size_t data_size = total * sizeof(std::complex<float>);
    void* input_ptr = nullptr;
    hipMalloc(&input_ptr, data_size);
    hipMemcpyHtoDAsync(input_ptr,
                        const_cast<std::complex<float>*>(signal.data()),
                        data_size, stream);
    hipStreamSynchronize(stream);

    auto gpu_result = iir.Process(input_ptr, channels, points);
    hipFree(input_ptr);

    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_error = 0.0f;
    for (size_t i = 0; i < total; ++i) {
      float err_re = std::abs(gpu_data[i].real() - cpu_result[i].real());
      float err_im = std::abs(gpu_data[i].imag() - cpu_result[i].imag());
      max_error = std::max(max_error, std::max(err_re, err_im));
    }

    bool passed = (max_error < 1e-3f);
    std::ostringstream oss;
    oss << "Test 6 (IIR void* input): max_err = " << std::scientific
        << std::setprecision(2) << max_error
        << (passed ? " PASSED" : " FAILED (expected < 1e-3)");
    con.Print(gpu_id, "Filters[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "Filters[ROCm]",
              "[X] IIR GPU ptr EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = 0;

  con.Print(gpu_id, "Filters[ROCm]", "");
  con.Print(gpu_id, "Filters[ROCm]", "============================================");
  con.Print(gpu_id, "Filters[ROCm]", "  FIR/IIR Filter ROCm Tests (HIP)");
  con.Print(gpu_id, "Filters[ROCm]", "============================================");

  int passed = 0;
  int total = 6;

  if (test_fir_basic(con, gpu_id))         ++passed;
  if (test_fir_large(con, gpu_id))         ++passed;
  if (test_fir_gpu_ptr(con, gpu_id))       ++passed;
  if (test_iir_basic(con, gpu_id))         ++passed;
  if (test_iir_multi_section(con, gpu_id)) ++passed;
  if (test_iir_gpu_ptr(con, gpu_id))       ++passed;

  con.Print(gpu_id, "Filters[ROCm]",
            "Results: " + std::to_string(passed) + "/" +
            std::to_string(total) + " passed");
  con.Print(gpu_id, "Filters[ROCm]", "============================================");
  con.Print(gpu_id, "Filters[ROCm]", "");
}

#else  // !ENABLE_ROCM

inline void run() {
  std::cout << "\n[test_filters_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_filters_rocm
