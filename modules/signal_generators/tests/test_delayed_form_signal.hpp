#pragma once

/**
 * @file test_delayed_form_signal.hpp
 * @brief Тесты DelayedFormSignalGenerator (Farrow 48×5)
 *
 * 1. Целая задержка (integer delay) — GPU vs CPU shift
 * 2. Дробная задержка (fractional delay) — GPU vs CPU Lagrange
 * 3. Multi-channel — 4 антенны с разными задержками
 * 4. Нулевая задержка — совпадает с FormSignalGenerator
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/delayed_form_signal_generator.hpp"
#include "generators/form_signal_generator.hpp"
#include "params/form_params.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_delayed_form_signal {

// ════════════════════════════════════════════════════════════════════════════
// CPU reference: getX without noise
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GetXClean(
    double fs, uint32_t points, double f0, double amplitude,
    double phase, double fdev, double norm_val) {

  double dt = 1.0 / fs;
  double ti = static_cast<double>(points) * dt;
  std::vector<std::complex<float>> out(points);

  for (uint32_t i = 0; i < points; ++i) {
    double t = static_cast<double>(i) * dt;

    if (t < 0.0 || t > ti - dt) {
      out[i] = {0.0f, 0.0f};
      continue;
    }

    double t_centered = t - ti * 0.5;
    double ph = 2.0 * M_PI * f0 * t
              + M_PI * fdev / ti * (t_centered * t_centered)
              + phase;

    out[i] = {
        static_cast<float>(amplitude * norm_val * std::cos(ph)),
        static_cast<float>(amplitude * norm_val * std::sin(ph))
    };
  }

  return out;
}

// ════════════════════════════════════════════════════════════════════════════
// Built-in Lagrange matrix (first 3 rows for basic validation)
// Full matrix is embedded in delayed_form_signal_generator.cpp
// ════════════════════════════════════════════════════════════════════════════

static const float kLagrangeMatrix[48][5] = {
  { 0.0f,     1.0f,     0.0f,     0.0f,     0.0f},
  {-0.0052f,  1.0417f, -0.0417f,  0.0052f,  0.0f},
  {-0.01f,    1.08f,   -0.08f,    0.01f,    0.0f},
  {-0.0143f,  1.1143f, -0.1143f,  0.0143f,  0.0f},
  {-0.018f,   1.144f,  -0.144f,   0.018f,   0.0f},
  {-0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f},
  {-0.0228f,  1.1827f, -0.1827f,  0.0228f,  0.0f},
  {-0.0239f,  1.1914f, -0.1914f,  0.0239f,  0.0f},
  {-0.024f,   1.2f,    -0.2f,     0.024f,   0.0f},
  {-0.0231f,  1.1914f, -0.1914f,  0.0231f,  0.0f},
  {-0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f},
  {-0.0169f,  1.1198f, -0.1198f,  0.0169f,  0.0f},
  {-0.0111f,  1.0432f, -0.0432f,  0.0111f,  0.0f},
  { 0.0026f,  0.9323f,  0.0677f, -0.0026f,  0.0f},
  { 0.0229f,  0.7812f,  0.2188f, -0.0229f,  0.0f},
  { 0.0507f,  0.5823f,  0.4177f, -0.0507f,  0.0f},
  { 0.0859f,  0.3281f,  0.6719f, -0.0859f,  0.0f},
  { 0.1276f,  0.0104f,  0.9896f, -0.1276f,  0.0f},
  { 0.175f,  -0.3802f,  1.3802f, -0.175f,   0.0f},
  { 0.2274f, -0.8385f,  1.8385f, -0.2274f,  0.0f},
  { 0.2839f, -1.3567f,  2.3567f, -0.2839f,  0.0f},
  { 0.3438f, -1.9375f,  2.9375f, -0.3438f,  0.0f},
  { 0.4063f, -2.5846f,  3.5846f, -0.4063f,  0.0f},
  { 0.4705f, -3.2917f,  4.2917f, -0.4705f,  0.0f},
  { 0.5355f, -4.0521f,  5.0521f, -0.5355f,  0.0f},
  { 0.6f,    -4.8594f,  5.8594f, -0.6f,     0.0f},
  { 0.6628f, -5.7083f,  6.7083f, -0.6628f,  0.0f},
  { 0.7227f, -6.592f,   7.592f,  -0.7227f,  0.0f},
  { 0.7786f, -7.5052f,  8.5052f, -0.7786f,  0.0f},
  { 0.8293f, -8.4411f,  9.4411f, -0.8293f,  0.0f},
  { 0.8734f, -9.3937f, 10.3937f, -0.8734f,  0.0f},
  { 0.9102f,-10.3567f, 11.3567f, -0.9102f,  0.0f},
  { 0.9384f,-11.3229f, 12.3229f, -0.9384f,  0.0f},
  { 0.957f, -12.2857f, 13.2857f, -0.957f,   0.0f},
  { 0.9648f,-13.2386f, 14.2386f, -0.9648f,  0.0f},
  { 0.9609f,-14.1748f, 15.1748f, -0.9609f,  0.0f},
  { 0.9446f,-14.9877f, 16.0f,    -0.9446f,  0.0f},
  { 0.9141f,-15.6684f, 16.75f,   -0.9141f,  0.0f},
  { 0.8684f,-16.2096f, 17.4219f, -0.8684f,  0.0f},
  { 0.8066f,-16.6039f, 18.0078f, -0.8066f,  0.0f},
  { 0.7275f,-16.8438f, 18.5f,    -0.7275f,  0.0f},
  { 0.6299f,-16.9219f, 19.0f,    -0.6299f,  0.0f},
  { 0.5126f,-16.8301f, 19.3984f, -0.5126f,  0.0f},
  { 0.3745f,-16.5605f, 19.6875f, -0.3745f,  0.0f},
  { 0.2143f,-16.0955f, 19.875f,  -0.2143f,  0.0f},
  { 0.0307f,-15.4175f, 20.0f,    -0.0307f,  0.0f},
  {-0.1816f,-14.5086f, 20.0234f,  0.1816f,  0.0f},
  {-0.4347f,-13.3521f, 20.0547f,  0.4347f,  0.0f}
};

// CPU reference: apply delay using Lagrange 48×5 (DelayedFormSignal_Kernel_CORRECT)
// read_pos = n - delay_samples, center = floor(read_pos), frac = read_pos - center, row = frac*48
inline std::vector<std::complex<float>> ApplyDelayRef(
    const std::vector<std::complex<float>>& input,
    float delay_samples) {

  int N = static_cast<int>(input.size());
  std::vector<std::complex<float>> output(N, {0.0f, 0.0f});

  for (int n = 0; n < N; ++n) {
    float read_pos = static_cast<float>(n) - delay_samples;
    if (read_pos < 0.0f) continue;  // output[n] = 0

    int center = static_cast<int>(std::floor(read_pos));
    float frac = read_pos - static_cast<float>(center);
    int row = static_cast<int>(frac * 48.0f) % 48;
    const float* L = kLagrangeMatrix[row];

    std::complex<float> val(0.0f, 0.0f);
    for (int k = 0; k < 5; ++k) {
      int idx = center - 1 + k;
      if (idx >= 0 && idx < N) {
        val += L[k] * input[idx];
      }
    }
    output[n] = val;
  }

  return output;
}

inline float MaxError(const std::complex<float>* a,
                      const std::complex<float>* b, size_t n) {
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = std::abs(a[i] - b[i]);
    if (d > max_err) max_err = d;
  }
  return max_err;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Integer delay
// ════════════════════════════════════════════════════════════════════════════

inline bool TestIntegerDelay(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [DelayedSig 1] Integer delay (5 samples)...\n";

  double fs = 1e6;
  uint32_t points = 4096;
  double f0 = 50000.0;
  double norm_val = 1.0 / std::sqrt(2.0);

  // 5 мкс при fs=1MHz → 5.0 сэмплов (целая)
  float delay_us = 5.0f;
  float delay_samples = delay_us * 1e-6f * static_cast<float>(fs);

  signal_gen::FormParams p;
  p.fs = fs;
  p.antennas = 1;
  p.points = points;
  p.f0 = f0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.norm = norm_val;

  signal_gen::DelayedFormSignalGenerator gen(backend);
  gen.SetParams(p);
  gen.SetDelays({delay_us});

  auto gpu_data = gen.GenerateToCpu();

  auto clean = GetXClean(fs, points, f0, 1.0, 0.0, 0.0, norm_val);
  auto ref = ApplyDelayRef(clean, delay_samples);

  float max_err = MaxError(gpu_data[0].data(), ref.data(), points);
  bool pass = max_err < 1e-2f;

  std::cout << "    delay_samples = " << delay_samples << "\n";
  std::cout << "    Max error: " << std::scientific << std::setprecision(2)
            << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Fractional delay
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFractionalDelay(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [DelayedSig 2] Fractional delay (2.7 samples)...\n";

  double fs = 1e6;
  uint32_t points = 4096;
  double f0 = 50000.0;
  double norm_val = 1.0 / std::sqrt(2.0);

  float delay_us = 2.7f;
  float delay_samples = delay_us * 1e-6f * static_cast<float>(fs);

  signal_gen::FormParams p;
  p.fs = fs;
  p.antennas = 1;
  p.points = points;
  p.f0 = f0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.norm = norm_val;

  signal_gen::DelayedFormSignalGenerator gen(backend);
  gen.SetParams(p);
  gen.SetDelays({delay_us});

  auto gpu_data = gen.GenerateToCpu();

  auto clean = GetXClean(fs, points, f0, 1.0, 0.0, 0.0, norm_val);
  auto ref = ApplyDelayRef(clean, delay_samples);

  float max_err = MaxError(gpu_data[0].data(), ref.data(), points);
  bool pass = max_err < 1e-2f;

  std::cout << "    delay_samples = " << delay_samples << "\n";
  std::cout << "    Max error: " << std::scientific << std::setprecision(2)
            << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Multi-channel
// ════════════════════════════════════════════════════════════════════════════

inline bool TestMultiChannel(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [DelayedSig 3] Multi-channel (4 antennas)...\n";

  double fs = 1e6;
  uint32_t points = 4096;
  uint32_t antennas = 4;
  double f0 = 50000.0;
  double norm_val = 1.0 / std::sqrt(2.0);

  std::vector<float> delays = {0.0f, 1.5f, 3.0f, 4.5f};

  signal_gen::FormParams p;
  p.fs = fs;
  p.antennas = antennas;
  p.points = points;
  p.f0 = f0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.norm = norm_val;

  signal_gen::DelayedFormSignalGenerator gen(backend);
  gen.SetParams(p);
  gen.SetDelays(delays);

  auto gpu_data = gen.GenerateToCpu();
  auto clean = GetXClean(fs, points, f0, 1.0, 0.0, 0.0, norm_val);

  bool all_pass = true;

  for (uint32_t ch = 0; ch < antennas; ++ch) {
    float delay_samples = delays[ch] * 1e-6f * static_cast<float>(fs);
    auto ref = ApplyDelayRef(clean, delay_samples);
    float err = MaxError(gpu_data[ch].data(), ref.data(), points);
    bool pass = err < 1e-2f;
    if (!pass) all_pass = false;

    std::cout << "    ch" << ch << ": delay=" << std::setprecision(1)
              << delays[ch] << "us (" << delay_samples << " samp)"
              << " err=" << std::scientific << std::setprecision(2)
              << err << std::fixed << " " << (pass ? "OK" : "FAIL") << "\n";
  }

  std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
  return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Zero delay = FormSignalGenerator
// ════════════════════════════════════════════════════════════════════════════

inline bool TestZeroDelay(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [DelayedSig 4] Zero delay = FormSignalGenerator...\n";

  signal_gen::FormParams p;
  p.fs = 1e6;
  p.antennas = 1;
  p.points = 4096;
  p.f0 = 100000.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.norm = 1.0 / std::sqrt(2.0);

  // Delayed with 0 delay
  signal_gen::DelayedFormSignalGenerator dgen(backend);
  dgen.SetParams(p);
  dgen.SetDelays({0.0f});
  auto delayed = dgen.GenerateToCpu();

  // Original
  signal_gen::FormSignalGenerator fgen(backend);
  fgen.SetParams(p);
  auto original = fgen.GenerateToCpu();

  float max_err = MaxError(
      delayed[0].data(), original[0].data(), p.points);
  bool pass = max_err < 1e-4f;

  std::cout << "    Max error vs FormSignalGenerator: " << std::scientific
            << std::setprecision(2) << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n";
  std::cout << "════════════════════════════════════════════════════════════\n";
  std::cout << "  DelayedFormSignalGenerator Tests (Farrow 48x5)\n";
  std::cout << "════════════════════════════════════════════════════════════\n";

  try {
    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clGetPlatformIDs failed");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clGetDeviceIDs failed");

    cl_context context = clCreateContext(
        nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("clCreateContext failed");

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device, props, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) {
      clReleaseContext(context);
      throw std::runtime_error("clCreateCommandQueue failed");
    }

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name,
                    nullptr);
    std::cout << "  GPU: " << dev_name << "\n";

    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->InitializeFromExternalContext(context, device, queue);

    int passed = 0, total = 0;
    auto runTest = [&](bool result) { total++; if (result) passed++; };

    runTest(TestIntegerDelay(backend.get()));
    runTest(TestFractionalDelay(backend.get()));
    runTest(TestMultiChannel(backend.get()));
    runTest(TestZeroDelay(backend.get()));

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  DelayedFormSig Results: " << passed << "/" << total
              << " tests passed\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";

    backend.reset();
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return (passed == total) ? 0 : 1;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

} // namespace test_delayed_form_signal
