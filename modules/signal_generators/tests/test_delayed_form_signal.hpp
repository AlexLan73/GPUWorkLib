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

// 4-точечная кубическая Lagrange-интерполяция, узлы {-1,0,1,2}
// Соответствует kBuiltinLagrangeMatrix в delayed_form_signal_generator.cpp
// Генерация: tools/compute_lagrange_matrix.py
static const float kLagrangeMatrix[48][5] = {
  {-0.00000000f,  1.00000000f,  0.00000000f, -0.00000000f,  0.00000000f},  // Row  0
  {-0.00672894f,  0.98915383f,  0.02104583f, -0.00347072f,  0.00000000f},  // Row  1
  {-0.01303289f,  0.97746672f,  0.04249855f, -0.00693239f,  0.00000000f},  // Row  2
  {-0.01892090f,  0.96496582f,  0.06433105f, -0.01037598f,  0.00000000f},  // Row  3
  {-0.02440201f,  0.95167824f,  0.08651620f, -0.01379244f,  0.00000000f},  // Row  4
  {-0.02948526f,  0.93763111f,  0.10902687f, -0.01717273f,  0.00000000f},  // Row  5
  {-0.03417969f,  0.92285156f,  0.13183594f, -0.02050781f,  0.00000000f},  // Row  6
  {-0.03849435f,  0.90736672f,  0.15491627f, -0.02378864f,  0.00000000f},  // Row  7
  {-0.04243827f,  0.89120370f,  0.17824074f, -0.02700617f,  0.00000000f},  // Row  8
  {-0.04602051f,  0.87438965f,  0.20178223f, -0.03015137f,  0.00000000f},  // Row  9
  {-0.04925010f,  0.85695168f,  0.22551360f, -0.03321518f,  0.00000000f},  // Row 10
  {-0.05213608f,  0.83891692f,  0.24940773f, -0.03618857f,  0.00000000f},  // Row 11
  {-0.05468750f,  0.82031250f,  0.27343750f, -0.03906250f,  0.00000000f},  // Row 12
  {-0.05691340f,  0.80116555f,  0.29757577f, -0.04182792f,  0.00000000f},  // Row 13
  {-0.05882282f,  0.78150318f,  0.32179543f, -0.04447579f,  0.00000000f},  // Row 14
  {-0.06042480f,  0.76135254f,  0.34606934f, -0.04699707f,  0.00000000f},  // Row 15
  {-0.06172840f,  0.74074074f,  0.37037037f, -0.04938272f,  0.00000000f},  // Row 16
  {-0.06274263f,  0.71969491f,  0.39467140f, -0.05162369f,  0.00000000f},  // Row 17
  {-0.06347656f,  0.69824219f,  0.41894531f, -0.05371094f,  0.00000000f},  // Row 18
  {-0.06393922f,  0.67640969f,  0.44316497f, -0.05563543f,  0.00000000f},  // Row 19
  {-0.06413966f,  0.65422454f,  0.46730324f, -0.05738812f,  0.00000000f},  // Row 20
  {-0.06408691f,  0.63171387f,  0.49133301f, -0.05895996f,  0.00000000f},  // Row 21
  {-0.06379003f,  0.60890480f,  0.51522714f, -0.06034192f,  0.00000000f},  // Row 22
  {-0.06325804f,  0.58582447f,  0.53895851f, -0.06152494f,  0.00000000f},  // Row 23
  {-0.06250000f,  0.56250000f,  0.56250000f, -0.06250000f,  0.00000000f},  // Row 24 (μ=0.5)
  {-0.06152494f,  0.53895851f,  0.58582447f, -0.06325804f,  0.00000000f},  // Row 25
  {-0.06034192f,  0.51522714f,  0.60890480f, -0.06379003f,  0.00000000f},  // Row 26
  {-0.05895996f,  0.49133301f,  0.63171387f, -0.06408691f,  0.00000000f},  // Row 27
  {-0.05738812f,  0.46730324f,  0.65422454f, -0.06413966f,  0.00000000f},  // Row 28
  {-0.05563543f,  0.44316497f,  0.67640969f, -0.06393922f,  0.00000000f},  // Row 29
  {-0.05371094f,  0.41894531f,  0.69824219f, -0.06347656f,  0.00000000f},  // Row 30
  {-0.05162369f,  0.39467140f,  0.71969491f, -0.06274263f,  0.00000000f},  // Row 31
  {-0.04938272f,  0.37037037f,  0.74074074f, -0.06172840f,  0.00000000f},  // Row 32
  {-0.04699707f,  0.34606934f,  0.76135254f, -0.06042480f,  0.00000000f},  // Row 33
  {-0.04447579f,  0.32179543f,  0.78150318f, -0.05882282f,  0.00000000f},  // Row 34
  {-0.04182792f,  0.29757577f,  0.80116555f, -0.05691340f,  0.00000000f},  // Row 35
  {-0.03906250f,  0.27343750f,  0.82031250f, -0.05468750f,  0.00000000f},  // Row 36
  {-0.03618857f,  0.24940773f,  0.83891692f, -0.05213608f,  0.00000000f},  // Row 37
  {-0.03321518f,  0.22551360f,  0.85695168f, -0.04925010f,  0.00000000f},  // Row 38
  {-0.03015137f,  0.20178223f,  0.87438965f, -0.04602051f,  0.00000000f},  // Row 39
  {-0.02700617f,  0.17824074f,  0.89120370f, -0.04243827f,  0.00000000f},  // Row 40
  {-0.02378864f,  0.15491627f,  0.90736672f, -0.03849435f,  0.00000000f},  // Row 41
  {-0.02050781f,  0.13183594f,  0.92285156f, -0.03417969f,  0.00000000f},  // Row 42
  {-0.01717273f,  0.10902687f,  0.93763111f, -0.02948526f,  0.00000000f},  // Row 43
  {-0.01379244f,  0.08651620f,  0.95167824f, -0.02440201f,  0.00000000f},  // Row 44
  {-0.01037598f,  0.06433105f,  0.96496582f, -0.01892090f,  0.00000000f},  // Row 45
  {-0.00693239f,  0.04249855f,  0.97746672f, -0.01303289f,  0.00000000f},  // Row 46
  {-0.00347072f,  0.02104583f,  0.98915383f, -0.00672894f,  0.00000000f},  // Row 47
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
