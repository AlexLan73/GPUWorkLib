#pragma once

/**
 * @file test_form_signal.hpp
 * @brief Тесты FormSignalGenerator
 *
 * 1. FormSignal без шума, 1 канал — GPU vs CPU reference (getX)
 * 2. FormSignal окно — нулевые значения за пределами [0, ti-dt]
 * 3. FormSignal multi-channel — 8 антенн с TAU_STEP
 * 4. FormSignal шум — статистическая проверка (mean, variance)
 * 5. FormParams парсер — из строки
 * 6. FormSignal chirp (fdev != 0)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/form_signal_generator.hpp"
#include "params/form_params.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_form_signal {

// ════════════════════════════════════════════════════════════════════════════
// CPU reference (Python getX equivalent)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief CPU reference implementation of getX formula (без шума)
 *
 * X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
 * X = 0 if t < 0 or t > ti - dt
 */
inline std::vector<std::complex<float>> GetXReference(
    double fs, uint32_t points, double f0, double amplitude,
    double phase, double fdev, double norm_val, double tau) {

  double dt = 1.0 / fs;
  double ti = static_cast<double>(points) * dt;
  std::vector<std::complex<float>> out(points);

  for (uint32_t i = 0; i < points; ++i) {
    double t = static_cast<double>(i) * dt + tau;

    if (t < 0.0 || t > ti - dt) {
      out[i] = {0.0f, 0.0f};
      continue;
    }

    double t_centered = t - ti * 0.5;
    double ph = 2.0 * M_PI * f0 * t
              + M_PI * fdev / ti * (t_centered * t_centered)
              + phase;

    float re = static_cast<float>(amplitude * norm_val * std::cos(ph));
    float im = static_cast<float>(amplitude * norm_val * std::sin(ph));
    out[i] = {re, im};
  }

  return out;
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
// Test 1: FormSignal без шума — GPU vs CPU reference
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormSignalNoNoise(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormSig 1] No noise, 1 channel — GPU vs CPU ref...\n";

  signal_gen::FormParams p;
  p.fs = 12e6;
  p.antennas = 1;
  p.points = 4096;
  p.f0 = 1e6;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.phase = 0.3;
  p.fdev = 0.0;
  p.norm = 1.0 / std::sqrt(2.0);

  signal_gen::FormSignalGenerator gen(backend);
  gen.SetParams(p);

  auto gpu_data = gen.GenerateToCpu();

  // CPU reference
  auto cpu_ref = GetXReference(
      p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, 0.0);

  float max_err = MaxError(
      gpu_data[0].data(), cpu_ref.data(), p.points);

  bool pass = max_err < 1e-3f;

  std::cout << "    fs=" << p.fs << " f0=" << p.f0
            << " points=" << p.points << "\n";
  std::cout << "    Max error: " << std::scientific << std::setprecision(2)
            << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Окно — проверка нулей за пределами [0, ti-dt]
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormSignalWindow(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormSig 2] Window test (tau shift)...\n";

  // tau=-0.1 → t = sample_id * dt - 0.1
  // При sample_id < 100: t < 0 → за пределами окна → X=0
  // При sample_id >= 100: t >= 0 → в окне → X != 0
  signal_gen::FormParams p;
  p.fs = 1000.0;
  p.antennas = 1;
  p.points = 1000;
  p.f0 = 100.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.tau_base = -0.1;  // negative delay → first 100 samples X=0

  signal_gen::FormSignalGenerator gen(backend);
  gen.SetParams(p);

  auto data = gen.GenerateToCpu();

  // Первые 100 отсчётов должны быть нулевыми (t < 0)
  int zero_count = 0;
  for (int i = 0; i < 100; ++i) {
    if (std::abs(data[0][i]) < 1e-6f) zero_count++;
  }

  // Отсчёты после задержки должны быть ненулевыми (t >= 0, в окне)
  int nonzero_count = 0;
  for (int i = 110; i < 500; ++i) {
    if (std::abs(data[0][i]) > 0.01f) nonzero_count++;
  }

  bool pass = (zero_count >= 99) && (nonzero_count > 350);

  std::cout << "    tau=-0.1s, dt=0.001s\n";
  std::cout << "    Zeros in first 100 samples: " << zero_count << "/100\n";
  std::cout << "    Non-zeros in [110..500]: " << nonzero_count << "/390\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Multi-channel с TAU_STEP
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormSignalMultiChannel(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormSig 3] Multi-channel (8 antennas, TAU_STEP)...\n";

  signal_gen::FormParams p;
  p.fs = 10000.0;
  p.antennas = 8;
  p.points = 2048;
  p.f0 = 500.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.tau_base = 0.0;
  p.tau_step = 0.01;  // 10 ms step → antenna 0: 0ms, antenna 7: 70ms

  signal_gen::FormSignalGenerator gen(backend);
  gen.SetParams(p);

  auto data = gen.GenerateToCpu();

  bool all_pass = true;

  for (uint32_t a = 0; a < p.antennas; ++a) {
    double tau = p.tau_base + a * p.tau_step;
    auto cpu_ref = GetXReference(
        p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, tau);

    float err = MaxError(data[a].data(), cpu_ref.data(), p.points);
    bool pass = err < 1e-3f;
    if (!pass) all_pass = false;

    std::cout << "    Antenna " << a << " (tau=" << std::setprecision(4)
              << tau << "s): err=" << std::scientific << std::setprecision(2)
              << err << std::fixed << " " << (pass ? "OK" : "FAIL") << "\n";
  }

  std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
  return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Шум — статистическая проверка
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormSignalNoise(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormSig 4] Noise statistics (an=1.0)...\n";

  signal_gen::FormParams p;
  p.fs = 10000.0;
  p.antennas = 1;
  p.points = 100000;
  p.f0 = 0.0;           // DC — без сигнальной компоненты синуса
  p.amplitude = 0.0;    // без чистого сигнала
  p.noise_amplitude = 1.0;
  p.norm = 1.0;         // norm=1 для чистой статистики
  p.noise_seed = 42;

  signal_gen::FormSignalGenerator gen(backend);
  gen.SetParams(p);

  auto data = gen.GenerateToCpu();

  // Mean
  double sum_re = 0, sum_im = 0;
  for (auto& d : data[0]) { sum_re += d.real(); sum_im += d.imag(); }
  double mean_re = sum_re / p.points;
  double mean_im = sum_im / p.points;

  // Variance
  double var_re = 0, var_im = 0;
  for (auto& d : data[0]) {
    var_re += (d.real() - mean_re) * (d.real() - mean_re);
    var_im += (d.imag() - mean_im) * (d.imag() - mean_im);
  }
  var_re /= p.points;
  var_im /= p.points;

  // Box-Muller: N(0,1) → variance = 1.0
  // noise = an * norm * N(0,1) → variance = an^2 * norm^2 = 1.0
  bool mean_ok = (std::abs(mean_re) < 0.05) && (std::abs(mean_im) < 0.05);
  bool var_ok = (std::abs(var_re - 1.0) < 0.1) && (std::abs(var_im - 1.0) < 0.1);

  bool pass = mean_ok && var_ok;

  std::cout << "    N=" << p.points << "\n";
  std::cout << "    Mean: re=" << std::setprecision(4) << mean_re
            << " im=" << mean_im << " (expected ~0)\n";
  std::cout << "    Variance: re=" << var_re << " im=" << var_im
            << " (expected ~1.0)\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: FormParams парсер
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormParamsParser() {
  std::cout << "\n  [FormSig 5] FormParams parser...\n";

  auto p = signal_gen::FormParams::ParseFromString(
      "f0=1e6,a=1.5,an=0.1,tau=0.001,antennas=8,points=2048,fs=10e6");

  bool pass = true;
  auto check = [&](const char* name, double actual, double expected) {
    bool ok = std::abs(actual - expected) < 1e-6;
    if (!ok) pass = false;
    std::cout << "    " << name << "=" << actual
              << " (expected " << expected << ") "
              << (ok ? "OK" : "FAIL") << "\n";
  };

  check("f0", p.f0, 1e6);
  check("amplitude", p.amplitude, 1.5);
  check("noise_amplitude", p.noise_amplitude, 0.1);
  check("tau_base", p.tau_base, 0.001);
  check("antennas", p.antennas, 8);
  check("points", p.points, 2048);
  check("fs", p.fs, 10e6);

  // Test unknown key → exception
  bool exception_thrown = false;
  try {
    signal_gen::FormParams::ParseFromString("unknown_key=123");
  } catch (const std::invalid_argument&) {
    exception_thrown = true;
  }
  std::cout << "    Unknown key throws: "
            << (exception_thrown ? "OK" : "FAIL") << "\n";
  if (!exception_thrown) pass = false;

  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Chirp (fdev != 0)
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFormSignalChirp(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormSig 6] Chirp (fdev != 0) GPU vs CPU...\n";

  signal_gen::FormParams p;
  p.fs = 100000.0;
  p.antennas = 1;
  p.points = 4096;
  p.f0 = 1000.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;
  p.fdev = 5000.0;  // 5 kHz chirp
  p.norm = 1.0 / std::sqrt(2.0);

  signal_gen::FormSignalGenerator gen(backend);
  gen.SetParams(p);

  auto gpu_data = gen.GenerateToCpu();

  auto cpu_ref = GetXReference(
      p.fs, p.points, p.f0, p.amplitude, p.phase, p.fdev, p.norm, 0.0);

  float max_err = MaxError(gpu_data[0].data(), cpu_ref.data(), p.points);
  bool pass = max_err < 1e-3f;

  std::cout << "    f0=" << p.f0 << " fdev=" << p.fdev << "\n";
  std::cout << "    Max error: " << std::scientific << std::setprecision(2)
            << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n";
  std::cout << "════════════════════════════════════════════════════════════\n";
  std::cout << "  FormSignalGenerator Tests\n";
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

    runTest(TestFormSignalNoNoise(backend.get()));
    runTest(TestFormSignalWindow(backend.get()));
    runTest(TestFormSignalMultiChannel(backend.get()));
    runTest(TestFormSignalNoise(backend.get()));
    runTest(TestFormParamsParser());
    runTest(TestFormSignalChirp(backend.get()));

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  FormSig Results: " << passed << "/" << total
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

} // namespace test_form_signal
