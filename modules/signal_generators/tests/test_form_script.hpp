#pragma once

/**
 * @file test_form_script.hpp
 * @brief Тесты FormScriptGenerator + on-disk kernel cache
 *
 * 1. DSL генерация — проверка текста GenerateScript()
 * 2. Compile + Generate — GPU vs CPU reference (сравнение с FormSignalGenerator)
 * 3. SaveKernel — проверка файлов на диске (.cl + binary + manifest)
 * 4. LoadKernel — загрузка и генерация (тот же результат)
 * 5. Versioning — повторный SaveKernel → _00 файлы
 * 6. ListKernels — список из manifest.json
 * 7. Chirp + noise — сложная конфигурация через FormScriptGenerator
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/form_script_generator.hpp"
#include "generators/form_signal_generator.hpp"
#include "params/form_params.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_form_script {

// ════════════════════════════════════════════════════════════════════════════
// Helper: MaxError between flat buffers
// ════════════════════════════════════════════════════════════════════════════

inline float MaxError(const std::vector<std::vector<std::complex<float>>>& a,
                      const std::vector<std::vector<std::complex<float>>>& b) {
  float max_err = 0.0f;
  for (size_t ch = 0; ch < a.size() && ch < b.size(); ++ch) {
    for (size_t i = 0; i < a[ch].size() && i < b[ch].size(); ++i) {
      float d = std::abs(a[ch][i] - b[ch][i]);
      if (d > max_err) max_err = d;
    }
  }
  return max_err;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup helper: remove test kernel files
// ════════════════════════════════════════════════════════════════════════════

inline void CleanupTestKernel(const std::string& name) {
  auto kdir = signal_gen::FormScriptGenerator::GetKernelsDir();
  auto bdir = signal_gen::FormScriptGenerator::GetKernelsBinDir();

  // Remove main files
  fs::remove(kdir + "/" + name + ".cl");
  fs::remove(bdir + "/" + name + "_opencl.bin");

  // Remove versioned files _00.._09
  for (int i = 0; i < 10; ++i) {
    char buf[8];
    snprintf(buf, sizeof(buf), "_%02d", i);
    std::string s(buf);
    fs::remove(kdir + "/" + name + s + ".cl");
    fs::remove(bdir + "/" + name + "_opencl" + s + ".bin");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: DSL генерация
// ════════════════════════════════════════════════════════════════════════════

inline bool TestDSLGeneration(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 1] DSL generation...\n";

  signal_gen::FormParams p;
  p.fs = 12e6;
  p.antennas = 8;
  p.points = 4096;
  p.f0 = 1e6;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.1;
  p.tau_base = 0.0;
  p.tau_step = 0.0001;

  signal_gen::FormScriptGenerator gen(backend);
  gen.SetParams(p);

  auto script = gen.GenerateScript();

  bool has_params  = script.find("[Params]") != std::string::npos;
  bool has_defs    = script.find("[Defs]") != std::string::npos;
  bool has_signal  = script.find("[Signal]") != std::string::npos;
  bool has_antennas = script.find("ANTENNAS = 8") != std::string::npos;
  bool has_points  = script.find("POINTS = 4096") != std::string::npos;
  bool has_f0      = script.find("F0 = 1") != std::string::npos;
  bool has_tau     = script.find("TAU_STEP") != std::string::npos;
  bool has_noise   = script.find("randn_re") != std::string::npos;

  bool pass = has_params && has_defs && has_signal && has_antennas
              && has_points && has_f0 && has_tau && has_noise;

  std::cout << "    [Params]: " << (has_params ? "OK" : "FAIL") << "\n";
  std::cout << "    [Defs]:   " << (has_defs ? "OK" : "FAIL") << "\n";
  std::cout << "    [Signal]: " << (has_signal ? "OK" : "FAIL") << "\n";
  std::cout << "    ANTENNAS: " << (has_antennas ? "OK" : "FAIL") << "\n";
  std::cout << "    TAU_STEP: " << (has_tau ? "OK" : "FAIL") << "\n";
  std::cout << "    Noise:    " << (has_noise ? "OK" : "FAIL") << "\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";

  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Compile + Generate vs FormSignalGenerator
// ════════════════════════════════════════════════════════════════════════════

inline bool TestCompileAndGenerate(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 2] Compile + Generate vs FormSignalGenerator...\n";

  signal_gen::FormParams p;
  p.fs = 12e6;
  p.antennas = 4;
  p.points = 2048;
  p.f0 = 500000.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.0;  // без шума для детерминистичного сравнения
  p.phase = 0.5;
  p.norm = 1.0 / std::sqrt(2.0);

  // FormScriptGenerator
  signal_gen::FormScriptGenerator script_gen(backend);
  script_gen.SetParams(p);
  script_gen.Compile();

  auto script_data = script_gen.GenerateToCpu();

  // FormSignalGenerator (reference)
  signal_gen::FormSignalGenerator form_gen(backend);
  form_gen.SetParams(p);
  auto form_data = form_gen.GenerateToCpu();

  float max_err = MaxError(script_data, form_data);
  bool pass = max_err < 1e-3f;

  std::cout << "    Antennas=" << p.antennas << " Points=" << p.points << "\n";
  std::cout << "    Max error (Script vs Form): " << std::scientific
            << std::setprecision(2) << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: SaveKernel — файлы на диске
// ════════════════════════════════════════════════════════════════════════════

inline bool TestSaveKernel(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 3] SaveKernel — disk files...\n";

  const std::string test_name = "test_save_cw";
  CleanupTestKernel(test_name);

  signal_gen::FormParams p;
  p.fs = 10e6;
  p.antennas = 2;
  p.points = 1024;
  p.f0 = 100000.0;
  p.amplitude = 1.0;

  signal_gen::FormScriptGenerator gen(backend);
  gen.SetParams(p);
  gen.Compile();
  gen.SaveKernel(test_name, "Test CW signal for unit test");

  auto kdir = signal_gen::FormScriptGenerator::GetKernelsDir();
  auto bdir = signal_gen::FormScriptGenerator::GetKernelsBinDir();

  bool cl_exists = fs::exists(kdir + "/" + test_name + ".cl");
  bool bin_exists = fs::exists(bdir + "/" + test_name + "_opencl.bin");
  bool manifest_exists = fs::exists(kdir + "/manifest.json");

  // Check manifest contains entry
  bool manifest_has_entry = false;
  if (manifest_exists) {
    std::ifstream f(kdir + "/manifest.json");
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    manifest_has_entry = content.find(test_name) != std::string::npos;
  }

  // Check .cl file has kernel source
  bool cl_has_source = false;
  if (cl_exists) {
    std::ifstream f(kdir + "/" + test_name + ".cl");
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    cl_has_source = content.find("form_script_signal") != std::string::npos;
  }

  bool pass = cl_exists && bin_exists && manifest_exists
              && manifest_has_entry && cl_has_source;

  std::cout << "    .cl file:      " << (cl_exists ? "OK" : "FAIL") << "\n";
  std::cout << "    .bin file:     " << (bin_exists ? "OK" : "FAIL") << "\n";
  std::cout << "    manifest.json: " << (manifest_exists ? "OK" : "FAIL") << "\n";
  std::cout << "    Manifest entry:" << (manifest_has_entry ? "OK" : "FAIL") << "\n";
  std::cout << "    Source valid:  " << (cl_has_source ? "OK" : "FAIL") << "\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";

  // Don't cleanup yet — Test 4 needs these files
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: LoadKernel — загрузка и генерация
// ════════════════════════════════════════════════════════════════════════════

inline bool TestLoadKernel(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 4] LoadKernel + Generate...\n";

  const std::string test_name = "test_save_cw";

  // First: generate reference data with same params
  signal_gen::FormParams p;
  p.fs = 10e6;
  p.antennas = 2;
  p.points = 1024;
  p.f0 = 100000.0;
  p.amplitude = 1.0;

  signal_gen::FormScriptGenerator ref_gen(backend);
  ref_gen.SetParams(p);
  ref_gen.Compile();
  auto ref_data = ref_gen.GenerateToCpu();

  // Now: load from disk and generate
  signal_gen::FormScriptGenerator load_gen(backend);
  load_gen.SetParams(p);  // need params for GetAntennas/GetPoints
  load_gen.LoadKernel(test_name);

  bool is_ready = load_gen.IsReady();
  if (!is_ready) {
    std::cout << "    LoadKernel failed: not ready\n";
    std::cout << "    FAIL\n";
    CleanupTestKernel(test_name);
    return false;
  }

  auto load_data = load_gen.GenerateToCpu();

  float max_err = MaxError(ref_data, load_data);
  bool pass = (max_err < 1e-3f) && is_ready;

  std::cout << "    IsReady after LoadKernel: " << (is_ready ? "OK" : "FAIL") << "\n";
  std::cout << "    Max error (loaded vs compiled): " << std::scientific
            << std::setprecision(2) << max_err << "\n";
  std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";

  CleanupTestKernel(test_name);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: Versioning — повторный SaveKernel создает _00 файлы
// ════════════════════════════════════════════════════════════════════════════

inline bool TestVersioning(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 5] Versioning (_00 rename)...\n";

  const std::string test_name = "test_ver_cw";
  CleanupTestKernel(test_name);

  auto kdir = signal_gen::FormScriptGenerator::GetKernelsDir();
  auto bdir = signal_gen::FormScriptGenerator::GetKernelsBinDir();

  signal_gen::FormParams p;
  p.fs = 10e6;
  p.antennas = 1;
  p.points = 512;
  p.f0 = 50000.0;

  signal_gen::FormScriptGenerator gen(backend);
  gen.SetParams(p);
  gen.Compile();

  // Save #1
  gen.SaveKernel(test_name, "Version 1");

  bool v1_cl = fs::exists(kdir + "/" + test_name + ".cl");
  bool v1_bin = fs::exists(bdir + "/" + test_name + "_opencl.bin");

  // Save #2 (same name → old → _00)
  p.f0 = 100000.0;
  gen.SetParams(p);
  gen.Compile();
  gen.SaveKernel(test_name, "Version 2");

  bool v2_cl = fs::exists(kdir + "/" + test_name + ".cl");
  bool v2_bin = fs::exists(bdir + "/" + test_name + "_opencl.bin");
  bool old_cl = fs::exists(kdir + "/" + test_name + "_00.cl");
  bool old_bin = fs::exists(bdir + "/" + test_name + "_opencl_00.bin");

  // Save #3 (same name → old → _01)
  p.f0 = 200000.0;
  gen.SetParams(p);
  gen.Compile();
  gen.SaveKernel(test_name, "Version 3");

  bool old2_cl = fs::exists(kdir + "/" + test_name + "_01.cl");
  bool old2_bin = fs::exists(bdir + "/" + test_name + "_opencl_01.bin");

  bool pass = v1_cl && v1_bin && v2_cl && v2_bin
              && old_cl && old_bin && old2_cl && old2_bin;

  std::cout << "    Save #1: cl=" << v1_cl << " bin=" << v1_bin << "\n";
  std::cout << "    Save #2: cl=" << v2_cl << " bin=" << v2_bin
            << " old_00=" << old_cl << "/" << old_bin << "\n";
  std::cout << "    Save #3: old_01=" << old2_cl << "/" << old2_bin << "\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";

  CleanupTestKernel(test_name);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: ListKernels
// ════════════════════════════════════════════════════════════════════════════

inline bool TestListKernels(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 6] ListKernels...\n";

  const std::string name1 = "test_list_a";
  const std::string name2 = "test_list_b";
  CleanupTestKernel(name1);
  CleanupTestKernel(name2);

  signal_gen::FormParams p;
  p.fs = 10e6;
  p.antennas = 1;
  p.points = 256;

  signal_gen::FormScriptGenerator gen(backend);

  // Save two kernels
  gen.SetParams(p);
  gen.Compile();
  gen.SaveKernel(name1, "Signal A");

  p.f0 = 1e6;
  gen.SetParams(p);
  gen.Compile();
  gen.SaveKernel(name2, "Signal B");

  auto names = gen.ListKernels();
  bool has_a = false, has_b = false;
  for (auto& n : names) {
    if (n == name1) has_a = true;
    if (n == name2) has_b = true;
  }

  bool pass = has_a && has_b;

  std::cout << "    Total kernels in manifest: " << names.size() << "\n";
  std::cout << "    " << name1 << ": " << (has_a ? "found" : "NOT FOUND") << "\n";
  std::cout << "    " << name2 << ": " << (has_b ? "found" : "NOT FOUND") << "\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";

  CleanupTestKernel(name1);
  CleanupTestKernel(name2);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 7: Chirp + noise — complex config
// ════════════════════════════════════════════════════════════════════════════

inline bool TestChirpWithNoise(drv_gpu_lib::IBackend* backend) {
  std::cout << "\n  [FormScript 7] Chirp + noise...\n";

  signal_gen::FormParams p;
  p.fs = 100000.0;
  p.antennas = 4;
  p.points = 4096;
  p.f0 = 1000.0;
  p.amplitude = 1.0;
  p.noise_amplitude = 0.1;
  p.fdev = 5000.0;
  p.norm = 1.0 / std::sqrt(2.0);
  p.tau_step = 0.001;
  p.noise_seed = 42;

  signal_gen::FormScriptGenerator gen(backend);
  gen.SetParams(p);
  gen.Compile();

  auto data = gen.GenerateToCpu();

  // Basic checks
  bool correct_channels = (data.size() == 4);
  bool correct_points = (data[0].size() == 4096);

  // Check signal is non-trivial
  float max_val = 0.0f;
  for (auto& ch : data) {
    for (auto& s : ch) {
      float mag = std::abs(s);
      if (mag > max_val) max_val = mag;
    }
  }
  bool has_signal = max_val > 0.1f;

  // Check different channels are different (tau_step > 0)
  float ch_diff = 0.0f;
  for (size_t i = 0; i < 100; ++i) {
    ch_diff += std::abs(data[0][i] - data[1][i]);
  }
  bool channels_differ = ch_diff > 1.0f;

  bool pass = correct_channels && correct_points && has_signal && channels_differ;

  std::cout << "    Channels: " << data.size()
            << " Points: " << data[0].size() << "\n";
  std::cout << "    Max amplitude: " << std::setprecision(4) << max_val << "\n";
  std::cout << "    Channel diff (0 vs 1): " << std::setprecision(4)
            << ch_diff << "\n";
  std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n";
  std::cout << "════════════════════════════════════════════════════════════\n";
  std::cout << "  FormScriptGenerator Tests (Этап 2)\n";
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

    runTest(TestDSLGeneration(backend.get()));
    runTest(TestCompileAndGenerate(backend.get()));
    runTest(TestSaveKernel(backend.get()));
    runTest(TestLoadKernel(backend.get()));
    runTest(TestVersioning(backend.get()));
    runTest(TestListKernels(backend.get()));
    runTest(TestChirpWithNoise(backend.get()));

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  FormScript Results: " << passed << "/" << total
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

} // namespace test_form_script
