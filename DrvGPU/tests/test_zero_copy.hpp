#pragma once

/**
 * @file test_zero_copy.hpp
 * @brief Тесты ZeroCopy bridge (OpenCL → ROCm)
 *
 * Тесты:
 * 1. detect_method — определение лучшего ZeroCopy метода
 * 2. export_dma_buf — экспорт cl_mem → dma-buf fd
 * 3. export_gpu_va — экспорт cl_mem → GPU VA (AMD-only)
 * 4. bridge_import — импорт через ZeroCopyBridge
 * 5. data_integrity — запись в cl_mem, чтение через hip_ptr
 * 6. bridge_lifecycle — создание, перемещение, освобождение
 *
 * @note Запускать ТОЛЬКО на Linux + AMD GPU с ROCm!
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../backends/opencl/opencl_backend.hpp"
#include "../backends/opencl/opencl_export.hpp"
#include "../backends/rocm/rocm_backend.hpp"
#include "../backends/rocm/zero_copy_bridge.hpp"
#include "../logger/logger.hpp"

#include <CL/cl.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

namespace test_zero_copy {

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

static void print_test(const std::string& name, bool passed) {
  std::cout << "  [ZeroCopy] " << name << ": "
            << (passed ? "PASSED" : "FAILED") << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Detect ZeroCopy method
// ════════════════════════════════════════════════════════════════════════════

static void test_detect_method() {
  using namespace drv_gpu_lib;

  OpenCLBackend cl_backend;
  cl_backend.Initialize(0);

  cl_device_id device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());

  auto method = DetectBestZeroCopyMethod(device);
  std::cout << "  [ZeroCopy] Detected method: "
            << ZeroCopyMethodToString(method) << "\n";

  // Метод должен быть определён (хотя бы NONE)
  bool passed = true;  // Просто проверяем, что не крэшится

  // Проверка отдельных capabilities
  bool has_dma_buf = SupportsDmaBufExport(device);
  bool has_amd_va = SupportsAmdGpuVA(device);
  std::cout << "  [ZeroCopy]   DMA-BUF support: " << (has_dma_buf ? "YES" : "NO") << "\n";
  std::cout << "  [ZeroCopy]   AMD GPU VA support: " << (has_amd_va ? "YES" : "NO") << "\n";

  cl_backend.Cleanup();
  print_test("detect_method", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Export cl_mem → dma-buf fd
// ════════════════════════════════════════════════════════════════════════════

static void test_export_dma_buf() {
  using namespace drv_gpu_lib;

  OpenCLBackend cl_backend;
  cl_backend.Initialize(0);

  cl_device_id device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());
  if (!SupportsDmaBufExport(device)) {
    std::cout << "  [ZeroCopy] export_dma_buf: SKIPPED (no dma-buf support)\n";
    cl_backend.Cleanup();
    return;
  }

  // Выделяем OpenCL буфер
  const size_t buf_size = 1024 * sizeof(float);
  void* cl_buf = cl_backend.Allocate(buf_size);

  // Экспортируем
  int fd = ExportClBufferToFd(static_cast<cl_mem>(cl_buf));
  bool passed = (fd >= 0);

  std::cout << "  [ZeroCopy]   dma-buf fd = " << fd << "\n";

  cl_backend.Free(cl_buf);
  cl_backend.Cleanup();
  print_test("export_dma_buf", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: Export cl_mem → GPU VA (AMD-only)
// ════════════════════════════════════════════════════════════════════════════

static void test_export_gpu_va() {
  using namespace drv_gpu_lib;

  OpenCLBackend cl_backend;
  cl_backend.Initialize(0);

  cl_device_id device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());
  if (!SupportsAmdGpuVA(device)) {
    std::cout << "  [ZeroCopy] export_gpu_va: SKIPPED (no AMD GPU VA support)\n";
    cl_backend.Cleanup();
    return;
  }

  const size_t buf_size = 1024 * sizeof(float);
  void* cl_buf = cl_backend.Allocate(buf_size);

  void* gpu_va = ExportClBufferToGpuVA(static_cast<cl_mem>(cl_buf));
  bool passed = (gpu_va != nullptr);

  std::cout << "  [ZeroCopy]   GPU VA = 0x" << std::hex
            << reinterpret_cast<uintptr_t>(gpu_va) << std::dec << "\n";

  cl_backend.Free(cl_buf);
  cl_backend.Cleanup();
  print_test("export_gpu_va", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Bridge import (universal)
// ════════════════════════════════════════════════════════════════════════════

static void test_bridge_import() {
  using namespace drv_gpu_lib;

  OpenCLBackend cl_backend;
  cl_backend.Initialize(0);

  ROCmBackend rocm_backend;
  rocm_backend.Initialize(0);

  cl_device_id cl_device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());
  auto method = DetectBestZeroCopyMethod(cl_device);

  if (method == ZeroCopyMethod::NONE) {
    std::cout << "  [ZeroCopy] bridge_import: SKIPPED (no ZeroCopy method available)\n";
    rocm_backend.Cleanup();
    cl_backend.Cleanup();
    return;
  }

  const size_t buf_size = 1024 * sizeof(float);
  void* cl_buf = cl_backend.Allocate(buf_size);

  bool passed = false;
  try {
    ZeroCopyBridge bridge;
    bridge.ImportFromOpenCl(static_cast<cl_mem>(cl_buf), buf_size, cl_device);

    passed = bridge.IsActive() && bridge.GetHipPtr() != nullptr;
    std::cout << "  [ZeroCopy]   Method: " << ZeroCopyMethodToString(bridge.GetMethod()) << "\n";
    std::cout << "  [ZeroCopy]   HIP ptr: 0x" << std::hex
              << reinterpret_cast<uintptr_t>(bridge.GetHipPtr()) << std::dec << "\n";
  } catch (const std::exception& e) {
    std::cout << "  [ZeroCopy]   Exception: " << e.what() << "\n";
  }

  cl_backend.Free(cl_buf);
  rocm_backend.Cleanup();
  cl_backend.Cleanup();
  print_test("bridge_import", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: Data integrity (write via OpenCL, read via HIP)
// ════════════════════════════════════════════════════════════════════════════

static void test_data_integrity() {
  using namespace drv_gpu_lib;

  OpenCLBackend cl_backend;
  cl_backend.Initialize(0);

  ROCmBackend rocm_backend;
  rocm_backend.Initialize(0);

  cl_device_id cl_device = static_cast<cl_device_id>(cl_backend.GetNativeDevice());
  auto method = DetectBestZeroCopyMethod(cl_device);

  if (method == ZeroCopyMethod::NONE) {
    std::cout << "  [ZeroCopy] data_integrity: SKIPPED (no ZeroCopy method)\n";
    rocm_backend.Cleanup();
    cl_backend.Cleanup();
    return;
  }

  const size_t N = 1024;
  const size_t buf_size = N * sizeof(float);

  // 1. Подготовить данные
  std::vector<float> input(N);
  for (size_t i = 0; i < N; ++i) {
    input[i] = static_cast<float>(i) * 0.5f + 1.0f;
  }

  // 2. Записать в OpenCL
  void* cl_buf = cl_backend.Allocate(buf_size);
  cl_backend.MemcpyHostToDevice(cl_buf, input.data(), buf_size);

  // 3. clFinish — данные в VRAM
  cl_backend.Synchronize();

  // 4. ZeroCopy import
  bool passed = false;
  try {
    ZeroCopyBridge bridge;
    bridge.ImportFromOpenCl(static_cast<cl_mem>(cl_buf), buf_size, cl_device);

    // 5. Прочитать через HIP
    std::vector<float> output(N, 0.0f);
    hipError_t err = hipMemcpy(output.data(), bridge.GetHipPtr(),
                                buf_size, hipMemcpyDeviceToHost);

    if (err == hipSuccess) {
      // 6. Сравнить
      float max_error = 0.0f;
      for (size_t i = 0; i < N; ++i) {
        float diff = std::fabs(input[i] - output[i]);
        if (diff > max_error) max_error = diff;
      }

      passed = (max_error < 1e-6f);
      std::cout << "  [ZeroCopy]   Max error: " << max_error << "\n";
    } else {
      std::cout << "  [ZeroCopy]   hipMemcpy error: " << hipGetErrorString(err) << "\n";
    }
  } catch (const std::exception& e) {
    std::cout << "  [ZeroCopy]   Exception: " << e.what() << "\n";
  }

  cl_backend.Free(cl_buf);
  rocm_backend.Cleanup();
  cl_backend.Cleanup();
  print_test("data_integrity", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Bridge lifecycle (create, move, release)
// ════════════════════════════════════════════════════════════════════════════

static void test_bridge_lifecycle() {
  using namespace drv_gpu_lib;

  bool passed = true;

  // Тест 1: Пустой bridge
  {
    ZeroCopyBridge bridge;
    passed &= !bridge.IsActive();
    passed &= (bridge.GetHipPtr() == nullptr);
    passed &= (bridge.GetSize() == 0);
    passed &= (bridge.GetMethod() == ZeroCopyMethod::NONE);
  }

  // Тест 2: Move
  {
    ZeroCopyBridge a;
    ZeroCopyBridge b = std::move(a);
    passed &= !a.IsActive();
    passed &= !b.IsActive();
  }

  // Тест 3: Release на пустом bridge
  {
    ZeroCopyBridge bridge;
    bridge.Release();  // Не должен крэшиться
    passed &= !bridge.IsActive();
  }

  print_test("bridge_lifecycle", passed);
}

// ════════════════════════════════════════════════════════════════════════════
// Run all
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
  std::cout << "\n========== ZeroCopy Bridge Tests ==========\n";

  test_detect_method();
  test_export_dma_buf();
  test_export_gpu_va();
  test_bridge_import();
  test_data_integrity();
  test_bridge_lifecycle();

  std::cout << "========== ZeroCopy Bridge Tests Done ==========\n\n";
}

}  // namespace test_zero_copy

#endif  // ENABLE_ROCM
