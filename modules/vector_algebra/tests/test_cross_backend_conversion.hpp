#pragma once
#if ENABLE_ROCM

/**
 * @file test_cross_backend_conversion.hpp
 * @brief Cross-backend тесты: 85×85 матрица, все пути данных (5.10)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#include <cassert>
#include <complex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "cholesky_inverter_rocm.hpp"
#include "DrvGPU/interface/i_backend.hpp"
#include "DrvGPU/interface/input_data.hpp"
#include "services/console_output.hpp"

// Для другого ROCm контекста
#include "backends/rocm/rocm_backend.hpp"

// Helpers из основных тестов
#include "test_cholesky_inverter_rocm.hpp"

namespace vector_algebra::tests {

// ════════════════════════════════════════════════════════════════════════════
// 5.10.1: TestConvert_VectorInput — эталон
// ════════════════════════════════════════════════════════════════════════════

inline void TestConvert_VectorInput(drv_gpu_lib::IBackend* backend,
                                     SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestConvert_VectorInput [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 85;
  auto A = MakePositiveDefiniteHermitian(n, 777);

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = A;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(input);

  auto A_inv = result.AsVector();
  double err = FrobeniusError(A, A_inv, n);
  if (err >= 1e-3) {
    throw std::runtime_error(
        "TestConvert_VectorInput FAILED [" + std::string(ModeName(mode)) +
        "]: error=" + std::to_string(err));
  }

  con.Print(0, "VecAlg", "TestConvert_VectorInput PASSED [" +
            std::string(ModeName(mode)) + "] error=" + std::to_string(err));
}

// ════════════════════════════════════════════════════════════════════════════
// 5.10.2: TestConvert_HipInput — другой ROCm контекст
// ════════════════════════════════════════════════════════════════════════════

inline void TestConvert_HipInput(drv_gpu_lib::IBackend* backend,
                                  SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestConvert_HipInput [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 85;
  auto A = MakePositiveDefiniteHermitian(n, 777);  // Same seed = same ref
  const size_t bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);

  // Reference: CPU path
  drv_gpu_lib::InputData<std::vector<std::complex<float>>> cpu_input;
  cpu_input.antenna_count = 1;
  cpu_input.n_point = static_cast<uint32_t>(n * n);
  cpu_input.data = A;

  CholeskyInverterROCm ref_inverter(backend, mode);
  auto ref_result = ref_inverter.Invert(cpu_input);
  auto ref_vec = ref_result.AsVector();

  // Другой ROCm контекст на том же GPU
  drv_gpu_lib::ROCmBackend other_rocm;
  other_rocm.Initialize(backend->GetDeviceIndex());

  void* d_foreign = other_rocm.Allocate(bytes);
  other_rocm.MemcpyHostToDevice(d_foreign, A.data(), bytes);

  drv_gpu_lib::InputData<void*> hip_input;
  hip_input.antenna_count = 1;
  hip_input.n_point = static_cast<uint32_t>(n * n);
  hip_input.data = d_foreign;
  hip_input.gpu_memory_bytes = bytes;

  // Инвертировать через НАШ backend (данные от другого контекста)
  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(hip_input);
  auto test_vec = result.AsVector();

  other_rocm.Free(d_foreign);
  other_rocm.Cleanup();

  // Сравнить с эталоном
  double diff = 0.0;
  for (size_t i = 0; i < ref_vec.size(); ++i) {
    diff += std::norm(
        static_cast<std::complex<double>>(ref_vec[i]) -
        static_cast<std::complex<double>>(test_vec[i]));
  }
  diff = std::sqrt(diff);

  if (diff >= 1e-3) {
    throw std::runtime_error(
        "TestConvert_HipInput FAILED [" + std::string(ModeName(mode)) +
        "]: diff vs reference=" + std::to_string(diff));
  }

  con.Print(0, "VecAlg", "TestConvert_HipInput PASSED [" +
            std::string(ModeName(mode)) + "] diff=" + std::to_string(diff));
}

// ════════════════════════════════════════════════════════════════════════════
// 5.10.3: TestConvert_ClMemInput — SKIP
// ════════════════════════════════════════════════════════════════════════════

inline void TestConvert_ClMemInput(drv_gpu_lib::IBackend* /*backend*/,
                                    SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestConvert_ClMemInput [" +
            std::string(ModeName(mode)) +
            "] SKIPPED: requires HybridGPUContext");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.10.4: TestConvert_OutputFormats — AsVector vs AsHipPtr
// ════════════════════════════════════════════════════════════════════════════

inline void TestConvert_OutputFormats(drv_gpu_lib::IBackend* backend,
                                       SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestConvert_OutputFormats [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 85;
  auto A = MakePositiveDefiniteHermitian(n, 555);
  const size_t bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = A;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(input);

  // Path 1: AsVector()
  auto vec1 = result.AsVector();

  // Path 2: AsHipPtr() → manual hipMemcpy
  void* hip_ptr = result.AsHipPtr();
  std::vector<std::complex<float>> vec2(static_cast<size_t>(n) * n);
  hipMemcpy(vec2.data(), hip_ptr, bytes, hipMemcpyDeviceToHost);

  // Compare
  double diff = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    diff += std::norm(
        static_cast<std::complex<double>>(vec1[i]) -
        static_cast<std::complex<double>>(vec2[i]));
  }
  diff = std::sqrt(diff);

  if (diff >= 1e-7) {
    throw std::runtime_error(
        "TestConvert_OutputFormats FAILED [" + std::string(ModeName(mode)) +
        "]: AsVector vs AsHipPtr diff=" + std::to_string(diff));
  }

  con.Print(0, "VecAlg", "TestConvert_OutputFormats PASSED [" +
            std::string(ModeName(mode)) + "] diff=" + std::to_string(diff));
}

}  // namespace vector_algebra::tests

#endif  // ENABLE_ROCM
