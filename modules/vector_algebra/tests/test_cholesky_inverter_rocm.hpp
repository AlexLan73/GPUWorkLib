#pragma once
#if ENABLE_ROCM

/**
 * @file test_cholesky_inverter_rocm.hpp
 * @brief Функциональные тесты CholeskyInverterROCm (Task_11 v2)
 *
 * Каждый тест принимает SymmetrizeMode — запускается для Roundtrip и GpuKernel.
 * 10 тестов: Identity, 341, void*, cl_mem(SKIP), batch CPU/GPU, sizes, access.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "cholesky_inverter_rocm.hpp"
#include "DrvGPU/interface/i_backend.hpp"
#include "DrvGPU/interface/input_data.hpp"
#include "services/console_output.hpp"

namespace vector_algebra::tests {

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

inline const char* ModeName(SymmetrizeMode mode) {
  return mode == SymmetrizeMode::Roundtrip ? "Roundtrip" : "GpuKernel";
}

/// Создать HPD матрицу: A = B*B^H + n*I
inline std::vector<std::complex<float>> MakePositiveDefiniteHermitian(
    int n, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<std::complex<float>> B(static_cast<size_t>(n) * n);
  for (auto& v : B) v = {dist(rng), dist(rng)};

  std::vector<std::complex<float>> A(static_cast<size_t>(n) * n, {0, 0});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::complex<float> sum(0, 0);
      for (int k = 0; k < n; ++k) {
        sum += B[static_cast<size_t>(i) * n + k] *
               std::conj(B[static_cast<size_t>(j) * n + k]);
      }
      A[static_cast<size_t>(i) * n + j] = sum;
    }
    A[static_cast<size_t>(i) * n + i] +=
        std::complex<float>(static_cast<float>(n), 0.0f);
  }
  return A;
}

/// ||A * B - I||_F (float64 accumulation)
inline double FrobeniusError(
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B, int n) {
  double err = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        sum += static_cast<std::complex<double>>(
                   A[static_cast<size_t>(i) * n + k]) *
               static_cast<std::complex<double>>(
                   B[static_cast<size_t>(k) * n + j]);
      }
      if (i == j) sum -= 1.0;
      err += std::norm(sum);
    }
  }
  return std::sqrt(err);
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.10: TestResolveMatrixSize — без GPU
// ════════════════════════════════════════════════════════════════════════════

inline void TestResolveMatrixSize() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestResolveMatrixSize");

  auto check = [](int n_point, int expected) {
    int n = static_cast<int>(
        std::round(std::sqrt(static_cast<double>(n_point))));
    assert(n == expected);
    assert(n * n == n_point);
  };
  check(341 * 341, 341);
  check(64 * 64, 64);
  check(5 * 5, 5);
  check(256 * 256, 256);

  con.Print(0, "VecAlg", "TestResolveMatrixSize PASSED");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.1: TestCpuIdentity — I(5×5)
// ════════════════════════════════════════════════════════════════════════════

inline void TestCpuIdentity(drv_gpu_lib::IBackend* backend,
                             SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestCpuIdentity [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 5;
  std::vector<std::complex<float>> identity(n * n, {0, 0});
  for (int i = 0; i < n; ++i)
    identity[static_cast<size_t>(i) * n + i] = {1.0f, 0.0f};

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = n * n;
  input.data = identity;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(input);

  assert(result.matrix_size == n);
  assert(result.batch_count == 1);

  auto A_inv = result.AsVector();
  double err = FrobeniusError(identity, A_inv, n);
  if (err >= 1e-5) {
    throw std::runtime_error(
        "TestCpuIdentity FAILED [" + std::string(ModeName(mode)) +
        "]: error=" + std::to_string(err) + " >= 1e-5");
  }

  auto mat = result.matrix();
  assert(static_cast<int>(mat.size()) == n);
  assert(static_cast<int>(mat[0].size()) == n);

  con.Print(0, "VecAlg", "TestCpuIdentity PASSED [" +
            std::string(ModeName(mode)) + "] error=" + std::to_string(err));
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.2: TestCpu341 — HPD(341)
// ════════════════════════════════════════════════════════════════════════════

inline void TestCpu341(drv_gpu_lib::IBackend* backend, SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestCpu341 [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 341;
  auto A = MakePositiveDefiniteHermitian(n, 42);

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = A;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(input);
  auto A_inv = result.AsVector();

  double err = FrobeniusError(A, A_inv, n);
  if (err >= 1e-2) {
    throw std::runtime_error(
        "TestCpu341 FAILED [" + std::string(ModeName(mode)) +
        "]: error=" + std::to_string(err) + " >= 1e-2");
  }

  con.Print(0, "VecAlg", "TestCpu341 PASSED [" +
            std::string(ModeName(mode)) + "] error=" + std::to_string(err));
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.3: TestGpuVoidPtr341
// ════════════════════════════════════════════════════════════════════════════

inline void TestGpuVoidPtr341(drv_gpu_lib::IBackend* backend,
                               SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestGpuVoidPtr341 [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 341;
  auto A = MakePositiveDefiniteHermitian(n, 123);
  const size_t bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);

  void* gpu_in = backend->Allocate(bytes);
  backend->MemcpyHostToDevice(gpu_in, A.data(), bytes);

  drv_gpu_lib::InputData<void*> input;
  input.antenna_count = 1;
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = gpu_in;
  input.gpu_memory_bytes = bytes;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.Invert(input);

  auto A_inv = result.AsVector();
  backend->Free(gpu_in);

  double err = FrobeniusError(A, A_inv, n);
  if (err >= 1e-2) {
    throw std::runtime_error(
        "TestGpuVoidPtr341 FAILED [" + std::string(ModeName(mode)) +
        "]: error=" + std::to_string(err));
  }

  con.Print(0, "VecAlg", "TestGpuVoidPtr341 PASSED [" +
            std::string(ModeName(mode)) + "] error=" + std::to_string(err));
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.4: TestZeroCopyClMem — SKIP
// ════════════════════════════════════════════════════════════════════════════

inline void TestZeroCopyClMem(drv_gpu_lib::IBackend* /*backend*/,
                               SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestZeroCopyClMem [" +
            std::string(ModeName(mode)) +
            "] SKIPPED: requires HybridGPUContext");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.5: TestBatchCpu_4x64
// ════════════════════════════════════════════════════════════════════════════

inline void TestBatchCpu_4x64(drv_gpu_lib::IBackend* backend,
                               SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestBatchCpu_4x64 [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 64;
  constexpr int batch = 4;

  std::vector<std::complex<float>> flat;
  flat.reserve(static_cast<size_t>(batch) * n * n);
  std::vector<std::vector<std::complex<float>>> matrices(batch);
  for (int k = 0; k < batch; ++k) {
    matrices[k] =
        MakePositiveDefiniteHermitian(n, static_cast<unsigned>(k + 10));
    flat.insert(flat.end(), matrices[k].begin(), matrices[k].end());
  }

  drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = static_cast<uint32_t>(batch);
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = flat;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.InvertBatch(input, n);

  assert(result.matrix_size == n);
  assert(result.batch_count == batch);

  auto flat_inv = result.AsVector();

  for (int k = 0; k < batch; ++k) {
    std::vector<std::complex<float>> A_inv_k(
        flat_inv.begin() + static_cast<ptrdiff_t>(k) * n * n,
        flat_inv.begin() + static_cast<ptrdiff_t>(k + 1) * n * n);

    double err = FrobeniusError(matrices[k], A_inv_k, n);
    if (err >= 1e-3) {
      throw std::runtime_error(
          "TestBatchCpu_4x64 FAILED [" + std::string(ModeName(mode)) +
          "] matrix[" + std::to_string(k) +
          "] error=" + std::to_string(err));
    }
  }

  con.Print(0, "VecAlg", "TestBatchCpu_4x64 PASSED [" +
            std::string(ModeName(mode)) + "]");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.6: TestBatchGpu_4x64
// ════════════════════════════════════════════════════════════════════════════

inline void TestBatchGpu_4x64(drv_gpu_lib::IBackend* backend,
                               SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestBatchGpu_4x64 [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 64;
  constexpr int batch = 4;
  const size_t total_bytes =
      static_cast<size_t>(batch) * n * n * sizeof(std::complex<float>);

  std::vector<std::complex<float>> flat;
  flat.reserve(static_cast<size_t>(batch) * n * n);
  std::vector<std::vector<std::complex<float>>> matrices(batch);
  for (int k = 0; k < batch; ++k) {
    matrices[k] =
        MakePositiveDefiniteHermitian(n, static_cast<unsigned>(k + 50));
    flat.insert(flat.end(), matrices[k].begin(), matrices[k].end());
  }

  void* gpu_in = backend->Allocate(total_bytes);
  backend->MemcpyHostToDevice(gpu_in, flat.data(), total_bytes);

  drv_gpu_lib::InputData<void*> input;
  input.antenna_count = static_cast<uint32_t>(batch);
  input.n_point = static_cast<uint32_t>(n * n);
  input.data = gpu_in;
  input.gpu_memory_bytes = total_bytes;

  CholeskyInverterROCm inverter(backend, mode);
  auto result = inverter.InvertBatch(input, n);

  auto flat_inv = result.AsVector();
  backend->Free(gpu_in);

  for (int k = 0; k < batch; ++k) {
    std::vector<std::complex<float>> A_inv_k(
        flat_inv.begin() + static_cast<ptrdiff_t>(k) * n * n,
        flat_inv.begin() + static_cast<ptrdiff_t>(k + 1) * n * n);

    double err = FrobeniusError(matrices[k], A_inv_k, n);
    if (err >= 1e-3) {
      throw std::runtime_error(
          "TestBatchGpu_4x64 FAILED [" + std::string(ModeName(mode)) +
          "] matrix[" + std::to_string(k) +
          "] error=" + std::to_string(err));
    }
  }

  con.Print(0, "VecAlg", "TestBatchGpu_4x64 PASSED [" +
            std::string(ModeName(mode)) + "]");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.7: TestBatchSizes — batch 1,4,8,16
// ════════════════════════════════════════════════════════════════════════════

inline void TestBatchSizes(drv_gpu_lib::IBackend* backend,
                            SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestBatchSizes [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 64;
  const int batch_sizes[] = {1, 4, 8, 16};

  for (int batch : batch_sizes) {
    std::vector<std::complex<float>> flat;
    for (int k = 0; k < batch; ++k) {
      auto A = MakePositiveDefiniteHermitian(
          n, static_cast<unsigned>(k + 100));
      flat.insert(flat.end(), A.begin(), A.end());
    }

    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = static_cast<uint32_t>(batch);
    input.n_point = static_cast<uint32_t>(n * n);
    input.data = flat;

    CholeskyInverterROCm inverter(backend, mode);
    auto result = inverter.InvertBatch(input, n);

    assert(result.batch_count == batch);
    assert(result.matrix_size == n);

    con.Print(0, "VecAlg", "  batch=" + std::to_string(batch) + " OK");
  }

  con.Print(0, "VecAlg", "TestBatchSizes PASSED [" +
            std::string(ModeName(mode)) + "]");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.8: TestMatrixSizes — n=32,64,128,256
// ════════════════════════════════════════════════════════════════════════════

inline void TestMatrixSizes(drv_gpu_lib::IBackend* backend,
                             SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestMatrixSizes [" +
            std::string(ModeName(mode)) + "]");

  constexpr int batch = 4;
  const int sizes[] = {32, 64, 128, 256};

  for (int n : sizes) {
    std::vector<std::complex<float>> flat;
    std::vector<std::vector<std::complex<float>>> matrices(batch);
    for (int k = 0; k < batch; ++k) {
      matrices[k] = MakePositiveDefiniteHermitian(
          n, static_cast<unsigned>(k + 200));
      flat.insert(flat.end(), matrices[k].begin(), matrices[k].end());
    }

    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = static_cast<uint32_t>(batch);
    input.n_point = static_cast<uint32_t>(n * n);
    input.data = flat;

    CholeskyInverterROCm inverter(backend, mode);
    auto result = inverter.InvertBatch(input, n);

    auto flat_inv = result.AsVector();
    for (int k = 0; k < batch; ++k) {
      std::vector<std::complex<float>> A_inv_k(
          flat_inv.begin() + static_cast<ptrdiff_t>(k) * n * n,
          flat_inv.begin() + static_cast<ptrdiff_t>(k + 1) * n * n);

      double err = FrobeniusError(matrices[k], A_inv_k, n);
      if (err >= 1e-2) {
        throw std::runtime_error(
            "TestMatrixSizes FAILED [" + std::string(ModeName(mode)) +
            "] n=" + std::to_string(n) + " matrix[" + std::to_string(k) +
            "] error=" + std::to_string(err));
      }
    }

    con.Print(0, "VecAlg", "  n=" + std::to_string(n) + " OK");
  }

  con.Print(0, "VecAlg", "TestMatrixSizes PASSED [" +
            std::string(ModeName(mode)) + "]");
}

// ════════════════════════════════════════════════════════════════════════════
// 5.6.9: TestResultAccess — .matrix() / .matrices() / AsHipPtr()
// ════════════════════════════════════════════════════════════════════════════

inline void TestResultAccess(drv_gpu_lib::IBackend* backend,
                              SymmetrizeMode mode) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "TestResultAccess [" +
            std::string(ModeName(mode)) + "]");

  constexpr int n = 5;

  // Single
  {
    auto A = MakePositiveDefiniteHermitian(n, 99);
    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = 1;
    input.n_point = static_cast<uint32_t>(n * n);
    input.data = A;

    CholeskyInverterROCm inverter(backend, mode);
    auto result = inverter.Invert(input);

    auto mat = result.matrix();
    assert(static_cast<int>(mat.size()) == n);
    assert(static_cast<int>(mat[0].size()) == n);
    assert(result.AsHipPtr() != nullptr);
  }

  // Batched
  {
    constexpr int batch = 3;
    std::vector<std::complex<float>> flat;
    for (int k = 0; k < batch; ++k) {
      auto A = MakePositiveDefiniteHermitian(
          n, static_cast<unsigned>(k + 300));
      flat.insert(flat.end(), A.begin(), A.end());
    }

    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = static_cast<uint32_t>(batch);
    input.n_point = static_cast<uint32_t>(n * n);
    input.data = flat;

    CholeskyInverterROCm inverter(backend, mode);
    auto result = inverter.InvertBatch(input, n);

    auto mats = result.matrices();
    assert(static_cast<int>(mats.size()) == batch);
    assert(static_cast<int>(mats[0].size()) == n);
    assert(static_cast<int>(mats[0][0].size()) == n);

    auto vec = result.AsVector();
    assert(static_cast<int>(vec.size()) == batch * n * n);
  }

  con.Print(0, "VecAlg", "TestResultAccess PASSED [" +
            std::string(ModeName(mode)) + "]");
}

}  // namespace vector_algebra::tests

#endif  // ENABLE_ROCM
