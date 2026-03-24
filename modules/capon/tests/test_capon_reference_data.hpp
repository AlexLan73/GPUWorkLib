#pragma once

/**
 * @file test_capon_reference_data.hpp
 * @brief Тесты CaponProcessor на реальных данных (MATLAB сигнал + физические координаты)
 *
 * Загружает файлы из Doc_Addition/Capon/capon_test/build/:
 *   - x_data.txt, y_data.txt  — координаты антенных секций (340 значений)
 *   - signal_matlab.txt       — MATLAB сигнал (341 строка x 1000 комплексных чисел)
 *   - z_values.txt            — эталонные значения рельефа (4 блока x 37x37 = 5476)
 *
 * Физические параметры (из эталонной реализации):
 *   f0 = 3918e6 + 3.15e6 = 3921150000 Hz
 *   c  = 299792458 m/s
 *   getU(x, y, u, v, f0) = exp(j * 2pi * (x*u + y*v) * f0/c)
 *
 * Отличие формул:
 *   Эталон (ArrayFire):  R = Y*Y^H       + mu*I   (нет деления на N)
 *   GPU (CaponProcessor): R = (1/N)*Y*Y^H + mu*I   (есть деление на N)
 *   -> для сравнения форм рельефа используется нормировка
 *
 * Тесты:
 *   01 — загрузка и валидация файлов
 *   02 — GPU рельеф на реальных данных (P=85, N=1000, M=1369): > 0 и конечный
 *   03 — CPU эталон vs GPU (P=8, N=64, M=16): относительная погрешность < 0.5%
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-23
 */

#if ENABLE_ROCM

#include "capon_processor.hpp"
#include "capon_test_helpers.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>
#include <algorithm>

namespace test_capon_reference_data {

using cx = std::complex<float>;

// Используем общие утилиты из capon_test_helpers
using capon_test_helpers::kF0;
using capon_test_helpers::kC;
using capon_test_helpers::kDataDir;
using capon_test_helpers::LoadRealVector;
using capon_test_helpers::LoadSignalMatlab;
using capon_test_helpers::MakePhysicalSteering;
using capon_test_helpers::MakeScanGrid1D;

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "Capon", msg);
}

inline drv_gpu_lib::IBackend* GetTestBackend() {
  return &capon_test_helpers::GetROCmBackend();
}

// ============================================================================
// CPU эталон (для test_03) — специфичен для этого файла
// ============================================================================

/// Разложение Холецкого нижняя треугольная L (column-major, in-place)
inline bool CholeskyLower(std::vector<cx>& A, uint32_t P) {
  for (uint32_t j = 0; j < P; ++j) {
    float diag = A[j * P + j].real();
    for (uint32_t k = 0; k < j; ++k) {
      cx ljk = A[k * P + j];  // L[j,k] column-major: (row=j, col=k)
      diag -= (ljk * std::conj(ljk)).real();
    }
    if (diag <= 0.0f) return false;
    A[j * P + j] = cx(std::sqrt(diag), 0.0f);

    for (uint32_t i = j + 1; i < P; ++i) {
      cx sum = A[j * P + i];
      for (uint32_t k = 0; k < j; ++k) {
        sum -= A[k * P + i] * std::conj(A[k * P + j]);
      }
      A[j * P + i] = sum / A[j * P + j];
    }
  }
  return true;
}

/// Решить L*x = b (нижнетреугольная, column-major L)
inline std::vector<cx> ForwardSolve(const std::vector<cx>& L,
                                    uint32_t P,
                                    const std::vector<cx>& b) {
  std::vector<cx> x(P);
  for (uint32_t i = 0; i < P; ++i) {
    cx sum = b[i];
    for (uint32_t k = 0; k < i; ++k) {
      sum -= L[k * P + i] * x[k];
    }
    x[i] = sum / L[i * P + i];
  }
  return x;
}

/**
 * @brief CPU Capon рельеф (GPU формула с делением на N)
 *
 * R = Y*Y^H/N + mu*I
 * L*L^H = R  (Cholesky)
 * x_m = L^{-1} * u_m
 * z[m] = 1 / ||x_m||^2
 */
inline std::vector<float> CpuCaponRelief(
    const std::vector<cx>& Y, uint32_t P, uint32_t N,
    const std::vector<cx>& U, uint32_t M,
    float mu) {
  // 1. R = Y*Y^H/N + mu*I
  std::vector<cx> R(static_cast<size_t>(P) * P, cx(0.0f));
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t i = 0; i < P; ++i) {
      cx y_i = Y[static_cast<size_t>(n) * P + i];
      for (uint32_t j = 0; j < P; ++j) {
        cx y_j = Y[static_cast<size_t>(n) * P + j];
        R[static_cast<size_t>(j) * P + i] += y_i * std::conj(y_j);
      }
    }
  }
  float inv_N = 1.0f / static_cast<float>(N);
  for (auto& r : R) r *= inv_N;
  for (uint32_t i = 0; i < P; ++i) {
    R[static_cast<size_t>(i) * P + i] += cx(mu, 0.0f);
  }

  // 2. Cholesky
  if (!CholeskyLower(R, P)) {
    return std::vector<float>(M, 0.0f);
  }

  // 3. z[m] = 1 / ||L^{-1}*u_m||^2
  std::vector<float> z(M);
  for (uint32_t m = 0; m < M; ++m) {
    std::vector<cx> u_m(P);
    for (uint32_t p = 0; p < P; ++p) {
      u_m[p] = U[static_cast<size_t>(m) * P + p];
    }
    std::vector<cx> x = ForwardSolve(R, P, u_m);
    float norm2 = 0.0f;
    for (auto& v : x) {
      norm2 += v.real() * v.real() + v.imag() * v.imag();
    }
    z[m] = (norm2 > 1e-30f) ? (1.0f / norm2) : 0.0f;
  }
  return z;
}

// ============================================================================
// Test 01: Загрузка и валидация файлов
// ============================================================================

inline void test_01_load_files() {
  TestPrint("[test_capon_reference_data::01] Load and validate reference files");

  std::vector<float> x_vals, y_vals;
  const bool x_ok = LoadRealVector(kDataDir + "x_data.txt", x_vals);
  const bool y_ok = LoadRealVector(kDataDir + "y_data.txt", y_vals);

  if (!x_ok || !y_ok) {
    TestPrint("[test_capon_reference_data::01] SKIP: x_data.txt / y_data.txt not found");
    return;
  }

  if (x_vals.size() < 85) {
    TestPrint("[test_capon_reference_data::01] SKIP: x_data has < 85 elements");
    return;
  }
  if (y_vals.size() < 85) {
    TestPrint("[test_capon_reference_data::01] SKIP: y_data has < 85 elements");
    return;
  }

  std::vector<cx> signal_test;
  const bool sig_ok = LoadSignalMatlab(kDataDir + "signal_matlab.txt",
                                       2, 3, signal_test);
  if (!sig_ok) {
    TestPrint("[test_capon_reference_data::01] SKIP: signal_matlab.txt not found or bad format");
    return;
  }

  std::ostringstream msg;
  msg << "[test_capon_reference_data::01] PASS"
      << "  x.size=" << x_vals.size()
      << "  y.size=" << y_vals.size()
      << "  signal parsed OK (2 rows x 3 cols)";
  TestPrint(msg.str());
}

// ============================================================================
// Test 02: GPU рельеф на реальных данных (P=85, N=1000, M=1369)
// ============================================================================

inline void test_02_physical_relief_properties() {
  TestPrint("[test_capon_reference_data::02] GPU Capon on real data (P=85, N=1000, M=1369)");

  std::vector<float> x_vals, y_vals;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_vals) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_vals)) {
    TestPrint("[test_capon_reference_data::02] SKIP: coordinate files not found");
    return;
  }

  const uint32_t P = 85;
  const uint32_t N = 1000;

  if (x_vals.size() < P || y_vals.size() < P) {
    TestPrint("[test_capon_reference_data::02] SKIP: not enough antenna elements");
    return;
  }
  x_vals.resize(P);
  y_vals.resize(P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[test_capon_reference_data::02] SKIP: signal_matlab.txt not found or parse error");
    return;
  }

  // 2D сетка 37x37 = 1369 направлений
  auto u0_seq = MakeScanGrid1D(3.25, 0.00312);
  const uint32_t Nu = static_cast<uint32_t>(u0_seq.size());

  const uint32_t M = Nu * Nu;
  std::vector<float> u_dirs(M), v_dirs(M);
  for (uint32_t iv = 0; iv < Nu; ++iv) {
    for (uint32_t iu = 0; iu < Nu; ++iu) {
      uint32_t m = iv * Nu + iu;
      u_dirs[m] = u0_seq[iu];
      v_dirs[m] = u0_seq[iv];
    }
  }

  auto steering = MakePhysicalSteering(x_vals, y_vals,
                                       u_dirs, v_dirs, kF0, kC);

  const float mu_gpu = 1.0f;

  capon::CaponParams params{P, N, M, mu_gpu};

  auto* backend = GetTestBackend();
  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(signal, steering, params);

  if (result.relief.size() != M) {
    TestPrint("[test_capon_reference_data::02] FAIL: relief size mismatch");
    return;
  }

  float min_val = result.relief[0];
  float max_val = result.relief[0];
  bool all_positive = true;
  bool all_finite   = true;
  for (float v : result.relief) {
    if (!std::isfinite(v))      { all_finite   = false; }
    if (v <= 0.0f)              { all_positive = false; }
    if (v < min_val) min_val = v;
    if (v > max_val) max_val = v;
  }

  if (!all_finite) {
    TestPrint("[test_capon_reference_data::02] FAIL: non-finite values in relief");
    return;
  }
  if (!all_positive) {
    TestPrint("[test_capon_reference_data::02] FAIL: non-positive values in relief");
    return;
  }

  std::ostringstream msg;
  msg << "[test_capon_reference_data::02] PASS"
      << "  M=" << M << "  Nu=" << Nu
      << "  relief_min=" << min_val
      << "  relief_max=" << max_val;
  TestPrint(msg.str());
}

// ============================================================================
// Test 03: CPU эталон vs GPU (P=8, N=64, M=16)
// ============================================================================

inline void test_03_cpu_vs_gpu_small_p() {
  TestPrint("[test_capon_reference_data::03] CPU reference vs GPU (P=8, N=64, M=16)");

  std::vector<float> x_vals, y_vals;
  if (!LoadRealVector(kDataDir + "x_data.txt", x_vals) ||
      !LoadRealVector(kDataDir + "y_data.txt", y_vals)) {
    TestPrint("[test_capon_reference_data::03] SKIP: coordinate files not found");
    return;
  }

  const uint32_t P = 8;
  const uint32_t N = 64;
  const uint32_t M = 16;

  if (x_vals.size() < P || y_vals.size() < P) {
    TestPrint("[test_capon_reference_data::03] SKIP: not enough antenna elements");
    return;
  }

  std::vector<float> x8(x_vals.begin(), x_vals.begin() + P);
  std::vector<float> y8(y_vals.begin(), y_vals.begin() + P);

  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[test_capon_reference_data::03] SKIP: signal_matlab.txt not found");
    return;
  }

  const double ulim = std::sin(3.25 * M_PI / 180.0);
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    u_dirs[m] = static_cast<float>(
        -ulim + 2.0 * ulim * m / (M - 1));
  }

  auto steering = MakePhysicalSteering(x8, y8, u_dirs, v_dirs, kF0, kC);

  const float mu = 100.0f;

  capon::CaponParams params{P, N, M, mu};
  auto* backend = GetTestBackend();
  capon::CaponProcessor processor(backend);
  auto gpu_result = processor.ComputeRelief(signal, steering, params);

  if (gpu_result.relief.size() != M) {
    TestPrint("[test_capon_reference_data::03] FAIL: GPU relief size mismatch");
    return;
  }

  auto cpu_z = CpuCaponRelief(signal, P, N, steering, M, mu);

  float max_rel_error = 0.0f;
  uint32_t worst_m    = 0;
  for (uint32_t m = 0; m < M; ++m) {
    float ref = std::max(cpu_z[m], 1e-6f);
    float err = std::abs(gpu_result.relief[m] - cpu_z[m]) / ref;
    if (err > max_rel_error) {
      max_rel_error = err;
      worst_m = m;
    }
  }

  const float kTolerance = 0.005f;

  if (max_rel_error > kTolerance) {
    std::ostringstream msg;
    msg << "[test_capon_reference_data::03] FAIL"
        << "  max_rel_error=" << max_rel_error * 100.0f << "%"
        << "  at m=" << worst_m
        << "  (tolerance=" << kTolerance * 100.0f << "%)";
    TestPrint(msg.str());

    TestPrint("  Diagnostic (gpu vs cpu):");
    for (uint32_t m = 0; m < std::min(M, 5u); ++m) {
      std::ostringstream d;
      d << "    m=" << m
        << "  gpu=" << gpu_result.relief[m]
        << "  cpu=" << cpu_z[m];
      TestPrint(d.str());
    }
    return;
  }

  std::ostringstream msg;
  msg << "[test_capon_reference_data::03] PASS"
      << "  max_rel_error=" << max_rel_error * 100.0f << "%"
      << "  (tolerance=" << kTolerance * 100.0f << "%)";
  TestPrint(msg.str());
}

// ============================================================================
// run()
// ============================================================================

inline void run() {
  TestPrint("=== test_capon_reference_data ===");
  test_01_load_files();
  test_02_physical_relief_properties();
  test_03_cpu_vs_gpu_small_p();
  TestPrint("=== test_capon_reference_data DONE ===");
}

}  // namespace test_capon_reference_data

#endif  // ENABLE_ROCM
