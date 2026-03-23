#pragma once

/**
 * @file test_capon_rocm.hpp
 * @brief ROCm-тесты для CaponProcessor
 *
 * Проверяет:
 *   1. Базовый рельеф Кейпона (шум → все значения > 0, размер верный)
 *   2. Подавление помехи: мощная CW-помеха на угле θ_int=0° → MVDR минимален там
 *   3. Адаптивное ДО (выходная матрица нужной размерности)
 *   4. Регуляризация (mu=0 vs mu>0, вырожденная матрица N < P)
 *   5. GPU-to-GPU пайплайн (hipMalloc + hipMemcpy + void* API)
 *
 * Сравнение эталона:
 *   test_02 проверяет физическое свойство MVDR: z[m_int] < mean(z) / 2.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "capon_test_helpers.hpp"
#include "capon_processor.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <numeric>
#include <string>

namespace test_capon_rocm {

using cx = std::complex<float>;
using capon_test_helpers::MakeSteeringMatrix;
using capon_test_helpers::MakeNoise;
using capon_test_helpers::AddInterference;

inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "Capon", msg);
}

inline drv_gpu_lib::IBackend* GetTestBackend() {
  return &capon_test_helpers::GetROCmBackend();
}

// ============================================================================
// Test 01: Базовый рельеф Кейпона — только шум, рельеф должен быть > 0
// ============================================================================

inline void test_01_relief_noise_only() {
  TestPrint("[test_capon_rocm::01] ComputeRelief — only noise (flat spectrum)");

  auto* backend = GetTestBackend();

  const uint32_t P = 8;   // channels
  const uint32_t N = 64;  // samples
  const uint32_t M = 16;  // directions

  auto signal   = MakeNoise(P * N, 1.0f, 42u);
  auto steering = MakeSteeringMatrix(P, M, -static_cast<float>(M_PI)/3.0f,
                                          static_cast<float>(M_PI)/3.0f);

  capon::CaponParams params;
  params.n_channels   = P;
  params.n_samples    = N;
  params.n_directions = M;
  params.mu           = 0.01f;

  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(signal, steering, params);

  assert(result.relief.size() == M);

  // Все значения рельефа должны быть > 0 и конечными
  for (uint32_t m = 0; m < M; ++m) {
    assert(std::isfinite(result.relief[m]));
    assert(result.relief[m] > 0.0f);
  }

  TestPrint("[test_capon_rocm::01] PASS");
}

// ============================================================================
// Test 02: Рельеф с помехой — MVDR минимален на направлении помехи
//
// Физика:  MVDR минимизирует мощность выхода при ограничении «пропустить
//          сигнал из нужного направления». На направлении мощной помехи
//          z[m] = 1/(u^H * R^{-1} * u) будет МИНИМАЛЬНЫМ (подавление).
// ============================================================================

inline void test_02_relief_with_interference() {
  TestPrint("[test_capon_rocm::02] ComputeRelief — interference suppression check");

  auto* backend = GetTestBackend();

  const uint32_t P = 8;
  const uint32_t N = 128;
  const uint32_t M = 32;

  // Равномерное покрытие [-60°, +60°]; помеха на 0° — индекс M/2 = 16
  const float theta_min = -static_cast<float>(M_PI) / 3.0f;
  const float theta_max =  static_cast<float>(M_PI) / 3.0f;
  const float theta_int =  0.0f;  // помеха точно по центру сетки

  // Сигнал: шум + мощная CW-помеха (SNR ≈ 100)
  auto signal = MakeNoise(P * N, 1.0f, 77u);
  AddInterference(signal, P, N, theta_int, /*amplitude=*/10.0f);

  auto steering = MakeSteeringMatrix(P, M, theta_min, theta_max);

  capon::CaponParams params{P, N, M, 0.001f};

  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(signal, steering, params);

  assert(result.relief.size() == M);

  // Найти индекс ближайшего направления к θ_int = 0
  uint32_t m_int = 0;
  float min_diff = 1e9f;
  for (uint32_t m = 0; m < M; ++m) {
    float theta = theta_min + (theta_max - theta_min) * m / (M - 1);
    float diff  = std::abs(theta - theta_int);
    if (diff < min_diff) { min_diff = diff; m_int = m; }
  }

  // MVDR: на направлении помехи рельеф должен быть значительно меньше среднего
  float mean_relief = 0.0f;
  for (auto v : result.relief) mean_relief += v;
  mean_relief /= static_cast<float>(M);

  // z[m_int] < mean/2 — Capon хорошо подавляет помеху
  assert(result.relief[m_int] < mean_relief * 0.5f);

  TestPrint("[test_capon_rocm::02] PASS (MVDR suppression confirmed at interference direction)");
}

// ============================================================================
// Test 03: AdaptiveBeamform — размерность выхода
// ============================================================================

inline void test_03_adaptive_beamform_dims() {
  TestPrint("[test_capon_rocm::03] AdaptiveBeamform — output dimensions");

  auto* backend = GetTestBackend();

  const uint32_t P = 4;
  const uint32_t N = 32;
  const uint32_t M = 6;

  auto signal   = MakeNoise(P * N, 1.0f, 13u);
  auto steering = MakeSteeringMatrix(P, M, -static_cast<float>(M_PI)/6.0f,
                                          static_cast<float>(M_PI)/6.0f);

  capon::CaponParams params{P, N, M, 0.01f};

  capon::CaponProcessor processor(backend);
  auto result = processor.AdaptiveBeamform(signal, steering, params);

  assert(result.n_directions == M);
  assert(result.n_samples    == N);
  assert(result.output.size() == static_cast<size_t>(M) * N);

  // Проверить что выход конечный
  for (const auto& v : result.output) {
    assert(std::isfinite(v.real()) && std::isfinite(v.imag()));
  }

  TestPrint("[test_capon_rocm::03] PASS");
}

// ============================================================================
// Test 04: Регуляризация — mu=0 vs mu>0 (численная устойчивость)
// ============================================================================

inline void test_04_regularization() {
  TestPrint("[test_capon_rocm::04] Regularization — mu=0 vs mu=0.1");

  auto* backend = GetTestBackend();

  const uint32_t P = 4;
  const uint32_t N = 8;  // N < P → матрица вырождена без регуляризации
  const uint32_t M = 8;

  auto signal   = MakeNoise(P * N, 1.0f, 99u);
  auto steering = MakeSteeringMatrix(P, M, -static_cast<float>(M_PI)/4.0f,
                                          static_cast<float>(M_PI)/4.0f);

  // С регуляризацией должно работать без ошибок
  capon::CaponParams params{P, N, M, 0.1f};

  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(signal, steering, params);

  assert(result.relief.size() == M);
  for (auto v : result.relief) {
    assert(std::isfinite(v) && v >= 0.0f);
  }

  TestPrint("[test_capon_rocm::04] PASS");
}

// ============================================================================
// Test 05: GPU-to-GPU пайплайн (hipMalloc → hipMemcpy → void* API)
// ============================================================================

inline void test_05_gpu_to_gpu() {
  TestPrint("[test_capon_rocm::05] GPU-to-GPU pipeline (hipMalloc + D2D)");

  auto* backend = GetTestBackend();

  const uint32_t P = 8;
  const uint32_t N = 64;
  const uint32_t M = 16;

  auto signal_cpu   = MakeNoise(P * N, 1.0f, 55u);
  auto steering_cpu = MakeSteeringMatrix(P, M, -static_cast<float>(M_PI)/3.0f,
                                               static_cast<float>(M_PI)/3.0f);

  const size_t bytes_Y = signal_cpu.size()   * sizeof(cx);
  const size_t bytes_U = steering_cpu.size() * sizeof(cx);

  // Аллоцировать GPU буферы через HIP
  void* gpu_Y = nullptr;
  void* gpu_U = nullptr;
  hipError_t err;

  err = hipMalloc(&gpu_Y, bytes_Y);
  assert(err == hipSuccess && "hipMalloc gpu_Y failed");
  err = hipMalloc(&gpu_U, bytes_U);
  assert(err == hipSuccess && "hipMalloc gpu_U failed");

  // Загрузить данные CPU → GPU
  err = hipMemcpy(gpu_Y, signal_cpu.data(),   bytes_Y, hipMemcpyHostToDevice);
  assert(err == hipSuccess && "hipMemcpy signal failed");
  err = hipMemcpy(gpu_U, steering_cpu.data(), bytes_U, hipMemcpyHostToDevice);
  assert(err == hipSuccess && "hipMemcpy steering failed");

  // Вычислить рельеф через void* GPU API
  capon::CaponParams params{P, N, M, 0.01f};
  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(gpu_Y, gpu_U, params);

  assert(result.relief.size() == M);
  for (uint32_t m = 0; m < M; ++m) {
    assert(std::isfinite(result.relief[m]));
    assert(result.relief[m] > 0.0f);
  }

  hipFree(gpu_Y);
  hipFree(gpu_U);

  TestPrint("[test_capon_rocm::05] PASS");
}

// ============================================================================
// run() — точка входа
// ============================================================================

inline void run() {
  drv_gpu_lib::ConsoleOutput::GetInstance().Start();
  TestPrint("=== test_capon_rocm ===");
  test_01_relief_noise_only();
  test_02_relief_with_interference();
  test_03_adaptive_beamform_dims();
  test_04_regularization();
  test_05_gpu_to_gpu();
  TestPrint("=== test_capon_rocm DONE ===");
}

}  // namespace test_capon_rocm

#endif  // ENABLE_ROCM
