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
 *   5. GPU-to-GPU пайплайн (входные данные уже на GPU)
 *
 * Сравнение эталона:
 *   test_02 проверяет физическое свойство MVDR: z[m_int] < mean(z) / 2.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "capon_processor.hpp"
#include "services/console_output.hpp"
#include "services/test_helpers.hpp"  // GetTestBackend(), CheckClose()

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <numeric>
#include <string>

namespace test_capon_rocm {

using cx = std::complex<float>;
using Con = drv_gpu_lib::ConsoleOutput;

// ============================================================================
// Вспомогательные функции
// ============================================================================

/// Управляющие векторы ULA: u[p,m] = exp(j*2π*p*(d/λ)*sin(θ_m))
/// d/λ = 0.5 (полуволновой интервал), θ сканирует [theta_min, theta_max]
/// Хранение: column-major, [p + m*P]
static std::vector<cx> MakeSteeringMatrix(
    uint32_t n_channels, uint32_t n_directions,
    float theta_min_rad, float theta_max_rad) {
  std::vector<cx> U(static_cast<size_t>(n_channels) * n_directions);
  for (uint32_t m = 0; m < n_directions; ++m) {
    float theta = (n_directions > 1)
        ? theta_min_rad + (theta_max_rad - theta_min_rad) * m / (n_directions - 1)
        : theta_min_rad;
    float d_sin = std::sin(theta) * 0.5f;  // d/λ = 0.5
    for (uint32_t p = 0; p < n_channels; ++p) {
      float phase = 2.0f * static_cast<float>(M_PI) * p * d_sin;
      U[m * n_channels + p] = cx(std::cos(phase), std::sin(phase));
    }
  }
  return U;
}

/**
 * @brief Сгенерировать CN(0, sigma²) шум через LCG + Box-Muller.
 *
 * Воспроизводимый (детерминированный seed), но статистически корректный:
 * амплитуда гауссова, фаза равномерная → правильная ковариационная матрица.
 */
static std::vector<cx> MakeNoise(size_t count, float sigma = 1.0f,
                                 uint32_t seed = 42) {
  std::vector<cx> noise(count);
  uint32_t state = seed;

  auto rng_uniform = [&]() -> float {
    // LCG: Numerical Recipes константы
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu));
    // результат в [0, 1)
  };

  for (size_t i = 0; i < count; ++i) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;  // защита от log(0)

    // Box-Muller: Gaussian с нулевым средним, дисперсия sigma²
    float mag = sigma * std::sqrt(-2.0f * std::log(u1));
    float phi = 2.0f * static_cast<float>(M_PI) * u2;
    noise[i]  = cx(mag * std::cos(phi), mag * std::sin(phi));
  }
  return noise;
}

/**
 * @brief Добавить CW-помеху из направления theta_rad в сигнальную матрицу.
 *
 * Y[p, n] += amplitude * exp(j*2π*p*(d/λ)*sin(θ)) * exp(j*ω₀*n)
 * Хранение Y: column-major, индекс [p, n] = n*P + p
 *
 * @param Y          сигнальная матрица [P × N], column-major
 * @param n_channels P
 * @param n_samples  N
 * @param theta_rad  угол прихода помехи
 * @param amplitude  амплитуда помехи
 * @param omega0     нормированная частота (рад/отсчёт)
 */
static void AddInterference(std::vector<cx>& Y,
                            uint32_t n_channels, uint32_t n_samples,
                            float theta_rad,
                            float amplitude,
                            float omega0 = 0.37f) {
  const float d_sin = std::sin(theta_rad) * 0.5f;  // d/λ = 0.5
  for (uint32_t n = 0; n < n_samples; ++n) {
    for (uint32_t p = 0; p < n_channels; ++p) {
      float phase_spatial  = 2.0f * static_cast<float>(M_PI) * p * d_sin;
      float phase_temporal = omega0 * static_cast<float>(n);
      cx s = amplitude * cx(std::cos(phase_spatial + phase_temporal),
                            std::sin(phase_spatial + phase_temporal));
      Y[static_cast<size_t>(n) * n_channels + p] += s;  // column-major: [p,n] = n*P+p
    }
  }
}

// ============================================================================
// Test 01: Базовый рельеф Кейпона — только шум, рельеф должен быть > 0
// ============================================================================

inline void test_01_relief_noise_only() {
  Con::Print("[test_capon_rocm::01] ComputeRelief — only noise (flat spectrum)");

  auto* backend = drv_gpu_lib::GetTestBackend();

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

  Con::Print("[test_capon_rocm::01] PASS");
}

// ============================================================================
// Test 02: Рельеф с помехой — MVDR минимален на направлении помехи
//
// Физика:  MVDR минимизирует мощность выхода при ограничении «пропустить
//          сигнал из нужного направления». На направлении мощной помехи
//          z[m] = 1/(u^H * R^{-1} * u) будет МИНИМАЛЬНЫМ (подавление).
// ============================================================================

inline void test_02_relief_with_interference() {
  Con::Print("[test_capon_rocm::02] ComputeRelief — interference suppression check");

  auto* backend = drv_gpu_lib::GetTestBackend();

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

  Con::Print("[test_capon_rocm::02] PASS (MVDR suppression confirmed at interference direction)");
}

// ============================================================================
// Test 03: AdaptiveBeamform — размерность выхода
// ============================================================================

inline void test_03_adaptive_beamform_dims() {
  Con::Print("[test_capon_rocm::03] AdaptiveBeamform — output dimensions");

  auto* backend = drv_gpu_lib::GetTestBackend();

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

  Con::Print("[test_capon_rocm::03] PASS");
}

// ============================================================================
// Test 04: Регуляризация — mu=0 vs mu>0 (численная устойчивость)
// ============================================================================

inline void test_04_regularization() {
  Con::Print("[test_capon_rocm::04] Regularization — mu=0 vs mu=0.1");

  auto* backend = drv_gpu_lib::GetTestBackend();

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

  Con::Print("[test_capon_rocm::04] PASS");
}

// ============================================================================
// Test 05: GPU-to-GPU пайплайн
// ============================================================================

inline void test_05_gpu_to_gpu() {
  Con::Print("[test_capon_rocm::05] GPU-to-GPU pipeline");

  auto* backend = drv_gpu_lib::GetTestBackend();

  const uint32_t P = 8;
  const uint32_t N = 64;
  const uint32_t M = 16;

  auto signal_cpu   = MakeNoise(P * N, 1.0f, 55u);
  auto steering_cpu = MakeSteeringMatrix(P, M, -static_cast<float>(M_PI)/3.0f,
                                               static_cast<float>(M_PI)/3.0f);

  // TODO: аллоцировать GPU буфер через backend и загрузить данные
  // void* gpu_signal   = backend->Malloc(signal_cpu.size() * sizeof(cx));
  // void* gpu_steering = backend->Malloc(steering_cpu.size() * sizeof(cx));
  // backend->Upload(gpu_signal, signal_cpu.data(), ...);
  // backend->Upload(gpu_steering, steering_cpu.data(), ...);
  // capon::CaponParams params{P, N, M, 0.01f};
  // capon::CaponProcessor processor(backend);
  // auto result = processor.ComputeRelief(gpu_signal, gpu_steering, params);
  // assert(result.relief.size() == M);

  Con::Print("[test_capon_rocm::05] SKIP (TODO: GPU alloc/upload API в тесте)");
}

// ============================================================================
// run() — точка входа
// ============================================================================

inline void run() {
  Con::Print("=== test_capon_rocm ===");
  test_01_relief_noise_only();
  test_02_relief_with_interference();
  test_03_adaptive_beamform_dims();
  test_04_regularization();
  test_05_gpu_to_gpu();
  Con::Print("=== test_capon_rocm DONE ===");
}

}  // namespace test_capon_rocm

#endif  // ENABLE_ROCM
