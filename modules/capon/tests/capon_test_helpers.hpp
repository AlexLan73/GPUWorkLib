#pragma once

/**
 * @file capon_test_helpers.hpp
 * @brief Общие утилиты тестов модуля capon
 *
 * Содержит:
 *   - GetROCmBackend()     — shared singleton ROCm backend (device 0)
 *   - MakeSteeringMatrix() — ULA steering: u[p,m] = exp(j*2π*p*0.5*sin(θ_m))
 *   - MakeNoise()          — CN(0, σ²) через LCG + Box-Muller
 *   - AddInterference()    — CW-помеха из направления θ
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-23
 */

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace capon_test_helpers {

using cx = std::complex<float>;

// ============================================================================
// Shared ROCm backend (singleton, device 0)
// ============================================================================

inline drv_gpu_lib::ROCmBackend& GetROCmBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  if (!backend.IsInitialized()) {
    backend.Initialize(0);
  }
  return backend;
}

// ============================================================================
// Steering matrix (ULA)
// ============================================================================

/// Управляющие векторы ULA: u[p,m] = exp(j*2π*p*(d/λ)*sin(θ_m))
/// d/λ = 0.5 (полуволновой интервал), θ сканирует [theta_min, theta_max]
/// Хранение: column-major, [p + m*P]
inline std::vector<cx> MakeSteeringMatrix(
    uint32_t n_channels, uint32_t n_directions,
    float theta_min_rad, float theta_max_rad) {
  std::vector<cx> U(static_cast<size_t>(n_channels) * n_directions);
  for (uint32_t m = 0; m < n_directions; ++m) {
    float theta = (n_directions > 1)
        ? theta_min_rad + (theta_max_rad - theta_min_rad) * m / (n_directions - 1)
        : theta_min_rad;
    float d_sin = std::sin(theta) * 0.5f;
    for (uint32_t p = 0; p < n_channels; ++p) {
      float phase = 2.0f * static_cast<float>(M_PI) * p * d_sin;
      U[m * n_channels + p] = cx(std::cos(phase), std::sin(phase));
    }
  }
  return U;
}

// ============================================================================
// Noise generator
// ============================================================================

/**
 * @brief Сгенерировать CN(0, sigma²) шум через LCG + Box-Muller.
 *
 * Воспроизводимый (детерминированный seed), но статистически корректный:
 * амплитуда гауссова, фаза равномерная → правильная ковариационная матрица.
 */
inline std::vector<cx> MakeNoise(size_t count, float sigma = 1.0f,
                                 uint32_t seed = 42) {
  std::vector<cx> noise(count);
  uint32_t state = seed;

  auto rng_uniform = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu));
  };

  for (size_t i = 0; i < count; ++i) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;  // защита от log(0)

    float mag = sigma * std::sqrt(-2.0f * std::log(u1));
    float phi = 2.0f * static_cast<float>(M_PI) * u2;
    noise[i]  = cx(mag * std::cos(phi), mag * std::sin(phi));
  }
  return noise;
}

// ============================================================================
// Interference
// ============================================================================

/**
 * @brief Добавить CW-помеху из направления theta_rad в сигнальную матрицу.
 *
 * Y[p, n] += amplitude * exp(j*2π*p*(d/λ)*sin(θ)) * exp(j*ω₀*n)
 * Хранение Y: column-major, индекс [p, n] = n*P + p
 */
inline void AddInterference(std::vector<cx>& Y,
                            uint32_t n_channels, uint32_t n_samples,
                            float theta_rad,
                            float amplitude,
                            float omega0 = 0.37f) {
  const float d_sin = std::sin(theta_rad) * 0.5f;
  for (uint32_t n = 0; n < n_samples; ++n) {
    for (uint32_t p = 0; p < n_channels; ++p) {
      float phase_spatial  = 2.0f * static_cast<float>(M_PI) * p * d_sin;
      float phase_temporal = omega0 * static_cast<float>(n);
      cx s = amplitude * cx(std::cos(phase_spatial + phase_temporal),
                            std::sin(phase_spatial + phase_temporal));
      Y[static_cast<size_t>(n) * n_channels + p] += s;
    }
  }
}

}  // namespace capon_test_helpers

#endif  // ENABLE_ROCM
