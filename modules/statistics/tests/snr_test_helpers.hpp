#pragma once

/**
 * @file snr_test_helpers.hpp
 * @brief Test utilities for SNR-estimator (SNR_08)
 *
 * Генерация CW/AWGN сигналов, GPU upload helpers, shared backend.
 * Namespace: snr_test_helpers
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace snr_test_helpers {

using cx = std::complex<float>;

/**
 * @brief Сгенерировать комплексный CW (тон после дечирпа LFM).
 * @param n_samples  число отсчётов
 * @param freq_norm  нормированная частота ∈ (-0.5, 0.5) (f_d / f_s)
 * @param amplitude  амплитуда A
 *
 * Формула: s[n] = A * exp(2π * freq_norm * n * j)
 */
inline std::vector<cx> MakeDechirpedCW(
    uint32_t n_samples, float freq_norm, float amplitude)
{
  std::vector<cx> signal(n_samples);
  const float two_pi_f = 2.0f * 3.14159265358979323846f * freq_norm;
  for (uint32_t n = 0; n < n_samples; ++n) {
    float phase = two_pi_f * static_cast<float>(n);
    signal[n] = cx(amplitude * std::cos(phase),
                   amplitude * std::sin(phase));
  }
  return signal;
}

/**
 * @brief Сгенерировать комплексный AWGN через LCG + Box-Muller.
 * @param n_samples    число отсчётов
 * @param noise_power  σ² (дисперсия комплексного шума, E[|z|²] = σ²)
 * @param seed         LCG seed
 *
 * Реальная и мнимая части независимы, каждая ~ N(0, σ²/2).
 * Детерминированная генерация (LCG), поэтому тесты воспроизводимы.
 */
inline std::vector<cx> MakeNoise(
    uint32_t n_samples, float noise_power, uint32_t seed = 42u)
{
  std::vector<cx> noise(n_samples);
  uint32_t state = seed ? seed : 1u;
  auto rng_uniform = [&]() -> float {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
  };

  const float sigma = std::sqrt(noise_power / 2.0f);
  for (uint32_t i = 0; i < n_samples; ++i) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-30f) u1 = 1e-30f;  // защита log(0)
    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    noise[i] = cx(sigma * r * std::cos(theta),
                  sigma * r * std::sin(theta));
  }
  return noise;
}

/// Добавить AWGN к сигналу in-place.
inline void AddNoise(std::vector<cx>& signal,
                     float noise_power, uint32_t seed = 0u)
{
  auto noise = MakeNoise(static_cast<uint32_t>(signal.size()),
                         noise_power, seed ? seed : 1u);
  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] += noise[i];
  }
}

/**
 * @brief Скопировать CPU complex vector в свежий hipMalloc буфер.
 * Caller должен освободить через FreeGpu().
 */
inline void* CopyToGpu(const std::vector<cx>& data) {
  void* gpu_ptr = nullptr;
  size_t bytes = data.size() * sizeof(cx);
  hipError_t err = hipMalloc(&gpu_ptr, bytes);
  if (err != hipSuccess || !gpu_ptr) {
    throw std::runtime_error(
        "snr_test_helpers::CopyToGpu: hipMalloc failed (" +
        std::to_string(bytes / (1024u * 1024u)) + " MB)");
  }
  err = hipMemcpy(gpu_ptr, data.data(), bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(gpu_ptr);
    throw std::runtime_error("snr_test_helpers::CopyToGpu: hipMemcpy H2D failed");
  }
  return gpu_ptr;
}

/// Освободить hipMalloc буфер (nullptr-safe).
inline void FreeGpu(void* ptr) {
  if (ptr) hipFree(ptr);
}

/**
 * @brief Shared test ROCm backend (singleton, device 0).
 *
 * NB: Создаётся один раз на процесс — не destroy'ится (singleton lifetime).
 * Это ОК для тестов (один device 0 на сессию).
 */
inline drv_gpu_lib::IBackend* GetTestBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  static bool initialized = false;
  if (!initialized) {
    backend.Initialize(0);
    initialized = true;
  }
  return &backend;
}

}  // namespace snr_test_helpers

#endif  // ENABLE_ROCM
