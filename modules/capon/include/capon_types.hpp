#pragma once

/**
 * @file capon_types.hpp
 * @brief Типы данных модуля Capon — параметры, результаты, индексы буферов.
 *
 * Алгоритм Кейпона (MVDR — Minimum Variance Distortionless Response):
 *   1. Ковариационная матрица: R = (1/N) * Y * Y^H + μI
 *   2. Обращение матрицы:      R^{-1}  (rocSOLVER POTRF+POTRI)
 *   3. Рельеф Кейпона:         z[m] = 1 / Re(u_m^H * R^{-1} * u_m)
 *   4. Адаптивное ДО:          w = R^{-1} * U,  Y_out = w^H * Y
 *
 *   Y — матрица принятого сигнала [n_channels × n_samples]
 *   U — матрица управляющих векторов  [n_channels × n_directions]
 *   R — ковариационная матрица  [n_channels × n_channels]
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#include <complex>
#include <cstdint>
#include <vector>

namespace capon {

// ============================================================================
// Параметры алгоритма
// ============================================================================

struct CaponParams {
  uint32_t n_channels;    ///< P — число антенных каналов (строки Y и U)
  uint32_t n_samples;     ///< N — число временных отсчётов (столбцы Y)
  uint32_t n_directions;  ///< M — число направлений (столбцы U)
  float    mu = 0.0f;     ///< Коэффициент регуляризации (диагональная загрузка)
};

// ============================================================================
// Результаты
// ============================================================================

/// Рельеф Кейпона — M вещественных значений (пространственный спектр)
struct CaponReliefResult {
  std::vector<float> relief;  ///< Re(1 / (u^H * R^{-1} * u)) — размер n_directions
};

/// Адаптивное диаграммообразование — матрица [n_directions × n_samples]
struct CaponBeamResult {
  uint32_t n_directions = 0;
  uint32_t n_samples    = 0;
  std::vector<std::complex<float>> output;  ///< n_directions × n_samples (row-major)
};

// Индексы разделяемых буферов (GpuContext shared buffers) — только ROCm
#if ENABLE_ROCM
namespace shared_buf {
  static constexpr size_t kSignal    = 0;
  static constexpr size_t kSteering  = 1;
  static constexpr size_t kCovMatrix = 2;
  static constexpr size_t kWeight    = 3;
  static constexpr size_t kOutput    = 4;
  static constexpr size_t kCount     = 5;
}  // namespace shared_buf
#endif  // ENABLE_ROCM

}  // namespace capon
