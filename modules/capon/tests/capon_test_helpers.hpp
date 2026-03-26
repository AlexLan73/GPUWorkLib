#pragma once

/**
 * @file capon_test_helpers.hpp
 * @brief Общие утилиты тестов модуля capon
 *
 * Содержит:
 *   - GetROCmBackend()       — shared singleton ROCm backend (device 0)
 *   - MakeSteeringMatrix()   — ULA steering: u[p,m] = exp(j*2pi*p*0.5*sin(theta_m))
 *   - MakeNoise()            — CN(0, sigma^2) через LCG + Box-Muller
 *   - AddInterference()      — CW-помеха из направления theta
 *
 * Загрузка данных заказчика:
 *   - kDataDir               — путь к файлам данных
 *   - kF0, kC                — физические константы антенной решётки
 *   - LoadRealVector()       — загрузка вещественного вектора из файла
 *   - ParseMatlabComplex()   — парсер MATLAB complex: "6.17+18.74i"
 *   - LoadSignalMatlab()     — загрузка сигнальной матрицы [P x N]
 *   - MakePhysicalSteering() — управляющие векторы по реальным координатам
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
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

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
// Физические константы (из эталонной реализации заказчика)
// ============================================================================

static constexpr double kF0 = 3918e6 + 3.15e6;   // 3921150000 Hz
static constexpr double kC  = 299792458.0;        // скорость света, м/с

// ============================================================================
// Путь к данным заказчика (относительно корня проекта)
//
// Данные лежат в modules/capon/tests/data/ — локальная копия из прототипа.
// Оригинал: Doc_Addition/Capon/capon_test/build/
// См. памятку: modules/capon/tests/data/README_DATA.md
// ============================================================================

static const std::string kDataDir =
    "modules/capon/tests/data/";

// ============================================================================
// Steering matrix (ULA — синтетическая)
// ============================================================================

/// Управляющие векторы ULA: u[p,m] = exp(j*2pi*p*(d/lambda)*sin(theta_m))
/// d/lambda = 0.5 (полуволновой интервал), theta сканирует [theta_min, theta_max]
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
 * @brief Сгенерировать CN(0, sigma^2) шум через LCG + Box-Muller.
 *
 * Воспроизводимый (детерминированный seed), но статистически корректный:
 * амплитуда гауссова, фаза равномерная -> правильная ковариационная матрица.
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
 * Y[p, n] += amplitude * exp(j*2pi*p*(d/lambda)*sin(theta)) * exp(j*omega0*n)
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

// ============================================================================
// Загрузка данных заказчика — файлы из Doc_Addition/Capon/capon_test/build/
// ============================================================================

/// Загрузить вектор вещественных чисел из файла (одно число на строку/через пробел)
inline bool LoadRealVector(const std::string& path,
                           std::vector<float>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  double v;
  while (f >> v) {
    out.push_back(static_cast<float>(v));
  }
  return !out.empty();
}

/// Разобрать одно комплексное число в формате MATLAB: "реальная+мнимаяi"
/// Поддерживает оба знака: "6.17+18.74i" и "-9.80-30.59i"
inline bool ParseMatlabComplex(const std::string& token, cx& result) {
  if (token.empty()) return false;
  if (token.back() != 'i') return false;
  std::string s = token.substr(0, token.size() - 1);  // обрезать 'i'

  // Ищем последний + или -, стоящий НЕ после 'e'/'E'
  int split_pos = -1;
  for (int i = static_cast<int>(s.size()) - 1; i > 0; --i) {
    char c = s[i];
    if ((c == '+' || c == '-') &&
        s[i - 1] != 'e' && s[i - 1] != 'E') {
      split_pos = i;
      break;
    }
  }
  if (split_pos <= 0) return false;

  std::string real_str = s.substr(0, split_pos);
  std::string imag_str = s.substr(split_pos);  // включает знак

  try {
    float re = std::stof(real_str);
    float im = std::stof(imag_str);
    result = cx(re, im);
    return true;
  } catch (...) {
    return false;
  }
}

/**
 * @brief Загрузить сигнальную матрицу из signal_matlab.txt
 *
 * Формат: 341 строка, каждая содержит 1000 комплексных чисел в MATLAB формате.
 * Первые P строк -> P каналов, первые N столбцов -> N отсчётов.
 * Хранение: column-major Y[n*P + p] -> для CaponProcessor.
 *
 * @param path   путь к файлу
 * @param P      число строк (каналов) для загрузки
 * @param N      число столбцов (отсчётов) для загрузки
 * @param out    выходной вектор [P*N], column-major
 * @return true при успехе
 */
inline bool LoadSignalMatlab(const std::string& path,
                             uint32_t P, uint32_t N,
                             std::vector<cx>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;

  // Буфер: row-major строки x столбцы -> потом транспонировать в column-major
  std::vector<std::vector<cx>> rows;
  rows.reserve(P);

  std::string line;
  uint32_t row_idx = 0;
  while (std::getline(f, line) && row_idx < P) {
    std::istringstream ss(line);
    std::vector<cx> row;
    row.reserve(N);
    std::string token;
    while (ss >> token && row.size() < N) {
      cx val;
      if (!ParseMatlabComplex(token, val)) return false;
      row.push_back(val);
    }
    if (row.size() < N) return false;
    rows.push_back(std::move(row));
    ++row_idx;
  }
  if (rows.size() < P) return false;

  // Транспонировать в column-major: Y[p + n*P]
  out.resize(static_cast<size_t>(P) * N);
  for (uint32_t p = 0; p < P; ++p) {
    for (uint32_t n = 0; n < N; ++n) {
      out[static_cast<size_t>(n) * P + p] = rows[p][n];
    }
  }
  return true;
}

// ============================================================================
// Управляющие векторы по реальным координатам антенной решётки
// ============================================================================

/**
 * @brief Построить матрицу управляющих векторов по физическим координатам
 *
 * U[p, m] = exp(j * 2pi * (x[p]*u_m + y[p]*v_m) * f0/c)
 *
 * Нормализация: getU в эталоне делит на sqrt(y_sub_prm.numdims()),
 * для 1D вектора numdims()=1 -> sqrt(1)=1 -> нет нормализации.
 *
 * Хранение column-major: U[m*P + p]
 *
 * @param x          координаты x[0..P-1]
 * @param y          координаты y[0..P-1]
 * @param u_dirs     u-координаты направлений [M]
 * @param v_dirs     v-координаты направлений [M]
 * @param f0         несущая частота, Гц
 * @param c          скорость распространения, м/с
 * @return вектор [P*M], column-major
 */
inline std::vector<cx> MakePhysicalSteering(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& u_dirs,
    const std::vector<float>& v_dirs,
    double f0, double c) {
  const uint32_t P = static_cast<uint32_t>(x.size());
  const uint32_t M = static_cast<uint32_t>(u_dirs.size());
  assert(u_dirs.size() == v_dirs.size());
  assert(x.size() == y.size());

  std::vector<cx> U(static_cast<size_t>(P) * M);
  const double k = 2.0 * M_PI * f0 / c;

  for (uint32_t m = 0; m < M; ++m) {
    double um = static_cast<double>(u_dirs[m]);
    double vm = static_cast<double>(v_dirs[m]);
    for (uint32_t p = 0; p < P; ++p) {
      double phase = k * (static_cast<double>(x[p]) * um +
                          static_cast<double>(y[p]) * vm);
      U[static_cast<size_t>(m) * P + p] =
          cx(static_cast<float>(std::cos(phase)),
             static_cast<float>(std::sin(phase)));
    }
  }
  return U;
}

/**
 * @brief Построить 1D сетку направлений для сканирования Кейпона
 *
 * Возвращает Nu равномерных точек от -ulim до +ulim с шагом u_step.
 * ulim = sin(angle_deg * pi/180).
 *
 * @param angle_deg   угол сканирования (градусы), default = 3.25
 * @param u_step      шаг сетки, default = 0.00312
 * @return вектор u-значений [Nu]
 */
inline std::vector<float> MakeScanGrid1D(double angle_deg = 3.25,
                                          double u_step = 0.00312) {
  const double ulim = std::sin(angle_deg * M_PI / 180.0);
  std::vector<float> u0;
  for (double v = -ulim; v <= ulim + u_step * 0.5; v += u_step) {
    u0.push_back(static_cast<float>(v));
  }
  return u0;
}

}  // namespace capon_test_helpers

#endif  // ENABLE_ROCM
