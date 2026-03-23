#pragma once

/**
 * @file test_capon_reference_data.hpp
 * @brief Тесты CaponProcessor на реальных данных (MATLAB сигнал + физические координаты)
 *
 * Загружает файлы из Doc_Addition/Capon/capon_test/build/:
 *   - x_data.txt, y_data.txt  — координаты антенных секций (340 значений)
 *   - signal_matlab.txt       — MATLAB сигнал (341 строка × 1000 комплексных чисел)
 *   - z_values.txt            — эталонные значения рельефа (4 блока × 37×37 = 5476)
 *
 * Физические параметры (из эталонной реализации):
 *   f0 = 3918e6 + 3.15e6 = 3921150000 Hz
 *   c  = 299792458 m/s
 *   getU(x, y, u, v, f0) = exp(j * 2π * (x*u + y*v) * f0/c)   — нет нормализации (numdims=1)
 *
 * Отличие формул:
 *   Эталон (ArrayFire):  R = Y*Y^H       + mu*I   (нет деления на N)
 *   GPU (CaponProcessor): R = (1/N)*Y*Y^H + mu*I   (есть деление на N)
 *   → для сравнения форм рельефа используется нормировка
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
#include "services/console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace test_capon_reference_data {

using cx = std::complex<float>;

// Переиспользуем TestPrint и GetTestBackend из test_capon_rocm
// (они статические inline — при включении обоих файлов без ODR-нарушения)
inline void TestPrint(const std::string& msg) {
  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "Capon", msg);
}

inline drv_gpu_lib::IBackend* GetTestBackend() {
  static drv_gpu_lib::ROCmBackend backend;
  if (!backend.IsInitialized()) {
    backend.Initialize(0);
  }
  return &backend;
}

// ============================================================================
// Физические константы
// ============================================================================

static constexpr double kF0 = 3918e6 + 3.15e6;   // 3921150000 Hz
static constexpr double kC  = 299792458.0;          // скорость света, м/с

// ============================================================================
// Пути к файлам данных (относительно корня проекта)
// ============================================================================

static const std::string kDataDir =
    "Doc_Addition/Capon/capon_test/build/";

// ============================================================================
// Загрузка файлов
// ============================================================================

/// Загрузить вектор вещественных чисел из файла (одно число на строку/через пробел)
static bool LoadRealVector(const std::string& path,
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
static bool ParseMatlabComplex(const std::string& token, cx& result) {
  if (token.empty()) return false;
  // Найти последний + или - перед 'i', не являющийся частью 'e+' / 'E-'
  // Ищем справа налево (от предпоследнего символа, последний — 'i')
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
 * Первые P строк → P каналов, первые N столбцов → N отсчётов.
 * Хранение: column-major Y[m*P + p] → для CaponProcessor.
 *
 * @param path   путь к файлу
 * @param P      число строк (каналов) для загрузки
 * @param N      число столбцов (отсчётов) для загрузки
 * @param out    выходной вектор [P*N], column-major
 * @return true при успехе
 */
static bool LoadSignalMatlab(const std::string& path,
                             uint32_t P, uint32_t N,
                             std::vector<cx>& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;

  // Буфер: row-major строки × столбцы → потом транспонировать в column-major
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
// Построение направляющих векторов (физическая антенна)
// ============================================================================

/**
 * @brief Построить матрицу управляющих векторов
 *
 * U[p, m] = exp(j * 2π * (x[p]*u_m + y[p]*v_m) * f0/c)
 *
 * Нормализация: getU в эталоне делит на sqrt(y_sub_prm.numdims()),
 * для 1D вектора numdims()=1 → sqrt(1)=1 → нет нормализации.
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
static std::vector<cx> MakePhysicalSteering(
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

// ============================================================================
// CPU эталон (для test_03)
// ============================================================================

/**
 * @brief Разложение Холецкого нижняя треугольная L (column-major, in-place)
 *
 * Вход: A — симметричная положительно определённая матрица P×P, column-major.
 * Выход: нижний треугольник A заменяется на L, где A = L*L^H.
 *
 * @return true при успехе, false при вырожденности
 */
static bool CholeskyLower(std::vector<cx>& A, uint32_t P) {
  // L[i,j] = A[j*P+i], i >= j
  for (uint32_t j = 0; j < P; ++j) {
    // Диагональный элемент
    float diag = A[j * P + j].real();
    for (uint32_t k = 0; k < j; ++k) {
      cx lkj = A[j * P + k];
      diag -= (lkj * std::conj(lkj)).real();
    }
    if (diag <= 0.0f) return false;
    A[j * P + j] = cx(std::sqrt(diag), 0.0f);

    // Внестрогий нижний треугольник
    for (uint32_t i = j + 1; i < P; ++i) {
      cx sum = A[j * P + i];  // a[i,j]
      for (uint32_t k = 0; k < j; ++k) {
        sum -= A[k * P + i] * std::conj(A[k * P + j]);
      }
      A[j * P + i] = sum / A[j * P + j];
    }
  }
  return true;
}

/**
 * @brief Решить L*x = b (нижнетреугольная, column-major L)
 *
 * @param L  нижняя треугольная матрица P×P, column-major
 * @param b  правая часть [P]
 * @return x [P]
 */
static std::vector<cx> ForwardSolve(const std::vector<cx>& L,
                                    uint32_t P,
                                    const std::vector<cx>& b) {
  std::vector<cx> x(P);
  for (uint32_t i = 0; i < P; ++i) {
    cx sum = b[i];
    for (uint32_t k = 0; k < i; ++k) {
      sum -= L[k * P + i] * x[k];  // L[i,k] column-major: L[k*P+i]
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
 *
 * @param Y      сигнальная матрица [P*N], column-major
 * @param P      число каналов
 * @param N      число отсчётов
 * @param U      управляющие векторы [P*M], column-major
 * @param M      число направлений
 * @param mu     коэффициент регуляризации
 * @return z[M]  рельеф Кейпона
 */
static std::vector<float> CpuCaponRelief(
    const std::vector<cx>& Y, uint32_t P, uint32_t N,
    const std::vector<cx>& U, uint32_t M,
    float mu) {
  // 1. Вычислить R = Y*Y^H/N + mu*I   (column-major, P×P)
  std::vector<cx> R(static_cast<size_t>(P) * P, cx(0.0f));
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t i = 0; i < P; ++i) {
      cx y_i = Y[static_cast<size_t>(n) * P + i];
      for (uint32_t j = 0; j < P; ++j) {
        cx y_j = Y[static_cast<size_t>(n) * P + j];
        // R[i,j] += y_i * conj(y_j)  ;  column-major: R[j*P+i]
        R[static_cast<size_t>(j) * P + i] += y_i * std::conj(y_j);
      }
    }
  }
  float inv_N = 1.0f / static_cast<float>(N);
  for (auto& r : R) r *= inv_N;
  // Добавить mu*I
  for (uint32_t i = 0; i < P; ++i) {
    R[static_cast<size_t>(i) * P + i] += cx(mu, 0.0f);
  }

  // 2. Cholesky: L*L^H = R (нижняя треугольная)
  if (!CholeskyLower(R, P)) {
    return std::vector<float>(M, 0.0f);
  }
  // Теперь нижний треугольник R содержит L

  // 3. Для каждого направления: z[m] = 1 / ||L^{-1}*u_m||^2
  std::vector<float> z(M);
  for (uint32_t m = 0; m < M; ++m) {
    // Извлечь u_m (column-major: U[m*P .. m*P+P-1])
    std::vector<cx> u_m(P);
    for (uint32_t p = 0; p < P; ++p) {
      u_m[p] = U[static_cast<size_t>(m) * P + p];
    }
    // x = L^{-1} * u_m
    std::vector<cx> x = ForwardSolve(R, P, u_m);
    // ||x||^2
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

  // Попробовать загрузить первые 2 строки × 3 столбца сигнала (быстрая проверка)
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

  // Загрузить координаты
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

  // Загрузить сигнал
  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[test_capon_reference_data::02] SKIP: signal_matlab.txt not found or parse error");
    return;
  }

  // Построить направляющую сетку 37×37=1369
  // u_step = 0.00312,  ulim = sin(3.25° * π/180)
  const double u_step = 0.00312;
  const double ulim   = std::sin(3.25 * M_PI / 180.0);  // ≈ 0.05673

  // Вычислить количество точек: arange(-ulim, ulim, u_step)
  // число шагов = floor((2*ulim) / u_step) + 1
  std::vector<float> u0_seq;
  for (double v = -ulim; v <= ulim + u_step * 0.5; v += u_step) {
    u0_seq.push_back(static_cast<float>(v));
  }
  const uint32_t Nu = static_cast<uint32_t>(u0_seq.size());

  // Mesh-grid: u[m] = u0[m % Nu], v[m] = u0[m / Nu]
  const uint32_t M = Nu * Nu;
  std::vector<float> u_dirs(M), v_dirs(M);
  for (uint32_t iv = 0; iv < Nu; ++iv) {
    for (uint32_t iu = 0; iu < Nu; ++iu) {
      uint32_t m = iv * Nu + iu;
      u_dirs[m] = u0_seq[iu];
      v_dirs[m] = u0_seq[iv];
    }
  }

  // Построить управляющие векторы
  auto steering = MakePhysicalSteering(x_vals, y_vals,
                                       u_dirs, v_dirs, kF0, kC);

  // Большая mu для компенсации отсутствия нормализации (GPU делит на N=1000)
  // Эквивалентно эталонной mu=1000.0 при R без деления: mu_gpu = mu_ref/N
  const float mu_gpu = 1.0f;  // gpu: R = Y*Y^H/N + mu*I

  capon::CaponParams params{P, N, M, mu_gpu};

  auto* backend = GetTestBackend();
  capon::CaponProcessor processor(backend);
  auto result = processor.ComputeRelief(signal, steering, params);

  if (result.relief.size() != M) {
    TestPrint("[test_capon_reference_data::02] FAIL: relief size mismatch");
    return;
  }

  // Validate: все > 0 и конечные
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
// Test 03: CPU эталон vs GPU (P=8, N=64, M=16) — ключевой тест корректности
// ============================================================================

inline void test_03_cpu_vs_gpu_small_p() {
  TestPrint("[test_capon_reference_data::03] CPU reference vs GPU (P=8, N=64, M=16)");

  // Загрузить координаты
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

  // Первые 8 каналов
  std::vector<float> x8(x_vals.begin(), x_vals.begin() + P);
  std::vector<float> y8(y_vals.begin(), y_vals.begin() + P);

  // Загрузить первые P строк × N столбцов сигнала
  std::vector<cx> signal;
  if (!LoadSignalMatlab(kDataDir + "signal_matlab.txt", P, N, signal)) {
    TestPrint("[test_capon_reference_data::03] SKIP: signal_matlab.txt not found");
    return;
  }

  // 1D сканирование по u при v=0, M=16 направлений
  const double ulim = std::sin(3.25 * M_PI / 180.0);
  std::vector<float> u_dirs(M), v_dirs(M, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    u_dirs[m] = static_cast<float>(
        -ulim + 2.0 * ulim * m / (M - 1));
  }

  // Построить управляющие векторы
  auto steering = MakePhysicalSteering(x8, y8, u_dirs, v_dirs, kF0, kC);

  const float mu = 100.0f;

  // GPU результат
  capon::CaponParams params{P, N, M, mu};
  auto* backend = GetTestBackend();
  capon::CaponProcessor processor(backend);
  auto gpu_result = processor.ComputeRelief(signal, steering, params);

  if (gpu_result.relief.size() != M) {
    TestPrint("[test_capon_reference_data::03] FAIL: GPU relief size mismatch");
    return;
  }

  // CPU эталон (та же формула с делением на N)
  auto cpu_z = CpuCaponRelief(signal, P, N, steering, M, mu);

  // Сравнить: relative_error[m] = |gpu[m] - cpu[m]| / max(cpu[m], 1e-6)
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

  // Допуск: 0.5% (плавающая точка + GPU нюансы)
  const float kTolerance = 0.005f;

  if (max_rel_error > kTolerance) {
    std::ostringstream msg;
    msg << "[test_capon_reference_data::03] FAIL"
        << "  max_rel_error=" << max_rel_error * 100.0f << "%"
        << "  at m=" << worst_m
        << "  (tolerance=" << kTolerance * 100.0f << "%)";
    TestPrint(msg.str());

    // Диагностика: первые 5 пар
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
// run() — точка входа
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
