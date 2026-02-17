#pragma once

/**
 * @file delayed_form_signal_generator.hpp
 * @brief DelayedFormSignalGenerator — генератор с дробной задержкой (Farrow 48×5)
 *
 * Алгоритм:
 *   1. Генерация чистого сигнала (getX, noise=0) через FormSignalGenerator
 *   2. Применение дробной задержки (Lagrange 48×5) per-antenna на GPU
 *   3. Добавление шума (Philox + Box-Muller) если noise_amplitude > 0
 *
 * Задержка задаётся в микросекундах (float) на каждую антенну.
 * Матрица 48×5 — предвычисленные Lagrange-коэффициенты (встроена в код
 * или загружается из JSON).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "../params/form_params.hpp"
#include "form_signal_generator.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

/**
 * @class DelayedFormSignalGenerator
 * @brief GPU-генератор с дробной задержкой Farrow (Lagrange 48×5)
 *
 * @code
 * DelayedFormSignalGenerator gen(backend);
 *
 * FormParams params;
 * params.fs = 12e6;
 * params.f0 = 1e6;
 * params.antennas = 8;
 * params.points = 4096;
 * params.amplitude = 1.0;
 * params.noise_amplitude = 0.1;  // шум добавляется ПОСЛЕ задержки
 * gen.SetParams(params);
 *
 * // Задержки в микросекундах (по одной на антенну)
 * gen.SetDelays({0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f, 10.5f});
 *
 * // GPU
 * auto input = gen.GenerateInputData();
 * clReleaseMemObject(input.data);
 *
 * // CPU
 * auto cpu_data = gen.GenerateToCpu();
 * @endcode
 */
class DelayedFormSignalGenerator {
public:
  explicit DelayedFormSignalGenerator(drv_gpu_lib::IBackend* backend);
  ~DelayedFormSignalGenerator();

  // No copy
  DelayedFormSignalGenerator(const DelayedFormSignalGenerator&) = delete;
  DelayedFormSignalGenerator& operator=(const DelayedFormSignalGenerator&) = delete;

  // Move
  DelayedFormSignalGenerator(DelayedFormSignalGenerator&& other) noexcept;
  DelayedFormSignalGenerator& operator=(DelayedFormSignalGenerator&& other) noexcept;

  /// Установить параметры сигнала (FormParams)
  void SetParams(const FormParams& params);

  /// Установить параметры из строки
  void SetParamsFromString(const std::string& params_str);

  /**
   * @brief Установить задержки per-antenna в микросекундах
   * @param delay_us Вектор задержек (float), delay_us.size() == antennas
   * @throws std::invalid_argument если размер не совпадает с antennas
   */
  void SetDelays(const std::vector<float>& delay_us);

  /**
   * @brief Загрузить матрицу Lagrange из JSON-файла
   * @param json_path Путь к файлу (формат: { "data": [[...], ...] })
   *
   * Если не вызвано — используется встроенная матрица 48×5.
   */
  void LoadMatrix(const std::string& json_path);

  /**
   * @brief Генерация на GPU: сигнал + задержка + шум
   * @return InputData<cl_mem>
   * @note Вызывающий код освобождает result.data через clReleaseMemObject()
   */
  drv_gpu_lib::InputData<cl_mem> GenerateInputData();

  /**
   * @brief Генерация с возвратом на CPU
   * @return vector[antenna_id][sample_id] complex<float>
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ══════════════════════════════════════════════════════════════════════
  // Getters
  // ══════════════════════════════════════════════════════════════════════

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }
  const std::vector<float>& GetDelays() const { return delay_us_; }

private:
  void CompileDelayKernel();
  void UploadMatrix();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;
  std::vector<float> delay_us_;

  // Lagrange matrix 48×5 (240 floats)
  std::vector<float> lagrange_matrix_;
  bool matrix_loaded_ = false;

  // OpenCL resources
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program delay_program_ = nullptr;
  cl_mem matrix_buf_ = nullptr;    // __constant: 48×5 floats

  // FormSignalGenerator for clean signal generation
  FormSignalGenerator signal_gen_;
};

} // namespace signal_gen
