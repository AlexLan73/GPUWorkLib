#pragma once

/**
 * @file form_signal_generator.hpp
 * @brief FormSignalGenerator — мультиканальный генератор комплексных сигналов
 *
 * Формула getX:
 *   X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
 *     + an*norm*(randn + j*randn)
 *   X = 0 при t < 0 или t > ti - dt
 *
 * Поддержка:
 * - Мультиканальная генерация (antennas каналов параллельно)
 * - Per-channel задержка: FIXED / LINEAR / RANDOM
 * - Шум: Philox-2x32 + Box-Muller (встроен в kernel)
 * - Chirp: fdev != 0 дает ЛЧМ-модуляцию
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

/**
 * @class FormSignalGenerator
 * @brief GPU-генератор комплексных сигналов по формуле getX
 *
 * Standalone класс (не ISignalGenerator) — свой API для мультиканальности.
 *
 * @code
 * FormSignalGenerator gen(backend);
 *
 * FormParams params;
 * params.fs = 12e6;
 * params.f0 = 1e6;
 * params.antennas = 8;
 * params.points = 4096;
 * params.amplitude = 1.0;
 * params.noise_amplitude = 0.1;
 * params.tau_base = 0.0;
 * params.tau_step = 0.0001;
 * gen.SetParams(params);
 *
 * // GPU
 * cl_mem gpu_buf = gen.Generate();
 * // ... use gpu_buf [antennas * points * sizeof(complex<float>)] ...
 * clReleaseMemObject(gpu_buf);
 *
 * // CPU (vector per channel)
 * auto cpu_data = gen.GenerateToCpu();
 * // cpu_data[antenna_id][sample_id]
 * @endcode
 */
class FormSignalGenerator {
public:
  explicit FormSignalGenerator(drv_gpu_lib::IBackend* backend);
  ~FormSignalGenerator();

  // No copy
  FormSignalGenerator(const FormSignalGenerator&) = delete;
  FormSignalGenerator& operator=(const FormSignalGenerator&) = delete;

  // Move
  FormSignalGenerator(FormSignalGenerator&& other) noexcept;
  FormSignalGenerator& operator=(FormSignalGenerator&& other) noexcept;

  /// Установить параметры
  void SetParams(const FormParams& params) { params_ = params; }

  /// Установить параметры из строки "f0=1e6,a=1.0,an=0.1,tau=0.001"
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  /**
   * @brief Генерация на GPU
   * @return cl_mem [antennas * points * sizeof(complex<float>)]
   * @note Вызывающий код должен освободить через clReleaseMemObject()!
   */
  cl_mem Generate();

  /**
   * @brief Генерация с возвратом на CPU (по каналам)
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

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;

  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen
