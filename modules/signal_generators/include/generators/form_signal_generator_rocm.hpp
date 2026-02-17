#pragma once

/**
 * @file form_signal_generator_rocm.hpp
 * @brief ROCm/HIP stub for FormSignalGenerator
 *
 * Placeholder for future AMD GPU support (HIP kernels).
 * All methods throw std::runtime_error("ROCm not implemented").
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"

#include <vector>
#include <complex>
#include <string>
#include <stdexcept>

namespace signal_gen {

/**
 * @class FormSignalGeneratorROCm
 * @brief ROCm stub — throws on any call
 *
 * Same API as FormSignalGenerator (OpenCL).
 * To implement: replace throws with HIP kernel logic (form_signal.hip).
 */
class FormSignalGeneratorROCm {
public:
  explicit FormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend)
      : backend_(backend) {}

  ~FormSignalGeneratorROCm() = default;

  // No copy
  FormSignalGeneratorROCm(const FormSignalGeneratorROCm&) = delete;
  FormSignalGeneratorROCm& operator=(const FormSignalGeneratorROCm&) = delete;

  /// Установить параметры
  void SetParams(const FormParams& params) { params_ = params; }

  /// Установить параметры из строки
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  /**
   * @brief Генерация на GPU (ROCm) — NOT IMPLEMENTED
   * @throws std::runtime_error always
   */
  void* Generate() {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::Generate(): ROCm backend not implemented. "
        "Requires HIP kernels (kernels/rocm/form_signal.hip).");
  }

  /**
   * @brief Генерация с возвратом на CPU — NOT IMPLEMENTED
   * @throws std::runtime_error always
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::GenerateToCpu(): ROCm backend not implemented.");
  }

  // ══════════════════════════════════════════════════════════════════════
  // Getters (work without GPU)
  // ══════════════════════════════════════════════════════════════════════

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

private:
  drv_gpu_lib::IBackend* backend_ = nullptr;
  FormParams params_;
};

} // namespace signal_gen
