#pragma once

/**
 * @file delayed_form_signal_generator_rocm.hpp
 * @brief ROCm: FormSignalGeneratorROCm + LchFarrowROCm pipeline
 *
 * Pipeline:
 *   1. FormSignalGeneratorROCm → clean signal (GPU)
 *   2. LchFarrowROCm → fractional delay (Lagrange 48x5) per-antenna
 *   3. Noise added through LchFarrowROCm::SetNoise()
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "../params/form_params.hpp"
#include "form_signal_generator_rocm.hpp"
#include "lch_farrow_rocm.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

class DelayedFormSignalGeneratorROCm {
public:
  explicit DelayedFormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~DelayedFormSignalGeneratorROCm() = default;

  DelayedFormSignalGeneratorROCm(const DelayedFormSignalGeneratorROCm&) = delete;
  DelayedFormSignalGeneratorROCm& operator=(const DelayedFormSignalGeneratorROCm&) = delete;

  void SetParams(const FormParams& params);
  void SetDelays(const std::vector<float>& delay_us);
  void LoadMatrix(const std::string& json_path) { lch_farrow_.LoadMatrix(json_path); }

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  const std::vector<float>& GetDelays() const { return lch_farrow_.GetDelays(); }

  /// GPU generate: signal + delay → InputData<void*> (caller must hipFree)
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /// CPU generate → vector[antenna][sample]
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

private:
  drv_gpu_lib::IBackend* backend_;
  FormParams params_;
  FormSignalGeneratorROCm signal_gen_;
  lch_farrow::LchFarrowROCm lch_farrow_;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class DelayedFormSignalGeneratorROCm {
public:
  explicit DelayedFormSignalGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void SetParams(const FormParams&) {}
  void SetDelays(const std::vector<float>&) {}
  drv_gpu_lib::InputData<void*> GenerateInputData() {
    throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("DelayedFormSignalGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
