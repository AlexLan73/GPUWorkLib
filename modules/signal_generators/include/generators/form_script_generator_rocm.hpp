#pragma once

/**
 * @file form_script_generator_rocm.hpp
 * @brief ROCm: FormParams → DSL script → ScriptGeneratorROCm → GPU signal
 *
 * Simplified ROCm port of FormScriptGenerator.
 * Converts FormParams to DSL text, then uses ScriptGeneratorROCm to compile and execute.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "../params/form_params.hpp"
#include "script_generator_rocm.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <string>

namespace signal_gen {

class FormScriptGeneratorROCm {
public:
  explicit FormScriptGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~FormScriptGeneratorROCm() = default;

  void SetParams(const FormParams& params);
  void SetParamsFromString(const std::string& params_str);

  /// Generate on GPU → InputData<void*> (caller must hipFree)
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /// Generate → read back to CPU
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }

  /// Get generated HIP kernel source (for debugging)
  const std::string& GetKernelSource() const { return script_gen_.GetKernelSource(); }

private:
  /// Build DSL script text from FormParams
  std::string BuildScript() const;

  drv_gpu_lib::IBackend* backend_;
  FormParams params_;
  ScriptGeneratorROCm script_gen_;
  bool compiled_ = false;
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
class FormScriptGeneratorROCm {
public:
  explicit FormScriptGeneratorROCm(drv_gpu_lib::IBackend*) {}
  void SetParams(const FormParams&) {}
  drv_gpu_lib::InputData<void*> GenerateInputData() {
    throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("FormScriptGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
