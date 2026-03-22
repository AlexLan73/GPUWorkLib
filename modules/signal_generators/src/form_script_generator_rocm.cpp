/**
 * @file form_script_generator_rocm.cpp
 * @brief ROCm: FormParams → DSL → ScriptGeneratorROCm
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "generators/form_script_generator_rocm.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

FormScriptGeneratorROCm::FormScriptGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend), script_gen_(backend) {
}

void FormScriptGeneratorROCm::SetParams(const FormParams& params) {
  params_ = params;
  compiled_ = false;
}

void FormScriptGeneratorROCm::SetParamsFromString(const std::string& params_str) {
  params_ = FormParams::ParseFromString(params_str);
  compiled_ = false;
}

std::string FormScriptGeneratorROCm::BuildScript() const {
  std::ostringstream ss;
  ss << "[Params]\n";
  ss << "ANTENNAS = " << params_.antennas << "\n";
  ss << "POINTS = " << params_.points << "\n\n";

  ss << "[Defs]\n";
  ss << std::setprecision(12);
  ss << "float dt = " << (1.0 / params_.fs) << ";\n";
  ss << "float f0 = " << params_.f0 << ";\n";
  ss << "float amp = " << params_.amplitude << ";\n";
  ss << "float phi = " << params_.phase << ";\n";
  ss << "float fdev = " << params_.fdev << ";\n";
  ss << "float norm_val = " << params_.norm << ";\n";
  ss << "float t = (float)T * dt;\n";
  ss << "float freq = f0 + (float)ID * fdev;\n";
  ss << "float phase = 2.0f * M_PI_F * freq * t + phi;\n\n";

  ss << "[Signal]\n";
  ss << "float cos_val = __cosf(phase);\n";
  ss << "float sin_val = __sinf(phase);\n";
  ss << "res_re = amp * cos_val * norm_val;\n";
  ss << "res_im = amp * sin_val * norm_val;\n";

  return ss.str();
}

drv_gpu_lib::InputData<void*> FormScriptGeneratorROCm::GenerateInputData() {
  if (!compiled_) {
    std::string script = BuildScript();
    script_gen_.LoadScript(script);
    compiled_ = true;
  }

  void* gpu_ptr = script_gen_.Generate();

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = params_.antennas;
  result.n_point = params_.points;
  result.data = gpu_ptr;
  result.gpu_memory_bytes = static_cast<size_t>(params_.antennas) * params_.points
                            * sizeof(std::complex<float>);
  result.sample_rate = static_cast<float>(params_.fs);
  return result;
}

std::vector<std::vector<std::complex<float>>>
FormScriptGeneratorROCm::GenerateToCpu() {
  auto input = GenerateInputData();
  void* gpu_buf = input.data;
  size_t total = static_cast<size_t>(params_.antennas) * params_.points;
  std::vector<std::complex<float>> flat(total);
  hipMemcpyDtoH(flat.data(), gpu_buf, total * sizeof(std::complex<float>));
  hipFree(gpu_buf);

  std::vector<std::vector<std::complex<float>>> out(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t off = static_cast<size_t>(a) * params_.points;
    out[a].assign(flat.begin() + off, flat.begin() + off + params_.points);
  }
  return out;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
