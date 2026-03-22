/**
 * @file lfm_conjugate_generator_rocm.cpp
 * @brief LfmConjugateGeneratorROCm — Ref03 implementation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "generators/lfm_conjugate_generator_rocm.hpp"
#include "kernels/lfm_kernels_rocm.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

static const std::vector<std::string> kConjKernelNames = {
  "generate_lfm_conjugate"
};

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

LfmConjugateGeneratorROCm::LfmConjugateGeneratorROCm(
    drv_gpu_lib::IBackend* backend,
    const LfmParams& params)
    : ctx_(backend, "LfmConj", "modules/signal_generators/kernels")
    , params_(params) {
}

// ════════════════════════════════════════════════════════════════════════════
// Lazy compilation
// ════════════════════════════════════════════════════════════════════════════

void LfmConjugateGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetLfmSource_rocm(), kConjKernelNames);
  compiled_ = true;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Generation
// ════════════════════════════════════════════════════════════════════════════

void* LfmConjugateGeneratorROCm::GenerateToGpu() {
  EnsureCompiled();

  uint32_t n_point = static_cast<uint32_t>(system_.length);
  float sample_rate = static_cast<float>(system_.fs);
  float f_start = static_cast<float>(params_.f_start);
  float f_end = static_cast<float>(params_.f_end);
  float duration = static_cast<float>(n_point) / sample_rate;
  float chirp_rate = (f_end - f_start) / duration;

  size_t buffer_size = n_point * sizeof(std::complex<float>);
  void* output = nullptr;
  hipError_t err = hipMalloc(&output, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error("LfmConjugateGeneratorROCm: hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  unsigned int grid = (n_point + kBlockSize - 1) / kBlockSize;
  void* args[] = { &output, &n_point, &sample_rate, &f_start, &chirp_rate };

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("generate_lfm_conjugate"),
      grid, 1, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    (void)hipFree(output);
    throw std::runtime_error("LfmConjugateGeneratorROCm: kernel launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(ctx_.stream());
  return output;  // CALLER OWNS — must hipFree
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Generation (reference)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> LfmConjugateGeneratorROCm::GenerateToCpu() {
  size_t n = system_.length;
  if (n == 0) return {};

  double fs = system_.fs;
  double duration = static_cast<double>(n) / fs;
  double mu = (params_.f_end - params_.f_start) / duration;

  std::vector<std::complex<float>> result(n);
  for (size_t i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / fs;
    double phase = -(M_PI * mu * t * t + 2.0 * M_PI * params_.f_start * t);
    result[i] = std::complex<float>(
        static_cast<float>(std::cos(phase)),
        static_cast<float>(std::sin(phase)));
  }
  return result;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
