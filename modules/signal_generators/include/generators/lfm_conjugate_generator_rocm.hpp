#pragma once

/**
 * @file lfm_conjugate_generator_rocm.hpp
 * @brief LfmConjugateGeneratorROCm — conjugate LFM reference signal (ROCm/HIP)
 *
 * Formula: s_ref*(t) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
 * where mu = (f_end - f_start) / T, T = num_samples / sample_rate
 *
 * Used by HeterodyneDechirp as reference for dechirp processing:
 *   s_dc = s_rx(t) * s_ref*(t)  →  tone at f_beat = mu*tau
 *
 * Ref03: GpuContext for kernel compilation.
 * Kernel: generate_lfm_conjugate (in lfm_kernels_rocm.hpp)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
#include "interface/i_backend.hpp"
#include "interface/gpu_context.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

class LfmConjugateGeneratorROCm {
public:
  explicit LfmConjugateGeneratorROCm(drv_gpu_lib::IBackend* backend,
                                      const LfmParams& params);
  ~LfmConjugateGeneratorROCm() = default;

  LfmConjugateGeneratorROCm(const LfmConjugateGeneratorROCm&) = delete;
  LfmConjugateGeneratorROCm& operator=(const LfmConjugateGeneratorROCm&) = delete;
  LfmConjugateGeneratorROCm(LfmConjugateGeneratorROCm&&) noexcept = default;
  LfmConjugateGeneratorROCm& operator=(LfmConjugateGeneratorROCm&&) noexcept = default;

  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Generate conjugate LFM on GPU (ROCm)
   * @return HIP device pointer [num_samples × complex<float>]
   *         CALLER OWNS — must hipFree!
   */
  void* GenerateToGpu();

  /**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector<complex<float>>, length = system_.length
   */
  std::vector<std::complex<float>> GenerateToCpu();

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  LfmParams params_;
  SystemSampling system_;
  bool compiled_ = false;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM — Windows stub

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
#include "interface/i_backend.hpp"
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {

class LfmConjugateGeneratorROCm {
public:
  explicit LfmConjugateGeneratorROCm(drv_gpu_lib::IBackend*, const LfmParams&) {}
  void SetParams(const LfmParams&) {}
  void SetSampling(const SystemSampling&) {}
  void* GenerateToGpu() { throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled"); }
  std::vector<std::complex<float>> GenerateToCpu() {
    throw std::runtime_error("LfmConjugateGeneratorROCm: ROCm not enabled");
  }
};

}  // namespace signal_gen

#endif  // ENABLE_ROCM
