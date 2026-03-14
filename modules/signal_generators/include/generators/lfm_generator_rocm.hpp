#pragma once

/**
 * @file lfm_generator_rocm.hpp
 * @brief LfmGeneratorROCm — LFM (chirp) signal generator (ROCm/HIP)
 *
 * ROCm port of LfmGenerator (OpenCL). Same algorithm, HIP runtime.
 * Uses Ref03 GpuContext for kernel compilation.
 *
 * s(t) = amplitude * exp(j * (pi * chirp_rate * t^2 + 2*pi*f_start*t))
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "../params/signal_request.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "interface/gpu_context.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

class LfmGeneratorROCm {
public:
  explicit LfmGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~LfmGeneratorROCm() = default;

  LfmGeneratorROCm(const LfmGeneratorROCm&) = delete;
  LfmGeneratorROCm& operator=(const LfmGeneratorROCm&) = delete;
  LfmGeneratorROCm(LfmGeneratorROCm&&) noexcept = default;
  LfmGeneratorROCm& operator=(LfmGeneratorROCm&&) noexcept = default;

  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const LfmParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const LfmParams& params,
      uint32_t beam_count);

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include "../params/signal_request.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class LfmGeneratorROCm {
public:
  explicit LfmGeneratorROCm(drv_gpu_lib::IBackend*) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const LfmParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const LfmParams&, uint32_t) {
    throw std::runtime_error("LfmGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
