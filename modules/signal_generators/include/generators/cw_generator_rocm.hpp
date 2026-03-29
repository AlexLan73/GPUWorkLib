#pragma once

/**
 * @file cw_generator_rocm.hpp
 * @brief CwGeneratorROCm — CW (sinusoid) signal generator (ROCm/HIP)
 *
 * ROCm port of CwGenerator (OpenCL). Same algorithm, HIP runtime.
 * Uses Ref03 GpuContext for kernel compilation.
 *
 * s(t) = amplitude * exp(j * (2*pi*freq*t + initial_phase))
 * For multi-beam: freq_i = base_freq + beam_id * freq_step
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

/// @ingroup grp_signal_generators
class CwGeneratorROCm {
public:
  explicit CwGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~CwGeneratorROCm() = default;

  CwGeneratorROCm(const CwGeneratorROCm&) = delete;
  CwGeneratorROCm& operator=(const CwGeneratorROCm&) = delete;
  CwGeneratorROCm(CwGeneratorROCm&&) noexcept = default;
  CwGeneratorROCm& operator=(CwGeneratorROCm&&) noexcept = default;

  /**
   * @brief Generate CW signal on GPU
   * @param system Sampling parameters (fs, length)
   * @param params CW parameters (f0, phase, amplitude, freq_step)
   * @param beam_count Number of beams
   * @param prof_events Optional profiling (nullptr = no overhead)
   * @return InputData<void*> with GPU signal (caller must hipFree result.data)
   */
  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const CwParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  /// CPU reference implementation
  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const CwParams& params,
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
class CwGeneratorROCm {
public:
  explicit CwGeneratorROCm(drv_gpu_lib::IBackend*) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const CwParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const CwParams&, uint32_t) {
    throw std::runtime_error("CwGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
