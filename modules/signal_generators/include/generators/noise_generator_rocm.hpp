#pragma once

/**
 * @file noise_generator_rocm.hpp
 * @brief NoiseGeneratorROCm — Noise generator (ROCm/HIP)
 *
 * ROCm port of NoiseGenerator (OpenCL). Same algorithm, HIP runtime.
 * Uses Ref03 GpuContext for kernel compilation.
 *
 * Philox-2x32-10 PRNG + Box-Muller for Gaussian noise.
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
#include <random>

namespace signal_gen {

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

class NoiseGeneratorROCm {
public:
  explicit NoiseGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~NoiseGeneratorROCm() = default;

  NoiseGeneratorROCm(const NoiseGeneratorROCm&) = delete;
  NoiseGeneratorROCm& operator=(const NoiseGeneratorROCm&) = delete;
  NoiseGeneratorROCm(NoiseGeneratorROCm&&) noexcept = default;
  NoiseGeneratorROCm& operator=(NoiseGeneratorROCm&&) noexcept = default;

  drv_gpu_lib::InputData<void*> GenerateToGpu(
      const SystemSampling& system,
      const NoiseParams& params,
      uint32_t beam_count,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<std::complex<float>> GenerateToCpu(
      const SystemSampling& system,
      const NoiseParams& params,
      uint32_t beam_count);

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;
  std::mt19937 rng_{std::random_device{}()};
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
class NoiseGeneratorROCm {
public:
  explicit NoiseGeneratorROCm(drv_gpu_lib::IBackend*) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(const SystemSampling&, const NoiseParams&, uint32_t, void* = nullptr) {
    throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> GenerateToCpu(const SystemSampling&, const NoiseParams&, uint32_t) {
    throw std::runtime_error("NoiseGeneratorROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
