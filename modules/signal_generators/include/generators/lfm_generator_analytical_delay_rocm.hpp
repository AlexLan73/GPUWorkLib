#pragma once

/**
 * @file lfm_generator_analytical_delay_rocm.hpp
 * @brief ROCm LFM generator with analytical per-antenna delay
 *
 * s(t) = A * exp(j * [pi*mu*(t-tau)^2 + 2*pi*f_start*(t-tau)])
 * where tau = delay_us[antenna] * 1e-6 (microseconds → seconds)
 * output = 0 when t < tau (signal hasn't arrived yet)
 *
 * Ref03: GpuContext for compilation. Kernel: generate_lfm_analytical_delay.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
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

class LfmGeneratorAnalyticalDelayROCm {
public:
  LfmGeneratorAnalyticalDelayROCm(drv_gpu_lib::IBackend* backend,
                                    const LfmParams& params);
  ~LfmGeneratorAnalyticalDelayROCm() = default;

  LfmGeneratorAnalyticalDelayROCm(const LfmGeneratorAnalyticalDelayROCm&) = delete;
  LfmGeneratorAnalyticalDelayROCm& operator=(const LfmGeneratorAnalyticalDelayROCm&) = delete;
  LfmGeneratorAnalyticalDelayROCm(LfmGeneratorAnalyticalDelayROCm&&) noexcept = default;
  LfmGeneratorAnalyticalDelayROCm& operator=(LfmGeneratorAnalyticalDelayROCm&&) noexcept = default;

  void SetParams(const LfmParams& params) { params_ = params; }
  void SetSampling(const SystemSampling& system) { system_ = system; }
  void SetDelays(const std::vector<float>& delay_us) { delay_us_ = delay_us; }

  const LfmParams& GetParams() const { return params_; }
  const SystemSampling& GetSampling() const { return system_; }
  const std::vector<float>& GetDelays() const { return delay_us_; }
  uint32_t GetAntennas() const { return static_cast<uint32_t>(delay_us_.size()); }

  /// GPU generate → InputData<void*> (caller must hipFree result.data)
  drv_gpu_lib::InputData<void*> GenerateToGpu(ROCmProfEvents* prof_events = nullptr);

  /// CPU reference
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  LfmParams params_;
  SystemSampling system_;
  std::vector<float> delay_us_;
  bool compiled_ = false;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include <stdexcept>
#include <vector>
#include <complex>

namespace signal_gen {
class LfmGeneratorAnalyticalDelayROCm {
public:
  LfmGeneratorAnalyticalDelayROCm(drv_gpu_lib::IBackend*, const LfmParams&) {}
  void SetParams(const LfmParams&) {}
  void SetSampling(const SystemSampling&) {}
  void SetDelays(const std::vector<float>&) {}
  drv_gpu_lib::InputData<void*> GenerateToGpu(void* = nullptr) {
    throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
  }
  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("LfmGeneratorAnalyticalDelayROCm: ROCm not enabled");
  }
};
}  // namespace signal_gen

#endif  // ENABLE_ROCM
