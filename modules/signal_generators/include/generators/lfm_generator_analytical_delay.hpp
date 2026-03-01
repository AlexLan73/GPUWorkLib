#pragma once

/**
 * @file lfm_generator_analytical_delay.hpp
 * @brief LfmGeneratorAnalyticalDelay - LFM signal with analytical fractional delay
 *
 * Generates LFM (chirp) signal with per-antenna delay by time substitution:
 *   S_delayed(t) = A * exp(j * (pi * k * t_local^2 + 2*pi * f_start * t_local))
 *   where t_local = t - tau, and output = 0 when t < tau.
 *
 * This is the "ideal" delay — no interpolation artifacts.
 * Used as reference for testing LchFarrow and beamforming applications.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <utility>

namespace signal_gen {

/**
 * @class LfmGeneratorAnalyticalDelay
 * @brief GPU/CPU LFM generator with analytical per-antenna delay
 *
 * @code
 * LfmGeneratorAnalyticalDelay gen(backend);
 * gen.SetParams(lfm_params);
 * gen.SetSampling(system);
 * gen.SetDelays({0.0f, 0.27f, 0.54f});  // microseconds
 *
 * // GPU
 * auto result = gen.GenerateToGpu();
 * clReleaseMemObject(result.data);
 *
 * // CPU
 * auto cpu = gen.GenerateToCpu();
 * @endcode
 */
class LfmGeneratorAnalyticalDelay {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  LfmGeneratorAnalyticalDelay(drv_gpu_lib::IBackend* backend,
                               const LfmParams& params);
  ~LfmGeneratorAnalyticalDelay();

  // No copy
  LfmGeneratorAnalyticalDelay(const LfmGeneratorAnalyticalDelay&) = delete;
  LfmGeneratorAnalyticalDelay& operator=(const LfmGeneratorAnalyticalDelay&) = delete;

  // Move
  LfmGeneratorAnalyticalDelay(LfmGeneratorAnalyticalDelay&& other) noexcept;
  LfmGeneratorAnalyticalDelay& operator=(LfmGeneratorAnalyticalDelay&& other) noexcept;

  /// Set LFM parameters
  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  /// Set sampling parameters (fs, length)
  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Set per-antenna delays in microseconds
   * @param delay_us Delays (float), one per antenna
   */
  void SetDelays(const std::vector<float>& delay_us);

  /**
   * @brief Generate on GPU
   * @return InputData<cl_mem> with [antennas * points] complex signal
   * @note Caller must release result.data via clReleaseMemObject()
   */
  drv_gpu_lib::InputData<cl_mem> GenerateToGpu();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *
   * Собирает события: "Kernel" (lfm_analytical_delay.cl)
   */
  drv_gpu_lib::InputData<cl_mem> GenerateToGpu(ProfEvents* prof_events);

  /**
   * @brief Generate on CPU (reference)
   * @return [antenna][sample] complex<float>
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  uint32_t GetAntennas() const {
    return static_cast<uint32_t>(delay_us_.size());
  }
  const std::vector<float>& GetDelays() const { return delay_us_; }

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  LfmParams params_;
  SystemSampling system_;
  std::vector<float> delay_us_;

  // OpenCL
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen
