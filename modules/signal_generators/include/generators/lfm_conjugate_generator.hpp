#pragma once

/**
 * @file lfm_conjugate_generator.hpp
 * @brief Conjugate LFM generator: conj(s_tx) at tau=0
 *
 * Formula: s_ref*(t) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
 * where mu = (f_end - f_start) / T = B/T
 *
 * Used as reference signal for dechirp processing:
 *   s_dc = s_rx(t) * s_ref*(t)   // result = tone at f_beat = mu*tau
 *
 * Analog of LfmGeneratorAnalyticalDelay but (1) always delay=0, (2) conjugate
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "../params/signal_request.hpp"
#include "../params/system_sampling.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cstdint>
#include <utility>

namespace signal_gen {

/**
 * @class LfmConjugateGenerator
 * @brief GPU/CPU conjugate LFM generator for dechirp reference
 *
 * @code
 * LfmConjugateGenerator gen(backend, lfm_params);
 * gen.SetSampling(system);
 *
 * // GPU
 * cl_mem ref = gen.GenerateToGpu();
 * // ... use in dechirp pipeline ...
 * clReleaseMemObject(ref);
 *
 * // CPU
 * auto ref_cpu = gen.GenerateToCpu();
 * @endcode
 */
class LfmConjugateGenerator {
public:
  /// Тип для сбора OpenCL событий профилирования (имя → cl_event)
  using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

  LfmConjugateGenerator(drv_gpu_lib::IBackend* backend,
                         const LfmParams& params);
  ~LfmConjugateGenerator();

  // No copy
  LfmConjugateGenerator(const LfmConjugateGenerator&) = delete;
  LfmConjugateGenerator& operator=(const LfmConjugateGenerator&) = delete;

  // Move
  LfmConjugateGenerator(LfmConjugateGenerator&& other) noexcept;
  LfmConjugateGenerator& operator=(LfmConjugateGenerator&& other) noexcept;

  /// Set LFM parameters
  void SetParams(const LfmParams& params) { params_ = params; }
  const LfmParams& GetParams() const { return params_; }

  /// Set sampling parameters (fs, length)
  void SetSampling(const SystemSampling& system) { system_ = system; }
  const SystemSampling& GetSampling() const { return system_; }

  /**
   * @brief Generate conjugate LFM on GPU
   * @return cl_mem with [num_samples] complex signal (conj LFM)
   * @note Caller must release via clReleaseMemObject()
   */
  cl_mem GenerateToGpu();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *
   * Собирает события: "Kernel" (lfm_conjugate.cl)
   */
  cl_mem GenerateToGpu(ProfEvents* prof_events);

  /**
   * @brief Generate conjugate LFM on CPU (reference)
   * @return vector of complex<float>, length = system_.length
   */
  std::vector<std::complex<float>> GenerateToCpu();

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  LfmParams params_;
  SystemSampling system_;

  // OpenCL
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
};

} // namespace signal_gen