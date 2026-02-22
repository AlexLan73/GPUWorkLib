#pragma once

/**
 * @file heterodyne_dechirp.hpp
 * @brief Heterodyne dechirp LFM - public facade API
 *
 * USAGE (normal mode):
 *   auto& backend = drv.GetBackend();
 *   HeterodyneDechirp het(backend);
 *   het.SetParams(params);
 *   auto result = het.Process(rx_matrix);  // num_antennas * num_samples
 *
 * USAGE (external OpenCL context):
 *   het.SetParams(params);
 *   auto result = het.ProcessExternal(cl_mem_ptr, params);
 *
 * PIPELINE:
 *   1. LfmConjugateGenerator -> s_ref* (GPU)
 *   2. dechirp_multiply.cl   -> s_dc = s_rx * s_ref*  (GPU)
 *   3. FFTProcessor           -> FFT spectrum           (GPU)
 *   4. SpectrumMaximaFinder  -> f_beat (peak)           (GPU)
 *   5. dechirp_correct.cl    -> frequency compensation  (GPU)
 *   6. Verification: spectrum -> DC                      (GPU)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "i_heterodyne_processor.hpp"
#include "heterodyne_params.hpp"
#include "common/backend_type.hpp"
#include "generators/lfm_conjugate_generator.hpp"
#include "params/signal_request.hpp"
#include "params/system_sampling.hpp"
#include <memory>

namespace drv_gpu_lib {

class IBackend;

class HeterodyneDechirp {
public:
  /**
   * @param backend         Pointer to DrvGPU backend (does not own)
   * @param compute_backend OpenCL (default) or ROCm (stub)
   */
  explicit HeterodyneDechirp(
      IBackend* backend,
      BackendType compute_backend = BackendType::OPENCL);

  HeterodyneDechirp(const HeterodyneDechirp&) = delete;
  HeterodyneDechirp& operator=(const HeterodyneDechirp&) = delete;

  /** Set LFM parameters (must be called before Process) */
  void SetParams(const HeterodyneParams& params);

  /**
   * Full pipeline from CPU data.
   * Input: s_rx flat matrix [num_antennas * num_samples], complex float
   *
   * Pipeline:
   *   1. Generate conj(LFM) reference (OPT-4: cached)
   *   2. Dechirp multiply: s_dc = s_rx * conj(s_tx) (OPT-3: GPU ref)
   *   3. FFT -> find peak -> f_beat
   *   4. Calculate range + SNR
   */
  HeterodyneResult Process(
      const std::vector<std::complex<float>>& rx_data);

  /**
   * External OpenCL context variant.
   * rx_cl_mem is a pointer to cl_mem (external program owns buffer).
   * params are passed from CPU (metadata: fs, B, N, antennas).
   */
  HeterodyneResult ProcessExternal(
      void* rx_cl_mem,
      const HeterodyneParams& params);

  /** Last result (cached) */
  const HeterodyneResult& GetLastResult() const { return last_result_; }

  /** Get current params */
  const HeterodyneParams& GetParams() const { return params_; }

private:
  HeterodyneResult BuildResult(
      const std::vector<std::complex<float>>& dc_data,
      const HeterodyneParams& params);

  /** OPT-4: Lazy-init and cache the conjugate LFM generator */
  void EnsureConjugateGenerator();

  std::unique_ptr<IHeterodyneProcessor> processor_;
  IBackend*                             backend_ = nullptr;
  HeterodyneParams                      params_;
  HeterodyneResult                      last_result_;

  // OPT-4: Cached conjugate LFM generator (rebuilt only on SetParams)
  std::unique_ptr<signal_gen::LfmConjugateGenerator> conj_gen_;
  bool params_dirty_ = true;  // true = need to rebuild conj_gen_
};

}  // namespace drv_gpu_lib