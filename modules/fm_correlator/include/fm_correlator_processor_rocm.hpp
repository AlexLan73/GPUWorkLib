#pragma once

/**
 * @file fm_correlator_processor_rocm.hpp
 * @brief ROCm backend for FM Correlator (hipFFT + HIP kernels)
 *
 * GPU pipeline:
 *   Stream 0: H2D(ref) -> apply_cyclic_shifts -> C2C FFT(ref)
 *   Stream 1: H2D(inp) -> R2C FFT(inp)
 *   Sync both streams
 *   Stream 0: multiply_conj_fused -> C2R IFFT -> extract_magnitudes -> D2H
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator_types.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hipfft/hipfft.h>

namespace drv_gpu_lib {

class FMCorrelatorProcessorROCm {
public:
  explicit FMCorrelatorProcessorROCm(IBackend* backend);
  ~FMCorrelatorProcessorROCm();

  FMCorrelatorProcessorROCm(const FMCorrelatorProcessorROCm&) = delete;
  FMCorrelatorProcessorROCm& operator=(const FMCorrelatorProcessorROCm&) = delete;

  void SetParams(const FMCorrelatorParams& params);

  void PrepareReference(const std::vector<float>& ref);

  FMCorrelatorResult Process(const std::vector<float>& inp);

  FMCorrelatorResult RunTestPattern(int shift_step);

  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  bool IsReferencePrepared() const { return ref_prepared_; }

private:
  void CompileKernels();
  void AllocateBuffers();
  void FreeBuffers();
  void CreatePlans();
  void DestroyPlans();
  void ReleaseAll();

  FMCorrelatorResult RunCorrelationPipeline();

  IBackend* backend_;
  FMCorrelatorParams params_;

  hipStream_t stream0_ = nullptr;
  hipStream_t stream1_ = nullptr;

  // hipFFT plans (persistent — created in SetParams, not in Process)
  hipfftHandle plan_ref_  = 0;
  hipfftHandle plan_inp_  = 0;
  hipfftHandle plan_corr_ = 0;
  bool plans_created_ = false;

  // GPU buffers
  float*  d_ref_float_   = nullptr;  // [N]
  void*   d_ref_complex_ = nullptr;  // [K * N] float2, in-place with d_ref_fft
  void*   d_inp_fft_     = nullptr;  // [S * (N/2+1)] float2
  float*  d_inp_float_   = nullptr;  // [S * N]
  void*   d_corr_fft_    = nullptr;  // [S * K * (N/2+1)] float2
  float*  d_corr_time_   = nullptr;  // [S * K * N]
  float*  d_peaks_       = nullptr;  // [S * K * n_kg]

  // hiprtc compiled kernels (single module for all 4 kernels)
  hipModule_t   module_ = nullptr;
  hipFunction_t fn_apply_shifts_    = nullptr;
  hipFunction_t fn_multiply_conj_   = nullptr;
  hipFunction_t fn_extract_mag_     = nullptr;
  hipFunction_t fn_gen_test_inputs_ = nullptr;
  bool kernels_compiled_ = false;

  bool ref_prepared_ = false;
  bool buffers_allocated_ = false;
  int  current_batch_S_ = 0;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
