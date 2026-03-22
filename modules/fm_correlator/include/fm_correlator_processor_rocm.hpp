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
#include "interface/gpu_context.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

namespace drv_gpu_lib {

// ROCm-бэкенд FM-коррелятора: hipFFT + hiprtc-кернелы.
// Владеет GPU-буферами, hipFFT-планами и HIP-потоками.
// Порядок вызовов: SetParams() → PrepareReference() → Process()/RunTestPattern().
class FMCorrelatorProcessorROCm {
public:
  // Создаёт два HIP-потока для параллельной обработки ref и inp.
  // backend — не владеем.
  explicit FMCorrelatorProcessorROCm(IBackend* backend);
  ~FMCorrelatorProcessorROCm();

  FMCorrelatorProcessorROCm(const FMCorrelatorProcessorROCm&) = delete;
  FMCorrelatorProcessorROCm& operator=(const FMCorrelatorProcessorROCm&) = delete;

  // Move semantics
  FMCorrelatorProcessorROCm(FMCorrelatorProcessorROCm&& other) noexcept;
  FMCorrelatorProcessorROCm& operator=(FMCorrelatorProcessorROCm&& other) noexcept;

  // Применить параметры: compile (первый раз), free старые буферы/планы,
  // allocate под новые размеры. Сбрасывает ref_prepared_ — эталон потеряется.
  void SetParams(const FMCorrelatorParams& params);

  // H2D(ref) → apply_cyclic_shifts (GPU) → C2C FFT батча K на stream0.
  // После этого d_ref_complex_ содержит спектры всех K сдвигов.
  void PrepareReference(const std::vector<float>& ref);

  // H2D(inp) на stream1, R2C FFT на stream1; sync; RunCorrelationPipeline().
  FMCorrelatorResult Process(const std::vector<float>& inp);

  // GPU-генерация inp (circshift(ref, s*shift_step)) на stream1, R2C FFT;
  // sync; RunCorrelationPipeline(). Нет H2D для входных данных — только GPU.
  FMCorrelatorResult RunTestPattern(int shift_step);

  // Авто-батчинг: делит total_signals на батчи по BatchManager,
  // для каждого батча переконфигурирует num_signals и вызывает Process().
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  bool IsReferencePrepared() const { return ref_prepared_; }

private:
  void EnsureCompiled();
  void AllocateBuffers();
  void FreeBuffers();
  void CreatePlans();
  void DestroyPlans();
  void ReleaseAll();

  FMCorrelatorResult RunCorrelationPipeline();
  FMCorrelatorResult ProcessFromPtr(const float* data, int num_signals);

  GpuContext ctx_;             ///< Ref03: compilation, disk cache (NOT stream — we use 2 own streams)
  IBackend*  backend_;         ///< Non-owning, for hipMalloc
  FMCorrelatorParams params_;
  bool compiled_ = false;

  // 2-stream pipeline: ref on stream0, inp on stream1 (true GPU concurrency)
  hipStream_t stream0_ = nullptr;
  hipStream_t stream1_ = nullptr;

  // hipFFT plans (bound to streams via hipfftSetStream)
  hipfftHandle plan_ref_  = 0;
  hipfftHandle plan_inp_  = 0;
  hipfftHandle plan_corr_ = 0;
  bool plans_created_ = false;

  // GPU buffers
  float*  d_ref_float_   = nullptr;
  void*   d_ref_complex_ = nullptr;
  float*  d_inp_float_   = nullptr;
  void*   d_inp_fft_     = nullptr;
  void*   d_corr_fft_    = nullptr;
  float*  d_corr_time_   = nullptr;
  float*  d_peaks_       = nullptr;

  bool ref_prepared_ = false;
  bool buffers_allocated_ = false;
  int  current_batch_S_ = 0;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM — Windows stub

#include "fm_correlator_types.hpp"
#include "interface/i_backend.hpp"
#include <stdexcept>
#include <vector>

namespace drv_gpu_lib {

class FMCorrelatorProcessorROCm {
public:
  explicit FMCorrelatorProcessorROCm(IBackend*) {}
  ~FMCorrelatorProcessorROCm() = default;

  void SetParams(const FMCorrelatorParams&) {
    throw std::runtime_error("FMCorrelatorProcessorROCm: ROCm not enabled");
  }
  void PrepareReference(const std::vector<float>&) {
    throw std::runtime_error("FMCorrelatorProcessorROCm: ROCm not enabled");
  }
  FMCorrelatorResult Process(const std::vector<float>&) {
    throw std::runtime_error("FMCorrelatorProcessorROCm: ROCm not enabled");
  }
  FMCorrelatorResult RunTestPattern(int) {
    throw std::runtime_error("FMCorrelatorProcessorROCm: ROCm not enabled");
  }
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>&, int) {
    throw std::runtime_error("FMCorrelatorProcessorROCm: ROCm not enabled");
  }
  bool IsReferencePrepared() const { return false; }
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
