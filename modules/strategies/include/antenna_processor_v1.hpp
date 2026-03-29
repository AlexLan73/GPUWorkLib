#pragma once

/**
 * @file antenna_processor_v1.hpp
 * @brief AntennaProcessor_v1 - concrete ROCm implementation of antenna array pipeline
 *
 * Pipeline:
 *   1. d_S already on GPU
 *   2. Debug 2.1: stats on d_S (Stream 1, parallel)
 *   3. GEMM: X = W * S via hipBLAS (Stream 2)
 *   4. Debug 2.2: stats on d_X (Stream 3, parallel with Window+FFT)
 *   5. Window (Hamming) + FFT (hipFFT batch) -> d_spectrum (Stream 2)
 *   6. Debug 2.3: stats on |spectrum| (Stream 4)
 *   7. Post-FFT scenarios on shared d_spectrum (Stream 4):
 *      - Step2.1: OneMax + Parabola (no phase)
 *      - Step2.2: AllMaxima (limit=1000)
 *      - Step2.3: GlobalMinMax
 *
 * NOT final: AntennaProcessorTest inherits from this.
 *
 * @date 2026-03-07
 */

#include "antenna_processor.hpp"
#include "interfaces/i_checkpoint_save.hpp"
#include "interfaces/i_post_fft_scenario.hpp"
#include "checkpoint/null_checkpoint_save.hpp"

#if ENABLE_ROCM
#include "interface/gpu_context.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#endif

#include <memory>
#include <vector>
#include <complex>

// Forward declarations
namespace drv_gpu_lib   { class IBackend; }
namespace statistics    { class StatisticsProcessor; }
namespace antenna_fft   { class AllMaximaPipelineROCm; }
namespace fft_processor { class ComplexToMagPhaseROCm; }

namespace strategies {

/// @ingroup grp_strategies
class AntennaProcessor_v1 : public AntennaProcessor {
public:
  explicit AntennaProcessor_v1(
      drv_gpu_lib::IBackend* backend,
      const AntennaProcessorConfig& cfg);

  ~AntennaProcessor_v1() override;

  // No copy
  AntennaProcessor_v1(const AntennaProcessor_v1&) = delete;
  AntennaProcessor_v1& operator=(const AntennaProcessor_v1&) = delete;

  // AntennaProcessor interface
  AntennaResult process(const void* d_S, const void* d_W) override;

  void set_scenario_mode(PostFftScenarioMode mode) override { cfg_.scenario_mode = mode; }
  void set_pre_input_stats(StatisticsSet stats) override { cfg_.pre_input_stats = stats; }
  void set_post_gemm_stats(StatisticsSet stats) override { cfg_.post_gemm_stats = stats; }
  void set_post_fft_stats(StatisticsSet stats) override  { cfg_.post_fft_stats  = stats; }
  void set_debug_mode(bool enabled) override { cfg_.debug_mode = enabled; }

  const AntennaProcessorConfig& config() const override { return cfg_; }
  int gpu_id() const override;

  // Checkpoint setter
  void set_checkpoint_save(std::unique_ptr<ICheckpointSave> save);

  /**
   * @brief Upload external weight matrix to GPU (managed by this class)
   * @param W Flat row-major [n_ant x n_ant] complex<float> matrix
   *
   * After this call, use get_managed_weights_ptr() to obtain the GPU pointer.
   * The buffer is freed in the destructor (not by the caller).
   */
  void set_external_weights(const std::vector<std::complex<float>>& W);

  /// GPU pointer to the last uploaded external weights (nullptr if not set)
  void* get_managed_weights_ptr() const { return d_W_managed_; }

protected:
  // Step methods for AntennaProcessorTest to call individually
  void do_debug_point_21(const void* d_S, AntennaResult& result);
  void do_gemm(const void* d_S, const void* d_W);
  void do_debug_point_22(AntennaResult& result);
  void do_window_fft();
  void do_debug_point_23(AntennaResult& result);
  void do_run_post_fft_scenarios(AntennaResult& result);
  /// 3-stream parallel variant of do_run_post_fft_scenarios (for benchmark 3.6)
  void do_run_post_fft_parallel(AntennaResult& result);

  // Access to internal buffers (for AntennaProcessorTest)
  void*    get_d_X() const { return d_X_; }
  void*    get_d_spectrum() const { return d_spectrum_; }
  void*    get_d_magnitudes() const { return d_magnitudes_; }
  uint32_t get_nFFT() const { return nFFT_; }

private:
  void allocate_buffers();
  void release_buffers();
  void ensure_compiled();
  void create_fft_plan();
  uint32_t compute_nFFT(uint32_t n_samples) const;

  // Backend
  drv_gpu_lib::IBackend* backend_ = nullptr;
  AntennaProcessorConfig cfg_;

#if ENABLE_ROCM
  // Ref03: GpuContext for kernel compilation (replaces manual hiprtc + KernelCacheService)
  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;

  // HIP streams (7 for multi-stream pipeline)
  hipStream_t stream_main_    = nullptr;
  hipStream_t stream_debug1_  = nullptr;
  hipStream_t stream_debug2_  = nullptr;
  hipStream_t stream_debug3_  = nullptr;
  hipStream_t stream_bench3a_ = nullptr;
  hipStream_t stream_bench3b_ = nullptr;
  hipStream_t stream_bench3c_ = nullptr;

  // HIP events (inter-stream sync)
  hipEvent_t event_gemm_done_ = nullptr;
  hipEvent_t event_fft_done_  = nullptr;
  hipEvent_t event_c1_done_   = nullptr;
  hipEvent_t event_c2_done_   = nullptr;

  // hipBLAS
  hipblasHandle_t hipblas_handle_ = nullptr;

  // hipFFT
  hipfftHandle fft_plan_ = 0;
  bool fft_plan_created_ = false;

  // GPU buffers (raw — migrated to BufferSet later)
  void* d_X_          = nullptr;
  void* d_fft_input_  = nullptr;
  void* d_spectrum_   = nullptr;
  void* d_magnitudes_ = nullptr;
  void* d_hamming_window_ = nullptr;
  void* d_one_max_results_ = nullptr;
  void* d_minmax_results_  = nullptr;
#endif

  // Externally supplied weight matrix (managed GPU buffer — freed by release_buffers)
  void* d_W_managed_ = nullptr;

  // Sizes
  uint32_t nFFT_ = 0;

  // Components
  std::unique_ptr<statistics::StatisticsProcessor> stats_processor_;
  std::unique_ptr<antenna_fft::AllMaximaPipelineROCm> all_maxima_pipeline_;
  std::unique_ptr<fft_processor::ComplexToMagPhaseROCm> complex_to_mag_;
  std::unique_ptr<ICheckpointSave> checkpoint_;

  static constexpr uint32_t kBlockSize = 256;
};

}  // namespace strategies
