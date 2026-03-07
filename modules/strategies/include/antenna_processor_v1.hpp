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
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#endif

#include <memory>
#include <vector>
#include <complex>

// Forward declarations
namespace drv_gpu_lib { class IBackend; class KernelCacheService; }
namespace statistics  { class StatisticsProcessor; }
namespace antenna_fft { class AllMaximaPipelineROCm; }

namespace strategies {

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

protected:
  // Step methods for AntennaProcessorTest to call individually
  void do_debug_point_21(const void* d_S, AntennaResult& result);
  void do_gemm(const void* d_S, const void* d_W);
  void do_debug_point_22(AntennaResult& result);
  void do_window_fft();
  void do_debug_point_23(AntennaResult& result);
  void do_run_post_fft_scenarios(AntennaResult& result);

  // Access to internal buffers (for AntennaProcessorTest)
  void*    get_d_X() const { return d_X_; }
  void*    get_d_spectrum() const { return d_spectrum_; }
  void*    get_d_magnitudes() const { return d_magnitudes_; }
  uint32_t get_nFFT() const { return nFFT_; }

private:
  void allocate_buffers();
  void release_buffers();
  void compile_kernels();
  void create_fft_plan();
  uint32_t compute_nFFT(uint32_t n_samples) const;

  // Backend
  drv_gpu_lib::IBackend* backend_ = nullptr;
  AntennaProcessorConfig cfg_;

#if ENABLE_ROCM
  // HIP streams
  hipStream_t stream_main_   = nullptr;  ///< Stream 2: GEMM -> Window+FFT
  hipStream_t stream_debug1_ = nullptr;  ///< Stream 1: debug 2.1 (stats d_S)
  hipStream_t stream_debug2_ = nullptr;  ///< Stream 3: debug 2.2 (stats d_X)
  hipStream_t stream_debug3_ = nullptr;  ///< Stream 4: debug 2.3 + post-FFT

  // HIP events
  hipEvent_t event_gemm_done_ = nullptr;
  hipEvent_t event_fft_done_  = nullptr;
  hipEvent_t event_c1_done_   = nullptr;
  hipEvent_t event_c2_done_   = nullptr;

  // hipBLAS
  hipblasHandle_t hipblas_handle_ = nullptr;

  // hipFFT
  hipfftHandle fft_plan_ = 0;
  bool fft_plan_created_ = false;

  // GPU buffers
  void* d_X_          = nullptr;  ///< [n_ant x n_samples] GEMM output
  void* d_fft_input_  = nullptr;  ///< [n_ant x nFFT] zero-padded for FFT
  void* d_spectrum_   = nullptr;  ///< [n_ant x nFFT] FFT output (shared by all post-FFT)
  void* d_magnitudes_ = nullptr;  ///< [n_ant x nFFT] float |spectrum|
  void* d_hamming_window_ = nullptr;  ///< [n_samples] precomputed Hamming window (float)

  // Pre-allocated result buffers (avoid Allocate/Free in hot path — P3)
  void* d_one_max_results_ = nullptr;  ///< [n_ant] OneMaxParabolaLite
  void* d_minmax_results_  = nullptr;  ///< [n_ant] MinMaxResult

  // hiprtc kernels
  hipModule_t   kernel_module_ = nullptr;
  hipFunction_t hamming_pad_kernel_  = nullptr;  ///< fused hamming+pad
  hipFunction_t magnitudes_kernel_   = nullptr;
  hipFunction_t minmax_kernel_       = nullptr;
  hipFunction_t one_max_kernel_      = nullptr;  ///< moved from hot path (P12)
  bool kernels_compiled_ = false;
#endif

  // Sizes
  uint32_t nFFT_ = 0;

  // Components
  std::unique_ptr<statistics::StatisticsProcessor> stats_processor_;
  std::unique_ptr<antenna_fft::AllMaximaPipelineROCm> all_maxima_pipeline_;
  std::unique_ptr<ICheckpointSave> checkpoint_;
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  static constexpr uint32_t kBlockSize = 256;
};

}  // namespace strategies
