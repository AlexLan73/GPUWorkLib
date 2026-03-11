#pragma once

/**
 * @file statistics_processor.hpp
 * @brief StatisticsProcessor -- statistical computations on complex signal data (ROCm/HIP)
 *
 * ROCm-only module. Computes per-beam statistics on complex float GPU data:
 * - Mean (complex) -- hierarchical reduction kernel
 * - Median (magnitude) -- rocPRIM radix sort + middle element
 * - Variance / STD (magnitude) -- single-pass Welford kernel
 * - ComputeStatistics -- one-pass mean + variance + std (Welford)
 *
 * Input: beam_count * n_point complex<float> (all antennas at once).
 *
 * IMPORTANT: This file compiles ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "statistics_types.hpp"
#include "statistics_sort_gpu.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <complex>
#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

// Forward declaration to avoid header pollution
namespace drv_gpu_lib { class KernelCacheService; }

namespace statistics {

class StatisticsProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Constructor
   * @param backend Pointer to IBackend (non-owning, must be ROCm backend)
   */
  explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);

  ~StatisticsProcessor();

  // No copying
  StatisticsProcessor(const StatisticsProcessor&) = delete;
  StatisticsProcessor& operator=(const StatisticsProcessor&) = delete;

  // Move semantics
  StatisticsProcessor(StatisticsProcessor&& other) noexcept;
  StatisticsProcessor& operator=(StatisticsProcessor&& other) noexcept;

  // =========================================================================
  // Public API -- CPU data (upload -> compute -> download)
  // =========================================================================

  /**
   * @brief Compute complex mean per beam (CPU data)
   * @param data beam_count * n_point complex<float>
   * @param params Statistics parameters
   * @return Vector of MeanResult (one per beam)
   */
  std::vector<MeanResult> ComputeMean(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  /**
   * @brief Compute median of magnitudes per beam (CPU data)
   * Uses rocPRIM radix sort + middle element.
   * @param data beam_count * n_point complex<float>
   * @param params Statistics parameters
   * @return Vector of MedianResult (one per beam)
   */
  std::vector<MedianResult> ComputeMedian(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  /**
   * @brief Compute full statistics per beam (CPU data)
   * Single-pass Welford: mean(complex) + variance(|z|) + std(|z|) + mean(|z|)
   * @param data beam_count * n_point complex<float>
   * @param params Statistics parameters
   * @return Vector of StatisticsResult (one per beam)
   */
  std::vector<StatisticsResult> ComputeStatistics(
      const std::vector<std::complex<float>>& data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU data (already on device)
  // =========================================================================

  /**
   * @brief Compute complex mean per beam (GPU data)
   * @param gpu_data Device pointer to complex<float> data
   * @param params Statistics parameters
   */
  std::vector<MeanResult> ComputeMean(
      void* gpu_data,
      const StatisticsParams& params);

  /**
   * @brief Compute median of magnitudes per beam (GPU data)
   */
  std::vector<MedianResult> ComputeMedian(
      void* gpu_data,
      const StatisticsParams& params);

  /**
   * @brief Compute full statistics per beam (GPU data)
   */
  std::vector<StatisticsResult> ComputeStatistics(
      void* gpu_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- GPU float data (magnitudes already computed)
  // =========================================================================

  /**
   * @brief Compute statistics on float magnitudes per beam (GPU data)
   * @param gpu_float_data Device pointer to float data [beam_count x n_point]
   * @param params Statistics parameters
   * @return StatisticsResult with mean_magnitude, variance, std_dev (mean complex = 0)
   */
  std::vector<StatisticsResult> ComputeStatisticsFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  /**
   * @brief Compute median of float magnitudes per beam (GPU data)
   * @param gpu_float_data Device pointer to float data [beam_count x n_point]
   * @param params Statistics parameters
   */
  std::vector<MedianResult> ComputeMedianFloat(
      void* gpu_float_data,
      const StatisticsParams& params);

  // =========================================================================
  // Public API -- CPU float data (vector<float> wrappers for tests)
  // =========================================================================

  /**
   * @brief Compute statistics on float magnitudes per beam (CPU data)
   *
   * Convenience wrapper for testing: uploads data via hipMalloc+hipMemcpy,
   * calls ComputeStatisticsFloat(void*, params), frees GPU buffer.
   *
   * @param data Flat float array [beam_count * n_point]
   * @param params Statistics parameters
   * @return StatisticsResult per beam
   */
  std::vector<StatisticsResult> ComputeStatisticsFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

  /**
   * @brief Compute median of float magnitudes per beam (CPU data)
   *
   * Convenience wrapper for testing: uploads data via hipMalloc+hipMemcpy,
   * calls ComputeMedianFloat(void*, params), frees GPU buffer.
   *
   * @param data Flat float array [beam_count * n_point]
   * @param params Statistics parameters
   * @return MedianResult per beam
   */
  std::vector<MedianResult> ComputeMedianFloat(
      const std::vector<float>& data,
      const StatisticsParams& params);

private:
  // =========================================================================
  // GPU Resources management
  // =========================================================================

  /// Compile HIP kernels via hiprtc
  void CompileKernels();

  /// Allocate working buffers for n_point samples, beam_count beams
  void AllocateBuffers(size_t beam_count, size_t n_point);

  /// Release all GPU resources
  void ReleaseResources();

  // =========================================================================
  // GPU Operations
  // =========================================================================

  /// Upload CPU data to input_buffer_
  void UploadData(const std::complex<float>* data, size_t count);

  /// Copy GPU data to input_buffer_ (D2D)
  void CopyGpuData(void* src, size_t count);

  /// Execute compute_magnitudes kernel: complex -> |z| per element
  void ExecuteMagnitudesKernel(size_t total_elements);

  /// Execute hierarchical mean reduction kernel for complex data
  void ExecuteMeanReduction(size_t beam_count, size_t n_point);

  /// Execute Welford statistics kernel (mean_mag + variance + std per beam)
  void ExecuteWelfordKernel(size_t beam_count, size_t n_point);

  /// TASK-1: Execute fused Welford kernel (reads only input, no magnitudes buffer)
  void ExecuteWelfordFusedKernel(size_t beam_count, size_t n_point);

  /// Execute sort + median extraction per beam
  void ExecuteMedianSort(size_t beam_count, size_t n_point);

  /// TASK-2: Execute extract_medians kernel → compact medians_compact_buf_
  void ExecuteExtractMediansKernel(size_t beam_count, size_t n_point);

  /// Execute Welford on float input (magnitudes already computed)
  void ExecuteWelfordFloatKernel(size_t beam_count, size_t n_point);

  /// Copy float GPU data to magnitudes_buf_ (D2D, float-sized)
  void CopyFloatGpuData(void* src, size_t count);

  // =========================================================================
  // Members
  // =========================================================================

  // Backend
  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;

  // GPU buffers
  void* input_buffer_        = nullptr;  ///< complex<float> input: beams * n_point
  void* magnitudes_buf_      = nullptr;  ///< float magnitudes: beams * n_point
  void* sort_buf_            = nullptr;  ///< float sorted output (rocprim writes here)
  void* sort_temp_buf_       = nullptr;  ///< rocPRIM segmented_radix_sort temp storage
  void* offsets_buf_         = nullptr;  ///< unsigned int[beam_count+1]: segment offsets
  void* reduce_buf_          = nullptr;  ///< float2 partial sums for mean reduction
  void* result_buf_          = nullptr;  ///< per-beam results (various types)
  void* medians_compact_buf_ = nullptr;  ///< TASK-2: compact float[beam_count] medians

  // hiprtc compiled kernels
  hipModule_t module_ = nullptr;
  hipFunction_t magnitudes_kernel_      = nullptr;  ///< complex -> |z|
  hipFunction_t mean_reduce_kernel_     = nullptr;  ///< hierarchical sum for complex mean
  hipFunction_t mean_final_kernel_      = nullptr;  ///< final mean division
  hipFunction_t welford_kernel_         = nullptr;  ///< Welford (magnitudes + input)
  hipFunction_t welford_fused_kernel_   = nullptr;  ///< TASK-1: fused, input only
  hipFunction_t welford_float_kernel_  = nullptr;  ///< Welford for float input (magnitudes)
  hipFunction_t extract_medians_kernel_ = nullptr;  ///< TASK-2: compact median extract
  bool kernels_compiled_ = false;

  // TASK-3: Disk cache for compiled HSACO
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  // State
  size_t current_beams_ = 0;
  size_t current_n_point_ = 0;
  size_t sort_temp_size_ = 0;
};

}  // namespace statistics

#endif  // ENABLE_ROCM
