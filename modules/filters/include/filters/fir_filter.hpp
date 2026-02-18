#pragma once

/**
 * @file fir_filter.hpp
 * @brief FirFilter - GPU FIR convolution filter (OpenCL)
 *
 * Applies FIR (Finite Impulse Response) filter to multi-channel complex signal.
 * Each channel filtered in parallel on GPU.
 *
 * Algorithm: Direct-form convolution
 *   y[ch][n] = sum_{k=0}^{N-1} h[k] * x[ch][n-k]
 *
 * Coefficients in __constant memory (<=16384 taps), auto-fallback to __global.
 *
 * Usage:
 * @code
 * FirFilter fir(backend);
 * fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});
 *
 * auto result = fir.Process(input_buf, channels, points);
 * // result.data is cl_mem, caller must clReleaseMemObject()
 *
 * // Or from JSON:
 * fir.LoadConfig("lowpass_64tap.json");
 * @endcode
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "types/filter_params.hpp"
#include "types/filter_types.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

/// Maximum taps for __constant memory (~64KB / 4 bytes = 16384)
static constexpr uint32_t kMaxConstantTaps = 16000;

class FirFilter {
public:
  explicit FirFilter(drv_gpu_lib::IBackend* backend);
  ~FirFilter();

  // No copy
  FirFilter(const FirFilter&) = delete;
  FirFilter& operator=(const FirFilter&) = delete;

  // Move
  FirFilter(FirFilter&& other) noexcept;
  FirFilter& operator=(FirFilter&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Load FIR coefficients from JSON file
   * @param json_path Path to JSON: { "type":"fir", "coefficients":[...] }
   */
  void LoadConfig(const std::string& json_path);

  /**
   * @brief Set FIR coefficients directly (Stage 1: scipy -> GPU)
   * @param coeffs h[k] array, length = num_taps
   * @throws std::invalid_argument if coeffs empty
   */
  void SetCoefficients(const std::vector<float>& coeffs);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Apply FIR filter on GPU
   * @param input_buf cl_mem with [channels * points] complex float (float2)
   * @param channels Number of parallel channels
   * @param points Samples per channel
   * @return InputData<cl_mem> with filtered signal
   * @note Caller must release result.data via clReleaseMemObject()
   * @note input_buf is NOT released by this method
   */
  drv_gpu_lib::InputData<cl_mem> Process(
      cl_mem input_buf, uint32_t channels, uint32_t points);

  /**
   * @brief CPU reference implementation (for validation)
   * @param input Flat [channels * points] complex vector
   * @param channels Number of channels
   * @param points Samples per channel
   * @return Filtered signal (same layout)
   */
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points);

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  uint32_t GetNumTaps() const { return static_cast<uint32_t>(coefficients_.size()); }
  const std::vector<float>& GetCoefficients() const { return coefficients_; }
  bool IsReady() const { return program_ != nullptr && !coefficients_.empty(); }

private:
  void CompileKernel();
  void UploadCoefficients();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  std::vector<float> coefficients_;
  bool use_global_coeffs_ = false;  ///< true if num_taps > kMaxConstantTaps

  // OpenCL resources
  cl_context       context_   = nullptr;
  cl_command_queue queue_     = nullptr;
  cl_device_id     device_    = nullptr;
  cl_program       program_   = nullptr;
  cl_mem           coeff_buf_ = nullptr;
};

} // namespace filters
