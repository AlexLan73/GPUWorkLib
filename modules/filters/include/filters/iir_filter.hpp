#pragma once

/**
 * @file iir_filter.hpp
 * @brief IirFilter - GPU IIR biquad cascade filter (OpenCL)
 *
 * Applies IIR (Infinite Impulse Response) filter as cascade of biquad sections.
 * Each channel filtered in parallel on GPU (data dependency within channel).
 *
 * Algorithm: Direct Form II Transposed (numerically stable)
 *   y[n] = b0*x[n] + w1
 *   w1   = b1*x[n] - a1*y[n] + w2
 *   w2   = b2*x[n] - a2*y[n]
 *
 * All sections processed in single kernel call per channel.
 *
 * Usage:
 * @code
 * IirFilter iir(backend);
 * iir.SetBiquadSections({{0.067, 0.135, 0.067, -1.143, 0.413}});
 *
 * auto result = iir.Process(input_buf, channels, points);
 * // result.data is cl_mem, caller must clReleaseMemObject()
 *
 * // Or from JSON:
 * iir.LoadConfig("butterworth4.json");
 * @endcode
 *
 * @note GPU IIR is efficient ONLY with many channels (>= 8).
 *       Single-channel IIR is faster on CPU!
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "services/kernel_cache_service.hpp"
#include "types/filter_params.hpp"
#include "types/filter_types.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <memory>

namespace filters {

class IirFilter {
public:
  explicit IirFilter(drv_gpu_lib::IBackend* backend);
  ~IirFilter();

  // No copy
  IirFilter(const IirFilter&) = delete;
  IirFilter& operator=(const IirFilter&) = delete;

  // Move
  IirFilter(IirFilter&& other) noexcept;
  IirFilter& operator=(IirFilter&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Load IIR config from JSON file
   * @param json_path Path to JSON: { "type":"iir", "sections":[...] }
   */
  void LoadConfig(const std::string& json_path);

  /**
   * @brief Set biquad sections directly (Stage 1: scipy -> GPU)
   * @param sections Vector of BiquadSection {b0, b1, b2, a1, a2}
   * @throws std::invalid_argument if sections empty
   */
  void SetBiquadSections(const std::vector<BiquadSection>& sections);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Apply IIR biquad cascade on GPU
   * @param input_buf cl_mem with [channels * points] complex float (float2)
   * @param channels Number of parallel channels
   * @param points Samples per channel
   * @return InputData<cl_mem> with filtered signal
   * @note Caller must release result.data via clReleaseMemObject()
   */
  drv_gpu_lib::InputData<cl_mem> Process(
      cl_mem input_buf, uint32_t channels, uint32_t points);

  /**
   * @brief CPU reference implementation
   * @param input Flat [channels * points] complex vector
   * @param channels Number of channels
   * @param points Samples per channel
   * @return Filtered signal
   */
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points);

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  uint32_t GetNumSections() const { return static_cast<uint32_t>(sections_.size()); }
  const std::vector<BiquadSection>& GetSections() const { return sections_; }
  bool IsReady() const { return program_ != nullptr && !sections_.empty(); }

private:
  void CompileKernel();
  void UploadSosMatrix();
  void ReleaseGpuResources();

  /// Get compiled binary from cl_program
  std::vector<uint8_t> GetProgramBinary() const;
  /// Create cl_program from binary blob
  void LoadFromBinary(const std::vector<uint8_t>& binary);

  drv_gpu_lib::IBackend* backend_ = nullptr;
  std::vector<BiquadSection> sections_;

  /// On-disk kernel cache
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  // OpenCL resources
  cl_context       context_   = nullptr;
  cl_command_queue queue_     = nullptr;
  cl_device_id     device_    = nullptr;
  cl_program       program_   = nullptr;
  cl_mem           sos_buf_   = nullptr;  ///< [num_sections * 5] float
};

} // namespace filters
