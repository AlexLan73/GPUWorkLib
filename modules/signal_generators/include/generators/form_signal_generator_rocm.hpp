#pragma once

/**
 * @file form_signal_generator_rocm.hpp
 * @brief FormSignalGeneratorROCm - multi-channel getX signal generator (ROCm/HIP)
 *
 * ROCm port of FormSignalGenerator (OpenCL). Same algorithm, HIP runtime:
 * - hiprtc for kernel compilation
 * - void* device pointers instead of cl_mem
 * - hipStream_t instead of cl_command_queue
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * Usage:
 * @code
 * FormSignalGeneratorROCm gen(rocm_backend);
 * gen.SetParams(params);
 *
 * auto result = gen.GenerateInputData();
 * // result.data is void* (HIP device pointer), caller must hipFree()
 *
 * auto cpu_data = gen.GenerateToCpu();
 * // cpu_data[antenna][sample] = complex<float>
 * @endcode
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "interface/gpu_context.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <utility>

namespace signal_gen {

/// Список событий профилирования ROCm (имя стадии + ROCmProfilingData)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/// @ingroup grp_signal_generators
class FormSignalGeneratorROCm {
public:
  explicit FormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend);
  ~FormSignalGeneratorROCm();

  // No copy
  FormSignalGeneratorROCm(const FormSignalGeneratorROCm&) = delete;
  FormSignalGeneratorROCm& operator=(const FormSignalGeneratorROCm&) = delete;

  // Move
  FormSignalGeneratorROCm(FormSignalGeneratorROCm&& other) noexcept;
  FormSignalGeneratorROCm& operator=(FormSignalGeneratorROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void SetParams(const FormParams& params) { params_ = params; }
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Generate signal on GPU (ROCm)
   * @return InputData<void*> with generated signal (caller must hipFree result.data)
   */
  drv_gpu_lib::InputData<void*> GenerateInputData();

  /**
   * @brief Генерация на GPU с опциональным сбором событий профилирования (ROCm)
   * @param prof_events nullptr → production (zero overhead); &vec → benchmark
   *
   * Собирает события: "Kernel" (generate_form_signal HIP kernel)
   */
  drv_gpu_lib::InputData<void*> GenerateInputData(ROCmProfEvents* prof_events);

  /**
   * @brief Generate signal on GPU, read back to CPU
   * @return vector[antenna][sample] = complex<float>
   */
  std::vector<std::vector<std::complex<float>>> GenerateToCpu();

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

private:
  void EnsureCompiled();

  drv_gpu_lib::GpuContext ctx_;
  FormParams params_;
  bool compiled_ = false;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace signal_gen

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include "../params/form_params.hpp"
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace signal_gen {

class FormSignalGeneratorROCm {
public:
  explicit FormSignalGeneratorROCm(drv_gpu_lib::IBackend*) {}
  ~FormSignalGeneratorROCm() = default;

  void SetParams(const FormParams& params) { params_ = params; }
  void SetParamsFromString(const std::string& params_str) {
    params_ = FormParams::ParseFromString(params_str);
  }

  drv_gpu_lib::InputData<void*> GenerateInputData() {
    throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
  }

  std::vector<std::vector<std::complex<float>>> GenerateToCpu() {
    throw std::runtime_error("FormSignalGeneratorROCm: ROCm not enabled");
  }

  const FormParams& GetParams() const { return params_; }
  uint32_t GetAntennas() const { return params_.antennas; }
  uint32_t GetPoints() const { return params_.points; }
  size_t GetTotalSamples() const {
    return static_cast<size_t>(params_.antennas) * params_.points;
  }

private:
  FormParams params_;
};

}  // namespace signal_gen

#endif  // ENABLE_ROCM
