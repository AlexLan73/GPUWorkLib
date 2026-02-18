#pragma once

/**
 * @file iir_filter_rocm.hpp
 * @brief IIR filter ROCm stub (not implemented yet)
 *
 * Placeholder for future ROCm/HIP IIR implementation.
 * All methods throw std::runtime_error.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "types/filter_params.hpp"

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend* /*backend*/) {}
  ~IirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("IirFilter ROCm backend not implemented");
  }

  void SetBiquadSections(const std::vector<BiquadSection>&) {
    throw std::runtime_error("IirFilter ROCm backend not implemented");
  }

  drv_gpu_lib::InputData<void*> Process(
      void* /*input_buf*/, uint32_t /*channels*/, uint32_t /*points*/) {
    throw std::runtime_error("IirFilter ROCm backend not implemented");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&,
      uint32_t, uint32_t) {
    throw std::runtime_error("IirFilter ROCm backend not implemented");
  }

  uint32_t GetNumSections() const { return 0; }
  bool IsReady() const { return false; }
};

} // namespace filters
