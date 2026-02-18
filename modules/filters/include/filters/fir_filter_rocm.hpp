#pragma once

/**
 * @file fir_filter_rocm.hpp
 * @brief FIR filter ROCm stub (not implemented yet)
 *
 * Placeholder for future ROCm/HIP FIR implementation.
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

class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend* /*backend*/) {}
  ~FirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("FirFilter ROCm backend not implemented");
  }

  void SetCoefficients(const std::vector<float>&) {
    throw std::runtime_error("FirFilter ROCm backend not implemented");
  }

  drv_gpu_lib::InputData<void*> Process(
      void* /*input_buf*/, uint32_t /*channels*/, uint32_t /*points*/) {
    throw std::runtime_error("FirFilter ROCm backend not implemented");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&,
      uint32_t, uint32_t) {
    throw std::runtime_error("FirFilter ROCm backend not implemented");
  }

  uint32_t GetNumTaps() const { return 0; }
  bool IsReady() const { return false; }
};

} // namespace filters
