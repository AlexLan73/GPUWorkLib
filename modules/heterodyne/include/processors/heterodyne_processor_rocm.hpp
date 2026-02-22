#pragma once

/**
 * @file heterodyne_processor_rocm.hpp
 * @brief ROCm stub for heterodyne processor (not implemented)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "../i_heterodyne_processor.hpp"
#include "interface/i_backend.hpp"

namespace drv_gpu_lib {

class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorROCm(IBackend* backend);
  ~HeterodyneProcessorROCm() = default;

  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override;

  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params) override;

  std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_cl_mem,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override;
};

}  // namespace drv_gpu_lib