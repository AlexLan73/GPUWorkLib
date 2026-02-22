#pragma once

/**
 * @file i_heterodyne_processor.hpp
 * @brief Strategy interface for heterodyne dechirp (OpenCL/ROCm)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "heterodyne_params.hpp"
#include <vector>
#include <complex>
#include <stdexcept>

namespace drv_gpu_lib {

class IHeterodyneProcessor {
public:
  virtual ~IHeterodyneProcessor() = default;

  /**
   * Dechirp: s_dc = s_rx * conj(s_tx) on GPU.
   *
   * @param rx_data   Matrix [num_antennas * num_samples] complex float (flat)
   * @param ref_data  Vector [num_samples] complex float = conj(s_tx)
   * @param params    LFM parameters
   * @return          Matrix [num_antennas * num_samples] complex float (dechirped)
   */
  virtual std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) = 0;

  /**
   * Frequency correction: multiply by exp(-j*2*pi*f_beat*t) per antenna
   *
   * @param dc_data     Dechirped data [num_antennas * num_samples]
   * @param f_beat_hz   Beat frequencies per antenna [num_antennas]
   * @param params      LFM parameters
   * @return            Corrected data [num_antennas * num_samples]
   */
  virtual std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params) = 0;

  /**
   * Dechirp from external GPU buffer (does NOT own the pointer!)
   * For integration with external OpenCL programs.
   *
   * @param rx_cl_mem  cl_mem - external buffer [num_antennas * num_samples]
   * @param ref_data   Reference signal (CPU -> GPU internally)
   * @param params     LFM parameters
   * @return           Dechirped data on CPU
   */
  virtual std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_cl_mem,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) = 0;

  /**
   * OPT-3: Dechirp with both rx and ref already on GPU (no PCIe for ref).
   * Default implementation falls back to DechirpFromGPU with CPU ref.
   */
  virtual std::vector<std::complex<float>> DechirpWithGPURef(
      void* /*rx_cl_mem*/, void* /*ref_cl_mem*/,
      const HeterodyneParams& /*params*/) {
    throw std::runtime_error("DechirpWithGPURef: not implemented");
  }
};

}  // namespace drv_gpu_lib