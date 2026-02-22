#pragma once

/**
 * @file heterodyne_processor_opencl.hpp
 * @brief OpenCL implementation of heterodyne dechirp processor
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "../i_heterodyne_processor.hpp"
#include "interface/i_backend.hpp"

#include <CL/cl.h>

namespace drv_gpu_lib {

class HeterodyneProcessorOpenCL : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorOpenCL(IBackend* backend);
  ~HeterodyneProcessorOpenCL();

  HeterodyneProcessorOpenCL(const HeterodyneProcessorOpenCL&) = delete;
  HeterodyneProcessorOpenCL& operator=(const HeterodyneProcessorOpenCL&) = delete;

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

  /** OPT-3: Dechirp with GPU-resident reference (no PCIe for ref) */
  std::vector<std::complex<float>> DechirpWithGPURef(
      void* rx_cl_mem,
      void* ref_cl_mem,
      const HeterodyneParams& params);

private:
  void CompileKernels();
  void ReleaseGpuResources();

  /** OPT-2: Allocate/reuse GPU buffers when size changes */
  void EnsureBuffers(int total_samples, int num_samples);

  IBackend*         backend_ = nullptr;
  cl_context        context_ = nullptr;
  cl_command_queue  queue_   = nullptr;
  cl_device_id      device_  = nullptr;

  // Programs (compiled once)
  cl_program        program_multiply_ = nullptr;
  cl_program        program_correct_  = nullptr;

  // OPT-1: Cached kernel objects (created once from programs)
  cl_kernel         kernel_multiply_ = nullptr;
  cl_kernel         kernel_correct_  = nullptr;

  // OPT-2: Cached GPU buffers (reused across calls)
  cl_mem            buf_rx_     = nullptr;
  cl_mem            buf_ref_    = nullptr;
  cl_mem            buf_dc_     = nullptr;   // output for dechirp / input for correct
  cl_mem            buf_corr_   = nullptr;   // output for correct
  cl_mem            buf_freq_   = nullptr;   // f_beat / phase_step per antenna
  int               cached_total_   = 0;     // antennas * samples
  int               cached_samples_ = 0;     // samples only (for ref)
  int               cached_antennas_ = 0;    // antennas only (for freq buf)
};

}  // namespace drv_gpu_lib