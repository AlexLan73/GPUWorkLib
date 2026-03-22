#pragma once

/**
 * @file heterodyne_processor_rocm.hpp
 * @brief ROCm/HIP implementation of heterodyne dechirp processor
 *
 * Port of HeterodyneProcessorOpenCL with HIP equivalents:
 * - hiprtc for kernel compilation (dechirp_multiply, dechirp_correct)
 * - void* device pointers instead of cl_mem
 * - hipStream_t instead of cl_command_queue
 *
 * Optimizations preserved from OpenCL version:
 *   OPT-1: Kernel objects cached (compiled once)
 *   OPT-2: GPU buffers cached and reused across calls
 *   OPT-3: DechirpWithGPURef() — both inputs on GPU
 *   OPT-5: 1D kernel launch (gid = ant*N + n)
 *   OPT-6: phase_step precomputed on CPU
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "../i_heterodyne_processor.hpp"
#include "interface/i_backend.hpp"
#include "interface/gpu_context.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>
#include <utility>
#include <vector>

namespace drv_gpu_lib {

using HeterodyneROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorROCm(IBackend* backend);
  ~HeterodyneProcessorROCm();

  // No copy
  HeterodyneProcessorROCm(const HeterodyneProcessorROCm&) = delete;
  HeterodyneProcessorROCm& operator=(const HeterodyneProcessorROCm&) = delete;

  // Move support
  HeterodyneProcessorROCm(HeterodyneProcessorROCm&& other) noexcept;
  HeterodyneProcessorROCm& operator=(HeterodyneProcessorROCm&& other) noexcept;

  /** Dechirp: s_dc = conj(s_rx * s_ref) on GPU */
  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override {
    return Dechirp(rx_data, ref_data, params, nullptr);
  }

  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /** Frequency correction: multiply by exp(j * phase_step * n) */
  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params) override {
    return Correct(dc_data, f_beat_hz, params, nullptr);
  }

  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /** Dechirp from external GPU buffer (void* = hipDeviceptr_t) */
  std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_gpu_ptr,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override {
    return DechirpFromGPU(rx_gpu_ptr, ref_data, params, nullptr);
  }

  std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_gpu_ptr,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /** OPT-3: Both rx and ref already on GPU */
  std::vector<std::complex<float>> DechirpWithGPURef(
      void* rx_gpu_ptr,
      void* ref_gpu_ptr,
      const HeterodyneParams& params) override {
    return DechirpWithGPURef(rx_gpu_ptr, ref_gpu_ptr, params, nullptr);
  }

  std::vector<std::complex<float>> DechirpWithGPURef(
      void* rx_gpu_ptr,
      void* ref_gpu_ptr,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

private:
  void EnsureCompiled();
  void ReleaseGpuResources();

  /** OPT-2: Allocate/reuse GPU buffers when size changes */
  void EnsureBuffers(int total_samples, int num_samples);

  GpuContext ctx_;  ///< Ref03: compilation, stream, disk cache
  IBackend*  backend_ = nullptr;  ///< Non-owning, for EnsureBuffers (hipMalloc)
  bool       compiled_ = false;

  // OPT-2: Cached GPU buffers (reused across calls, size-dependent)
  void*  buf_rx_   = nullptr;
  void*  buf_ref_  = nullptr;
  void*  buf_dc_   = nullptr;
  void*  buf_corr_ = nullptr;
  void*  buf_freq_ = nullptr;
  int    cached_total_    = 0;
  int    cached_samples_  = 0;
  int    cached_antennas_ = 0;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include "../i_heterodyne_processor.hpp"
#include "interface/i_backend.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorROCm(IBackend* /*backend*/) {}
  ~HeterodyneProcessorROCm() = default;

  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& /*rx_data*/,
      const std::vector<std::complex<float>>& /*ref_data*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::Dechirp: ROCm not enabled");
  }

  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& /*dc_data*/,
      const std::vector<float>& /*f_beat_hz*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::Correct: ROCm not enabled");
  }

  std::vector<std::complex<float>> DechirpFromGPU(
      void* /*rx_gpu_ptr*/,
      const std::vector<std::complex<float>>& /*ref_data*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::DechirpFromGPU: ROCm not enabled");
  }

  std::vector<std::complex<float>> DechirpWithGPURef(
      void* /*rx_gpu_ptr*/,
      void* /*ref_gpu_ptr*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::DechirpWithGPURef: ROCm not enabled");
  }
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
