#pragma once

/**
 * @file strategies_float_api.hpp
 * @brief CPU-vector API for post-FFT strategies computations
 *
 * Provides convenience wrappers accepting std::vector<float> magnitudes:
 *   - OneMaxParabolaFromFloat  — one peak + parabolic interpolation per beam
 *   - GlobalMinMaxFromFloat    — global MIN + MAX + dynamic range per beam
 *   - AllMaximaFromMagnitudes  — all local maxima per beam via AllMaximaPipeline
 *
 * Pattern (same as StatisticsProcessor CPU wrappers):
 *   hipMalloc → H2D → GPU kernel → D2H → hipFree
 *
 * IMPORTANT: ROCm-only. All code under #if ENABLE_ROCM.
 *
 * @date 2026-03-12
 */

#if ENABLE_ROCM

#include "result_types.hpp"
#include "interface/i_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "kernels/strategies_kernels_rocm.hpp"
#include "pipelines/all_maxima_pipeline_rocm.hpp"
#include "services/kernel_cache_service.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <memory>
#include <cstdint>

namespace strategies {

/**
 * @class StrategiesFloatApi
 * @brief Standalone post-FFT computations from CPU float magnitudes
 *
 * Compiles strategies kernels once in the constructor (cached via KernelCacheService).
 * Each method: H2D upload → kernel launch → D2H download (no persistent GPU state).
 */
class StrategiesFloatApi {
public:

  explicit StrategiesFloatApi(drv_gpu_lib::IBackend* backend)
      : backend_(backend)
  {
    if (!backend_ || !backend_->IsInitialized())
      throw std::runtime_error("StrategiesFloatApi: backend is null or not initialized");

    if (backend_->GetType() != drv_gpu_lib::BackendType::ROCm)
      throw std::runtime_error("StrategiesFloatApi: requires ROCm backend");

    stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
    if (!stream_)
      throw std::runtime_error("StrategiesFloatApi: failed to get HIP stream");

    all_maxima_ = std::make_unique<antenna_fft::AllMaximaPipelineROCm>(stream_, backend_);

    kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
        "modules/strategies/kernels", drv_gpu_lib::BackendType::ROCm);

    compile_kernels();
  }

  ~StrategiesFloatApi() {
    if (kernel_module_) hipModuleUnload(kernel_module_);
  }

  // No copy
  StrategiesFloatApi(const StrategiesFloatApi&) = delete;
  StrategiesFloatApi& operator=(const StrategiesFloatApi&) = delete;

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * @brief Find one peak + parabolic interpolation per beam (CPU float input)
   *
   * @param mags      Flat float magnitudes [n_ant * nFFT]
   * @param n_ant     Number of beams / antennas
   * @param nFFT      FFT size per beam
   * @param sample_rate Sampling frequency [Hz]
   * @return Vector of OneMaxParabolaLite (one per beam)
   */
  std::vector<OneMaxParabolaLite> OneMaxParabolaFromFloat(
      const std::vector<float>& mags,
      uint32_t n_ant, uint32_t nFFT, float sample_rate)
  {
    if (mags.size() != static_cast<size_t>(n_ant) * nFFT)
      throw std::invalid_argument("OneMaxParabolaFromFloat: mags.size() != n_ant * nFFT");

    const size_t mag_bytes     = mags.size() * sizeof(float);
    const size_t zeros_bytes   = static_cast<size_t>(n_ant) * nFFT * sizeof(float) * 2;  // float2
    const size_t result_bytes  = n_ant * sizeof(OneMaxParabolaLite);

    void* d_mag    = nullptr;
    void* d_zeros  = nullptr;
    void* d_result = nullptr;

    if (hipMalloc(&d_mag,    mag_bytes)    != hipSuccess ||
        hipMalloc(&d_zeros,  zeros_bytes)  != hipSuccess ||
        hipMalloc(&d_result, result_bytes) != hipSuccess)
    {
      hipFree(d_mag); hipFree(d_zeros); hipFree(d_result);
      throw std::runtime_error("OneMaxParabolaFromFloat: hipMalloc failed");
    }

    hipMemcpy(d_mag, mags.data(), mag_bytes, hipMemcpyHostToDevice);
    hipMemset(d_zeros, 0, zeros_bytes);

    // one_max_no_phase kernel: args = (d_magnitudes, d_spectrum, d_results, n_ant, nFFT, sample_rate)
    void* args[] = { &d_mag, &d_zeros, &d_result, &n_ant, &nFFT, &sample_rate };
    hipModuleLaunchKernel(
        one_max_kernel_,
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream_,
        args, nullptr);

    hipStreamSynchronize(stream_);

    std::vector<OneMaxParabolaLite> out(n_ant);
    hipMemcpy(out.data(), d_result, result_bytes, hipMemcpyDeviceToHost);

    hipFree(d_mag);
    hipFree(d_zeros);
    hipFree(d_result);

    return out;
  }

  /**
   * @brief Compute global MIN + MAX + dynamic_range per beam (CPU float input)
   *
   * @param mags      Flat float magnitudes [n_ant * nFFT]
   * @param n_ant     Number of beams / antennas
   * @param nFFT      FFT size per beam
   * @param sample_rate Sampling frequency [Hz]
   * @return Vector of MinMaxResult (one per beam)
   */
  std::vector<MinMaxResult> GlobalMinMaxFromFloat(
      const std::vector<float>& mags,
      uint32_t n_ant, uint32_t nFFT, float sample_rate)
  {
    if (mags.size() != static_cast<size_t>(n_ant) * nFFT)
      throw std::invalid_argument("GlobalMinMaxFromFloat: mags.size() != n_ant * nFFT");

    const size_t mag_bytes    = mags.size() * sizeof(float);
    const size_t result_bytes = n_ant * sizeof(MinMaxResult);

    void* d_mag    = nullptr;
    void* d_result = nullptr;

    if (hipMalloc(&d_mag,    mag_bytes)    != hipSuccess ||
        hipMalloc(&d_result, result_bytes) != hipSuccess)
    {
      hipFree(d_mag); hipFree(d_result);
      throw std::runtime_error("GlobalMinMaxFromFloat: hipMalloc failed");
    }

    hipMemcpy(d_mag, mags.data(), mag_bytes, hipMemcpyHostToDevice);

    // global_minmax kernel: args = (d_magnitudes, d_results, n_ant, nFFT, sample_rate)
    void* args[] = { &d_mag, &d_result, &n_ant, &nFFT, &sample_rate };
    hipModuleLaunchKernel(
        minmax_kernel_,
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream_,
        args, nullptr);

    hipStreamSynchronize(stream_);

    std::vector<MinMaxResult> out(n_ant);
    hipMemcpy(out.data(), d_result, result_bytes, hipMemcpyDeviceToHost);

    hipFree(d_mag);
    hipFree(d_result);

    return out;
  }

  /**
   * @brief Find all local maxima per beam (CPU float input)
   *
   * @param mags             Flat float magnitudes [beam_count * nFFT]
   * @param beam_count       Number of beams
   * @param nFFT             FFT size per beam
   * @param sample_rate      Sampling frequency [Hz]
   * @param dest             Output destination (CPU or GPU)
   * @param search_start     First bin to search (default 1 to skip DC)
   * @param search_end       Last bin (0 = nFFT/2)
   * @param max_maxima_per_beam  Max number of maxima per beam
   * @return AllMaximaResult with beams vector
   */
  antenna_fft::AllMaximaResult AllMaximaFromMagnitudes(
      const std::vector<float>& mags,
      uint32_t beam_count, uint32_t nFFT, float sample_rate,
      antenna_fft::OutputDestination dest = antenna_fft::OutputDestination::CPU,
      uint32_t search_start = 1,
      uint32_t search_end = 0,
      uint32_t max_maxima_per_beam = 1000)
  {
    if (mags.size() != static_cast<size_t>(beam_count) * nFFT)
      throw std::invalid_argument("AllMaximaFromMagnitudes: mags.size() != beam_count * nFFT");

    const size_t mag_bytes   = mags.size() * sizeof(float);
    const size_t zeros_bytes = static_cast<size_t>(beam_count) * nFFT * sizeof(float) * 2;  // float2

    void* d_mag   = nullptr;
    void* d_zeros = nullptr;

    if (hipMalloc(&d_mag,   mag_bytes)   != hipSuccess ||
        hipMalloc(&d_zeros, zeros_bytes) != hipSuccess)
    {
      hipFree(d_mag); hipFree(d_zeros);
      throw std::runtime_error("AllMaximaFromMagnitudes: hipMalloc failed");
    }

    hipMemcpy(d_mag, mags.data(), mag_bytes, hipMemcpyHostToDevice);
    hipMemset(d_zeros, 0, zeros_bytes);

    auto result = all_maxima_->Execute(
        d_mag,
        d_zeros,
        beam_count,
        nFFT,
        sample_rate,
        dest,
        search_start,
        search_end,
        max_maxima_per_beam);

    hipFree(d_mag);
    hipFree(d_zeros);

    return result;
  }

private:
  // =========================================================================
  // Kernel compilation (same strategy as AntennaProcessor_v1)
  // =========================================================================

  void compile_kernels() {
    if (kernels_compiled_) return;

    constexpr const char* kCacheName = "strategies_kernels";
    const char* src = kernels::GetStrategiesHIPKernelSource();

    // Try loading from disk cache
    if (kernel_cache_) {
      auto entry = kernel_cache_->Load(kCacheName);
      if (entry && entry->has_binary()) {
        hipError_t err = hipModuleLoadData(&kernel_module_, entry->binary.data());
        if (err == hipSuccess) {
          err = hipModuleGetFunction(&minmax_kernel_,   kernel_module_, "global_minmax");
          if (err == hipSuccess)
            err = hipModuleGetFunction(&one_max_kernel_, kernel_module_, "one_max_no_phase");
          if (err == hipSuccess) {
            kernels_compiled_ = true;
            return;
          }
        }
        if (kernel_module_) { hipModuleUnload(kernel_module_); kernel_module_ = nullptr; }
      }
    }

    // Compile from source
    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, src, "strategies_kernels.hip", 0, nullptr, nullptr);

    std::string arch_name;
    try {
      auto* rb = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
      arch_name = rb->GetCore().GetArchName();
    } catch (...) {}

    int warp_size = 32;
    try {
      auto* rwb = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
      warp_size = rwb->GetCore().GetWarpSize();
    } catch (...) {}
    std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
    std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
    std::vector<const char*> opts = { "-O3", warp_define.c_str(), "-DBLOCK_SIZE=256" };
    if (!arch_flag.empty()) opts.push_back(arch_flag.c_str());

    hiprtcResult res = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (res != HIPRTC_SUCCESS) {
      size_t sz = 0; hiprtcGetProgramLogSize(prog, &sz);
      std::string log(sz, '\0'); hiprtcGetProgramLog(prog, log.data());
      hiprtcDestroyProgram(&prog);
      throw std::runtime_error("StrategiesFloatApi kernel compile error:\n" + log);
    }

    size_t code_size = 0;
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipModuleLoadData(&kernel_module_, code.data());
    hipModuleGetFunction(&minmax_kernel_,   kernel_module_, "global_minmax");
    hipModuleGetFunction(&one_max_kernel_,  kernel_module_, "one_max_no_phase");

    kernels_compiled_ = true;
  }

  // =========================================================================
  // Members
  // =========================================================================

  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;

  hipModule_t   kernel_module_  = nullptr;
  hipFunction_t minmax_kernel_  = nullptr;
  hipFunction_t one_max_kernel_ = nullptr;
  bool          kernels_compiled_ = false;

  std::unique_ptr<antenna_fft::AllMaximaPipelineROCm> all_maxima_;
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  static constexpr uint32_t kBlockSize = 256;
};

}  // namespace strategies

#endif  // ENABLE_ROCM
