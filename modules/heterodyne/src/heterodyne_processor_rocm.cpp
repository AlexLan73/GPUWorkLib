/**
 * @file heterodyne_processor_rocm.cpp
 * @brief ROCm/HIP implementation of heterodyne dechirp processor
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CONTENTS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PART 1: Constructor, Destructor, Move
 * PART 2: EnsureBuffers — OPT-2 buffer caching
 * PART 3: Dechirp — s_dc = conj(s_rx * s_ref)
 * PART 4: Correct — frequency correction exp(j * phase_step * n)
 * PART 5: DechirpFromGPU — external GPU buffer
 * PART 6: DechirpWithGPURef — OPT-3 both inputs on GPU
 * PART 7: CompileKernels — hiprtc compilation
 * PART 8: ReleaseGpuResources — cleanup
 *
 * Key differences from OpenCL version:
 * - hiprtc for runtime kernel compilation
 * - hipMalloc/hipFree for device memory
 * - hipMemcpy H2D/D2H instead of clEnqueueWriteBuffer/ReadBuffer
 * - hipModuleLaunchKernel instead of clEnqueueNDRangeKernel
 * - Stream-ordered execution via hipStream_t
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "processors/heterodyne_processor_rocm.hpp"
#include "kernels/heterodyne_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor, Move
// ════════════════════════════════════════════════════════════════════════════

HeterodyneProcessorROCm::HeterodyneProcessorROCm(IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "HeterodyneProcessorROCm: backend is null or not initialized");
  }

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        "HeterodyneProcessorROCm: failed to get HIP stream from backend");
  }

  CompileKernels();
}

HeterodyneProcessorROCm::~HeterodyneProcessorROCm() {
  ReleaseGpuResources();
}

HeterodyneProcessorROCm::HeterodyneProcessorROCm(
    HeterodyneProcessorROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , module_multiply_(other.module_multiply_)
    , module_correct_(other.module_correct_)
    , kernel_multiply_(other.kernel_multiply_)
    , kernel_correct_(other.kernel_correct_)
    , kernels_compiled_(other.kernels_compiled_)
    , buf_rx_(other.buf_rx_)
    , buf_ref_(other.buf_ref_)
    , buf_dc_(other.buf_dc_)
    , buf_corr_(other.buf_corr_)
    , buf_freq_(other.buf_freq_)
    , cached_total_(other.cached_total_)
    , cached_samples_(other.cached_samples_)
    , cached_antennas_(other.cached_antennas_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_multiply_ = nullptr;
  other.module_correct_ = nullptr;
  other.kernel_multiply_ = nullptr;
  other.kernel_correct_ = nullptr;
  other.kernels_compiled_ = false;
  other.buf_rx_ = nullptr;
  other.buf_ref_ = nullptr;
  other.buf_dc_ = nullptr;
  other.buf_corr_ = nullptr;
  other.buf_freq_ = nullptr;
  other.cached_total_ = 0;
  other.cached_samples_ = 0;
  other.cached_antennas_ = 0;
}

HeterodyneProcessorROCm& HeterodyneProcessorROCm::operator=(
    HeterodyneProcessorROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    module_multiply_ = other.module_multiply_;
    module_correct_ = other.module_correct_;
    kernel_multiply_ = other.kernel_multiply_;
    kernel_correct_ = other.kernel_correct_;
    kernels_compiled_ = other.kernels_compiled_;
    buf_rx_ = other.buf_rx_;
    buf_ref_ = other.buf_ref_;
    buf_dc_ = other.buf_dc_;
    buf_corr_ = other.buf_corr_;
    buf_freq_ = other.buf_freq_;
    cached_total_ = other.cached_total_;
    cached_samples_ = other.cached_samples_;
    cached_antennas_ = other.cached_antennas_;
    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_multiply_ = nullptr;
    other.module_correct_ = nullptr;
    other.kernel_multiply_ = nullptr;
    other.kernel_correct_ = nullptr;
    other.kernels_compiled_ = false;
    other.buf_rx_ = nullptr;
    other.buf_ref_ = nullptr;
    other.buf_dc_ = nullptr;
    other.buf_corr_ = nullptr;
    other.buf_freq_ = nullptr;
    other.cached_total_ = 0;
    other.cached_samples_ = 0;
    other.cached_antennas_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 2: EnsureBuffers — OPT-2 buffer caching
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::EnsureBuffers(int total_samples, int num_samples) {
  int antennas = (total_samples > 0 && num_samples > 0)
                 ? total_samples / num_samples : 0;
  hipError_t err;

  // Rx + DC + Corr buffers (total = antennas * samples)
  if (total_samples != cached_total_) {
    if (buf_rx_)   { hipFree(buf_rx_);   buf_rx_ = nullptr; }
    if (buf_dc_)   { hipFree(buf_dc_);   buf_dc_ = nullptr; }
    if (buf_corr_) { hipFree(buf_corr_); buf_corr_ = nullptr; }

    size_t bytes = static_cast<size_t>(total_samples) * sizeof(std::complex<float>);

    err = hipMalloc(&buf_rx_, bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: rx alloc failed: " +
          std::string(hipGetErrorString(err)));

    err = hipMalloc(&buf_dc_, bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: dc alloc failed: " +
          std::string(hipGetErrorString(err)));

    err = hipMalloc(&buf_corr_, bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: corr alloc failed: " +
          std::string(hipGetErrorString(err)));

    cached_total_ = total_samples;
  }

  // Ref buffer (num_samples)
  if (num_samples != cached_samples_) {
    if (buf_ref_) { hipFree(buf_ref_); buf_ref_ = nullptr; }

    size_t ref_bytes = static_cast<size_t>(num_samples) * sizeof(std::complex<float>);
    err = hipMalloc(&buf_ref_, ref_bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: ref alloc failed: " +
          std::string(hipGetErrorString(err)));

    cached_samples_ = num_samples;
  }

  // Freq/phase_step buffer (antennas)
  if (antennas != cached_antennas_) {
    if (buf_freq_) { hipFree(buf_freq_); buf_freq_ = nullptr; }

    size_t freq_bytes = static_cast<size_t>(antennas) * sizeof(float);
    err = hipMalloc(&buf_freq_, freq_bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: freq alloc failed: " +
          std::string(hipGetErrorString(err)));

    cached_antennas_ = antennas;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: Dechirp — s_dc = conj(s_rx * s_ref) on GPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::Dechirp(
    const std::vector<std::complex<float>>& rx_data,
    const std::vector<std::complex<float>>& ref_data,
    const HeterodyneParams& params) {

  int total = params.num_antennas * params.num_samples;
  if (static_cast<int>(rx_data.size()) != total) {
    throw std::runtime_error(
        "Dechirp: rx_data size mismatch: expected "
        + std::to_string(total) + ", got " + std::to_string(rx_data.size()));
  }
  if (static_cast<int>(ref_data.size()) != params.num_samples) {
    throw std::runtime_error(
        "Dechirp: ref_data size mismatch: expected "
        + std::to_string(params.num_samples) + ", got "
        + std::to_string(ref_data.size()));
  }

  EnsureBuffers(total, params.num_samples);

  size_t rx_bytes  = static_cast<size_t>(total) * sizeof(std::complex<float>);
  size_t ref_bytes = static_cast<size_t>(params.num_samples) * sizeof(std::complex<float>);

  hipError_t err;

  // Upload data to cached buffers
  err = hipMemcpyHtoDAsync(buf_rx_, const_cast<std::complex<float>*>(rx_data.data()),
                            rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: rx upload failed: " +
        std::string(hipGetErrorString(err)));

  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: ref upload failed: " +
        std::string(hipGetErrorString(err)));

  // OPT-1: Use cached kernel, OPT-5: 1D launch
  int n_pts = params.num_samples;
  int total_elem = total;

  void* args[] = { &buf_rx_, &buf_ref_, &buf_dc_, &n_pts, &total_elem };

  unsigned int grid_size = (static_cast<unsigned int>(total) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: kernel launch failed: " +
        std::string(hipGetErrorString(err)));

  // Read result
  std::vector<std::complex<float>> result(total);
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: read failed: " +
        std::string(hipGetErrorString(err)));

  hipStreamSynchronize(stream_);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: Correct — frequency correction
// OPT-6: phase_step precomputed on CPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::Correct(
    const std::vector<std::complex<float>>& dc_data,
    const std::vector<float>& f_beat_hz,
    const HeterodyneParams& params) {

  int total = params.num_antennas * params.num_samples;
  if (static_cast<int>(dc_data.size()) != total) {
    throw std::runtime_error("Correct: dc_data size mismatch");
  }
  if (static_cast<int>(f_beat_hz.size()) != params.num_antennas) {
    throw std::runtime_error("Correct: f_beat_hz size mismatch");
  }

  EnsureBuffers(total, params.num_samples);

  size_t data_bytes = static_cast<size_t>(total) * sizeof(std::complex<float>);

  hipError_t err;

  // Upload dc data
  err = hipMemcpyHtoDAsync(buf_dc_, const_cast<std::complex<float>*>(dc_data.data()),
                            data_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: dc upload failed: " +
        std::string(hipGetErrorString(err)));

  // OPT-6: Precompute phase_step on CPU: phase_step[ant] = -2*pi*f_beat/fs
  std::vector<float> phase_step(params.num_antennas);
  for (int i = 0; i < params.num_antennas; ++i) {
    phase_step[i] = static_cast<float>(-2.0 * M_PI * f_beat_hz[i] / params.sample_rate);
  }

  size_t freq_bytes = static_cast<size_t>(params.num_antennas) * sizeof(float);
  err = hipMemcpyHtoDAsync(buf_freq_, phase_step.data(), freq_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: phase_step upload failed: " +
        std::string(hipGetErrorString(err)));

  // OPT-1: Use cached kernel, OPT-5: 1D launch
  int n_pts = params.num_samples;
  int total_elem = total;

  void* args[] = { &buf_dc_, &buf_corr_, &buf_freq_, &n_pts, &total_elem };

  unsigned int grid_size = (static_cast<unsigned int>(total) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      kernel_correct_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: kernel launch failed: " +
        std::string(hipGetErrorString(err)));

  std::vector<std::complex<float>> result(total);
  err = hipMemcpyDtoHAsync(result.data(), buf_corr_, data_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: read failed: " +
        std::string(hipGetErrorString(err)));

  hipStreamSynchronize(stream_);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 5: DechirpFromGPU — external GPU buffer (void* = hipDeviceptr_t)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::DechirpFromGPU(
    void* rx_gpu_ptr,
    const std::vector<std::complex<float>>& ref_data,
    const HeterodyneParams& params) {

  if (!rx_gpu_ptr) {
    throw std::runtime_error("DechirpFromGPU: rx_gpu_ptr is null");
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes  = static_cast<size_t>(total) * sizeof(std::complex<float>);
  size_t ref_bytes = static_cast<size_t>(params.num_samples) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  hipError_t err;

  // Upload ref to cached buffer
  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: ref upload failed: " +
        std::string(hipGetErrorString(err)));

  int n_pts = params.num_samples;
  int total_elem = total;

  // Use external rx buffer directly (DO NOT free — caller owns it)
  void* args[] = { &rx_gpu_ptr, &buf_ref_, &buf_dc_, &n_pts, &total_elem };

  unsigned int grid_size = (static_cast<unsigned int>(total) + kBlockSize - 1) / kBlockSize;

  err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: kernel launch failed: " +
        std::string(hipGetErrorString(err)));

  std::vector<std::complex<float>> result(total);
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: read failed: " +
        std::string(hipGetErrorString(err)));

  hipStreamSynchronize(stream_);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 6: DechirpWithGPURef — OPT-3 both inputs on GPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::DechirpWithGPURef(
    void* rx_gpu_ptr,
    void* ref_gpu_ptr,
    const HeterodyneParams& params) {

  if (!rx_gpu_ptr || !ref_gpu_ptr) {
    throw std::runtime_error("DechirpWithGPURef: null GPU pointer");
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes = static_cast<size_t>(total) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  int n_pts = params.num_samples;
  int total_elem = total;

  void* args[] = { &rx_gpu_ptr, &ref_gpu_ptr, &buf_dc_, &n_pts, &total_elem };

  unsigned int grid_size = (static_cast<unsigned int>(total) + kBlockSize - 1) / kBlockSize;

  hipError_t err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: kernel launch failed: " +
        std::string(hipGetErrorString(err)));

  std::vector<std::complex<float>> result(total);
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: read failed: " +
        std::string(hipGetErrorString(err)));

  hipStreamSynchronize(stream_);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: CompileKernels — hiprtc compilation (OPT-1: cached)
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = ConsoleOutput::GetInstance();

  // --- dechirp_multiply ---
  {
    const char* source = kernels::GetDechirpMultiplySource_rocm();

    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(
        &prog, source, "dechirp_multiply.hip", 0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
      throw std::runtime_error(
          "CompileKernels: hiprtcCreateProgram(multiply) failed: " +
          std::string(hiprtcGetErrorString(rtcResult)));
    }

    const char* options[] = { "-O3" };
    rtcResult = hiprtcCompileProgram(prog, 1, options);
    if (rtcResult != HIPRTC_SUCCESS) {
      size_t logSize = 0;
      hiprtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      hiprtcGetProgramLog(prog, &log[0]);
      con.PrintError(0, "Heterodyne[ROCm]", "multiply kernel compile log:\n" + log);
      hiprtcDestroyProgram(&prog);
      throw std::runtime_error("CompileKernels: dechirp_multiply compilation failed");
    }

    size_t codeSize = 0;
    hiprtcGetCodeSize(prog, &codeSize);
    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipError_t hipErr = hipModuleLoadData(&module_multiply_, code.data());
    if (hipErr != hipSuccess) {
      throw std::runtime_error(
          "CompileKernels: hipModuleLoadData(multiply) failed: " +
          std::string(hipGetErrorString(hipErr)));
    }

    hipErr = hipModuleGetFunction(&kernel_multiply_, module_multiply_, "dechirp_multiply");
    if (hipErr != hipSuccess) {
      hipModuleUnload(module_multiply_);
      module_multiply_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: hipModuleGetFunction(dechirp_multiply) failed: " +
          std::string(hipGetErrorString(hipErr)));
    }
  }

  // --- dechirp_correct ---
  {
    const char* source = kernels::GetDechirpCorrectSource_rocm();

    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(
        &prog, source, "dechirp_correct.hip", 0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
      hipModuleUnload(module_multiply_);
      module_multiply_ = nullptr;
      kernel_multiply_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: hiprtcCreateProgram(correct) failed: " +
          std::string(hiprtcGetErrorString(rtcResult)));
    }

    const char* options[] = { "-O3" };
    rtcResult = hiprtcCompileProgram(prog, 1, options);
    if (rtcResult != HIPRTC_SUCCESS) {
      size_t logSize = 0;
      hiprtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      hiprtcGetProgramLog(prog, &log[0]);
      con.PrintError(0, "Heterodyne[ROCm]", "correct kernel compile log:\n" + log);
      hiprtcDestroyProgram(&prog);
      hipModuleUnload(module_multiply_);
      module_multiply_ = nullptr;
      kernel_multiply_ = nullptr;
      throw std::runtime_error("CompileKernels: dechirp_correct compilation failed");
    }

    size_t codeSize = 0;
    hiprtcGetCodeSize(prog, &codeSize);
    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipError_t hipErr = hipModuleLoadData(&module_correct_, code.data());
    if (hipErr != hipSuccess) {
      hipModuleUnload(module_multiply_);
      module_multiply_ = nullptr;
      kernel_multiply_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: hipModuleLoadData(correct) failed: " +
          std::string(hipGetErrorString(hipErr)));
    }

    hipErr = hipModuleGetFunction(&kernel_correct_, module_correct_, "dechirp_correct");
    if (hipErr != hipSuccess) {
      hipModuleUnload(module_correct_);
      module_correct_ = nullptr;
      hipModuleUnload(module_multiply_);
      module_multiply_ = nullptr;
      kernel_multiply_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: hipModuleGetFunction(dechirp_correct) failed: " +
          std::string(hipGetErrorString(hipErr)));
    }
  }

  kernels_compiled_ = true;
  con.Print(0, "Heterodyne[ROCm]",
      "HIP kernels compiled (dechirp_multiply, dechirp_correct)");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 8: ReleaseGpuResources — cleanup
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::ReleaseGpuResources() {
  // OPT-1: Release cached modules (kernels are invalidated with module)
  if (module_multiply_) {
    hipModuleUnload(module_multiply_);
    module_multiply_ = nullptr;
    kernel_multiply_ = nullptr;
  }
  if (module_correct_) {
    hipModuleUnload(module_correct_);
    module_correct_ = nullptr;
    kernel_correct_ = nullptr;
  }
  kernels_compiled_ = false;

  // OPT-2: Release cached buffers
  if (buf_rx_)   { hipFree(buf_rx_);   buf_rx_ = nullptr; }
  if (buf_ref_)  { hipFree(buf_ref_);  buf_ref_ = nullptr; }
  if (buf_dc_)   { hipFree(buf_dc_);   buf_dc_ = nullptr; }
  if (buf_corr_) { hipFree(buf_corr_); buf_corr_ = nullptr; }
  if (buf_freq_) { hipFree(buf_freq_); buf_freq_ = nullptr; }

  cached_total_ = 0;
  cached_samples_ = 0;
  cached_antennas_ = 0;
}

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM

// ════════════════════════════════════════════════════════════════════════════
// Stub — all methods are inline in the header (nothing to compile here)
// ════════════════════════════════════════════════════════════════════════════

#endif  // ENABLE_ROCM
