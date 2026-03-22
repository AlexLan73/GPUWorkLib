/**
 * @file fir_filter_rocm.cpp
 * @brief FirFilterROCm implementation - GPU FIR convolution (ROCm/HIP)
 *
 * Port of fir_filter.cpp (OpenCL) to HIP/ROCm.
 * Uses hiprtc for runtime kernel compilation, hipModuleLaunchKernel for dispatch.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "filters/fir_filter_rocm.hpp"
#include "kernels/fir_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"
#include "services/console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>

using fft_func_utils::MakeROCmDataFromEvents;

namespace filters {

// ========================================================================
// Constructor / Destructor
// ========================================================================

FirFilterROCm::FirFilterROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "FirFilterROCm: backend is null or not initialized");
  }

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        "FirFilterROCm: failed to get HIP stream from backend");
  }

  CompileKernel();
}

FirFilterROCm::~FirFilterROCm() {
  ReleaseGpuResources();
}

FirFilterROCm::FirFilterROCm(FirFilterROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , coefficients_(std::move(other.coefficients_))
    , module_(other.module_)
    , kernel_(other.kernel_)
    , kernel_compiled_(other.kernel_compiled_)
    , coeff_buf_(other.coeff_buf_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
  other.kernel_ = nullptr;
  other.kernel_compiled_ = false;
  other.coeff_buf_ = nullptr;
}

FirFilterROCm& FirFilterROCm::operator=(FirFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    coefficients_ = std::move(other.coefficients_);
    module_ = other.module_;
    kernel_ = other.kernel_;
    kernel_compiled_ = other.kernel_compiled_;
    coeff_buf_ = other.coeff_buf_;
    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.kernel_ = nullptr;
    other.kernel_compiled_ = false;
    other.coeff_buf_ = nullptr;
  }
  return *this;
}

// ========================================================================
// Configuration
// ========================================================================

void FirFilterROCm::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "fir") {
    throw std::runtime_error(
        "FirFilterROCm::LoadConfig: expected type 'fir', got '" + cfg.type + "'");
  }
  SetCoefficients(cfg.coefficients);
}

void FirFilterROCm::SetCoefficients(const std::vector<float>& coeffs) {
  if (coeffs.empty()) {
    throw std::invalid_argument("FirFilterROCm::SetCoefficients: empty coefficients");
  }

  coefficients_ = coeffs;
  UploadCoefficients();
}

// ========================================================================
// GPU Processing
// ========================================================================

drv_gpu_lib::InputData<void*>
FirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points,
                       ROCmProfEvents* prof_events) {
  if (!input_ptr) {
    throw std::invalid_argument("FirFilterROCm::Process: input_ptr is null");
  }
  if (channels == 0 || points == 0) {
    throw std::runtime_error("FirFilterROCm::Process: channels or points is 0");
  }
  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilterROCm::Process: no coefficients set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);
  hipError_t err;

  // Allocate output buffer
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::Process: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Kernel arguments
  unsigned int num_taps = static_cast<unsigned int>(coefficients_.size());
  unsigned int ch = channels;
  unsigned int pts = points;

  void* args[] = {
    &input_ptr,
    &output_ptr,
    &coeff_buf_,
    &num_taps,
    &ch,
    &pts
  };

  // 2D grid: X = samples, Y = channels (eliminates div/mod in kernel)
  unsigned int grid_x = static_cast<unsigned int>(
      (points + kBlockSize - 1) / kBlockSize);
  unsigned int grid_y = channels;

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }

  err = hipModuleLaunchKernel(
      kernel_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (prof_events) {
    hipEventRecord(ev_k_e, stream_);
  }

  if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "FirFilterROCm::Process: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "fir_filter")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = channels;
  result.n_point       = points;
  result.data          = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

drv_gpu_lib::InputData<void*>
FirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points,
    ROCmProfEvents* prof_events)
{
  size_t expected = static_cast<size_t>(channels) * points;
  if (data.size() < expected) {
    throw std::invalid_argument(
        "FirFilterROCm::ProcessFromCPU: input size " +
        std::to_string(data.size()) + " < expected " +
        std::to_string(expected));
  }

  size_t data_size = expected * sizeof(std::complex<float>);
  void* input_ptr = nullptr;
  hipError_t err = hipMalloc(&input_ptr, data_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::ProcessFromCPU: hipMalloc(input) failed");
  }

  // H2D Upload timing
  hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_up_s);
    hipEventCreate(&ev_up_e);
    hipEventRecord(ev_up_s, stream_);
  }

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, stream_);

  if (prof_events) {
    hipEventRecord(ev_up_e, stream_);
  }

  if (err != hipSuccess) {
    if (ev_up_s) { hipEventDestroy(ev_up_s); hipEventDestroy(ev_up_e); }
    (void)hipFree(input_ptr);
    throw std::runtime_error(
        "FirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
  }
  // No hipStreamSynchronize here: kernel in same stream will wait for H2D

  if (prof_events) {
    prof_events->push_back({"Upload",
        MakeROCmDataFromEvents(ev_up_s, ev_up_e, 0, "H2D")});
  }

  auto result = Process(input_ptr, channels, points, prof_events);

  (void)hipFree(input_ptr);
  return result;
}

// ========================================================================
// CPU Reference Implementation
// ========================================================================

std::vector<std::complex<float>>
FirFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilterROCm::ProcessCpu: no coefficients set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("FirFilterROCm::ProcessCpu: input too small");
  }

  std::vector<std::complex<float>> output(total, {0.0f, 0.0f});
  uint32_t num_taps = static_cast<uint32_t>(coefficients_.size());

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;
    for (uint32_t n = 0; n < points; ++n) {
      std::complex<float> acc(0.0f, 0.0f);
      for (uint32_t k = 0; k < num_taps; ++k) {
        int idx = static_cast<int>(n) - static_cast<int>(k);
        if (idx >= 0) {
          acc += coefficients_[k] * input[base + idx];
        }
      }
      output[base + n] = acc;
    }
  }

  return output;
}

// ========================================================================
// GPU Internals
// ========================================================================

void FirFilterROCm::CompileKernel() {
  if (kernel_compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const std::string cache_name = "fir_filter_cf32_rocm";

  // Try loading from KernelCacheService (HSACO fast path)
  {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    auto entry = cache.Load(cache_name);
    if (entry && entry->has_binary()) {
      hipError_t hipErr = hipModuleLoadData(&module_, entry->binary.data());
      if (hipErr == hipSuccess) {
        hipErr = hipModuleGetFunction(&kernel_, module_, "fir_filter_cf32");
        if (hipErr == hipSuccess) {
          kernel_compiled_ = true;
          con.Print(0, "FirFilter[ROCm]",
              "HIP kernel loaded from cache (fir_filter_cf32)");
          return;
        }
      }
    }
  }

  // Compile from source (hiprtc)
  const char* source = kernels::GetFirDirectSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "fir_filter_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  // -O3, --offload-arch=gfxXXXX, -DBLOCK_SIZE=256
  std::string arch_name;
  try {
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name = "";
  }
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
  std::string block_size_def = "-DBLOCK_SIZE=" + std::to_string(kBlockSize);
  std::vector<const char*> opts = {"-O3", block_size_def.c_str()};
  if (!arch_flag.empty())
    opts.push_back(arch_flag.c_str());

  rtcResult = hiprtcCompileProgram(prog,
      static_cast<int>(opts.size()), opts.data());
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    con.PrintError(0, "FirFilter[ROCm]", "Kernel compile log:\n" + log);

    (void)hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  (void)hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  hipErr = hipModuleGetFunction(&kernel_, module_, "fir_filter_cf32");
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: hipModuleGetFunction(fir_filter_cf32) failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  kernel_compiled_ = true;

  // Save to cache
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    std::vector<uint8_t> binary(code.begin(), code.end());
    cache.Save(cache_name, std::string(source), binary,
               arch_name, "FIR direct-form convolution");
  } catch (...) {}

  con.Print(0, "FirFilter[ROCm]",
      "HIP kernel compiled (fir_filter_cf32)" +
      (arch_name.empty() ? "" : " [" + arch_name + "]"));
}

void FirFilterROCm::UploadCoefficients() {
  if (coeff_buf_) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
  }

  size_t coeff_size = coefficients_.size() * sizeof(float);
  hipError_t err = hipMalloc(&coeff_buf_, coeff_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::UploadCoefficients: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  err = hipMemcpyHtoDAsync(coeff_buf_, coefficients_.data(),
                            coeff_size, stream_);
  if (err != hipSuccess) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
    throw std::runtime_error(
        "FirFilterROCm::UploadCoefficients: hipMemcpyHtoDAsync failed");
  }
  (void)hipStreamSynchronize(stream_);
}

void FirFilterROCm::ReleaseGpuResources() {
  if (module_) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    kernel_ = nullptr;
    kernel_compiled_ = false;
  }
  if (coeff_buf_) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
  }
}

}  // namespace filters

#endif  // ENABLE_ROCM
