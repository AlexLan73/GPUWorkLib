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
#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
FirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
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

  // Launch: 1D grid covering channels * points
  unsigned int grid_size = static_cast<unsigned int>(
      (total_points + kBlockSize - 1) / kBlockSize);

  err = hipModuleLaunchKernel(
      kernel_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error(
        "FirFilterROCm::Process: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  hipStreamSynchronize(stream_);

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
    uint32_t channels, uint32_t points)
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

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, stream_);
  if (err != hipSuccess) {
    hipFree(input_ptr);
    throw std::runtime_error(
        "FirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
  }
  hipStreamSynchronize(stream_);

  auto result = Process(input_ptr, channels, points);

  hipFree(input_ptr);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementation
// ════════════════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════════════════
// GPU Internals
// ════════════════════════════════════════════════════════════════════════════

void FirFilterROCm::CompileKernel() {
  if (kernel_compiled_) return;

  const char* source = kernels::GetFirDirectSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "fir_filter_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  const char* options[] = { "-O3" };
  rtcResult = hiprtcCompileProgram(prog, 1, options);
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.PrintError(0, "FirFilter[ROCm]", "Kernel compile log:\n" + log);

    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "FirFilterROCm::CompileKernel: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

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

  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "FirFilter[ROCm]",
      "HIP kernel compiled (fir_filter_cf32)");
}

void FirFilterROCm::UploadCoefficients() {
  if (coeff_buf_) {
    hipFree(coeff_buf_);
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
    hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
    throw std::runtime_error(
        "FirFilterROCm::UploadCoefficients: hipMemcpyHtoDAsync failed");
  }
  hipStreamSynchronize(stream_);
}

void FirFilterROCm::ReleaseGpuResources() {
  if (module_) {
    hipModuleUnload(module_);
    module_ = nullptr;
    kernel_ = nullptr;
    kernel_compiled_ = false;
  }
  if (coeff_buf_) {
    hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
  }
}

}  // namespace filters

#endif  // ENABLE_ROCM
