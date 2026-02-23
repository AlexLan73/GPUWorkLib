/**
 * @file iir_filter_rocm.cpp
 * @brief IirFilterROCm implementation - GPU IIR biquad cascade (ROCm/HIP)
 *
 * Port of iir_filter.cpp (OpenCL) to HIP/ROCm.
 * Uses hiprtc for runtime kernel compilation, hipModuleLaunchKernel for dispatch.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "filters/iir_filter_rocm.hpp"
#include "kernels/iir_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

IirFilterROCm::IirFilterROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "IirFilterROCm: backend is null or not initialized");
  }

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        "IirFilterROCm: failed to get HIP stream from backend");
  }

  CompileKernel();
}

IirFilterROCm::~IirFilterROCm() {
  ReleaseGpuResources();
}

IirFilterROCm::IirFilterROCm(IirFilterROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , sections_(std::move(other.sections_))
    , module_(other.module_)
    , kernel_(other.kernel_)
    , kernel_compiled_(other.kernel_compiled_)
    , sos_buf_(other.sos_buf_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
  other.kernel_ = nullptr;
  other.kernel_compiled_ = false;
  other.sos_buf_ = nullptr;
}

IirFilterROCm& IirFilterROCm::operator=(IirFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    sections_ = std::move(other.sections_);
    module_ = other.module_;
    kernel_ = other.kernel_;
    kernel_compiled_ = other.kernel_compiled_;
    sos_buf_ = other.sos_buf_;
    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.kernel_ = nullptr;
    other.kernel_compiled_ = false;
    other.sos_buf_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void IirFilterROCm::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "iir") {
    throw std::runtime_error(
        "IirFilterROCm::LoadConfig: expected type 'iir', got '" + cfg.type + "'");
  }
  SetBiquadSections(cfg.sections);
}

void IirFilterROCm::SetBiquadSections(const std::vector<BiquadSection>& sections) {
  if (sections.empty()) {
    throw std::invalid_argument("IirFilterROCm::SetBiquadSections: empty sections");
  }

  sections_ = sections;
  UploadSosMatrix();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
IirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr) {
    throw std::invalid_argument("IirFilterROCm::Process: input_ptr is null");
  }
  if (channels == 0 || points == 0) {
    throw std::runtime_error("IirFilterROCm::Process: channels or points is 0");
  }
  if (sections_.empty()) {
    throw std::runtime_error("IirFilterROCm::Process: no biquad sections set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);
  hipError_t err;

  // Allocate output buffer
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::Process: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Kernel arguments
  unsigned int num_sec = static_cast<unsigned int>(sections_.size());
  unsigned int ch = channels;
  unsigned int pts = points;

  void* args[] = {
    &input_ptr,
    &output_ptr,
    &sos_buf_,
    &num_sec,
    &ch,
    &pts
  };

  // Launch: 1D grid, one thread per channel
  unsigned int grid_size = static_cast<unsigned int>(
      (channels + kBlockSize - 1) / kBlockSize);

  err = hipModuleLaunchKernel(
      kernel_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error(
        "IirFilterROCm::Process: hipModuleLaunchKernel failed: " +
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
IirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points)
{
  size_t expected = static_cast<size_t>(channels) * points;
  if (data.size() < expected) {
    throw std::invalid_argument(
        "IirFilterROCm::ProcessFromCPU: input size " +
        std::to_string(data.size()) + " < expected " +
        std::to_string(expected));
  }

  size_t data_size = expected * sizeof(std::complex<float>);
  void* input_ptr = nullptr;
  hipError_t err = hipMalloc(&input_ptr, data_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::ProcessFromCPU: hipMalloc(input) failed");
  }

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, stream_);
  if (err != hipSuccess) {
    hipFree(input_ptr);
    throw std::runtime_error(
        "IirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
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
IirFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (sections_.empty()) {
    throw std::runtime_error("IirFilterROCm::ProcessCpu: no biquad sections set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("IirFilterROCm::ProcessCpu: input too small");
  }

  // Copy input as working buffer
  std::vector<std::complex<float>> output(input.begin(), input.begin() + total);

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    for (const auto& sec : sections_) {
      // Direct Form II Transposed state
      std::complex<float> w1(0.0f, 0.0f);
      std::complex<float> w2(0.0f, 0.0f);

      for (uint32_t n = 0; n < points; ++n) {
        std::complex<float> x = output[base + n];
        std::complex<float> y = sec.b0 * x + w1;
        w1 = sec.b1 * x - sec.a1 * y + w2;
        w2 = sec.b2 * x - sec.a2 * y;
        output[base + n] = y;
      }
    }
  }

  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Internals
// ════════════════════════════════════════════════════════════════════════════

void IirFilterROCm::CompileKernel() {
  if (kernel_compiled_) return;

  const char* source = kernels::GetIirBiquadCascadeSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "iir_filter_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "IirFilterROCm::CompileKernel: hiprtcCreateProgram failed: " +
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
    con.PrintError(0, "IirFilter[ROCm]", "Kernel compile log:\n" + log);

    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "IirFilterROCm::CompileKernel: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::CompileKernel: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  hipErr = hipModuleGetFunction(&kernel_, module_, "iir_biquad_cascade_cf32");
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::CompileKernel: hipModuleGetFunction(iir_biquad_cascade_cf32) failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  kernel_compiled_ = true;

  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "IirFilter[ROCm]",
      "HIP kernel compiled (iir_biquad_cascade_cf32)");
}

void IirFilterROCm::UploadSosMatrix() {
  if (sos_buf_) {
    hipFree(sos_buf_);
    sos_buf_ = nullptr;
  }

  // Pack sections into flat float array: [sec*5 + {b0,b1,b2,a1,a2}]
  std::vector<float> sos_flat;
  sos_flat.reserve(sections_.size() * 5);
  for (const auto& sec : sections_) {
    sos_flat.push_back(sec.b0);
    sos_flat.push_back(sec.b1);
    sos_flat.push_back(sec.b2);
    sos_flat.push_back(sec.a1);
    sos_flat.push_back(sec.a2);
  }

  size_t sos_size = sos_flat.size() * sizeof(float);
  hipError_t err = hipMalloc(&sos_buf_, sos_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::UploadSosMatrix: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  err = hipMemcpyHtoDAsync(sos_buf_, sos_flat.data(), sos_size, stream_);
  if (err != hipSuccess) {
    hipFree(sos_buf_);
    sos_buf_ = nullptr;
    throw std::runtime_error(
        "IirFilterROCm::UploadSosMatrix: hipMemcpyHtoDAsync failed");
  }
  hipStreamSynchronize(stream_);
}

void IirFilterROCm::ReleaseGpuResources() {
  if (module_) {
    hipModuleUnload(module_);
    module_ = nullptr;
    kernel_ = nullptr;
    kernel_compiled_ = false;
  }
  if (sos_buf_) {
    hipFree(sos_buf_);
    sos_buf_ = nullptr;
  }
}

}  // namespace filters

#endif  // ENABLE_ROCM
