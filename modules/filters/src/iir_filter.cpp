/**
 * @file iir_filter.cpp
 * @brief IirFilter implementation - GPU IIR biquad cascade (OpenCL)
 *
 * Cascade of biquad sections, Direct Form II Transposed.
 * All sections processed in single kernel call.
 * Parallelism: across channels (1 work-item per channel).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "filters/iir_filter.hpp"
#include "kernels/iir_kernels.hpp"

#include "common/backend_type.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

// Path to filters kernels/ set by CMake (fallback: relative)
#ifndef FILTERS_KERNELS_DIR
#define FILTERS_KERNELS_DIR "modules/filters/kernels"
#endif

IirFilter::IirFilter(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "IirFilter: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
      FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::OPENCL);

  CompileKernel();
}

IirFilter::~IirFilter() {
  ReleaseGpuResources();
}

IirFilter::IirFilter(IirFilter&& other) noexcept
    : backend_(other.backend_)
    , sections_(std::move(other.sections_))
    , kernel_cache_(std::move(other.kernel_cache_))
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_)
    , sos_buf_(other.sos_buf_) {
  other.program_ = nullptr;
  other.sos_buf_ = nullptr;
}

IirFilter& IirFilter::operator=(IirFilter&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_  = other.backend_;
    sections_ = std::move(other.sections_);
    kernel_cache_ = std::move(other.kernel_cache_);
    context_  = other.context_;
    queue_    = other.queue_;
    device_   = other.device_;
    program_  = other.program_;
    sos_buf_  = other.sos_buf_;
    other.program_ = nullptr;
    other.sos_buf_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void IirFilter::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "iir") {
    throw std::runtime_error(
        "IirFilter::LoadConfig: expected type 'iir', got '" + cfg.type + "'");
  }
  SetBiquadSections(cfg.sections);
}

void IirFilter::SetBiquadSections(const std::vector<BiquadSection>& sections) {
  if (sections.empty()) {
    throw std::invalid_argument("IirFilter::SetBiquadSections: empty sections");
  }

  sections_ = sections;
  UploadSosMatrix();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem>
IirFilter::Process(cl_mem input_buf, uint32_t channels, uint32_t points,
                   ProfEvents* prof_events) {
  if (channels == 0 || points == 0) {
    throw std::runtime_error("IirFilter::Process: channels or points is 0");
  }
  if (sections_.empty()) {
    throw std::runtime_error("IirFilter::Process: no biquad sections set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size  = total_points * sizeof(std::complex<float>);

  cl_int err;

  // Output buffer
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "IirFilter: clCreateBuffer(output) failed: " + std::to_string(err));
  }

  // Create kernel
  cl_kernel k = clCreateKernel(program_, "iir_biquad_cascade_cf32", &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "IirFilter: clCreateKernel failed: " + std::to_string(err));
  }

  // Set arguments
  uint32_t num_sec = static_cast<uint32_t>(sections_.size());
  uint32_t pts = points;

  err  = clSetKernelArg(k, 0, sizeof(cl_mem),   &input_buf);
  err |= clSetKernelArg(k, 1, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(k, 2, sizeof(cl_mem),   &sos_buf_);
  err |= clSetKernelArg(k, 3, sizeof(uint32_t), &num_sec);
  err |= clSetKernelArg(k, 4, sizeof(uint32_t), &pts);

  if (err != CL_SUCCESS) {
    clReleaseKernel(k);
    clReleaseMemObject(output_buf);
    throw std::runtime_error("IirFilter: clSetKernelArg failed");
  }

  // Launch: 1D NDRange [channels]
  // IIR: one work-item per channel (data dependency within channel)
  size_t local_size  = std::min(static_cast<size_t>(channels), static_cast<size_t>(256));
  size_t global_size = ((static_cast<size_t>(channels) + local_size - 1) / local_size) * local_size;

  cl_event kernel_event = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, k, 1, nullptr,
      &global_size, &local_size,
      0, nullptr, &kernel_event);

  clReleaseKernel(k);

  if (err != CL_SUCCESS) {
    if (kernel_event) clReleaseEvent(kernel_event);
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "IirFilter: enqueue failed: " + std::to_string(err));
  }

  clFinish(queue_);

  CollectOrRelease(kernel_event, "Kernel", prof_events);

  // Build result
  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count   = channels;
  result.n_point         = points;
  result.data            = output_buf;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementation
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
IirFilter::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (sections_.empty()) {
    throw std::runtime_error("IirFilter::ProcessCpu: no biquad sections set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("IirFilter::ProcessCpu: input too small");
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

void IirFilter::CompileKernel() {
  const std::string kernel_name = "iir_filter_cf32";

  // Try loading from cache (binary fast path)
  try {
    auto entry = kernel_cache_->Load(kernel_name);
    if (entry.has_binary()) {
      LoadFromBinary(entry.binary);
      return;
    }
  } catch (...) {
    // Cache miss or load error — compile from source
  }

  // Compile from source
  cl_int err;
  const char* source = kernels::GetIirBiquadCascadeSource_opencl();
  size_t source_len = strlen(source);

  program_ = clCreateProgramWithSource(
      context_, 1, &source, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "IirFilter: clCreateProgramWithSource failed: " + std::to_string(err));
  }

  err = clBuildProgram(
      program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error(
        "IirFilter kernel build failed:\n" + std::string(log.data()));
  }

  // Save to cache for next time
  try {
    auto binary = GetProgramBinary();
    kernel_cache_->Save(kernel_name, std::string(source),
                        binary, "", "IIR biquad cascade");
  } catch (...) {
    // Non-critical: cache save failed
  }
}

std::vector<uint8_t> IirFilter::GetProgramBinary() const {
  if (!program_) {
    throw std::runtime_error("IirFilter::GetProgramBinary: no compiled program");
  }

  size_t binary_size = 0;
  cl_int err = clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                 sizeof(size_t), &binary_size, nullptr);
  if (err != CL_SUCCESS || binary_size == 0) {
    throw std::runtime_error("IirFilter::GetProgramBinary: cannot get binary size");
  }

  std::vector<uint8_t> binary(binary_size);
  uint8_t* ptrs[] = { binary.data() };
  err = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                          sizeof(uint8_t*), ptrs, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("IirFilter::GetProgramBinary: cannot get binary");
  }

  return binary;
}

void IirFilter::LoadFromBinary(const std::vector<uint8_t>& binary) {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }

  cl_int err, binary_status;
  size_t bin_size = binary.size();
  const unsigned char* bin_ptr = binary.data();

  program_ = clCreateProgramWithBinary(
      context_, 1, &device_, &bin_size, &bin_ptr, &binary_status, &err);

  if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
    program_ = nullptr;
    throw std::runtime_error(
        "IirFilter::LoadFromBinary: clCreateProgramWithBinary failed");
  }

  err = clBuildProgram(program_, 1, &device_,
                       "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error("IirFilter::LoadFromBinary: clBuildProgram failed");
  }
}

void IirFilter::UploadSosMatrix() {
  if (sos_buf_) {
    clReleaseMemObject(sos_buf_);
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

  cl_int err;
  sos_buf_ = clCreateBuffer(
      context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sos_flat.size() * sizeof(float),
      sos_flat.data(), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "IirFilter: upload SOS matrix failed: " + std::to_string(err));
  }
}

void IirFilter::ReleaseGpuResources() {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
  if (sos_buf_) {
    clReleaseMemObject(sos_buf_);
    sos_buf_ = nullptr;
  }
}

} // namespace filters
