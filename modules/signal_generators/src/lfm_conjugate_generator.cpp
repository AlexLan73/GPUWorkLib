/**
 * @file lfm_conjugate_generator.cpp
 * @brief Conjugate LFM signal generator implementation
 *
 * Formula:
 *   t = sample_id / fs
 *   chirp_rate = (f_end - f_start) / duration
 *   phase = -(pi * chirp_rate * t^2 + 2*pi * f_start * t)
 *   output = amplitude * exp(j * phase)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "generators/lfm_conjugate_generator.hpp"
#include "kernel_loader.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

LfmConjugateGenerator::LfmConjugateGenerator(
    drv_gpu_lib::IBackend* backend, const LfmParams& params)
    : backend_(backend), params_(params) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "LfmConjugateGenerator: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  CompileKernel();
}

LfmConjugateGenerator::~LfmConjugateGenerator() {
  ReleaseGpuResources();
}

LfmConjugateGenerator::LfmConjugateGenerator(
    LfmConjugateGenerator&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , system_(other.system_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_) {
  other.program_ = nullptr;
}

LfmConjugateGenerator& LfmConjugateGenerator::operator=(
    LfmConjugateGenerator&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    params_ = other.params_;
    system_ = other.system_;
    context_ = other.context_;
    queue_ = other.queue_;
    device_ = other.device_;
    program_ = other.program_;
    other.program_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU generation
// ════════════════════════════════════════════════════════════════════════════

cl_mem LfmConjugateGenerator::GenerateToGpu() {
  uint32_t points = static_cast<uint32_t>(system_.length);

  if (points == 0) {
    throw std::runtime_error(
        "LfmConjugateGenerator::GenerateToGpu: num_samples is 0");
  }

  size_t buffer_size = static_cast<size_t>(points) * sizeof(std::complex<float>);

  cl_int err;

  // Output buffer
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LfmConjugateGenerator: clCreateBuffer(output) failed: "
        + std::to_string(err));
  }

  // Create kernel
  cl_kernel k = clCreateKernel(program_, "generate_lfm_conjugate", &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LfmConjugateGenerator: clCreateKernel failed: "
        + std::to_string(err));
  }

  // Parameters
  uint32_t pts = points;
  float sr = static_cast<float>(system_.fs);
  float f_start = static_cast<float>(params_.f_start);
  double duration = static_cast<double>(points) / system_.fs;
  float chirp_rate = static_cast<float>(params_.GetChirpRate(duration));
  float amp = static_cast<float>(params_.amplitude);

  err  = clSetKernelArg(k, 0, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(k, 1, sizeof(uint32_t), &pts);
  err |= clSetKernelArg(k, 2, sizeof(float),    &sr);
  err |= clSetKernelArg(k, 3, sizeof(float),    &f_start);
  err |= clSetKernelArg(k, 4, sizeof(float),    &chirp_rate);
  err |= clSetKernelArg(k, 5, sizeof(float),    &amp);

  if (err != CL_SUCCESS) {
    clReleaseKernel(k);
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LfmConjugateGenerator: clSetKernelArg failed");
  }

  // Launch
  size_t local_size = 256;
  size_t global_size =
      ((static_cast<size_t>(points) + local_size - 1) / local_size) * local_size;

  err = clEnqueueNDRangeKernel(
      queue_, k, 1, nullptr,
      &global_size, &local_size, 0, nullptr, nullptr);

  clReleaseKernel(k);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LfmConjugateGenerator: enqueue failed: "
        + std::to_string(err));
  }

  clFinish(queue_);
  return output_buf;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU generation (reference)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
LfmConjugateGenerator::GenerateToCpu() {
  uint32_t points = static_cast<uint32_t>(system_.length);

  if (points == 0) {
    throw std::runtime_error(
        "LfmConjugateGenerator::GenerateToCpu: num_samples is 0");
  }

  double duration = static_cast<double>(system_.length) / system_.fs;
  double chirp_rate = params_.GetChirpRate(duration);
  double f_start = params_.f_start;
  double amp = params_.amplitude;
  double fs = system_.fs;

  std::vector<std::complex<float>> result(points);

  for (uint32_t n = 0; n < points; ++n) {
    double t = static_cast<double>(n) / fs;

    // Conjugate: negate the phase
    double phase = -(M_PI * chirp_rate * t * t
                   + 2.0 * M_PI * f_start * t);

    result[n] = std::complex<float>(
        static_cast<float>(amp * std::cos(phase)),
        static_cast<float>(amp * std::sin(phase)));
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void LfmConjugateGenerator::CompileKernel() {
  std::string source = LoadKernelFile("lfm_conjugate.cl");
  const char* src_ptr = source.c_str();
  size_t source_len = source.size();

  cl_int err;
  program_ = clCreateProgramWithSource(
      context_, 1, &src_ptr, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LfmConjugateGenerator: clCreateProgramWithSource failed");
  }

  err = clBuildProgram(
      program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
        nullptr);
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error(
        "LfmConjugateGenerator kernel build failed:\n"
        + std::string(log.data()));
  }
}

void LfmConjugateGenerator::ReleaseGpuResources() {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
}

} // namespace signal_gen