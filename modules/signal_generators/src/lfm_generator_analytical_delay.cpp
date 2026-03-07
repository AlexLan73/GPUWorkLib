/**
 * @file lfm_generator_analytical_delay.cpp
 * @brief LFM signal with analytical per-antenna fractional delay
 *
 * Formula:
 *   t = sample_id / fs
 *   tau = delay_us[antenna_id] * 1e-6  (seconds)
 *   if t < tau: output = 0
 *   else:
 *     t_local = t - tau
 *     chirp_rate = (f_end - f_start) / duration
 *     phase = pi * chirp_rate * t_local^2 + 2*pi * f_start * t_local
 *     output = amplitude * exp(j * phase)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "generators/lfm_generator_analytical_delay.hpp"
#include "kernel_loader.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {
namespace {

// Сохранить cl_event для профилирования или освободить (production path).
// Ключевое правило: вызывать ПОСЛЕ того как event использован как wait-dependency.
void CollectOrRelease(cl_event ev, const char* name,
                      LfmGeneratorAnalyticalDelay::ProfEvents* prof_events) {
  if (!ev) return;
  if (prof_events) prof_events->push_back({name, ev});
  else clReleaseEvent(ev);
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

LfmGeneratorAnalyticalDelay::LfmGeneratorAnalyticalDelay(
    drv_gpu_lib::IBackend* backend, const LfmParams& params)
    : backend_(backend), params_(params) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  CompileKernel();
}

LfmGeneratorAnalyticalDelay::~LfmGeneratorAnalyticalDelay() {
  ReleaseGpuResources();
}

LfmGeneratorAnalyticalDelay::LfmGeneratorAnalyticalDelay(
    LfmGeneratorAnalyticalDelay&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , system_(other.system_)
    , delay_us_(std::move(other.delay_us_))
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_) {
  other.program_ = nullptr;
}

LfmGeneratorAnalyticalDelay& LfmGeneratorAnalyticalDelay::operator=(
    LfmGeneratorAnalyticalDelay&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    params_ = other.params_;
    system_ = other.system_;
    delay_us_ = std::move(other.delay_us_);
    context_ = other.context_;
    queue_ = other.queue_;
    device_ = other.device_;
    program_ = other.program_;
    other.program_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// SetDelays
// ════════════════════════════════════════════════════════════════════════════

void LfmGeneratorAnalyticalDelay::SetDelays(const std::vector<float>& delay_us) {
  delay_us_ = delay_us;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU generation
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem>
LfmGeneratorAnalyticalDelay::GenerateToGpu() {
  return GenerateToGpu(nullptr);
}

drv_gpu_lib::InputData<cl_mem>
LfmGeneratorAnalyticalDelay::GenerateToGpu(ProfEvents* prof_events) {
  uint32_t antennas = GetAntennas();
  uint32_t points = static_cast<uint32_t>(system_.length);

  if (antennas == 0 || points == 0) {
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay::GenerateToGpu: antennas or points is 0");
  }

  size_t total_points = static_cast<size_t>(antennas) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  cl_int err;

  // Output buffer
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: clCreateBuffer(output) failed: "
        + std::to_string(err));
  }

  // Upload delays (CL_MEM_COPY_HOST_PTR — synchronous, no event needed)
  cl_mem delay_buf = clCreateBuffer(
      context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      delay_us_.size() * sizeof(float),
      const_cast<float*>(delay_us_.data()), &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: clCreateBuffer(delay) failed: "
        + std::to_string(err));
  }

  // Create kernel
  const char* kernel_name = params_.complex_iq
      ? "generate_lfm_analytical_delay"
      : "generate_lfm_analytical_delay_real";

  cl_kernel k = clCreateKernel(program_, kernel_name, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    clReleaseMemObject(delay_buf);
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: clCreateKernel failed: "
        + std::to_string(err));
  }

  // Parameters
  uint32_t ant = antennas;
  uint32_t pts = points;
  float sr = static_cast<float>(system_.fs);
  float f_start = static_cast<float>(params_.f_start);
  double duration = static_cast<double>(points) / system_.fs;
  float chirp_rate = static_cast<float>(params_.GetChirpRate(duration));
  float amp = static_cast<float>(params_.amplitude);

  err  = clSetKernelArg(k, 0, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(k, 1, sizeof(cl_mem),   &delay_buf);
  err |= clSetKernelArg(k, 2, sizeof(uint32_t), &ant);
  err |= clSetKernelArg(k, 3, sizeof(uint32_t), &pts);
  err |= clSetKernelArg(k, 4, sizeof(float),    &sr);
  err |= clSetKernelArg(k, 5, sizeof(float),    &f_start);
  err |= clSetKernelArg(k, 6, sizeof(float),    &chirp_rate);
  err |= clSetKernelArg(k, 7, sizeof(float),    &amp);

  if (err != CL_SUCCESS) {
    clReleaseKernel(k);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(delay_buf);
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: clSetKernelArg failed");
  }

  // 2D grid: dim0 = samples, dim1 = antennas (eliminates div/mod in kernel)
  size_t local_size[2]  = { 256, 1 };
  size_t global_size[2] = {
      ((static_cast<size_t>(points) + 255) / 256) * 256,
      static_cast<size_t>(antennas)
  };

  cl_event ev_kernel = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, k, 2, nullptr,
      global_size, local_size, 0, nullptr, prof_events ? &ev_kernel : nullptr);

  clReleaseKernel(k);
  clReleaseMemObject(delay_buf);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: enqueue failed: "
        + std::to_string(err));
  }

  CollectOrRelease(ev_kernel, "Kernel", prof_events);

  clFinish(queue_);

  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count    = antennas;
  result.n_point          = points;
  result.data             = output_buf;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate      = static_cast<float>(system_.fs);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU generation (reference)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
LfmGeneratorAnalyticalDelay::GenerateToCpu() {
  uint32_t antennas = GetAntennas();
  uint32_t points = static_cast<uint32_t>(system_.length);

  if (antennas == 0 || points == 0) {
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay::GenerateToCpu: antennas or points is 0");
  }

  double duration = static_cast<double>(system_.length) / system_.fs;
  double chirp_rate = params_.GetChirpRate(duration);
  double f_start = params_.f_start;
  double amp = params_.amplitude;
  double fs = system_.fs;

  std::vector<std::vector<std::complex<float>>> result(antennas);

  for (uint32_t a = 0; a < antennas; ++a) {
    result[a].resize(points, {0.0f, 0.0f});
    double tau = static_cast<double>(delay_us_[a]) * 1e-6;

    for (uint32_t n = 0; n < points; ++n) {
      double t = static_cast<double>(n) / fs;
      if (t < tau) continue;  // signal not arrived yet

      double t_local = t - tau;
      double phase = M_PI * chirp_rate * t_local * t_local
                   + 2.0 * M_PI * f_start * t_local;

      if (params_.complex_iq) {
        result[a][n] = std::complex<float>(
            static_cast<float>(amp * std::cos(phase)),
            static_cast<float>(amp * std::sin(phase)));
      } else {
        result[a][n] = std::complex<float>(
            static_cast<float>(amp * std::cos(phase)), 0.0f);
      }
    }
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void LfmGeneratorAnalyticalDelay::CompileKernel() {
  std::string source = LoadKernelFile("lfm_analytical_delay.cl");
  const char* src_ptr = source.c_str();
  size_t source_len = source.size();

  cl_int err;
  program_ = clCreateProgramWithSource(
      context_, 1, &src_ptr, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "LfmGeneratorAnalyticalDelay: clCreateProgramWithSource failed");
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
        "LfmGeneratorAnalyticalDelay kernel build failed:\n"
        + std::string(log.data()));
  }
}

void LfmGeneratorAnalyticalDelay::ReleaseGpuResources() {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
}

} // namespace signal_gen
