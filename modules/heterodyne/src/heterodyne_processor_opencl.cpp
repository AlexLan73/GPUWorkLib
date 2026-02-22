/**
 * @file heterodyne_processor_opencl.cpp
 * @brief OpenCL implementation of heterodyne dechirp processor
 *
 * GPU kernels:
 *   dechirp_multiply.cl: s_dc = s_rx * conj(s_ref)
 *   dechirp_correct.cl:  s_corrected = s_dc * exp(-j*phase_step*n)
 *
 * Optimizations applied:
 *   OPT-1: cl_kernel objects cached (not created per call)
 *   OPT-2: GPU buffers cached and reused across calls
 *   OPT-3: DechirpWithGPURef() — ref stays on GPU (no PCIe)
 *   OPT-5: 1D kernel launch (gid = ant*N + n)
 *   OPT-6: phase_step precomputed on CPU, kernel uses multiply instead of divide
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "processors/heterodyne_processor_opencl.hpp"
#include "kernels/kernel_loader.hpp"

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
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

HeterodyneProcessorOpenCL::HeterodyneProcessorOpenCL(IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "HeterodyneProcessorOpenCL: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  if (!context_ || !queue_ || !device_) {
    throw std::runtime_error(
        "HeterodyneProcessorOpenCL: failed to get OpenCL resources");
  }

  CompileKernels();
}

HeterodyneProcessorOpenCL::~HeterodyneProcessorOpenCL() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// OPT-2: EnsureBuffers — allocate/reuse GPU buffers
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorOpenCL::EnsureBuffers(int total_samples, int num_samples) {
  int antennas = (total_samples > 0 && num_samples > 0)
                 ? total_samples / num_samples : 0;
  cl_int err;

  // Rx + DC buffers (total = antennas * samples)
  if (total_samples != cached_total_) {
    if (buf_rx_)   { clReleaseMemObject(buf_rx_);   buf_rx_ = nullptr; }
    if (buf_dc_)   { clReleaseMemObject(buf_dc_);   buf_dc_ = nullptr; }
    if (buf_corr_) { clReleaseMemObject(buf_corr_); buf_corr_ = nullptr; }

    size_t bytes = static_cast<size_t>(total_samples) * sizeof(std::complex<float>);

    buf_rx_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("EnsureBuffers: rx alloc failed: " + std::to_string(err));

    buf_dc_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("EnsureBuffers: dc alloc failed: " + std::to_string(err));

    buf_corr_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("EnsureBuffers: corr alloc failed: " + std::to_string(err));

    cached_total_ = total_samples;
  }

  // Ref buffer (num_samples)
  if (num_samples != cached_samples_) {
    if (buf_ref_) { clReleaseMemObject(buf_ref_); buf_ref_ = nullptr; }

    size_t ref_bytes = static_cast<size_t>(num_samples) * sizeof(std::complex<float>);
    buf_ref_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, ref_bytes, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("EnsureBuffers: ref alloc failed: " + std::to_string(err));

    cached_samples_ = num_samples;
  }

  // Freq/phase_step buffer (antennas)
  if (antennas != cached_antennas_) {
    if (buf_freq_) { clReleaseMemObject(buf_freq_); buf_freq_ = nullptr; }

    size_t freq_bytes = static_cast<size_t>(antennas) * sizeof(float);
    buf_freq_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, freq_bytes, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("EnsureBuffers: freq alloc failed: " + std::to_string(err));

    cached_antennas_ = antennas;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Dechirp: s_dc = s_rx * conj(s_ref) on GPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorOpenCL::Dechirp(
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

  cl_int err;

  // Upload data to cached buffers
  err = clEnqueueWriteBuffer(queue_, buf_rx_, CL_FALSE, 0, rx_bytes,
                              rx_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Dechirp: rx upload failed: " + std::to_string(err));

  err = clEnqueueWriteBuffer(queue_, buf_ref_, CL_FALSE, 0, ref_bytes,
                              ref_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Dechirp: ref upload failed: " + std::to_string(err));

  // OPT-1: Use cached kernel
  int n_ant = params.num_antennas;
  int n_pts = params.num_samples;

  err  = clSetKernelArg(kernel_multiply_, 0, sizeof(cl_mem), &buf_rx_);
  err |= clSetKernelArg(kernel_multiply_, 1, sizeof(cl_mem), &buf_ref_);
  err |= clSetKernelArg(kernel_multiply_, 2, sizeof(cl_mem), &buf_dc_);
  err |= clSetKernelArg(kernel_multiply_, 3, sizeof(int), &n_pts);

  if (err != CL_SUCCESS)
    throw std::runtime_error("Dechirp: clSetKernelArg failed");

  // OPT-5: 1D launch
  size_t global = static_cast<size_t>(total);

  err = clEnqueueNDRangeKernel(queue_, kernel_multiply_, 1, nullptr, &global,
                                nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Dechirp: enqueue failed: " + std::to_string(err));

  // Read result
  std::vector<std::complex<float>> result(total);
  err = clEnqueueReadBuffer(queue_, buf_dc_, CL_TRUE, 0, rx_bytes,
                             result.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Dechirp: read failed: " + std::to_string(err));

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Correct: multiply by exp(-j * phase_step * n) per antenna
// OPT-6: phase_step precomputed on CPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorOpenCL::Correct(
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

  cl_int err;

  // Upload dc data
  err = clEnqueueWriteBuffer(queue_, buf_dc_, CL_FALSE, 0, data_bytes,
                              dc_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Correct: dc upload failed");

  // OPT-6: Precompute phase_step on CPU: phase_step[ant] = -2*pi*f_beat/fs
  std::vector<float> phase_step(params.num_antennas);
  for (int i = 0; i < params.num_antennas; ++i) {
    phase_step[i] = static_cast<float>(-2.0 * M_PI * f_beat_hz[i] / params.sample_rate);
  }

  size_t freq_bytes = static_cast<size_t>(params.num_antennas) * sizeof(float);
  err = clEnqueueWriteBuffer(queue_, buf_freq_, CL_FALSE, 0, freq_bytes,
                              phase_step.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Correct: phase_step upload failed");

  // OPT-1: Use cached kernel
  int n_pts = params.num_samples;

  err  = clSetKernelArg(kernel_correct_, 0, sizeof(cl_mem), &buf_dc_);
  err |= clSetKernelArg(kernel_correct_, 1, sizeof(cl_mem), &buf_corr_);
  err |= clSetKernelArg(kernel_correct_, 2, sizeof(cl_mem), &buf_freq_);
  err |= clSetKernelArg(kernel_correct_, 3, sizeof(int), &n_pts);

  if (err != CL_SUCCESS)
    throw std::runtime_error("Correct: clSetKernelArg failed");

  // OPT-5: 1D launch
  size_t global = static_cast<size_t>(total);

  err = clEnqueueNDRangeKernel(queue_, kernel_correct_, 1, nullptr, &global,
                                nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Correct: enqueue failed: " + std::to_string(err));

  std::vector<std::complex<float>> result(total);
  err = clEnqueueReadBuffer(queue_, buf_corr_, CL_TRUE, 0, data_bytes,
                             result.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Correct: read failed");

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// DechirpFromGPU: dechirp using external cl_mem buffer
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorOpenCL::DechirpFromGPU(
    void* rx_cl_mem,
    const std::vector<std::complex<float>>& ref_data,
    const HeterodyneParams& params) {

  if (!rx_cl_mem) {
    throw std::runtime_error("DechirpFromGPU: rx_cl_mem is null");
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes  = static_cast<size_t>(total) * sizeof(std::complex<float>);
  size_t ref_bytes = static_cast<size_t>(params.num_samples) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  cl_int err;
  cl_mem rx_buf = static_cast<cl_mem>(rx_cl_mem);  // external, DO NOT release

  // Upload ref to cached buffer
  err = clEnqueueWriteBuffer(queue_, buf_ref_, CL_FALSE, 0, ref_bytes,
                              ref_data.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpFromGPU: ref upload failed");

  int n_pts = params.num_samples;

  err  = clSetKernelArg(kernel_multiply_, 0, sizeof(cl_mem), &rx_buf);
  err |= clSetKernelArg(kernel_multiply_, 1, sizeof(cl_mem), &buf_ref_);
  err |= clSetKernelArg(kernel_multiply_, 2, sizeof(cl_mem), &buf_dc_);
  err |= clSetKernelArg(kernel_multiply_, 3, sizeof(int), &n_pts);

  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpFromGPU: clSetKernelArg failed");

  // OPT-5: 1D launch
  size_t global = static_cast<size_t>(total);

  err = clEnqueueNDRangeKernel(queue_, kernel_multiply_, 1, nullptr, &global,
                                nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpFromGPU: enqueue failed");

  std::vector<std::complex<float>> result(total);
  err = clEnqueueReadBuffer(queue_, buf_dc_, CL_TRUE, 0, rx_bytes,
                             result.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpFromGPU: read failed");

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// OPT-3: DechirpWithGPURef — both rx and ref are already on GPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorOpenCL::DechirpWithGPURef(
    void* rx_cl_mem,
    void* ref_cl_mem,
    const HeterodyneParams& params) {

  if (!rx_cl_mem || !ref_cl_mem) {
    throw std::runtime_error("DechirpWithGPURef: null cl_mem");
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes = static_cast<size_t>(total) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  cl_int err;
  cl_mem rx_buf  = static_cast<cl_mem>(rx_cl_mem);
  cl_mem ref_buf = static_cast<cl_mem>(ref_cl_mem);

  int n_pts = params.num_samples;

  err  = clSetKernelArg(kernel_multiply_, 0, sizeof(cl_mem), &rx_buf);
  err |= clSetKernelArg(kernel_multiply_, 1, sizeof(cl_mem), &ref_buf);
  err |= clSetKernelArg(kernel_multiply_, 2, sizeof(cl_mem), &buf_dc_);
  err |= clSetKernelArg(kernel_multiply_, 3, sizeof(int), &n_pts);

  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpWithGPURef: clSetKernelArg failed");

  size_t global = static_cast<size_t>(total);
  err = clEnqueueNDRangeKernel(queue_, kernel_multiply_, 1, nullptr, &global,
                                nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpWithGPURef: enqueue failed");

  std::vector<std::complex<float>> result(total);
  err = clEnqueueReadBuffer(queue_, buf_dc_, CL_TRUE, 0, rx_bytes,
                             result.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("DechirpWithGPURef: read failed");

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation (OPT-1: also create cached kernel objects)
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorOpenCL::CompileKernels() {
  cl_int err;

  // --- dechirp_multiply ---
  {
    std::string source = heterodyne::LoadKernelFile("dechirp_multiply.cl");
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();

    program_multiply_ = clCreateProgramWithSource(
        context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("CompileKernels: clCreateProgram(multiply) failed");

    err = clBuildProgram(program_multiply_, 1, &device_,
                          "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program_multiply_, device_,
          CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_multiply_, device_,
          CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      clReleaseProgram(program_multiply_);
      program_multiply_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: dechirp_multiply build failed:\n"
          + std::string(log.data()));
    }

    // OPT-1: Create kernel once
    kernel_multiply_ = clCreateKernel(program_multiply_, "dechirp_multiply", &err);
    if (err != CL_SUCCESS) {
      clReleaseProgram(program_multiply_);
      program_multiply_ = nullptr;
      throw std::runtime_error("CompileKernels: clCreateKernel(multiply) failed");
    }
  }

  // --- dechirp_correct ---
  {
    std::string source = heterodyne::LoadKernelFile("dechirp_correct.cl");
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();

    program_correct_ = clCreateProgramWithSource(
        context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
      clReleaseKernel(kernel_multiply_);
      kernel_multiply_ = nullptr;
      clReleaseProgram(program_multiply_);
      program_multiply_ = nullptr;
      throw std::runtime_error("CompileKernels: clCreateProgram(correct) failed");
    }

    err = clBuildProgram(program_correct_, 1, &device_,
                          "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program_correct_, device_,
          CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_correct_, device_,
          CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      clReleaseKernel(kernel_multiply_);
      kernel_multiply_ = nullptr;
      clReleaseProgram(program_multiply_);
      program_multiply_ = nullptr;
      clReleaseProgram(program_correct_);
      program_correct_ = nullptr;
      throw std::runtime_error(
          "CompileKernels: dechirp_correct build failed:\n"
          + std::string(log.data()));
    }

    // OPT-1: Create kernel once
    kernel_correct_ = clCreateKernel(program_correct_, "dechirp_correct", &err);
    if (err != CL_SUCCESS) {
      clReleaseKernel(kernel_multiply_);
      kernel_multiply_ = nullptr;
      clReleaseProgram(program_multiply_);
      program_multiply_ = nullptr;
      clReleaseProgram(program_correct_);
      program_correct_ = nullptr;
      throw std::runtime_error("CompileKernels: clCreateKernel(correct) failed");
    }
  }
}

void HeterodyneProcessorOpenCL::ReleaseGpuResources() {
  // OPT-1: Release cached kernels
  if (kernel_multiply_) { clReleaseKernel(kernel_multiply_); kernel_multiply_ = nullptr; }
  if (kernel_correct_)  { clReleaseKernel(kernel_correct_);  kernel_correct_ = nullptr; }

  // Release programs
  if (program_multiply_) { clReleaseProgram(program_multiply_); program_multiply_ = nullptr; }
  if (program_correct_)  { clReleaseProgram(program_correct_);  program_correct_ = nullptr; }

  // OPT-2: Release cached buffers
  if (buf_rx_)   { clReleaseMemObject(buf_rx_);   buf_rx_ = nullptr; }
  if (buf_ref_)  { clReleaseMemObject(buf_ref_);  buf_ref_ = nullptr; }
  if (buf_dc_)   { clReleaseMemObject(buf_dc_);   buf_dc_ = nullptr; }
  if (buf_corr_) { clReleaseMemObject(buf_corr_); buf_corr_ = nullptr; }
  if (buf_freq_) { clReleaseMemObject(buf_freq_); buf_freq_ = nullptr; }

  cached_total_ = 0;
  cached_samples_ = 0;
  cached_antennas_ = 0;
}

}  // namespace drv_gpu_lib
