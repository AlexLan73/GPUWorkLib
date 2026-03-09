/**
 * @file form_signal_generator.cpp
 * @brief Реализация FormSignalGenerator — мультиканальный getX на GPU
 *
 * Kernel: Philox-2x32 PRNG + Box-Muller + getX formula
 * Один проход по памяти: сигнал + шум в одном kernel.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#include "generators/form_signal_generator.hpp"
#include "kernel_loader.hpp"
#include "prof_utils.hpp"
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

FormSignalGenerator::FormSignalGenerator(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "FormSignalGenerator: backend is null or not initialized");
  }

  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  CompileKernel();
}

FormSignalGenerator::~FormSignalGenerator() {
  ReleaseGpuResources();
}

FormSignalGenerator::FormSignalGenerator(FormSignalGenerator&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_) {
  other.program_ = nullptr;
}

FormSignalGenerator& FormSignalGenerator::operator=(
    FormSignalGenerator&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    params_ = other.params_;
    context_ = other.context_;
    queue_ = other.queue_;
    device_ = other.device_;
    program_ = other.program_;
    other.program_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU генерация
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem> FormSignalGenerator::GenerateInputData() {
  return GenerateInputData(nullptr);
}

drv_gpu_lib::InputData<cl_mem>
FormSignalGenerator::GenerateInputData(ProfEvents* prof_events) {
  size_t total_points = GetTotalSamples();
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  cl_int err;
  cl_mem output = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FormSignalGenerator::Generate: clCreateBuffer failed: "
        + std::to_string(err));
  }

  cl_kernel k = clCreateKernel(program_, "generate_form_signal", &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(output);
    throw std::runtime_error(
        "FormSignalGenerator::Generate: clCreateKernel failed: "
        + std::to_string(err));
  }

  // Prepare args
  uint32_t ant = params_.antennas;
  uint32_t pts = params_.points;
  float dt = static_cast<float>(params_.GetDt());
  float ti = static_cast<float>(params_.GetDuration());
  float f0 = static_cast<float>(params_.f0);
  float amp = static_cast<float>(params_.amplitude);
  float an = static_cast<float>(params_.noise_amplitude);
  float phi = static_cast<float>(params_.phase);
  float fdev = static_cast<float>(params_.fdev);
  float norm_val = static_cast<float>(params_.norm);
  float tau_base = static_cast<float>(params_.tau_base);
  float tau_step = static_cast<float>(params_.tau_step);
  float tau_min = static_cast<float>(params_.tau_min);
  float tau_max = static_cast<float>(params_.tau_max);
  uint32_t tau_seed = params_.tau_seed;

  uint32_t noise_seed = params_.noise_seed;
  if (noise_seed == 0 && an > 0.0f) {
    noise_seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
        & 0xFFFFFFFF);
  }

  uint32_t tau_mode = static_cast<uint32_t>(params_.GetTauMode());

  err  = clSetKernelArg(k, 0,  sizeof(cl_mem),   &output);
  err |= clSetKernelArg(k, 1,  sizeof(uint32_t), &ant);
  err |= clSetKernelArg(k, 2,  sizeof(uint32_t), &pts);
  err |= clSetKernelArg(k, 3,  sizeof(float),    &dt);
  err |= clSetKernelArg(k, 4,  sizeof(float),    &ti);
  err |= clSetKernelArg(k, 5,  sizeof(float),    &f0);
  err |= clSetKernelArg(k, 6,  sizeof(float),    &amp);
  err |= clSetKernelArg(k, 7,  sizeof(float),    &an);
  err |= clSetKernelArg(k, 8,  sizeof(float),    &phi);
  err |= clSetKernelArg(k, 9,  sizeof(float),    &fdev);
  err |= clSetKernelArg(k, 10, sizeof(float),    &norm_val);
  err |= clSetKernelArg(k, 11, sizeof(float),    &tau_base);
  err |= clSetKernelArg(k, 12, sizeof(float),    &tau_step);
  err |= clSetKernelArg(k, 13, sizeof(float),    &tau_min);
  err |= clSetKernelArg(k, 14, sizeof(float),    &tau_max);
  err |= clSetKernelArg(k, 15, sizeof(uint32_t), &tau_seed);
  err |= clSetKernelArg(k, 16, sizeof(uint32_t), &noise_seed);
  err |= clSetKernelArg(k, 17, sizeof(uint32_t), &tau_mode);

  if (err != CL_SUCCESS) {
    clReleaseKernel(k);
    clReleaseMemObject(output);
    throw std::runtime_error(
        "FormSignalGenerator::Generate: clSetKernelArg failed");
  }

  // 2D grid: dim0 = samples, dim1 = antennas (eliminates div/mod in kernel)
  size_t local_size[2]  = { 256, 1 };
  size_t global_size[2] = {
      ((static_cast<size_t>(params_.points) + 255) / 256) * 256,
      static_cast<size_t>(params_.antennas)
  };

  cl_event ev_kernel = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, k, 2, nullptr,
      global_size, local_size, 0, nullptr, prof_events ? &ev_kernel : nullptr);
  clReleaseKernel(k);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output);
    throw std::runtime_error(
        "FormSignalGenerator::Generate: enqueue failed: "
        + std::to_string(err));
  }

  CollectOrRelease(ev_kernel, "Kernel", prof_events);

  clFinish(queue_);

  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count = params_.antennas;
  result.n_point       = params_.points;
  result.data          = output;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate   = static_cast<float>(params_.fs);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU генерация (по каналам)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
FormSignalGenerator::GenerateToCpu() {
  // Генерация на GPU → read back → split по каналам
  auto input = GenerateInputData();
  cl_mem gpu_buf = input.data;

  size_t total = GetTotalSamples();
  std::vector<std::complex<float>> flat(total);

  cl_int err = clEnqueueReadBuffer(
      queue_, gpu_buf, CL_TRUE, 0,
      total * sizeof(std::complex<float>),
      flat.data(), 0, nullptr, nullptr);
  clReleaseMemObject(gpu_buf);

  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FormSignalGenerator::GenerateToCpu: clEnqueueReadBuffer failed");
  }

  // Split flat → vector<vector<complex>>
  std::vector<std::vector<std::complex<float>>> result(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t offset = static_cast<size_t>(a) * params_.points;
    result[a].assign(
        flat.begin() + offset,
        flat.begin() + offset + params_.points);
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void FormSignalGenerator::CompileKernel() {
  // Load kernel from .cl files: prng.cl + form_signal.cl
  std::string source = LoadKernelWithPrng("form_signal.cl");
  const char* src_ptr = source.c_str();
  size_t source_len = source.size();

  cl_int err;
  program_ = clCreateProgramWithSource(
      context_, 1, &src_ptr, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FormSignalGenerator: clCreateProgramWithSource failed");
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
        "FormSignalGenerator kernel build failed:\n"
        + std::string(log.data()));
  }
}

void FormSignalGenerator::ReleaseGpuResources() {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
}

} // namespace signal_gen
