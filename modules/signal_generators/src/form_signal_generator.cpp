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
// OpenCL Kernel Source: Philox PRNG + FormSignal (getX)
// ════════════════════════════════════════════════════════════════════════════

static const char* FORM_SIGNAL_KERNEL_SOURCE = R"CL(

// ─────────────────────────────────────────────────────────────────────────
// Philox-2x32-10: counter-based PRNG (from NoiseGenerator)
// ─────────────────────────────────────────────────────────────────────────

uint2 philox2x32_round(uint2 ctr, uint key) {
    const uint PHILOX_M = 0xD2511F53u;
    uint hi = mul_hi(ctr.x, PHILOX_M);
    uint lo = ctr.x * PHILOX_M;
    return (uint2)(hi ^ key ^ ctr.y, lo);
}

uint2 philox2x32_10(uint2 ctr, uint key) {
    const uint PHILOX_BUMP = 0x9E3779B9u;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key);
    return ctr;
}

/// Uniform [0, 1) from Philox
float philox_uniform(uint id, uint seed) {
    uint2 ctr = (uint2)(id, seed);
    uint2 rnd = philox2x32_10(ctr, 0xAB12CD34u);
    return (float)(rnd.x) / 4294967296.0f;
}

// ─────────────────────────────────────────────────────────────────────────
// FormSignal kernel: getX formula
// ─────────────────────────────────────────────────────────────────────────
//
// X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))
//   + an*norm*(randn_re + j*randn_im)
// X = 0  if t < 0 or t > ti - dt
//
// gid = antenna_id * points + sample_id
//

__kernel void generate_form_signal(
    __global float2* output,
    const uint antennas,
    const uint points,
    const float dt,
    const float ti,
    const float f0,
    const float amplitude,
    const float noise_amplitude,
    const float phase_offset,
    const float fdev,
    const float norm_val,
    const float tau_base,
    const float tau_step,
    const float tau_min,
    const float tau_max,
    const uint tau_seed,
    const uint noise_seed,
    const uint tau_mode)
{
    const uint gid = get_global_id(0);
    const uint total = antennas * points;
    if (gid >= total) return;

    const uint antenna_id = gid / points;
    const uint sample_id  = gid % points;

    // ── Tau per-channel ──
    float tau;
    if (tau_mode == 0u) {
        // FIXED
        tau = tau_base;
    } else if (tau_mode == 1u) {
        // LINEAR
        tau = tau_base + (float)antenna_id * tau_step;
    } else {
        // RANDOM
        float u = philox_uniform(antenna_id, tau_seed);
        tau = tau_min + u * (tau_max - tau_min);
    }

    // ── Time ──
    float t = (float)sample_id * dt + tau;

    // ── Window: X=0 if outside [0, ti-dt] ──
    int in_window = (t >= 0.0f && t <= ti - dt) ? 1 : 0;

    if (in_window == 0) {
        output[gid] = (float2)(0.0f, 0.0f);
        return;
    }

    // ── Signal phase: 2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi ──
    float t_centered = t - ti * 0.5f;
    float phase = 2.0f * M_PI_F * f0 * t
                + M_PI_F * fdev / ti * (t_centered * t_centered)
                + phase_offset;

    float sig_re = amplitude * norm_val * cos(phase);
    float sig_im = amplitude * norm_val * sin(phase);

    // ── Noise (Philox + Box-Muller) ──
    float noise_re = 0.0f;
    float noise_im = 0.0f;

    if (noise_amplitude > 0.0f) {
        uint2 n_ctr = (uint2)(gid, noise_seed);
        uint2 n_rnd = philox2x32_10(n_ctr, 0xCD9E8D57u);

        float u1 = (float)(n_rnd.x) / 4294967296.0f + 1e-10f;
        float u2 = (float)(n_rnd.y) / 4294967296.0f;

        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * M_PI_F * u2;

        noise_re = noise_amplitude * norm_val * r * cos(theta);
        noise_im = noise_amplitude * norm_val * r * sin(theta);
    }

    output[gid] = (float2)(sig_re + noise_re, sig_im + noise_im);
}

)CL";

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

  size_t local_size = 256;
  size_t global_size =
      ((total_points + local_size - 1) / local_size) * local_size;

  err = clEnqueueNDRangeKernel(
      queue_, k, 1, nullptr,
      &global_size, &local_size, 0, nullptr, nullptr);
  clReleaseKernel(k);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output);
    throw std::runtime_error(
        "FormSignalGenerator::Generate: enqueue failed: "
        + std::to_string(err));
  }

  clFinish(queue_);

  return drv_gpu_lib::InputData<cl_mem>{
      .antenna_count = params_.antennas,
      .n_point = params_.points,
      .data = output,
      .gpu_memory_bytes = buffer_size,
      .sample_rate = static_cast<float>(params_.fs)
  };
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
  cl_int err;
  size_t source_len = strlen(FORM_SIGNAL_KERNEL_SOURCE);

  program_ = clCreateProgramWithSource(
      context_, 1, &FORM_SIGNAL_KERNEL_SOURCE, &source_len, &err);
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
