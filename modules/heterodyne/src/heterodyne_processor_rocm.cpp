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
#include "services/kernel_cache_service.hpp"
#include "backends/rocm/rocm_backend.hpp"

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
// Profiling helper: hipEvent → ROCmProfilingData (kind: 0=kernel, 1=copy)
// ════════════════════════════════════════════════════════════════════════════

static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind, const char* op_string = "")
{
  hipEventSynchronize(ev_end);
  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);
  hipEventDestroy(ev_start);
  hipEventDestroy(ev_end);
  drv_gpu_lib::ROCmProfilingData d{};
  uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
  d.start_ns    = 0;
  d.end_ns      = elapsed_ns;
  d.complete_ns = elapsed_ns;
  d.kind        = kind;
  d.op_string   = op_string;
  return d;
}

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
    , module_(other.module_)
    , kernel_multiply_(other.kernel_multiply_)
    , kernel_correct_(other.kernel_correct_)
    , kernels_compiled_(other.kernels_compiled_)
    , kernel_cache_(std::move(other.kernel_cache_))
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
  other.module_ = nullptr;
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
    module_ = other.module_;
    kernel_multiply_ = other.kernel_multiply_;
    kernel_correct_ = other.kernel_correct_;
    kernels_compiled_ = other.kernels_compiled_;
    kernel_cache_ = std::move(other.kernel_cache_);
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
    other.module_ = nullptr;
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
    if (buf_rx_)   { (void)hipFree(buf_rx_);   buf_rx_ = nullptr; }
    if (buf_dc_)   { (void)hipFree(buf_dc_);   buf_dc_ = nullptr; }
    if (buf_corr_) { (void)hipFree(buf_corr_); buf_corr_ = nullptr; }

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
    if (buf_ref_) { (void)hipFree(buf_ref_); buf_ref_ = nullptr; }

    size_t ref_bytes = static_cast<size_t>(num_samples) * sizeof(std::complex<float>);
    err = hipMalloc(&buf_ref_, ref_bytes);
    if (err != hipSuccess)
      throw std::runtime_error("EnsureBuffers: ref alloc failed: " +
          std::string(hipGetErrorString(err)));

    cached_samples_ = num_samples;
  }

  // Freq/phase_step buffer (antennas)
  if (antennas != cached_antennas_) {
    if (buf_freq_) { (void)hipFree(buf_freq_); buf_freq_ = nullptr; }

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
    const HeterodyneParams& params,
    HeterodyneROCmProfEvents* prof_events) {

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

  // Upload rx
  hipEvent_t ev_rx_s = nullptr, ev_rx_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_rx_s); hipEventCreate(&ev_rx_e);
    hipEventRecord(ev_rx_s, stream_);
  }
  err = hipMemcpyHtoDAsync(buf_rx_, const_cast<std::complex<float>*>(rx_data.data()),
                            rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: rx upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_rx_e, stream_);

  // Upload ref
  hipEvent_t ev_ref_s = nullptr, ev_ref_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_ref_s); hipEventCreate(&ev_ref_e);
    hipEventRecord(ev_ref_s, stream_);
  }
  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: ref upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ref_e, stream_);

  // OPT-1: Use cached kernel, OPT-5: 2D grid (x=sample, y=antenna)
  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  void* args[] = { &buf_rx_, &buf_ref_, &buf_dc_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }
  err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, stream_);

  // Download result
  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, stream_);
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, stream_);

  hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Upload_Rx",
        MakeROCmDataFromEvents(ev_rx_s,  ev_rx_e,  1, "H2D")});
    prof_events->push_back({"Upload_Ref",
        MakeROCmDataFromEvents(ev_ref_s, ev_ref_e, 1, "H2D")});
    prof_events->push_back({"Kernel_Multiply",
        MakeROCmDataFromEvents(ev_k_s,   ev_k_e,   0, "dechirp_multiply")});
    prof_events->push_back({"Download",
        MakeROCmDataFromEvents(ev_dl_s,  ev_dl_e,  1, "D2H")});
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: Correct — frequency correction
// OPT-6: phase_step precomputed on CPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::Correct(
    const std::vector<std::complex<float>>& dc_data,
    const std::vector<float>& f_beat_hz,
    const HeterodyneParams& params,
    HeterodyneROCmProfEvents* prof_events) {

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

  // Upload DC data
  hipEvent_t ev_dc_s = nullptr, ev_dc_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dc_s); hipEventCreate(&ev_dc_e);
    hipEventRecord(ev_dc_s, stream_);
  }
  err = hipMemcpyHtoDAsync(buf_dc_, const_cast<std::complex<float>*>(dc_data.data()),
                            data_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: dc upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dc_e, stream_);

  // OPT-6: Precompute phase_step on CPU: phase_step[ant] = -2*pi*f_beat/fs
  std::vector<float> phase_step(params.num_antennas);
  for (int i = 0; i < params.num_antennas; ++i) {
    phase_step[i] = static_cast<float>(-2.0 * M_PI * f_beat_hz[i] / params.sample_rate);
  }

  size_t freq_bytes = static_cast<size_t>(params.num_antennas) * sizeof(float);
  hipEvent_t ev_ps_s = nullptr, ev_ps_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_ps_s); hipEventCreate(&ev_ps_e);
    hipEventRecord(ev_ps_s, stream_);
  }
  err = hipMemcpyHtoDAsync(buf_freq_, phase_step.data(), freq_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: phase_step upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ps_e, stream_);

  // OPT-1: Use cached kernel, OPT-5: 2D grid (x=sample, y=antenna)
  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  void* args[] = { &buf_dc_, &buf_corr_, &buf_freq_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }
  err = hipModuleLaunchKernel(
      kernel_correct_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, stream_);

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, stream_);
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_corr_, data_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, stream_);

  hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Upload_DC",
        MakeROCmDataFromEvents(ev_dc_s, ev_dc_e, 1, "H2D")});
    prof_events->push_back({"Upload_PhaseStep",
        MakeROCmDataFromEvents(ev_ps_s, ev_ps_e, 1, "H2D")});
    prof_events->push_back({"Kernel_Correct",
        MakeROCmDataFromEvents(ev_k_s,  ev_k_e,  0, "dechirp_correct")});
    prof_events->push_back({"Download",
        MakeROCmDataFromEvents(ev_dl_s, ev_dl_e, 1, "D2H")});
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 5: DechirpFromGPU — external GPU buffer (void* = hipDeviceptr_t)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::DechirpFromGPU(
    void* rx_gpu_ptr,
    const std::vector<std::complex<float>>& ref_data,
    const HeterodyneParams& params,
    HeterodyneROCmProfEvents* prof_events) {

  if (!rx_gpu_ptr) {
    throw std::runtime_error("DechirpFromGPU: rx_gpu_ptr is null");
  }
  if (static_cast<int>(ref_data.size()) != params.num_samples) {
    throw std::runtime_error(
        "DechirpFromGPU: ref_data size mismatch: expected "
        + std::to_string(params.num_samples) + ", got "
        + std::to_string(ref_data.size()));
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes  = static_cast<size_t>(total) * sizeof(std::complex<float>);
  size_t ref_bytes = static_cast<size_t>(params.num_samples) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  hipError_t err;

  // Upload ref to cached buffer
  hipEvent_t ev_ref_s = nullptr, ev_ref_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_ref_s); hipEventCreate(&ev_ref_e);
    hipEventRecord(ev_ref_s, stream_);
  }
  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: ref upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ref_e, stream_);

  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  // Use external rx buffer directly (DO NOT free — caller owns it)
  void* args[] = { &rx_gpu_ptr, &buf_ref_, &buf_dc_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }
  err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, stream_);

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, stream_);
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, stream_);

  hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Upload_Ref",
        MakeROCmDataFromEvents(ev_ref_s, ev_ref_e, 1, "H2D")});
    prof_events->push_back({"Kernel_Multiply",
        MakeROCmDataFromEvents(ev_k_s,   ev_k_e,   0, "dechirp_multiply")});
    prof_events->push_back({"Download",
        MakeROCmDataFromEvents(ev_dl_s,  ev_dl_e,  1, "D2H")});
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 6: DechirpWithGPURef — OPT-3 both inputs on GPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> HeterodyneProcessorROCm::DechirpWithGPURef(
    void* rx_gpu_ptr,
    void* ref_gpu_ptr,
    const HeterodyneParams& params,
    HeterodyneROCmProfEvents* prof_events) {

  if (!rx_gpu_ptr || !ref_gpu_ptr) {
    throw std::runtime_error("DechirpWithGPURef: null GPU pointer");
  }

  int total = params.num_antennas * params.num_samples;
  size_t rx_bytes = static_cast<size_t>(total) * sizeof(std::complex<float>);

  EnsureBuffers(total, params.num_samples);

  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  void* args[] = { &rx_gpu_ptr, &ref_gpu_ptr, &buf_dc_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }
  hipError_t err = hipModuleLaunchKernel(
      kernel_multiply_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, stream_);

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, stream_);
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, stream_);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, stream_);

  hipStreamSynchronize(stream_);

  if (prof_events) {
    prof_events->push_back({"Kernel_Multiply",
        MakeROCmDataFromEvents(ev_k_s,  ev_k_e,  0, "dechirp_multiply")});
    prof_events->push_back({"Download",
        MakeROCmDataFromEvents(ev_dl_s, ev_dl_e, 1, "D2H")});
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: CompileKernels — hiprtc compilation (OPT-1: cached)
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = ConsoleOutput::GetInstance();

  // Инициализация кеша (lazy, один раз)
  if (!kernel_cache_) {
    kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
        "modules/heterodyne/kernels", drv_gpu_lib::BackendType::ROCm);
  }

  static constexpr const char* kKernelName = "heterodyne_kernels";

  // Загрузка модуля + извлечение двух функций
  auto loadModuleAndFunctions = [&](const void* data, size_t size) {
    hipError_t hipErr = hipModuleLoadData(&module_, data);
    if (hipErr != hipSuccess)
      throw std::runtime_error(
          "CompileKernels: hipModuleLoadData failed: " +
          std::string(hipGetErrorString(hipErr)));

    auto getFunc = [&](hipFunction_t* fn, const char* name) {
      hipErr = hipModuleGetFunction(fn, module_, name);
      if (hipErr != hipSuccess) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        kernel_multiply_ = nullptr;
        kernel_correct_ = nullptr;
        throw std::runtime_error(
            std::string("CompileKernels: hipModuleGetFunction(") +
            name + ") failed: " + hipGetErrorString(hipErr));
      }
    };

    getFunc(&kernel_multiply_, "dechirp_multiply");
    getFunc(&kernel_correct_,  "dechirp_correct");
  };

  // Шаг 1: попробовать загрузить из дискового кеша (~1-5 мс)
  if (kernel_cache_) {
    auto entry = kernel_cache_->Load(kKernelName);
    if (entry && entry->has_binary()) {
      loadModuleAndFunctions(entry->binary.data(), entry->binary.size());
      kernels_compiled_ = true;
      con.Print(0, "Heterodyne[ROCm]",
          "HIP kernels loaded from cache (multiply, correct)");
      return;
    }
  }

  // Шаг 2: компиляция из исходника через hiprtc (один модуль на оба ядра)
  const char* source = kernels::GetHeterodyneKernelSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "heterodyne_kernels.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "CompileKernels: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  // Получить целевую архитектуру GPU
  std::string arch_name;
  try {
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name = "";
  }

  int warp_size = 32;
  if (arch_name.find("gfx9") == 0) {
    warp_size = 64;
  }

  std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);

  std::vector<const char*> opts = { "-O3", warp_define.c_str() };
  if (!arch_flag.empty()) {
    opts.push_back(arch_flag.c_str());
  }

  rtcResult = hiprtcCompileProgram(prog,
      static_cast<int>(opts.size()), opts.data());
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    con.PrintError(0, "Heterodyne[ROCm]", "Kernel compile log:\n" + log);
    (void)hiprtcDestroyProgram(&prog);
    throw std::runtime_error("CompileKernels: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  (void)hiprtcDestroyProgram(&prog);

  loadModuleAndFunctions(code.data(), code.size());

  // Шаг 3: сохранить бинарь в кеш
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(kKernelName, std::string(source),
                          binary, arch_name, "heterodyne");
    } catch (...) {
      // Не критично
    }
  }

  kernels_compiled_ = true;
  con.Print(0, "Heterodyne[ROCm]",
      "HIP kernels compiled (multiply+correct, arch=" + arch_name + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 8: ReleaseGpuResources — cleanup
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::ReleaseGpuResources() {
  // OPT-1/10: Release single cached module (both kernels invalidated)
  if (module_) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    kernel_multiply_ = nullptr;
    kernel_correct_ = nullptr;
  }
  kernels_compiled_ = false;

  // OPT-2: Release cached buffers
  if (buf_rx_)   { (void)hipFree(buf_rx_);   buf_rx_ = nullptr; }
  if (buf_ref_)  { (void)hipFree(buf_ref_);  buf_ref_ = nullptr; }
  if (buf_dc_)   { (void)hipFree(buf_dc_);   buf_dc_ = nullptr; }
  if (buf_corr_) { (void)hipFree(buf_corr_); buf_corr_ = nullptr; }
  if (buf_freq_) { (void)hipFree(buf_freq_); buf_freq_ = nullptr; }

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
