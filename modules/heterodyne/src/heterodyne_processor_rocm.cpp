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
#include "rocm_profiling_helpers.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using fft_func_utils::MakeROCmDataFromEvents;

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor, Move
// ════════════════════════════════════════════════════════════════════════════

static const std::vector<std::string> kHetKernelNames = {
  "dechirp_multiply",
  "dechirp_correct"
};

HeterodyneProcessorROCm::HeterodyneProcessorROCm(IBackend* backend)
    : ctx_(backend, "Heterodyne", "modules/heterodyne/kernels")
    , backend_(backend) {
  EnsureCompiled();
}

HeterodyneProcessorROCm::~HeterodyneProcessorROCm() {
  ReleaseGpuResources();
}

HeterodyneProcessorROCm::HeterodyneProcessorROCm(
    HeterodyneProcessorROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , backend_(other.backend_)
    , compiled_(other.compiled_)
    , buf_rx_(other.buf_rx_)
    , buf_ref_(other.buf_ref_)
    , buf_dc_(other.buf_dc_)
    , buf_corr_(other.buf_corr_)
    , buf_freq_(other.buf_freq_)
    , cached_total_(other.cached_total_)
    , cached_samples_(other.cached_samples_)
    , cached_antennas_(other.cached_antennas_) {
  other.backend_ = nullptr;
  other.compiled_ = false;
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
    ctx_ = std::move(other.ctx_);
    backend_ = other.backend_;
    compiled_ = other.compiled_;
    buf_rx_ = other.buf_rx_;
    buf_ref_ = other.buf_ref_;
    buf_dc_ = other.buf_dc_;
    buf_corr_ = other.buf_corr_;
    buf_freq_ = other.buf_freq_;
    cached_total_ = other.cached_total_;
    cached_samples_ = other.cached_samples_;
    cached_antennas_ = other.cached_antennas_;
    other.backend_ = nullptr;
    other.compiled_ = false;
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
    hipEventRecord(ev_rx_s, ctx_.stream());
  }
  err = hipMemcpyHtoDAsync(buf_rx_, const_cast<std::complex<float>*>(rx_data.data()),
                            rx_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: rx upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_rx_e, ctx_.stream());

  // Upload ref
  hipEvent_t ev_ref_s = nullptr, ev_ref_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_ref_s); hipEventCreate(&ev_ref_e);
    hipEventRecord(ev_ref_s, ctx_.stream());
  }
  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: ref upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ref_e, ctx_.stream());

  // OPT-1: Use cached kernel, OPT-5: 2D grid (x=sample, y=antenna)
  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  void* args[] = { &buf_rx_, &buf_ref_, &buf_dc_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, ctx_.stream());
  }
  err = hipModuleLaunchKernel(
      ctx_.GetKernel("dechirp_multiply"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, ctx_.stream());

  // Download result
  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, ctx_.stream());
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Dechirp: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

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
    hipEventRecord(ev_dc_s, ctx_.stream());
  }
  err = hipMemcpyHtoDAsync(buf_dc_, const_cast<std::complex<float>*>(dc_data.data()),
                            data_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Correct: dc upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dc_e, ctx_.stream());

  // OPT-6: Precompute phase_step on CPU: phase_step[ant] = -2*pi*f_beat/fs
  std::vector<float> phase_step(params.num_antennas);
  for (int i = 0; i < params.num_antennas; ++i) {
    phase_step[i] = static_cast<float>(-2.0 * M_PI * f_beat_hz[i] / params.sample_rate);
  }

  size_t freq_bytes = static_cast<size_t>(params.num_antennas) * sizeof(float);
  hipEvent_t ev_ps_s = nullptr, ev_ps_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_ps_s); hipEventCreate(&ev_ps_e);
    hipEventRecord(ev_ps_s, ctx_.stream());
  }
  err = hipMemcpyHtoDAsync(buf_freq_, phase_step.data(), freq_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Correct: phase_step upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ps_e, ctx_.stream());

  // OPT-1: Use cached kernel, OPT-5: 2D grid (x=sample, y=antenna)
  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  void* args[] = { &buf_dc_, &buf_corr_, &buf_freq_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, ctx_.stream());
  }
  err = hipModuleLaunchKernel(
      ctx_.GetKernel("dechirp_correct"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("Correct: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, ctx_.stream());

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, ctx_.stream());
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_corr_, data_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("Correct: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

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
    hipEventRecord(ev_ref_s, ctx_.stream());
  }
  err = hipMemcpyHtoDAsync(buf_ref_, const_cast<std::complex<float>*>(ref_data.data()),
                            ref_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: ref upload failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_ref_e, ctx_.stream());

  int n_pts = params.num_samples;
  int n_ant = params.num_antennas;

  // Use external rx buffer directly (DO NOT free — caller owns it)
  void* args[] = { &rx_gpu_ptr, &buf_ref_, &buf_dc_, &n_pts, &n_ant };

  unsigned int grid_x = (static_cast<unsigned int>(n_pts) + kBlockSize - 1) / kBlockSize;
  unsigned int grid_y = static_cast<unsigned int>(n_ant);

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s); hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, ctx_.stream());
  }
  err = hipModuleLaunchKernel(
      ctx_.GetKernel("dechirp_multiply"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, ctx_.stream());

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, ctx_.stream());
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("DechirpFromGPU: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

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
    hipEventRecord(ev_k_s, ctx_.stream());
  }
  hipError_t err = hipModuleLaunchKernel(
      ctx_.GetKernel("dechirp_multiply"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_k_e, ctx_.stream());

  std::vector<std::complex<float>> result(total);
  hipEvent_t ev_dl_s = nullptr, ev_dl_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_dl_s); hipEventCreate(&ev_dl_e);
    hipEventRecord(ev_dl_s, ctx_.stream());
  }
  err = hipMemcpyDtoHAsync(result.data(), buf_dc_, rx_bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("DechirpWithGPURef: read failed: " +
        std::string(hipGetErrorString(err)));
  if (prof_events) hipEventRecord(ev_dl_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel_Multiply",
        MakeROCmDataFromEvents(ev_k_s,  ev_k_e,  0, "dechirp_multiply")});
    prof_events->push_back({"Download",
        MakeROCmDataFromEvents(ev_dl_s, ev_dl_e, 1, "D2H")});
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: Lazy compilation via GpuContext (Ref03)
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetHeterodyneKernelSource_rocm(), kHetKernelNames);
  compiled_ = true;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 8: ReleaseGpuResources — cleanup
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneProcessorROCm::ReleaseGpuResources() {
  // GpuContext manages kernel module — no manual hipModuleUnload
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
