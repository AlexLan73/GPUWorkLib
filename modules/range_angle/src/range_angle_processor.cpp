/**
 * @file range_angle_processor.cpp
 * @brief RangeAngleProcessor — Facade implementation (Ref03 Layer 6)
 *
 * Full pipeline implementation:
 *   1. DechirpWindowOp  — dechirp + Hamming window + zero-pad (hiprtc)
 *   2. RangeFftOp       — batched 1-D range FFT in-place (hipFFT)
 *   3. TransposeOp      — [n_ant × n_range_bins] → [n_range_bins × n_ant] (hiprtc)
 *   4. BeamFftOp        — 2-D spatial FFT + fftshift2d in-place (hipFFT + hiprtc)
 *   5. PeakSearchOp     — |.|^2 + CPU argmax + parabolic interpolation (hiprtc)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "range_angle_processor.hpp"
#include "kernels/range_angle_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>

namespace range_angle {

// ============================================================================
// Kernel names registered with GpuContext
// ============================================================================

static const std::vector<std::string> kKernelNames = {
  "hamming_window_kernel",
  "gen_ref_kernel",
  "dechirp_window_kernel",
  "transpose_complex_kernel",
  "fftshift2d_kernel",
  "magnitude_sq_kernel",
};

// ============================================================================
// Helpers
// ============================================================================

static uint32_t next_power_of_two(uint32_t n) {
  if (n == 0) return 1;
  uint32_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

static constexpr float kSpeedOfLight = 3e8f;

// ============================================================================
// Constructor / Destructor / Move
// ============================================================================

RangeAngleProcessor::RangeAngleProcessor(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "RangeAngle", "modules/range_angle/kernels") {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(backend ? backend->GetDeviceIndex() : 0,
            "RangeAngle", "RangeAngleProcessor created");
}

RangeAngleProcessor::~RangeAngleProcessor() {
  // Release Ops before GpuContext
  dechirp_op_.Release();
  range_fft_op_.Release();
  transpose_op_.Release();
  beam_fft_op_.Release();
  peak_op_.Release();
}

RangeAngleProcessor::RangeAngleProcessor(RangeAngleProcessor&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , dechirp_op_(std::move(other.dechirp_op_))
    , range_fft_op_(std::move(other.range_fft_op_))
    , transpose_op_(std::move(other.transpose_op_))
    , beam_fft_op_(std::move(other.beam_fft_op_))
    , peak_op_(std::move(other.peak_op_))
    , params_(other.params_)
    , compiled_(other.compiled_)
    , ref_built_(other.ref_built_) {
  other.compiled_  = false;
  other.ref_built_ = false;
}

RangeAngleProcessor& RangeAngleProcessor::operator=(RangeAngleProcessor&& other) noexcept {
  if (this != &other) {
    dechirp_op_.Release();
    range_fft_op_.Release();
    transpose_op_.Release();
    beam_fft_op_.Release();
    peak_op_.Release();

    ctx_          = std::move(other.ctx_);
    dechirp_op_   = std::move(other.dechirp_op_);
    range_fft_op_ = std::move(other.range_fft_op_);
    transpose_op_ = std::move(other.transpose_op_);
    beam_fft_op_  = std::move(other.beam_fft_op_);
    peak_op_      = std::move(other.peak_op_);
    params_       = other.params_;
    compiled_     = other.compiled_;
    ref_built_    = other.ref_built_;

    other.compiled_  = false;
    other.ref_built_ = false;
  }
  return *this;
}

// ============================================================================
// Configuration
// ============================================================================

void RangeAngleProcessor::SetParams(const RangeAngleParams& params) {
  params_    = params;
  compiled_  = false;
  ref_built_ = false;
  ComputeDerivedParams();
}

void RangeAngleProcessor::ComputeDerivedParams() {
  // nfft_range: next power of 2 if auto
  if (params_.nfft_range == 0) {
    params_.nfft_range = next_power_of_two(params_.n_samples);
  }

  // n_range_bins: use full one-sided FFT
  // bandwidth covers df = fs/n_samples per bin
  // useful bins = bandwidth / df = n_samples (but we keep all nfft/2 bins)
  params_.n_range_bins = params_.nfft_range / 2u;

  // range_res_m = c / (2 * bandwidth)
  float bw = params_.get_bandwidth();
  if (bw > 0.f) {
    params_.range_res_m = kSpeedOfLight / (2.f * bw);
  }
}

// ============================================================================
// Lazy kernel compilation
// ============================================================================

void RangeAngleProcessor::EnsureCompiled() {
  if (compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = ctx_.backend() ? ctx_.backend()->GetDeviceIndex() : 0;

  // Build extra defines: warp size from architecture
  std::vector<std::string> defines;
  if (ctx_.warp_size() == 64) {
    defines.push_back("-DWARP_SIZE=64");
  } else {
    defines.push_back("-DWARP_SIZE=32");
  }
  defines.push_back("-DBLOCK_SIZE=256");

  ctx_.CompileModule(
      kernels::GetRangeAngleKernelSource(),
      kKernelNames,
      defines);

  // Initialize Ops
  dechirp_op_.Initialize(ctx_);
  range_fft_op_.Initialize(ctx_);
  transpose_op_.Initialize(ctx_);
  beam_fft_op_.Initialize(ctx_);
  peak_op_.Initialize(ctx_);

  // Init Hamming window (GPU)
  dechirp_op_.InitWindow(params_.n_samples);

  // Create hipFFT plans
  range_fft_op_.InitPlan(params_.nfft_range, params_.get_n_antennas());
  beam_fft_op_.InitPlan(params_.n_ant_az, params_.n_ant_el, params_.n_range_bins);

  compiled_ = true;

  con.Print(gpu_id, "RangeAngle",
      "Kernels compiled OK. nfft_range=" + std::to_string(params_.nfft_range) +
      " n_range_bins=" + std::to_string(params_.n_range_bins) +
      " n_ant=" + std::to_string(params_.get_n_antennas()));
}

// ============================================================================
// Reference signal build (GPU-side LFM reference generation)
// ============================================================================

void RangeAngleProcessor::BuildRefSignal() {
  if (ref_built_) return;

  size_t ref_bytes = static_cast<size_t>(params_.n_samples) * 2u * sizeof(float);
  void* d_ref = ctx_.RequireShared(shared_buf::kRef, ref_bytes);

  // Generate conjugate LFM chirp on GPU
  // gen_ref_kernel: Grid(ceil(n_samples/256), 1, 1)  Block(256, 1, 1)
  uint32_t n_samples   = params_.n_samples;
  float    chirp_rate  = params_.get_chirp_rate();
  float    sample_rate = params_.sample_rate;

  uint32_t grid_x = (n_samples + 255u) / 256u;
  void* args[] = { &d_ref, &n_samples, &chirp_rate, &sample_rate };

  hipError_t err = hipModuleLaunchKernel(
      ctx_.GetKernel("gen_ref_kernel"),
      grid_x, 1, 1,
      256, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("RangeAngleProcessor::BuildRefSignal gen_ref_kernel: ") +
        hipGetErrorString(err));
  }

  ref_built_ = true;
}

// ============================================================================
// Data upload
// ============================================================================

void RangeAngleProcessor::UploadData(const std::complex<float>* data, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  void* gpu_buf = ctx_.RequireShared(shared_buf::kInput, bytes);

  hipError_t err = hipMemcpyAsync(
      gpu_buf, data, bytes,
      hipMemcpyHostToDevice, ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("RangeAngle UploadData: ") + hipGetErrorString(err));
  }
}

// ============================================================================
// Process — main pipeline
// ============================================================================

RangeAngleResult RangeAngleProcessor::Process(
    const std::vector<std::complex<float>>& data,
    bool download_result) {

  RangeAngleResult result;

  // Validate input size
  size_t expected = static_cast<size_t>(params_.get_n_antennas()) * params_.n_samples;
  if (data.size() != expected) {
    result.success       = false;
    result.error_message = "Input size mismatch: expected " +
                           std::to_string(expected) + " got " +
                           std::to_string(data.size());
    return result;
  }

  try {
    EnsureCompiled();
    BuildRefSignal();

    uint32_t n_ant   = params_.get_n_antennas();
    uint32_t nfft_r  = params_.nfft_range;
    uint32_t n_rbins = params_.n_range_bins;

    // Pre-allocate all shared buffers
    // kInput: raw IQ
    ctx_.RequireShared(shared_buf::kInput,
        static_cast<size_t>(n_ant) * params_.n_samples * 2u * sizeof(float));
    // kRef: LFM reference (already allocated in BuildRefSignal)
    // kDechirped: dechirp output with zero-pad (size = n_ant * nfft_r)
    ctx_.RequireShared(shared_buf::kDechirped,
        static_cast<size_t>(n_ant) * nfft_r * 2u * sizeof(float));
    // kTransposed: [n_rbins * n_ant] — same total elements as kDechirped
    //   but we use n_rbins * n_ant (n_rbins <= nfft_r/2)
    ctx_.RequireShared(shared_buf::kTransposed,
        static_cast<size_t>(n_rbins) * n_ant * 2u * sizeof(float));
    // kBeamFFT: shared with kTransposed (in-place), no separate alloc needed
    // kPowerCube: float power cube (allocated in PeakSearchOp::Execute)

    // Step 1: Upload data to kInput
    UploadData(data.data(), data.size());

    // Step 2: Dechirp + Hamming window → kDechirped (with zero-pad to nfft_r)
    dechirp_op_.Execute(n_ant, params_.n_samples, nfft_r);

    // Step 3: Range FFT in-place on kDechirped
    range_fft_op_.Execute();

    // Step 4: Transpose kDechirped[n_ant × nfft_r] → kTransposed[n_rbins × n_ant]
    //   We only transpose n_rbins columns (one-sided spectrum)
    transpose_op_.Execute(n_ant, n_rbins);

    // Step 5: 2-D beam FFT in-place on kTransposed + fftshift2d
    beam_fft_op_.Execute();

    // Step 6: |.|^2 + peak search
    peak_op_.Execute(params_, download_result);

    result = peak_op_.GetResult();
    result.n_range_bins   = n_rbins;
    result.n_ant_az       = params_.n_ant_az;
    result.n_ant_el       = params_.n_ant_el;
    result.gpu_power_cube = ctx_.GetShared(shared_buf::kPowerCube);

  } catch (const std::exception& ex) {
    result.success       = false;
    result.error_message = ex.what();
  }

  return result;
}

}  // namespace range_angle

#endif  // ENABLE_ROCM
