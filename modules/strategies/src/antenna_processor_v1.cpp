#include "antenna_processor_v1.hpp"
#include "interface/i_backend.hpp"
#include "kernels/strategies_kernels_rocm.hpp"

#if ENABLE_ROCM
#include "statistics_processor.hpp"
#include "pipelines/all_maxima_pipeline_rocm.hpp"
#include "complex_to_mag_phase_rocm.hpp"
#include "types/mag_phase_types.hpp"
#include "services/console_output.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
} while(0)

#define HIPBLAS_CHECK(call) do { \
    hipblasStatus_t err = (call); \
    if (err != HIPBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("hipBLAS error: " + std::to_string(static_cast<int>(err))); \
} while(0)

#define HIPFFT_CHECK(call) do { \
    hipfftResult err = (call); \
    if (err != HIPFFT_SUCCESS) \
        throw std::runtime_error("hipFFT error: " + std::to_string(static_cast<int>(err))); \
} while(0)

namespace strategies {

static const std::vector<std::string> kStrategyKernelNames = {
  "hamming_pad_fused",
  "compute_magnitudes",
  "global_minmax",
  "one_max_no_phase"
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

AntennaProcessor_v1::AntennaProcessor_v1(
    drv_gpu_lib::IBackend* backend,
    const AntennaProcessorConfig& cfg)
    : backend_(backend), cfg_(cfg)
    , ctx_(backend, "Strategies", "modules/strategies/kernels")
{
  // Create streams
  HIP_CHECK(hipStreamCreate(&stream_main_));
  HIP_CHECK(hipStreamCreate(&stream_debug1_));
  HIP_CHECK(hipStreamCreate(&stream_debug2_));
  HIP_CHECK(hipStreamCreate(&stream_debug3_));
  // Extra streams for parallel post-FFT benchmark (3.6)
  HIP_CHECK(hipStreamCreate(&stream_bench3a_));
  HIP_CHECK(hipStreamCreate(&stream_bench3b_));
  HIP_CHECK(hipStreamCreate(&stream_bench3c_));

  // Create events (disable timing for lower overhead where not needed)
  HIP_CHECK(hipEventCreateWithFlags(&event_gemm_done_, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_fft_done_,  hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_c1_done_,   hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_c2_done_,   hipEventDisableTiming));

  // Create hipBLAS handle, bind to main stream
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle_, stream_main_));

  // Compute nFFT
  nFFT_ = compute_nFFT(cfg_.n_samples);

  // Allocate GPU buffers (including pre-allocated result buffers P3)
  allocate_buffers();

  // Compile kernels via GpuContext (Ref03: cache, arch detection, warp_size)
  ensure_compiled();

  // Create FFT plan
  create_fft_plan();

  // Create statistics processor
  stats_processor_ = std::make_unique<statistics::StatisticsProcessor>(backend_);

  // Create AllMaxima pipeline (uses stream_debug3_ for post-FFT work)
  all_maxima_pipeline_ = std::make_unique<antenna_fft::AllMaximaPipelineROCm>(
      stream_debug3_, backend_);

  // Create ComplexToMagPhaseROCm for zero-alloc magnitude conversion
  complex_to_mag_ = std::make_unique<fft_processor::ComplexToMagPhaseROCm>(backend_);

  // Default checkpoint: NullCheckpointSave
  checkpoint_ = std::make_unique<NullCheckpointSave>();
}

AntennaProcessor_v1::~AntennaProcessor_v1() {
  release_buffers();

  if (fft_plan_created_) {
    hipfftDestroy(fft_plan_);
  }

  if (hipblas_handle_) {
    hipblasDestroy(hipblas_handle_);
  }

  // GpuContext manages kernel module — no manual hipModuleUnload

  if (event_gemm_done_) hipEventDestroy(event_gemm_done_);
  if (event_fft_done_)  hipEventDestroy(event_fft_done_);
  if (event_c1_done_)   hipEventDestroy(event_c1_done_);
  if (event_c2_done_)   hipEventDestroy(event_c2_done_);

  if (stream_main_)    hipStreamDestroy(stream_main_);
  if (stream_debug1_)  hipStreamDestroy(stream_debug1_);
  if (stream_debug2_)  hipStreamDestroy(stream_debug2_);
  if (stream_debug3_)  hipStreamDestroy(stream_debug3_);
  if (stream_bench3a_) hipStreamDestroy(stream_bench3a_);
  if (stream_bench3b_) hipStreamDestroy(stream_bench3b_);
  if (stream_bench3c_) hipStreamDestroy(stream_bench3c_);
}

int AntennaProcessor_v1::gpu_id() const {
  return backend_ ? backend_->GetDeviceIndex() : -1;
}

void AntennaProcessor_v1::set_checkpoint_save(std::unique_ptr<ICheckpointSave> save) {
  checkpoint_ = std::move(save);
}

// ============================================================================
// Buffer management
// ============================================================================

uint32_t AntennaProcessor_v1::compute_nFFT(uint32_t n_samples) const {
  // Next power of 2 >= n_samples, then x2 for zero-padding
  uint32_t p = 1;
  while (p < n_samples) p <<= 1;
  return p * 2;  // repeat_count = 2
}

void AntennaProcessor_v1::allocate_buffers() {
  const size_t n_ant = cfg_.n_ant;
  const size_t n_samples = cfg_.n_samples;
  const size_t cf_size = sizeof(float) * 2;  // complex<float> = 8 bytes

  // d_X: [n_ant x n_samples] complex float (GEMM output)
  d_X_ = backend_->Allocate(n_ant * n_samples * cf_size);

  // d_fft_input_: [n_ant x nFFT] complex float (zero-padded)
  d_fft_input_ = backend_->Allocate(n_ant * nFFT_ * cf_size);

  // d_spectrum_: [n_ant x nFFT] complex float (FFT output, shared by consumers)
  d_spectrum_ = backend_->Allocate(n_ant * nFFT_ * cf_size);

  // d_magnitudes_: [n_ant x nFFT] float (|spectrum|)
  d_magnitudes_ = backend_->Allocate(n_ant * nFFT_ * sizeof(float));

  // d_hamming_window_: precomputed Hamming window [n_samples] float (P10)
  std::vector<float> hamming(n_samples);
  const float N_m1 = static_cast<float>(n_samples - 1);
  constexpr float two_pi = 2.0f * 3.14159265358979f;
  for (uint32_t i = 0; i < n_samples; ++i) {
    hamming[i] = 0.54f - 0.46f * std::cos(two_pi * static_cast<float>(i) / N_m1);
  }
  d_hamming_window_ = backend_->Allocate(n_samples * sizeof(float));
  HIP_CHECK(hipMemcpy(d_hamming_window_, hamming.data(),
                        n_samples * sizeof(float), hipMemcpyHostToDevice));

  // Pre-allocated result buffers (P3 — avoid Allocate/Free in hot path)
  d_one_max_results_ = backend_->Allocate(n_ant * sizeof(OneMaxParabolaLite));
  d_minmax_results_  = backend_->Allocate(n_ant * sizeof(MinMaxResult));
}

void AntennaProcessor_v1::release_buffers() {
  if (d_X_)              { backend_->Free(d_X_);              d_X_ = nullptr; }
  if (d_fft_input_)      { backend_->Free(d_fft_input_);      d_fft_input_ = nullptr; }
  if (d_spectrum_)       { backend_->Free(d_spectrum_);        d_spectrum_ = nullptr; }
  if (d_magnitudes_)     { backend_->Free(d_magnitudes_);      d_magnitudes_ = nullptr; }
  if (d_hamming_window_) { backend_->Free(d_hamming_window_);  d_hamming_window_ = nullptr; }
  if (d_one_max_results_) { backend_->Free(d_one_max_results_); d_one_max_results_ = nullptr; }
  if (d_minmax_results_)  { backend_->Free(d_minmax_results_);  d_minmax_results_ = nullptr; }
  if (d_W_managed_)       { backend_->Free(d_W_managed_);       d_W_managed_ = nullptr; }
}

// ============================================================================
// SetExternalWeights — upload user-provided W matrix to GPU
// ============================================================================

void AntennaProcessor_v1::set_external_weights(
    const std::vector<std::complex<float>>& W)
{
  // Free old managed buffer if any
  if (d_W_managed_) {
    backend_->Free(d_W_managed_);
    d_W_managed_ = nullptr;
  }

  if (W.empty()) return;

  const size_t bytes = W.size() * sizeof(std::complex<float>);
  d_W_managed_ = backend_->Allocate(bytes);
  HIP_CHECK(hipMemcpy(d_W_managed_, W.data(), bytes, hipMemcpyHostToDevice));
}

// ============================================================================
// Kernel compilation — Ref03 via GpuContext (replaces ~100 lines hiprtc+cache)
// ============================================================================

void AntennaProcessor_v1::ensure_compiled() {
  if (compiled_) return;
  ctx_.CompileModule(
      kernels::GetStrategiesHIPKernelSource(),
      kStrategyKernelNames,
      {"-DBLOCK_SIZE=256"});
  compiled_ = true;
}

// ============================================================================
// FFT plan
// ============================================================================

void AntennaProcessor_v1::create_fft_plan() {
  if (fft_plan_created_) return;

  int n[1] = { static_cast<int>(nFFT_) };
  int batch = static_cast<int>(cfg_.n_ant);

  HIPFFT_CHECK(hipfftPlanMany(
      &fft_plan_,
      1,           // rank
      n,           // n
      n,           // inembed
      1,           // istride
      static_cast<int>(nFFT_),  // idist
      n,           // onembed
      1,           // ostride
      static_cast<int>(nFFT_),  // odist
      HIPFFT_C2C,  // type
      batch));     // batch

  HIPFFT_CHECK(hipfftSetStream(fft_plan_, stream_main_));
  fft_plan_created_ = true;
}

// ============================================================================
// Main pipeline: process()
// ============================================================================

AntennaResult AntennaProcessor_v1::process(const void* d_S, const void* d_W) {
  AntennaResult result;
  result.scenario_mode = cfg_.scenario_mode;

  auto t_start = std::chrono::high_resolution_clock::now();

  // --- Debug point 2.1: stats on d_S (Stream 1, parallel with GEMM) ---
  do_debug_point_21(d_S, result);

  // --- GEMM: X = W * S (Stream 2) ---
  do_gemm(d_S, d_W);

  // --- Debug point 2.2: stats on d_X (Stream 3, parallel with Window+FFT) ---
  // Wait for GEMM to complete before reading d_X
  HIP_CHECK(hipEventRecord(event_gemm_done_, stream_main_));
  HIP_CHECK(hipStreamWaitEvent(stream_debug2_, event_gemm_done_, 0));
  do_debug_point_22(result);

  // --- Window (Hamming) + FFT (Stream 2) ---
  do_window_fft();

  // --- Debug point 2.3: stats on |spectrum| (Stream 4) ---
  HIP_CHECK(hipEventRecord(event_fft_done_, stream_main_));
  HIP_CHECK(hipStreamWaitEvent(stream_debug3_, event_fft_done_, 0));
  do_debug_point_23(result);

  // --- Post-FFT scenarios (Stream 4, after FFT done) ---
  do_run_post_fft_scenarios(result);

  // Synchronize all streams before building result
  HIP_CHECK(hipEventRecord(event_c1_done_, stream_debug1_));
  HIP_CHECK(hipEventRecord(event_c2_done_, stream_debug2_));

  HIP_CHECK(hipEventSynchronize(event_c1_done_));
  HIP_CHECK(hipEventSynchronize(event_c2_done_));
  HIP_CHECK(hipStreamSynchronize(stream_debug3_));

  auto t_end = std::chrono::high_resolution_clock::now();
  result.perf.total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

  return result;
}

// ============================================================================
// Protected step methods (for AntennaProcessorTest)
// ============================================================================

void AntennaProcessor_v1::do_debug_point_21(const void* d_S, AntennaResult& result) {
  if (cfg_.pre_input_stats == StatPreset::NONE) return;

  statistics::StatisticsParams sp;
  sp.beam_count = cfg_.n_ant;
  sp.n_point    = cfg_.n_samples;

  // ComputeStatistics on d_S (const_cast needed because StatisticsProcessor takes void*)
  result.pre_input_stats = stats_processor_->ComputeStatistics(
      const_cast<void*>(d_S), sp);

  if (cfg_.pre_input_stats & STAT_MEDIAN) {
    result.pre_input_medians = stats_processor_->ComputeMedian(
        const_cast<void*>(d_S), sp);
  }

  checkpoint_->save_c1_signal(d_S, cfg_.n_ant, cfg_.n_samples,
                               cfg_.sample_rate, gpu_id());
}

void AntennaProcessor_v1::do_gemm(const void* d_S, const void* d_W) {
  // X[n_ant x n_samples] = W[n_ant x n_ant] * S[n_ant x n_samples]
  // Row-major trick: swap A/B in column-major hipBLAS
  const int M = static_cast<int>(cfg_.n_samples);
  const int N = static_cast<int>(cfg_.n_ant);
  const int K = static_cast<int>(cfg_.n_ant);

  hipComplex alpha = {1.0f, 0.0f};
  hipComplex beta  = {0.0f, 0.0f};

  HIPBLAS_CHECK(hipblasCgemm(
      hipblas_handle_,
      HIPBLAS_OP_N, HIPBLAS_OP_N,
      M, N, K,
      &alpha,
      static_cast<const hipComplex*>(d_S), M,
      static_cast<const hipComplex*>(d_W), K,
      &beta,
      static_cast<hipComplex*>(d_X_), M));
}

void AntennaProcessor_v1::do_debug_point_22(AntennaResult& result) {
  if (cfg_.post_gemm_stats == StatPreset::NONE) return;

  statistics::StatisticsParams sp;
  sp.beam_count = cfg_.n_ant;
  sp.n_point    = cfg_.n_samples;

  result.post_gemm_stats = stats_processor_->ComputeStatistics(d_X_, sp);

  if (cfg_.post_gemm_stats & STAT_MEDIAN) {
    result.post_gemm_medians = stats_processor_->ComputeMedian(d_X_, sp);
  }

  checkpoint_->save_c2_data(d_X_, cfg_.n_ant, cfg_.n_samples,
                             cfg_.sample_rate, gpu_id());
}

void AntennaProcessor_v1::do_window_fft() {
  uint32_t n_ant = cfg_.n_ant;
  uint32_t n_samples = cfg_.n_samples;
  const uint32_t total_fft = n_ant * nFFT_;

  // 1. Zero entire FFT input buffer asynchronously (P11)
  //    This eliminates the else-branch in the kernel
  HIP_CHECK(hipMemsetAsync(d_fft_input_, 0,
      static_cast<size_t>(total_fft) * sizeof(float) * 2, stream_main_));

  // 2. Fused Hamming window + copy to FFT input (P13+P10)
  //    2D grid: Y=beam, X=sample position — no div/mod (P6)
  {
    unsigned int grid_x = (n_samples + kBlockSize - 1) / kBlockSize;
    unsigned int grid_y = n_ant;
    void* args[] = { &d_X_, &d_fft_input_, &d_hamming_window_, &n_ant, &n_samples, &nFFT_ };
    HIP_CHECK(hipModuleLaunchKernel(
        ctx_.GetKernel("hamming_pad_fused"),
        grid_x, grid_y, 1,      // 2D grid
        kBlockSize, 1, 1,
        0, stream_main_,
        args, nullptr));
  }

  // 3. Execute batch FFT: d_fft_input_ -> d_spectrum_
  HIPFFT_CHECK(hipfftExecC2C(
      fft_plan_,
      static_cast<hipfftComplex*>(d_fft_input_),
      static_cast<hipfftComplex*>(d_spectrum_),
      HIPFFT_FORWARD));

  // 4. Compute magnitudes via fft_func (no extra alloc)
  //    Uses ComplexToMagPhaseROCm::ProcessMagnitudeToBuffer — writes to d_magnitudes_ directly
  {
    fft_processor::MagPhaseParams mp;
    mp.beam_count  = n_ant;
    mp.n_point     = nFFT_;
    mp.norm_coeff  = 0.0f;  // no normalization (inv_n = 1)
    complex_to_mag_->ProcessMagnitudeToBuffer(d_spectrum_, d_magnitudes_, mp);
  }
}

void AntennaProcessor_v1::do_debug_point_23(AntennaResult& result) {
  if (cfg_.post_fft_stats == StatPreset::NONE) return;

  statistics::StatisticsParams sp;
  sp.beam_count = cfg_.n_ant;
  sp.n_point    = nFFT_;

  // Stats on precomputed float magnitudes |spectrum|
  result.post_fft_stats = stats_processor_->ComputeStatisticsFloat(d_magnitudes_, sp);

  if (cfg_.post_fft_stats & STAT_MEDIAN) {
    result.post_fft_medians = stats_processor_->ComputeMedianFloat(d_magnitudes_, sp);
  }

  checkpoint_->save_c3_spectrum(d_spectrum_, cfg_.n_ant, nFFT_, gpu_id());
}

void AntennaProcessor_v1::do_run_post_fft_scenarios(AntennaResult& result) {
  uint32_t n_ant = cfg_.n_ant;
  float sr = cfg_.sample_rate;
  const auto mode = cfg_.scenario_mode;

  // Step2.1: OneMax + Parabola (no phase)
  // Uses pre-allocated d_one_max_results_ (P3) and cached ctx_.GetKernel("one_max_no_phase") (P12)
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::ONE_MAX_PARABOLA) {
    // 2D grid: Y=beam, X=1 block per beam (P6)
    void* args[] = { &d_magnitudes_, &d_spectrum_, &d_one_max_results_, &n_ant, &nFFT_, &sr };
    HIP_CHECK(hipModuleLaunchKernel(
        ctx_.GetKernel("one_max_no_phase"),
        1, n_ant, 1,             // grid: 1 block X, n_ant blocks Y
        kBlockSize, 1, 1,
        0, stream_debug3_,
        args, nullptr));

    // Copy back
    result.one_max.resize(n_ant);
    HIP_CHECK(hipStreamSynchronize(stream_debug3_));
    HIP_CHECK(hipMemcpy(result.one_max.data(), d_one_max_results_,
                         n_ant * sizeof(OneMaxParabolaLite), hipMemcpyDeviceToHost));
  }

  // Step2.2: AllMaxima via fft_func pipeline (AllMaximaPipelineROCm)
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::ALL_MAXIMA) {
    auto am_result = all_maxima_pipeline_->Execute(
        d_magnitudes_,          // float magnitudes [n_ant x nFFT]
        d_spectrum_,            // complex FFT data [n_ant x nFFT]
        n_ant,                  // beam_count
        nFFT_,                  // nFFT
        sr,                     // sample_rate
        antenna_fft::OutputDestination::CPU,  // copy results to CPU
        1,                      // search_start (skip DC bin)
        0,                      // search_end (0 = nFFT/2)
        1000);                  // max_maxima_per_beam

    result.all_maxima = std::move(am_result.beams);
  }

  // Step2.3: GlobalMinMax
  // Uses pre-allocated d_minmax_results_ (P3)
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::GLOBAL_MINMAX) {
    // 2D grid: Y=beam (P6)
    void* args[] = { &d_magnitudes_, &d_minmax_results_, &n_ant, &nFFT_, &sr };
    HIP_CHECK(hipModuleLaunchKernel(
        ctx_.GetKernel("global_minmax"),
        1, n_ant, 1,             // grid: 1 block X, n_ant blocks Y
        kBlockSize, 1, 1,
        0, stream_debug3_,
        args, nullptr));

    result.minmax.resize(n_ant);
    HIP_CHECK(hipStreamSynchronize(stream_debug3_));
    HIP_CHECK(hipMemcpy(result.minmax.data(), d_minmax_results_,
                         n_ant * sizeof(MinMaxResult), hipMemcpyDeviceToHost));

    checkpoint_->save_c3_minmax(result.minmax.data(), n_ant, gpu_id());
  }
}

// ============================================================================
// Parallel post-FFT scenarios (3 streams — for benchmark 3.6)
// OneMax → stream_bench3a_, AllMaxima → stream_bench3b_, MinMax → stream_bench3c_
// ============================================================================

void AntennaProcessor_v1::do_run_post_fft_parallel(AntennaResult& result) {
  uint32_t n_ant = cfg_.n_ant;
  float sr = cfg_.sample_rate;
  const auto mode = cfg_.scenario_mode;

  // Step2.1: OneMax on stream_bench3a_
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::ONE_MAX_PARABOLA) {
    void* args[] = { &d_magnitudes_, &d_spectrum_, &d_one_max_results_, &n_ant, &nFFT_, &sr };
    HIP_CHECK(hipModuleLaunchKernel(
        ctx_.GetKernel("one_max_no_phase"),
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream_bench3a_,
        args, nullptr));
  }

  // Step2.3: GlobalMinMax on stream_bench3c_ (launched early, alongside AllMaxima)
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::GLOBAL_MINMAX) {
    void* args[] = { &d_magnitudes_, &d_minmax_results_, &n_ant, &nFFT_, &sr };
    HIP_CHECK(hipModuleLaunchKernel(
        ctx_.GetKernel("global_minmax"),
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream_bench3c_,
        args, nullptr));
  }

  // Step2.2: AllMaxima uses stream_debug3_ (bound at construction)
  //          Runs concurrently with OneMax + MinMax above
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::ALL_MAXIMA) {
    auto am_result = all_maxima_pipeline_->Execute(
        d_magnitudes_,
        d_spectrum_,
        n_ant,
        nFFT_,
        sr,
        antenna_fft::OutputDestination::CPU,
        1, 0, 1000);
    result.all_maxima = std::move(am_result.beams);
  }

  // Sync all three streams and collect results
  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::ONE_MAX_PARABOLA) {
    HIP_CHECK(hipStreamSynchronize(stream_bench3a_));
    result.one_max.resize(n_ant);
    HIP_CHECK(hipMemcpy(result.one_max.data(), d_one_max_results_,
                         n_ant * sizeof(OneMaxParabolaLite), hipMemcpyDeviceToHost));
  }

  if (mode == PostFftScenarioMode::ALL_REQUIRED ||
      mode == PostFftScenarioMode::GLOBAL_MINMAX) {
    HIP_CHECK(hipStreamSynchronize(stream_bench3c_));
    result.minmax.resize(n_ant);
    HIP_CHECK(hipMemcpy(result.minmax.data(), d_minmax_results_,
                         n_ant * sizeof(MinMaxResult), hipMemcpyDeviceToHost));
  }
}

}  // namespace strategies

#else  // !ENABLE_ROCM

// ============================================================================
// Stub for non-ROCm builds (Windows)
// ============================================================================

namespace strategies {

AntennaProcessor_v1::AntennaProcessor_v1(
    drv_gpu_lib::IBackend* backend,
    const AntennaProcessorConfig& cfg)
    : backend_(backend), cfg_(cfg)
{
  checkpoint_ = std::make_unique<NullCheckpointSave>();
}

AntennaProcessor_v1::~AntennaProcessor_v1() = default;

int AntennaProcessor_v1::gpu_id() const { return -1; }

void AntennaProcessor_v1::set_checkpoint_save(std::unique_ptr<ICheckpointSave> save) {
  checkpoint_ = std::move(save);
}

AntennaResult AntennaProcessor_v1::process(const void*, const void*) {
  throw std::runtime_error("AntennaProcessor_v1: ROCm not enabled");
}

void AntennaProcessor_v1::do_debug_point_21(const void*, AntennaResult&) {}
void AntennaProcessor_v1::do_gemm(const void*, const void*) {}
void AntennaProcessor_v1::do_debug_point_22(AntennaResult&) {}
void AntennaProcessor_v1::do_window_fft() {}
void AntennaProcessor_v1::do_debug_point_23(AntennaResult&) {}
void AntennaProcessor_v1::do_run_post_fft_scenarios(AntennaResult&) {}

}  // namespace strategies

#endif  // ENABLE_ROCM
