/**
 * @file statistics_processor.cpp
 * @brief StatisticsProcessor — thin Facade implementation (Ref03)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 * All GPU logic is delegated to Op classes (Layer 5).
 * This file handles: data upload/download, kernel compilation trigger,
 * strategy selection (histogram vs radix sort), result readback.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-03-14 (v2 Ref03 Facade)
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "kernels/statistics_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"

#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <complex>

using fft_func_utils::MakeROCmDataFromEvents;

// All kernel names used by statistics module
static const std::vector<std::string> kKernelNames = {
  "compute_magnitudes",
  "mean_reduce_phase1",
  "mean_reduce_final",
  "welford_stats",
  "welford_fused",
  "extract_medians",
  "welford_float",
  "histogram_median_pass",
  "histogram_median_pass_complex",
  "find_median_bucket"
};

static constexpr unsigned int kBlockSize = 256;

// =========================================================================
// Constructor / Destructor / Move
// =========================================================================

StatisticsProcessor::StatisticsProcessor(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "Statistics", "modules/statistics/kernels") {
}

StatisticsProcessor::~StatisticsProcessor() {
  // Release Ops before GpuContext (Ops reference ctx_)
  mean_op_.Release();
  welford_fused_op_.Release();
  welford_float_op_.Release();
  median_sort_op_.Release();
  median_hist_op_.Release();
  median_hist_complex_op_.Release();
}

StatisticsProcessor::StatisticsProcessor(StatisticsProcessor&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , mean_op_(std::move(other.mean_op_))
    , welford_fused_op_(std::move(other.welford_fused_op_))
    , welford_float_op_(std::move(other.welford_float_op_))
    , median_sort_op_(std::move(other.median_sort_op_))
    , median_hist_op_(std::move(other.median_hist_op_))
    , median_hist_complex_op_(std::move(other.median_hist_complex_op_))
    , compiled_(other.compiled_) {
  other.compiled_ = false;
}

StatisticsProcessor& StatisticsProcessor::operator=(StatisticsProcessor&& other) noexcept {
  if (this != &other) {
    mean_op_.Release();
    welford_fused_op_.Release();
    welford_float_op_.Release();
    median_sort_op_.Release();
    median_hist_op_.Release();
    median_hist_complex_op_.Release();

    ctx_ = std::move(other.ctx_);
    mean_op_ = std::move(other.mean_op_);
    welford_fused_op_ = std::move(other.welford_fused_op_);
    welford_float_op_ = std::move(other.welford_float_op_);
    median_sort_op_ = std::move(other.median_sort_op_);
    median_hist_op_ = std::move(other.median_hist_op_);
    median_hist_complex_op_ = std::move(other.median_hist_complex_op_);
    compiled_ = other.compiled_;
    other.compiled_ = false;
  }
  return *this;
}

// =========================================================================
// Lazy compilation + Op initialization
// =========================================================================

void StatisticsProcessor::EnsureCompiled() {
  if (compiled_) return;

  // Extra defines for kernel compilation
  std::vector<std::string> defines = {
    "-DBLOCK_SIZE=" + std::to_string(kBlockSize)
  };

  ctx_.CompileModule(kernels::GetStatisticsKernelSource(), kKernelNames, defines);

  // Initialize all Ops with the compiled context
  mean_op_.Initialize(ctx_);
  welford_fused_op_.Initialize(ctx_);
  welford_float_op_.Initialize(ctx_);
  median_sort_op_.Initialize(ctx_);
  median_hist_op_.Initialize(ctx_);
  median_hist_complex_op_.Initialize(ctx_);

  compiled_ = true;
}

// =========================================================================
// Data transfer helpers
// =========================================================================

void StatisticsProcessor::UploadComplexData(const std::complex<float>* data, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  ctx_.RequireShared(statistics::shared_buf::kInput, bytes);

  hipError_t err = hipMemcpyHtoDAsync(
      ctx_.GetShared(statistics::shared_buf::kInput),
      const_cast<std::complex<float>*>(data),
      bytes, ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("StatisticsProcessor: upload failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::CopyComplexGpuData(void* src, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  ctx_.RequireShared(statistics::shared_buf::kInput, bytes);

  hipError_t err = hipMemcpyDtoDAsync(
      ctx_.GetShared(statistics::shared_buf::kInput),
      src, bytes, ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("StatisticsProcessor: D2D copy failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::UploadFloatData(const float* data, size_t count) {
  size_t bytes = count * sizeof(float);
  ctx_.RequireShared(statistics::shared_buf::kMagnitudes, bytes);

  hipError_t err = hipMemcpyHtoDAsync(
      ctx_.GetShared(statistics::shared_buf::kMagnitudes),
      const_cast<float*>(data),
      bytes, ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("StatisticsProcessor: float upload failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::CopyFloatGpuData(void* src, size_t count) {
  size_t bytes = count * sizeof(float);
  ctx_.RequireShared(statistics::shared_buf::kMagnitudes, bytes);

  hipError_t err = hipMemcpyDtoDAsync(
      ctx_.GetShared(statistics::shared_buf::kMagnitudes),
      src, bytes, ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("StatisticsProcessor: float D2D copy failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// =========================================================================
// Result readback helpers
// =========================================================================

std::vector<MeanResult> StatisticsProcessor::ReadMeanResults(uint32_t beam_count) {
  struct float2_t { float x, y; };
  std::vector<float2_t> raw(beam_count);
  hipError_t err = hipMemcpyDtoH(
      raw.data(),
      ctx_.GetShared(statistics::shared_buf::kResult),
      beam_count * sizeof(float2_t));
  if (err != hipSuccess) {
    throw std::runtime_error("ReadMeanResults: DtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MeanResult> results;
  results.reserve(beam_count);
  for (uint32_t i = 0; i < beam_count; ++i) {
    MeanResult r;
    r.beam_id = i;
    r.mean = std::complex<float>(raw[i].x, raw[i].y);
    results.push_back(r);
  }
  return results;
}

std::vector<StatisticsResult> StatisticsProcessor::ReadStatisticsResults(uint32_t beam_count) {
  struct WelfordResult {
    float mean_re, mean_im, mean_mag, variance, std_dev;
  };
  std::vector<WelfordResult> raw(beam_count);
  hipError_t err = hipMemcpyDtoH(
      raw.data(),
      ctx_.GetShared(statistics::shared_buf::kResult),
      beam_count * sizeof(WelfordResult));
  if (err != hipSuccess) {
    throw std::runtime_error("ReadStatisticsResults: DtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<StatisticsResult> results;
  results.reserve(beam_count);
  for (uint32_t i = 0; i < beam_count; ++i) {
    StatisticsResult r;
    r.beam_id = i;
    r.mean = std::complex<float>(raw[i].mean_re, raw[i].mean_im);
    r.mean_magnitude = raw[i].mean_mag;
    r.variance = raw[i].variance;
    r.std_dev = raw[i].std_dev;
    results.push_back(r);
  }
  return results;
}

std::vector<MedianResult> StatisticsProcessor::ReadMedianResults(uint32_t beam_count) {
  std::vector<float> medians_host(beam_count);
  hipError_t err = hipMemcpyDtoH(
      medians_host.data(),
      ctx_.GetShared(statistics::shared_buf::kMediansCompact),
      beam_count * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("ReadMedianResults: DtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MedianResult> results;
  results.reserve(beam_count);
  for (uint32_t b = 0; b < beam_count; ++b) {
    MedianResult r;
    r.beam_id = b;
    r.median_magnitude = medians_host[b];
    results.push_back(r);
  }
  return results;
}

// =========================================================================
// Public API — ComputeMean
// =========================================================================

std::vector<MeanResult> StatisticsProcessor::ComputeMean(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeMean: input size " + std::to_string(data.size()) +
                                 " != expected " + std::to_string(expected));
  }

  EnsureCompiled();
  UploadComplexData(data.data(), data.size());
  mean_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());
  return ReadMeanResults(params.beam_count);
}

std::vector<MeanResult> StatisticsProcessor::ComputeMean(
    void* gpu_data,
    const StatisticsParams& params) {
  if (!gpu_data) throw std::invalid_argument("ComputeMean: gpu_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyComplexGpuData(gpu_data, count);
  mean_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());
  return ReadMeanResults(params.beam_count);
}

// =========================================================================
// Public API — ComputeStatistics
// =========================================================================

std::vector<StatisticsResult> StatisticsProcessor::ComputeStatistics(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeStatistics: input size mismatch");
  }

  EnsureCompiled();
  UploadComplexData(data.data(), data.size());
  welford_fused_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());
  return ReadStatisticsResults(params.beam_count);
}

std::vector<StatisticsResult> StatisticsProcessor::ComputeStatistics(
    void* gpu_data,
    const StatisticsParams& params) {
  if (!gpu_data) throw std::invalid_argument("ComputeStatistics: gpu_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyComplexGpuData(gpu_data, count);
  welford_fused_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());
  return ReadStatisticsResults(params.beam_count);
}

// =========================================================================
// Public API — ComputeMedian (complex input)
// =========================================================================

std::vector<MedianResult> StatisticsProcessor::ComputeMedian(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeMedian: input size mismatch");
  }

  EnsureCompiled();
  UploadComplexData(data.data(), data.size());

  // Auto-select: histogram for large data, radix sort for small
  if (params.n_point > kHistogramThreshold) {
    median_hist_complex_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.Execute(params.beam_count, params.n_point);
  }

  hipStreamSynchronize(ctx_.stream());
  return ReadMedianResults(params.beam_count);
}

std::vector<MedianResult> StatisticsProcessor::ComputeMedian(
    void* gpu_data,
    const StatisticsParams& params) {
  if (!gpu_data) throw std::invalid_argument("ComputeMedian: gpu_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyComplexGpuData(gpu_data, count);

  if (params.n_point > kHistogramThreshold) {
    median_hist_complex_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.Execute(params.beam_count, params.n_point);
  }

  hipStreamSynchronize(ctx_.stream());
  return ReadMedianResults(params.beam_count);
}

// =========================================================================
// Public API — ComputeStatisticsFloat (float magnitudes input)
// =========================================================================

std::vector<StatisticsResult> StatisticsProcessor::ComputeStatisticsFloat(
    void* gpu_float_data,
    const StatisticsParams& params) {
  if (!gpu_float_data) throw std::invalid_argument("ComputeStatisticsFloat: gpu_float_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyFloatGpuData(gpu_float_data, count);
  welford_float_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());

  // WelfordFloat writes same 5-float struct but mean_re=0, mean_im=0
  auto results = ReadStatisticsResults(params.beam_count);
  for (auto& r : results) {
    r.mean = std::complex<float>(0.0f, 0.0f);
  }
  return results;
}

std::vector<StatisticsResult> StatisticsProcessor::ComputeStatisticsFloat(
    const std::vector<float>& data,
    const StatisticsParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeStatisticsFloat(vector): input size " +
        std::to_string(data.size()) + " != expected " + std::to_string(expected));
  }

  EnsureCompiled();
  // Upload directly to shared kMagnitudes buffer (reused, async, no hipMalloc/hipFree)
  UploadFloatData(data.data(), expected);
  welford_float_op_.Execute(params.beam_count, params.n_point);
  hipStreamSynchronize(ctx_.stream());

  auto results = ReadStatisticsResults(params.beam_count);
  for (auto& r : results) r.mean = std::complex<float>(0.0f, 0.0f);
  return results;
}

// =========================================================================
// Public API — ComputeMedianFloat (float magnitudes input)
// =========================================================================

std::vector<MedianResult> StatisticsProcessor::ComputeMedianFloat(
    void* gpu_float_data,
    const StatisticsParams& params) {
  if (!gpu_float_data) throw std::invalid_argument("ComputeMedianFloat: gpu_float_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyFloatGpuData(gpu_float_data, count);

  if (params.n_point > kHistogramThreshold) {
    median_hist_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.ExecuteFloat(params.beam_count, params.n_point);
  }

  hipStreamSynchronize(ctx_.stream());
  return ReadMedianResults(params.beam_count);
}

std::vector<MedianResult> StatisticsProcessor::ComputeMedianFloat(
    const std::vector<float>& data,
    const StatisticsParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeMedianFloat(vector): input size " +
        std::to_string(data.size()) + " != expected " + std::to_string(expected));
  }

  EnsureCompiled();
  UploadFloatData(data.data(), expected);

  if (params.n_point > kHistogramThreshold) {
    median_hist_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.ExecuteFloat(params.beam_count, params.n_point);
  }

  hipStreamSynchronize(ctx_.stream());
  return ReadMedianResults(params.beam_count);
}

// =========================================================================
// Private helper — MergeResults
// =========================================================================

std::vector<FullStatisticsResult> StatisticsProcessor::MergeResults(
    const std::vector<StatisticsResult>& stats,
    const std::vector<MedianResult>& medians)
{
  std::vector<FullStatisticsResult> out;
  out.reserve(stats.size());
  for (size_t i = 0; i < stats.size(); ++i) {
    FullStatisticsResult r;
    r.beam_id          = stats[i].beam_id;
    r.mean             = stats[i].mean;
    r.variance         = stats[i].variance;
    r.std_dev          = stats[i].std_dev;
    r.mean_magnitude   = stats[i].mean_magnitude;
    r.median_magnitude = medians[i].median_magnitude;
    out.push_back(r);
  }
  return out;
}

// =========================================================================
// Public API — ComputeAll (CPU complex data)
// =========================================================================

std::vector<FullStatisticsResult> StatisticsProcessor::ComputeAll(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params,
    StatisticsROCmProfEvents* prof_events)
{
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeAll: input size " +
                                std::to_string(data.size()) +
                                " != expected " + std::to_string(expected));
  }

  EnsureCompiled();

  hipEvent_t ev_up_s{},   ev_up_e{};
  hipEvent_t ev_welf_s{}, ev_welf_e{};
  hipEvent_t ev_med_s{},  ev_med_e{};
  if (prof_events) {
    hipEventCreate(&ev_up_s);   hipEventCreate(&ev_up_e);
    hipEventCreate(&ev_welf_s); hipEventCreate(&ev_welf_e);
    hipEventCreate(&ev_med_s);  hipEventCreate(&ev_med_e);
  }

  if (prof_events) hipEventRecord(ev_up_s, ctx_.stream());
  UploadComplexData(data.data(), data.size());
  if (prof_events) hipEventRecord(ev_up_e, ctx_.stream());

  if (prof_events) hipEventRecord(ev_welf_s, ctx_.stream());
  welford_fused_op_.Execute(params.beam_count, params.n_point);
  if (prof_events) hipEventRecord(ev_welf_e, ctx_.stream());

  if (prof_events) hipEventRecord(ev_med_s, ctx_.stream());
  if (params.n_point > kHistogramThreshold) {
    median_hist_complex_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.Execute(params.beam_count, params.n_point);
  }
  if (prof_events) hipEventRecord(ev_med_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Upload",
        MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
    prof_events->push_back({"Welford_Fused",
        MakeROCmDataFromEvents(ev_welf_s, ev_welf_e, 0, "welford_fused")});
    prof_events->push_back({"Median",
        MakeROCmDataFromEvents(ev_med_s,  ev_med_e,  0, "median")});
  }

  auto stats   = ReadStatisticsResults(params.beam_count);
  auto medians = ReadMedianResults(params.beam_count);
  return MergeResults(stats, medians);
}

// =========================================================================
// Public API — ComputeAll (GPU complex data)
// =========================================================================

std::vector<FullStatisticsResult> StatisticsProcessor::ComputeAll(
    void* gpu_data,
    const StatisticsParams& params,
    StatisticsROCmProfEvents* prof_events)
{
  if (!gpu_data) throw std::invalid_argument("ComputeAll: gpu_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;

  hipEvent_t ev_copy_s{}, ev_copy_e{};
  hipEvent_t ev_welf_s{}, ev_welf_e{};
  hipEvent_t ev_med_s{},  ev_med_e{};
  if (prof_events) {
    hipEventCreate(&ev_copy_s); hipEventCreate(&ev_copy_e);
    hipEventCreate(&ev_welf_s); hipEventCreate(&ev_welf_e);
    hipEventCreate(&ev_med_s);  hipEventCreate(&ev_med_e);
  }

  if (prof_events) hipEventRecord(ev_copy_s, ctx_.stream());
  CopyComplexGpuData(gpu_data, count);
  if (prof_events) hipEventRecord(ev_copy_e, ctx_.stream());

  if (prof_events) hipEventRecord(ev_welf_s, ctx_.stream());
  welford_fused_op_.Execute(params.beam_count, params.n_point);
  if (prof_events) hipEventRecord(ev_welf_e, ctx_.stream());

  if (prof_events) hipEventRecord(ev_med_s, ctx_.stream());
  if (params.n_point > kHistogramThreshold) {
    median_hist_complex_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.Execute(params.beam_count, params.n_point);
  }
  if (prof_events) hipEventRecord(ev_med_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"D2D_Copy",
        MakeROCmDataFromEvents(ev_copy_s, ev_copy_e, 1, "D2D")});
    prof_events->push_back({"Welford_Fused",
        MakeROCmDataFromEvents(ev_welf_s, ev_welf_e, 0, "welford_fused")});
    prof_events->push_back({"Median",
        MakeROCmDataFromEvents(ev_med_s,  ev_med_e,  0, "median")});
  }

  auto stats   = ReadStatisticsResults(params.beam_count);
  auto medians = ReadMedianResults(params.beam_count);
  return MergeResults(stats, medians);
}

// =========================================================================
// Public API — ComputeAllFloat (GPU float data)
// =========================================================================

std::vector<FullStatisticsResult> StatisticsProcessor::ComputeAllFloat(
    void* gpu_float_data,
    const StatisticsParams& params,
    StatisticsROCmProfEvents* prof_events)
{
  if (!gpu_float_data) throw std::invalid_argument("ComputeAllFloat: gpu_float_data is null");

  EnsureCompiled();
  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;

  hipEvent_t ev_copy_s{}, ev_copy_e{};
  hipEvent_t ev_welf_s{}, ev_welf_e{};
  hipEvent_t ev_med_s{},  ev_med_e{};
  if (prof_events) {
    hipEventCreate(&ev_copy_s); hipEventCreate(&ev_copy_e);
    hipEventCreate(&ev_welf_s); hipEventCreate(&ev_welf_e);
    hipEventCreate(&ev_med_s);  hipEventCreate(&ev_med_e);
  }

  if (prof_events) hipEventRecord(ev_copy_s, ctx_.stream());
  CopyFloatGpuData(gpu_float_data, count);
  if (prof_events) hipEventRecord(ev_copy_e, ctx_.stream());

  // Welford on float magnitudes → kResult
  if (prof_events) hipEventRecord(ev_welf_s, ctx_.stream());
  welford_float_op_.Execute(params.beam_count, params.n_point);
  if (prof_events) hipEventRecord(ev_welf_e, ctx_.stream());

  // Median on float magnitudes → kMediansCompact
  if (prof_events) hipEventRecord(ev_med_s, ctx_.stream());
  if (params.n_point > kHistogramThreshold) {
    median_hist_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.ExecuteFloat(params.beam_count, params.n_point);
  }
  if (prof_events) hipEventRecord(ev_med_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"D2D_Copy",
        MakeROCmDataFromEvents(ev_copy_s, ev_copy_e, 1, "D2D")});
    prof_events->push_back({"Welford_Float",
        MakeROCmDataFromEvents(ev_welf_s, ev_welf_e, 0, "welford_float")});
    prof_events->push_back({"Median",
        MakeROCmDataFromEvents(ev_med_s,  ev_med_e,  0, "median")});
  }

  // WelfordFloat writes mean_re=0, mean_im=0 — enforce explicitly
  auto stats = ReadStatisticsResults(params.beam_count);
  for (auto& r : stats) r.mean = std::complex<float>(0.0f, 0.0f);
  auto medians = ReadMedianResults(params.beam_count);
  return MergeResults(stats, medians);
}

// =========================================================================
// Public API — ComputeAllFloat (CPU float data)
// =========================================================================

std::vector<FullStatisticsResult> StatisticsProcessor::ComputeAllFloat(
    const std::vector<float>& data,
    const StatisticsParams& params)
{
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeAllFloat(vector): input size " +
        std::to_string(data.size()) + " != expected " + std::to_string(expected));
  }

  EnsureCompiled();
  // Upload directly to shared kMagnitudes (reused buffer, async H2D)
  UploadFloatData(data.data(), expected);

  // Welford on float magnitudes → kResult
  welford_float_op_.Execute(params.beam_count, params.n_point);

  // Median on float magnitudes → kMediansCompact
  if (params.n_point > kHistogramThreshold) {
    median_hist_op_.Execute(params.beam_count, params.n_point);
  } else {
    median_sort_op_.ExecuteFloat(params.beam_count, params.n_point);
  }

  hipStreamSynchronize(ctx_.stream());

  auto stats = ReadStatisticsResults(params.beam_count);
  for (auto& r : stats) r.mean = std::complex<float>(0.0f, 0.0f);
  auto medians = ReadMedianResults(params.beam_count);
  return MergeResults(stats, medians);
}

}  // namespace statistics

#endif  // ENABLE_ROCM
