/**
 * @file statistics_processor.cpp
 * @brief Implementation of StatisticsProcessor -- statistical computations using HIP kernels
 *
 * =========================================================================
 * CONTENTS
 * =========================================================================
 *
 * PART 1: Constructor, Destructor, Move Semantics
 * PART 2: Public API (ComputeMean, ComputeMedian, ComputeStatistics)
 * PART 3: GPU Resources (Allocate, CompileKernels, Release)
 * PART 4: GPU Operations (Upload, Magnitudes, MeanReduction, Welford, MedianSort)
 *
 * Optimizations v2 (2026-02-26):
 *   TASK-1: ComputeStatistics uses welford_fused (1 pass, no magnitudes buf)
 *   TASK-2: ComputeMedian uses extract_medians GPU kernel → 1 DtoH instead of 256
 *   TASK-3: hiprtc + disk HSACO cache (KernelCacheService) + --offload-arch flag
 *   TASK-4: Double-load, warp shuffle, __launch_bounds__, blocks_per_beam param
 *   TASK-5: hipMemcpyAsync in AllocateBuffers for offsets upload
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-02-26 (v2 optimized)
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "kernels/statistics_kernels_rocm.hpp"

#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace statistics {

// Block size for kernels
static constexpr unsigned int kBlockSize = 256;

// TASK-4.3: Double-load factor — each phase1 block covers 2*kBlockSize elements
// Must match the kernel source (which uses blockDim.x * 2 in double-load logic)
static constexpr unsigned int kDoubleLoadElements = kBlockSize * 2;  // 512

// =========================================================================
// PART 1: Constructor / Destructor / Move Semantics
// =========================================================================

StatisticsProcessor::StatisticsProcessor(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error("StatisticsProcessor: backend is null or not initialized");
  }

  if (backend_->GetType() != drv_gpu_lib::BackendType::ROCm) {
    throw std::runtime_error("StatisticsProcessor: requires ROCm backend, got different type");
  }

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error("StatisticsProcessor: failed to get HIP stream from backend");
  }

  // TASK-3: Initialize disk cache for compiled HSACO
  kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
      "modules/statistics/kernels", drv_gpu_lib::BackendType::ROCm);
}

StatisticsProcessor::~StatisticsProcessor() {
  ReleaseResources();
}

StatisticsProcessor::StatisticsProcessor(StatisticsProcessor&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , input_buffer_(other.input_buffer_)
    , magnitudes_buf_(other.magnitudes_buf_)
    , sort_buf_(other.sort_buf_)
    , sort_temp_buf_(other.sort_temp_buf_)
    , offsets_buf_(other.offsets_buf_)
    , reduce_buf_(other.reduce_buf_)
    , result_buf_(other.result_buf_)
    , medians_compact_buf_(other.medians_compact_buf_)
    , module_(other.module_)
    , magnitudes_kernel_(other.magnitudes_kernel_)
    , mean_reduce_kernel_(other.mean_reduce_kernel_)
    , mean_final_kernel_(other.mean_final_kernel_)
    , welford_kernel_(other.welford_kernel_)
    , welford_fused_kernel_(other.welford_fused_kernel_)
    , welford_float_kernel_(other.welford_float_kernel_)
    , extract_medians_kernel_(other.extract_medians_kernel_)
    , kernels_compiled_(other.kernels_compiled_)
    , current_beams_(other.current_beams_)
    , current_n_point_(other.current_n_point_)
    , sort_temp_size_(other.sort_temp_size_)
    , kernel_cache_(std::move(other.kernel_cache_)) {

  other.backend_               = nullptr;
  other.stream_                = nullptr;
  other.input_buffer_          = nullptr;
  other.magnitudes_buf_        = nullptr;
  other.sort_buf_              = nullptr;
  other.sort_temp_buf_         = nullptr;
  other.offsets_buf_           = nullptr;
  other.reduce_buf_            = nullptr;
  other.result_buf_            = nullptr;
  other.medians_compact_buf_   = nullptr;
  other.module_                = nullptr;
  other.magnitudes_kernel_     = nullptr;
  other.mean_reduce_kernel_    = nullptr;
  other.mean_final_kernel_     = nullptr;
  other.welford_kernel_        = nullptr;
  other.welford_fused_kernel_  = nullptr;
  other.welford_float_kernel_  = nullptr;
  other.extract_medians_kernel_ = nullptr;
  other.kernels_compiled_      = false;
  other.current_beams_         = 0;
  other.current_n_point_       = 0;
  other.sort_temp_size_        = 0;
}

StatisticsProcessor& StatisticsProcessor::operator=(StatisticsProcessor&& other) noexcept {
  if (this != &other) {
    ReleaseResources();

    backend_               = other.backend_;
    stream_                = other.stream_;
    input_buffer_          = other.input_buffer_;
    magnitudes_buf_        = other.magnitudes_buf_;
    sort_buf_              = other.sort_buf_;
    sort_temp_buf_         = other.sort_temp_buf_;
    offsets_buf_           = other.offsets_buf_;
    reduce_buf_            = other.reduce_buf_;
    result_buf_            = other.result_buf_;
    medians_compact_buf_   = other.medians_compact_buf_;
    module_                = other.module_;
    magnitudes_kernel_     = other.magnitudes_kernel_;
    mean_reduce_kernel_    = other.mean_reduce_kernel_;
    mean_final_kernel_     = other.mean_final_kernel_;
    welford_kernel_        = other.welford_kernel_;
    welford_fused_kernel_  = other.welford_fused_kernel_;
    welford_float_kernel_  = other.welford_float_kernel_;
    extract_medians_kernel_ = other.extract_medians_kernel_;
    kernels_compiled_      = other.kernels_compiled_;
    current_beams_         = other.current_beams_;
    current_n_point_       = other.current_n_point_;
    sort_temp_size_        = other.sort_temp_size_;
    kernel_cache_          = std::move(other.kernel_cache_);

    other.backend_               = nullptr;
    other.stream_                = nullptr;
    other.input_buffer_          = nullptr;
    other.magnitudes_buf_        = nullptr;
    other.sort_buf_              = nullptr;
    other.sort_temp_buf_         = nullptr;
    other.offsets_buf_           = nullptr;
    other.reduce_buf_            = nullptr;
    other.result_buf_            = nullptr;
    other.medians_compact_buf_   = nullptr;
    other.module_                = nullptr;
    other.magnitudes_kernel_     = nullptr;
    other.mean_reduce_kernel_    = nullptr;
    other.mean_final_kernel_     = nullptr;
    other.welford_kernel_        = nullptr;
    other.welford_fused_kernel_  = nullptr;
    other.extract_medians_kernel_ = nullptr;
    other.kernels_compiled_      = false;
    other.current_beams_         = 0;
    other.current_n_point_       = 0;
    other.sort_temp_size_        = 0;
  }
  return *this;
}

// =========================================================================
// PART 2: Public API
// =========================================================================

// --- ComputeMean (CPU data) ---
std::vector<MeanResult> StatisticsProcessor::ComputeMean(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params)
{
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeMean: input size " + std::to_string(data.size()) +
                                 " != expected " + std::to_string(expected));
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);
  UploadData(data.data(), data.size());
  ExecuteMeanReduction(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  // Read results: beam_count * float2_t (re, im)
  struct float2_t { float x, y; };
  std::vector<float2_t> raw(params.beam_count);
  hipError_t err = hipMemcpyDtoH(raw.data(), result_buf_,
                                   params.beam_count * sizeof(float2_t));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeMean: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MeanResult> results;
  results.reserve(params.beam_count);
  for (uint32_t i = 0; i < params.beam_count; ++i) {
    MeanResult r;
    r.beam_id = i;
    r.mean = std::complex<float>(raw[i].x, raw[i].y);
    results.push_back(r);
  }
  return results;
}

// --- ComputeMean (GPU data) ---
std::vector<MeanResult> StatisticsProcessor::ComputeMean(
    void* gpu_data,
    const StatisticsParams& params)
{
  if (!gpu_data) {
    throw std::invalid_argument("ComputeMean: gpu_data is null");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);

  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyGpuData(gpu_data, count);

  ExecuteMeanReduction(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  struct float2_t { float x, y; };
  std::vector<float2_t> raw(params.beam_count);
  hipError_t err = hipMemcpyDtoH(raw.data(), result_buf_,
                                   params.beam_count * sizeof(float2_t));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeMean: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MeanResult> results;
  results.reserve(params.beam_count);
  for (uint32_t i = 0; i < params.beam_count; ++i) {
    MeanResult r;
    r.beam_id = i;
    r.mean = std::complex<float>(raw[i].x, raw[i].y);
    results.push_back(r);
  }
  return results;
}

// --- ComputeMedian (CPU data) ---
// TASK-2: Uses extract_medians GPU kernel → 1 DtoH instead of 256
std::vector<MedianResult> StatisticsProcessor::ComputeMedian(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params)
{
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeMedian: input size mismatch");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);
  UploadData(data.data(), data.size());

  size_t total = static_cast<size_t>(params.beam_count) * params.n_point;
  ExecuteMagnitudesKernel(total);
  ExecuteMedianSort(params.beam_count, params.n_point);
  ExecuteExtractMediansKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  // TASK-2: One DtoH for all medians (beam_count floats)
  std::vector<float> medians_host(params.beam_count);
  hipError_t err = hipMemcpyDtoH(medians_host.data(), medians_compact_buf_,
                                   params.beam_count * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeMedian: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MedianResult> results;
  results.reserve(params.beam_count);
  for (uint32_t b = 0; b < params.beam_count; ++b) {
    MedianResult r;
    r.beam_id = b;
    r.median_magnitude = medians_host[b];
    results.push_back(r);
  }
  return results;
}

// --- ComputeMedian (GPU data) ---
std::vector<MedianResult> StatisticsProcessor::ComputeMedian(
    void* gpu_data,
    const StatisticsParams& params)
{
  if (!gpu_data) {
    throw std::invalid_argument("ComputeMedian: gpu_data is null");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);

  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyGpuData(gpu_data, count);

  ExecuteMagnitudesKernel(count);
  ExecuteMedianSort(params.beam_count, params.n_point);
  ExecuteExtractMediansKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  std::vector<float> medians_host(params.beam_count);
  hipError_t err = hipMemcpyDtoH(medians_host.data(), medians_compact_buf_,
                                   params.beam_count * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeMedian: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MedianResult> results;
  results.reserve(params.beam_count);
  for (uint32_t b = 0; b < params.beam_count; ++b) {
    MedianResult r;
    r.beam_id = b;
    r.median_magnitude = medians_host[b];
    results.push_back(r);
  }
  return results;
}

// --- ComputeStatistics (CPU data) ---
// TASK-1: Uses welford_fused — 1 pass over data, no separate magnitudes kernel
std::vector<StatisticsResult> StatisticsProcessor::ComputeStatistics(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params)
{
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ComputeStatistics: input size mismatch");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);
  UploadData(data.data(), data.size());

  // TASK-1: fused kernel — no ExecuteMagnitudesKernel needed
  ExecuteWelfordFusedKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  // Read WelfordResult per beam (5 floats each)
  struct WelfordResult {
    float mean_re, mean_im, mean_mag, variance, std_dev;
  };
  std::vector<WelfordResult> raw(params.beam_count);
  hipError_t err = hipMemcpyDtoH(raw.data(), result_buf_,
                                   params.beam_count * sizeof(WelfordResult));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeStatistics: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<StatisticsResult> results;
  results.reserve(params.beam_count);
  for (uint32_t i = 0; i < params.beam_count; ++i) {
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

// --- ComputeStatistics (GPU data) ---
std::vector<StatisticsResult> StatisticsProcessor::ComputeStatistics(
    void* gpu_data,
    const StatisticsParams& params)
{
  if (!gpu_data) {
    throw std::invalid_argument("ComputeStatistics: gpu_data is null");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);

  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyGpuData(gpu_data, count);

  // TASK-1: fused kernel — no ExecuteMagnitudesKernel needed
  ExecuteWelfordFusedKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  struct WelfordResult {
    float mean_re, mean_im, mean_mag, variance, std_dev;
  };
  std::vector<WelfordResult> raw(params.beam_count);
  hipError_t err = hipMemcpyDtoH(raw.data(), result_buf_,
                                   params.beam_count * sizeof(WelfordResult));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeStatistics: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<StatisticsResult> results;
  results.reserve(params.beam_count);
  for (uint32_t i = 0; i < params.beam_count; ++i) {
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

// =========================================================================
// PART 3: GPU Resources Management
// =========================================================================

// TASK-3: CompileKernels with disk HSACO cache + --offload-arch
void StatisticsProcessor::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  constexpr const char* kKernelName = "statistics_kernels";

  // ─── Try loading from disk cache ────────────────────────────────────────
  if (kernel_cache_) {
    try {
      auto entry = kernel_cache_->Load(kKernelName);
      if (entry.has_binary()) {
        hipError_t hip_err = hipModuleLoadData(&module_, entry.binary.data());
        if (hip_err == hipSuccess) {
          // Get all kernel function handles from cached module
          auto getKernel = [this](hipFunction_t* func, const char* name) {
            hipError_t err = hipModuleGetFunction(func, module_, name);
            if (err != hipSuccess) {
              throw std::runtime_error(std::string("getKernel(") + name +
                                        ") failed: " + hipGetErrorString(err));
            }
          };
          getKernel(&magnitudes_kernel_,      "compute_magnitudes");
          getKernel(&mean_reduce_kernel_,     "mean_reduce_phase1");
          getKernel(&mean_final_kernel_,      "mean_reduce_final");
          getKernel(&welford_kernel_,         "welford_stats");
          getKernel(&welford_fused_kernel_,   "welford_fused");
          getKernel(&extract_medians_kernel_, "extract_medians");
          getKernel(&welford_float_kernel_,  "welford_float");
          kernels_compiled_ = true;
          con.Print(0, "Statistics", "kernels loaded from cache (HSACO)");
          return;
        }
        // Cache might be stale (different arch), fall through to compile
        (void)hipModuleUnload(module_);
        module_ = nullptr;
      }
    } catch (...) {
      // Cache miss or corrupt — fall through to compile
    }
  }

  // ─── Get target architecture for --offload-arch ─────────────────────────
  std::string arch_name;
  try {
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name = "";
  }

  // ─── Compile via hiprtc ─────────────────────────────────────────────────
  const char* source = kernels::GetStatisticsKernelSource();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(&prog, source, "statistics_kernels.hip",
                                                0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error("CompileKernels: hiprtcCreateProgram failed: " +
                              std::to_string(static_cast<int>(rtcResult)));
  }

  // TASK-3: Compile with explicit arch + optimizations
  // P3: Determine WARP_SIZE by GPU architecture (CDNA=64, RDNA=32)
  int warp_size = 32;  // default: RDNA (gfx10xx, gfx11xx, gfx12xx)
  if (arch_name.find("gfx9") == 0) {
    warp_size = 64;  // CDNA / Vega (gfx900, gfx906, gfx908, gfx90a, gfx940, gfx942)
  }
  std::string warp_define  = "-DWARP_SIZE="  + std::to_string(warp_size);
  std::string block_define = "-DBLOCK_SIZE=" + std::to_string(kBlockSize);
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
  std::vector<const char*> opts = {"-O3", "-std=c++17",
                                    warp_define.c_str(), block_define.c_str()};
  if (!arch_flag.empty()) {
    opts.push_back(arch_flag.c_str());
  }

  rtcResult = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error("CompileKernels: compilation failed:\n" + log);
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess) {
    throw std::runtime_error("CompileKernels: hipModuleLoadData failed: " +
                              std::string(hipGetErrorString(hipErr)));
  }

  // Get kernel functions
  auto getKernel = [this](hipFunction_t* func, const char* name) {
    hipError_t err = hipModuleGetFunction(func, module_, name);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("CompileKernels: hipModuleGetFunction(") +
                                name + ") failed: " + hipGetErrorString(err));
    }
  };

  getKernel(&magnitudes_kernel_,      "compute_magnitudes");
  getKernel(&mean_reduce_kernel_,     "mean_reduce_phase1");
  getKernel(&mean_final_kernel_,      "mean_reduce_final");
  getKernel(&welford_kernel_,         "welford_stats");
  getKernel(&welford_fused_kernel_,   "welford_fused");
  getKernel(&extract_medians_kernel_, "extract_medians");
  getKernel(&welford_float_kernel_,  "welford_float");

  kernels_compiled_ = true;
  con.Print(0, "Statistics",
            "kernels compiled (" + std::to_string(codeSize) + " bytes HSACO" +
            (arch_name.empty() ? "" : ", " + arch_name) + ")");

  // ─── Save compiled binary to disk cache ─────────────────────────────────
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(kKernelName, source, binary, "", "statistics hiprtc kernels v2");
      con.Print(0, "Statistics", "kernels saved to cache");
    } catch (const std::exception& e) {
      con.Print(0, "Statistics", "warning: cache save failed: " + std::string(e.what()));
    }
  }
}

void StatisticsProcessor::AllocateBuffers(size_t beam_count, size_t n_point) {
  if (beam_count == current_beams_ && n_point == current_n_point_ && input_buffer_) {
    return;  // Already allocated for this size
  }

  // Free old buffers
  if (input_buffer_)        { (void)hipFree(input_buffer_);        input_buffer_ = nullptr; }
  if (magnitudes_buf_)      { (void)hipFree(magnitudes_buf_);      magnitudes_buf_ = nullptr; }
  if (sort_buf_)            { (void)hipFree(sort_buf_);            sort_buf_ = nullptr; }
  if (sort_temp_buf_)       { (void)hipFree(sort_temp_buf_);       sort_temp_buf_ = nullptr; }
  if (offsets_buf_)         { (void)hipFree(offsets_buf_);         offsets_buf_ = nullptr; }
  if (reduce_buf_)          { (void)hipFree(reduce_buf_);          reduce_buf_ = nullptr; }
  if (result_buf_)          { (void)hipFree(result_buf_);          result_buf_ = nullptr; }
  if (medians_compact_buf_) { (void)hipFree(medians_compact_buf_); medians_compact_buf_ = nullptr; }

  size_t total = beam_count * n_point;
  hipError_t err;

  // 1. Input buffer: complex<float> data
  err = hipMalloc(&input_buffer_, total * sizeof(std::complex<float>));
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: input_buffer hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // 2. Magnitudes buffer: float per element (used by ComputeMedian path)
  err = hipMalloc(&magnitudes_buf_, total * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: magnitudes_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // 3. Sort output buffer (rocprim writes here)
  err = hipMalloc(&sort_buf_, total * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: sort_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // 4. Segment offsets for rocprim::segmented_radix_sort_keys
  err = hipMalloc(&offsets_buf_, (beam_count + 1) * sizeof(unsigned int));
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: offsets_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }
  {
    std::vector<unsigned int> host_offsets(beam_count + 1);
    for (size_t i = 0; i <= beam_count; ++i)
      host_offsets[i] = static_cast<unsigned int>(i * n_point);
    // TASK-5: Use async copy (AllocateBuffers is called before Execute, stream order ok)
    err = hipMemcpyAsync(offsets_buf_, host_offsets.data(),
                         (beam_count + 1) * sizeof(unsigned int),
                         hipMemcpyHostToDevice, stream_);
    if (err != hipSuccess) {
      throw std::runtime_error("AllocateBuffers: offsets fill failed: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  // 5. Query and allocate rocprim temp storage for segmented sort
  {
    auto* d_offsets = static_cast<const unsigned int*>(offsets_buf_);
    hipError_t rc = gpu_sort::QuerySortTempSize(
        sort_temp_size_,
        d_offsets,
        d_offsets + 1,
        static_cast<unsigned int>(total),
        static_cast<unsigned int>(beam_count),
        stream_);
    if (rc != hipSuccess) {
      throw std::runtime_error("AllocateBuffers: QuerySortTempSize failed: " +
                                std::string(hipGetErrorString(rc)));
    }
    if (sort_temp_size_ > 0) {
      err = hipMalloc(&sort_temp_buf_, sort_temp_size_);
      if (err != hipSuccess) {
        throw std::runtime_error("AllocateBuffers: sort_temp hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
      }
    }
  }

  // 6. Reduce buffer: partial sums for mean reduction
  // TASK-4.3: Double-load → blocks_per_beam is halved
  size_t blocks_per_beam = (n_point + kDoubleLoadElements - 1) / kDoubleLoadElements;
  size_t reduce_count = beam_count * blocks_per_beam;
  err = hipMalloc(&reduce_buf_, reduce_count * 2 * sizeof(float));  // float2_t
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: reduce_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // 7. Result buffer: max of {beam_count * sizeof(float2_t), beam_count * sizeof(WelfordResult)}
  //    WelfordResult = 5 floats (largest)
  size_t result_size = beam_count * 5 * sizeof(float);
  err = hipMalloc(&result_buf_, result_size);
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: result_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // 8. TASK-2: Compact medians buffer (beam_count floats)
  err = hipMalloc(&medians_compact_buf_, beam_count * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("AllocateBuffers: medians_compact_buf hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  current_beams_   = beam_count;
  current_n_point_ = n_point;
}

void StatisticsProcessor::ReleaseResources() {
  if (input_buffer_)        { (void)hipFree(input_buffer_);        input_buffer_ = nullptr; }
  if (magnitudes_buf_)      { (void)hipFree(magnitudes_buf_);      magnitudes_buf_ = nullptr; }
  if (sort_buf_)            { (void)hipFree(sort_buf_);            sort_buf_ = nullptr; }
  if (sort_temp_buf_)       { (void)hipFree(sort_temp_buf_);       sort_temp_buf_ = nullptr; }
  if (offsets_buf_)         { (void)hipFree(offsets_buf_);         offsets_buf_ = nullptr; }
  if (reduce_buf_)          { (void)hipFree(reduce_buf_);          reduce_buf_ = nullptr; }
  if (result_buf_)          { (void)hipFree(result_buf_);          result_buf_ = nullptr; }
  if (medians_compact_buf_) { (void)hipFree(medians_compact_buf_); medians_compact_buf_ = nullptr; }

  if (module_) {
    (void)hipModuleUnload(module_);
    module_                = nullptr;
    magnitudes_kernel_     = nullptr;
    mean_reduce_kernel_    = nullptr;
    mean_final_kernel_     = nullptr;
    welford_kernel_        = nullptr;
    welford_fused_kernel_  = nullptr;
    welford_float_kernel_  = nullptr;
    extract_medians_kernel_ = nullptr;
    kernels_compiled_      = false;
  }

  current_beams_   = 0;
  current_n_point_ = 0;
  sort_temp_size_  = 0;
}

// =========================================================================
// PART 4: GPU Operations
// =========================================================================

void StatisticsProcessor::UploadData(const std::complex<float>* data, size_t count) {
  size_t data_size = count * sizeof(std::complex<float>);

  hipError_t err = hipMemcpyHtoDAsync(input_buffer_,
                                       const_cast<std::complex<float>*>(data),
                                       data_size, stream_);
  if (err != hipSuccess) {
    throw std::runtime_error("UploadData: hipMemcpyHtoDAsync failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::CopyGpuData(void* src, size_t count) {
  size_t data_size = count * sizeof(std::complex<float>);

  hipError_t err = hipMemcpyDtoDAsync(input_buffer_, src, data_size, stream_);
  if (err != hipSuccess) {
    throw std::runtime_error("CopyGpuData: hipMemcpyDtoDAsync failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// P3-D: double-load — each thread processes 2 elements, grid halved
void StatisticsProcessor::ExecuteMagnitudesKernel(size_t total_elements) {
  unsigned int total     = static_cast<unsigned int>(total_elements);
  unsigned int grid_size = (total + kBlockSize * 2 - 1) / (kBlockSize * 2);

  void* args[] = {
    &input_buffer_,
    &magnitudes_buf_,
    &total
  };

  hipError_t err = hipModuleLaunchKernel(
      magnitudes_kernel_,
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteMagnitudesKernel: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// TASK-4.2, 4.3: double-load, blocks_per_beam parameter
void StatisticsProcessor::ExecuteMeanReduction(size_t beam_count, size_t n_point) {
  unsigned int bc  = static_cast<unsigned int>(beam_count);
  unsigned int np  = static_cast<unsigned int>(n_point);

  // TASK-4.3: Double-load — blocks_per_beam is halved compared to v1
  unsigned int blocks_per_beam = (np + kDoubleLoadElements - 1) / kDoubleLoadElements;

  // P2-B: LDS is now static __shared__ (256+1 float per component) — no dynamic alloc
  size_t shared_mem = 0;

  // Phase 1: block-level reduction
  // P3-A: 2D grid — gridDimX=blocks_per_beam, gridDimY=beam_count (no div/mod in kernel)
  void* args1[] = {
    &input_buffer_,
    &reduce_buf_,
    &bc,
    &np
  };

  hipError_t err = hipModuleLaunchKernel(
      mean_reduce_kernel_,
      blocks_per_beam, bc, 1,
      kBlockSize, 1, 1,
      shared_mem, stream_,
      args1, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteMeanReduction phase1: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // Phase 2: final reduction (one block per beam)
  // Block size for final: min(blocks_per_beam, kBlockSize) rounded up to power of 2
  unsigned int final_block = 1;
  while (final_block < blocks_per_beam && final_block < kBlockSize) {
    final_block *= 2;
  }
  if (final_block > kBlockSize) final_block = kBlockSize;

  // P2-B: LDS is now static __shared__ (256+1 float per component) — no dynamic alloc
  size_t final_shared = 0;

  void* args2[] = {
    &reduce_buf_,
    &result_buf_,
    &bc,
    &blocks_per_beam,
    &np
  };

  err = hipModuleLaunchKernel(
      mean_final_kernel_,
      bc, 1, 1,
      final_block, 1, 1,
      final_shared, stream_,
      args2, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteMeanReduction final: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// Original Welford (kept for backward compat)
// P3-B: shared_mem with +1 padding per array
void StatisticsProcessor::ExecuteWelfordKernel(size_t beam_count, size_t n_point) {
  unsigned int bc = static_cast<unsigned int>(beam_count);
  unsigned int np = static_cast<unsigned int>(n_point);

  size_t shared_mem = 4 * (kBlockSize + 1) * sizeof(float);

  void* args[] = {
    &input_buffer_,
    &magnitudes_buf_,
    &result_buf_,
    &bc,
    &np
  };

  hipError_t err = hipModuleLaunchKernel(
      welford_kernel_,
      bc, 1, 1,
      kBlockSize, 1, 1,
      shared_mem, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteWelfordKernel: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// TASK-1: Fused Welford — reads only input[], no magnitudes buffer
// P3-B: shared_mem with +1 padding per array
void StatisticsProcessor::ExecuteWelfordFusedKernel(size_t beam_count, size_t n_point) {
  unsigned int bc = static_cast<unsigned int>(beam_count);
  unsigned int np = static_cast<unsigned int>(n_point);

  // One block per beam; 4 arrays of (kBlockSize+1) floats in shared mem
  size_t shared_mem = 4 * (kBlockSize + 1) * sizeof(float);

  void* args[] = {
    &input_buffer_,   // only input — no magnitudes_buf_!
    &result_buf_,
    &bc,
    &np
  };

  hipError_t err = hipModuleLaunchKernel(
      welford_fused_kernel_,
      bc, 1, 1,
      kBlockSize, 1, 1,
      shared_mem, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteWelfordFusedKernel: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// TASK-2: Extract medians GPU kernel (one thread per beam)
void StatisticsProcessor::ExecuteExtractMediansKernel(size_t beam_count, size_t n_point) {
  unsigned int bc = static_cast<unsigned int>(beam_count);
  unsigned int np = static_cast<unsigned int>(n_point);
  unsigned int grid = (bc + kBlockSize - 1) / kBlockSize;

  void* args[] = {
    &sort_buf_,             // sorted magnitudes (rocPRIM output)
    &medians_compact_buf_,  // compact output: beam_count floats
    &np,
    &bc
  };

  hipError_t err = hipModuleLaunchKernel(
      extract_medians_kernel_,
      grid, 1, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteExtractMediansKernel: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::ExecuteMedianSort(size_t beam_count, size_t n_point) {
  // GPU segmented radix sort: all beams sorted in ONE call (fully parallel).
  auto* d_offsets = static_cast<const unsigned int*>(offsets_buf_);

  hipError_t err = gpu_sort::ExecuteSort(
      sort_temp_buf_,
      sort_temp_size_,
      static_cast<const float*>(magnitudes_buf_),
      static_cast<float*>(sort_buf_),
      d_offsets,
      d_offsets + 1,
      static_cast<unsigned int>(beam_count * n_point),
      static_cast<unsigned int>(beam_count),
      stream_);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteMedianSort: GPU segmented sort failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

// =========================================================================
// PART 5: Float-input API (magnitudes already computed)
// =========================================================================

void StatisticsProcessor::CopyFloatGpuData(void* src, size_t count) {
  size_t data_size = count * sizeof(float);
  hipError_t err = hipMemcpyDtoDAsync(magnitudes_buf_, src, data_size, stream_);
  if (err != hipSuccess) {
    throw std::runtime_error("CopyFloatGpuData: hipMemcpyDtoDAsync failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

void StatisticsProcessor::ExecuteWelfordFloatKernel(size_t beam_count, size_t n_point) {
  unsigned int bc = static_cast<unsigned int>(beam_count);
  unsigned int np = static_cast<unsigned int>(n_point);

  // P3-B: 2 arrays of (kBlockSize+1) floats in shared mem (sum_mag + sum_sq)
  size_t shared_mem = 2 * (kBlockSize + 1) * sizeof(float);

  void* args[] = {
    &magnitudes_buf_,
    &result_buf_,
    &bc,
    &np
  };

  hipError_t err = hipModuleLaunchKernel(
      welford_float_kernel_,
      bc, 1, 1,
      kBlockSize, 1, 1,
      shared_mem, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error("ExecuteWelfordFloatKernel: launch failed: " +
                              std::string(hipGetErrorString(err)));
  }
}

std::vector<StatisticsResult> StatisticsProcessor::ComputeStatisticsFloat(
    void* gpu_float_data,
    const StatisticsParams& params)
{
  if (!gpu_float_data) {
    throw std::invalid_argument("ComputeStatisticsFloat: gpu_float_data is null");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);

  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyFloatGpuData(gpu_float_data, count);

  ExecuteWelfordFloatKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  struct WelfordResult {
    float mean_re, mean_im, mean_mag, variance, std_dev;
  };
  std::vector<WelfordResult> raw(params.beam_count);
  hipError_t err = hipMemcpyDtoH(raw.data(), result_buf_,
                                   params.beam_count * sizeof(WelfordResult));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeStatisticsFloat: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<StatisticsResult> results;
  results.reserve(params.beam_count);
  for (uint32_t i = 0; i < params.beam_count; ++i) {
    StatisticsResult r;
    r.beam_id = i;
    r.mean = std::complex<float>(0.0f, 0.0f);  // No complex mean for float input
    r.mean_magnitude = raw[i].mean_mag;
    r.variance = raw[i].variance;
    r.std_dev = raw[i].std_dev;
    results.push_back(r);
  }
  return results;
}

std::vector<MedianResult> StatisticsProcessor::ComputeMedianFloat(
    void* gpu_float_data,
    const StatisticsParams& params)
{
  if (!gpu_float_data) {
    throw std::invalid_argument("ComputeMedianFloat: gpu_float_data is null");
  }

  if (!kernels_compiled_) CompileKernels();
  AllocateBuffers(params.beam_count, params.n_point);

  size_t count = static_cast<size_t>(params.beam_count) * params.n_point;
  CopyFloatGpuData(gpu_float_data, count);

  // Sort magnitudes + extract medians (skip compute_magnitudes — already float)
  ExecuteMedianSort(params.beam_count, params.n_point);
  ExecuteExtractMediansKernel(params.beam_count, params.n_point);
  (void)hipStreamSynchronize(stream_);

  std::vector<float> medians_host(params.beam_count);
  hipError_t err = hipMemcpyDtoH(medians_host.data(), medians_compact_buf_,
                                   params.beam_count * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("ComputeMedianFloat: hipMemcpyDtoH failed: " +
                              std::string(hipGetErrorString(err)));
  }

  std::vector<MedianResult> results;
  results.reserve(params.beam_count);
  for (uint32_t b = 0; b < params.beam_count; ++b) {
    MedianResult r;
    r.beam_id = b;
    r.median_magnitude = medians_host[b];
    results.push_back(r);
  }
  return results;
}

}  // namespace statistics

#endif  // ENABLE_ROCM
