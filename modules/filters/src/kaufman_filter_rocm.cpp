/**
 * @file kaufman_filter_rocm.cpp
 * @brief KaufmanFilterROCm implementation (KAMA)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "filters/kaufman_filter_rocm.hpp"
#include "kernels/kaufman_kernels_rocm.hpp"
#include "console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cmath>
#include <cstring>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

KaufmanFilterROCm::KaufmanFilterROCm(
    drv_gpu_lib::IBackend* backend, unsigned int block_size)
    : backend_(backend), block_size_(block_size) {
  if (!backend_ || !backend_->IsInitialized())
    throw std::runtime_error("KaufmanFilterROCm: backend is null or not initialized");

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_)
    throw std::runtime_error("KaufmanFilterROCm: failed to get HIP stream");

  CompileKernel();
}

KaufmanFilterROCm::~KaufmanFilterROCm() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

KaufmanFilterROCm::KaufmanFilterROCm(KaufmanFilterROCm&& other) noexcept
    : backend_(other.backend_), stream_(other.stream_),
      module_(other.module_), kernel_(other.kernel_),
      kernel_compiled_(other.kernel_compiled_),
      params_(other.params_),
      fast_sc_(other.fast_sc_), slow_sc_(other.slow_sc_),
      cached_input_buf_(other.cached_input_buf_),
      cached_input_size_(other.cached_input_size_),
      block_size_(other.block_size_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
  other.kernel_ = nullptr;
  other.kernel_compiled_ = false;
  other.cached_input_buf_ = nullptr;
  other.cached_input_size_ = 0;
}

KaufmanFilterROCm& KaufmanFilterROCm::operator=(
    KaufmanFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    module_ = other.module_;
    kernel_ = other.kernel_;
    kernel_compiled_ = other.kernel_compiled_;
    params_ = other.params_;
    fast_sc_ = other.fast_sc_;
    slow_sc_ = other.slow_sc_;
    cached_input_buf_ = other.cached_input_buf_;
    cached_input_size_ = other.cached_input_size_;
    block_size_ = other.block_size_;

    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.kernel_ = nullptr;
    other.kernel_compiled_ = false;
    other.cached_input_buf_ = nullptr;
    other.cached_input_size_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::SetParams(const KaufmanParams& params) {
  SetParams(params.er_period, params.fast_period, params.slow_period);
}

void KaufmanFilterROCm::SetParams(uint32_t er_period,
                                   uint32_t fast_period,
                                   uint32_t slow_period) {
  if (er_period == 0)
    throw std::invalid_argument("KaufmanFilterROCm: er_period must be > 0");
  if (er_period > 128)
    throw std::invalid_argument("KaufmanFilterROCm: er_period max 128 (ring buffer limit)");
  if (fast_period == 0 || slow_period == 0)
    throw std::invalid_argument("KaufmanFilterROCm: fast/slow periods must be > 0");

  params_.er_period = er_period;
  params_.fast_period = fast_period;
  params_.slow_period = slow_period;

  fast_sc_ = 2.0f / static_cast<float>(fast_period + 1);
  slow_sc_ = 2.0f / static_cast<float>(slow_period + 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::CompileKernel() {
  if (kernel_compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const std::string cache_name = "kaufman_kernel_rocm";

  // ── Try loading from KernelCacheService (HSACO fast path) ──
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    auto entry = cache.Load(cache_name);
    if (entry.has_binary()) {
      hipError_t hipErr = hipModuleLoadData(
          &module_, entry.binary.data());
      if (hipErr == hipSuccess) {
        hipErr = hipModuleGetFunction(&kernel_, module_, "kaufman_kernel");
        if (hipErr == hipSuccess) {
          kernel_compiled_ = true;
          con.Print(0, "Kaufman[ROCm]",
              "HIP kernel loaded from cache (kaufman_kernel)");
          return;
        }
      }
    }
  } catch (...) {
    // Cache miss — compile from source
  }

  // ── Compile from source (hiprtc) ──
  const char* source = kernels::GetKaufmanSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "kaufman_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS)
    throw std::runtime_error("KaufmanFilterROCm: hiprtcCreateProgram failed");

  // Build compile options: -O3, --offload-arch, -DWARP_SIZE
  std::string arch_name;
  try {
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name = "";
  }
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
  std::string block_size_def = "-DBLOCK_SIZE=" + std::to_string(block_size_);
  std::vector<const char*> opts = {"-O3", "-DWARP_SIZE=32", block_size_def.c_str()};
  if (!arch_flag.empty())
    opts.push_back(arch_flag.c_str());

  rtcResult = hiprtcCompileProgram(prog,
      static_cast<int>(opts.size()), opts.data());
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    con.PrintError(0, "Kaufman[ROCm]", "Compile log:\n" + log);

    hiprtcDestroyProgram(&prog);
    throw std::runtime_error("KaufmanFilterROCm: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess)
    throw std::runtime_error("KaufmanFilterROCm: hipModuleLoadData failed");

  hipErr = hipModuleGetFunction(&kernel_, module_, "kaufman_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("KaufmanFilterROCm: hipModuleGetFunction(kaufman_kernel) failed");

  kernel_compiled_ = true;

  // ── Save to cache for next time ──
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    std::vector<uint8_t> binary(code.begin(), code.end());
    cache.Save(cache_name, std::string(source), binary,
               arch_name, "KAMA (Kaufman Adaptive MA)");
  } catch (...) {
    // Non-critical: cache save failure
  }

  con.Print(0, "Kaufman[ROCm]",
      "HIP kernel compiled (kaufman_kernel)" +
      (arch_name.empty() ? "" : " [" + arch_name + "]"));
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
KaufmanFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr || channels == 0 || points == 0)
    throw std::runtime_error("KaufmanFilterROCm::Process: invalid arguments");
  if (!kernel_compiled_)
    throw std::runtime_error("KaufmanFilterROCm::Process: kernel not compiled");

  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("KaufmanFilterROCm: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));

  unsigned int ch = channels;
  unsigned int pts = points;
  unsigned int N = params_.er_period;
  float fast = fast_sc_;
  float slow = slow_sc_;

  void* args[] = {
    &input_ptr, &output_ptr,
    &ch, &pts,
    &N, &fast, &slow
  };

  unsigned int grid_x = (channels + block_size_ - 1) / block_size_;

  err = hipModuleLaunchKernel(
      kernel_,
      grid_x, 1, 1,
      block_size_, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error("KaufmanFilterROCm: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  hipStreamSynchronize(stream_);

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = channels;
  result.n_point = points;
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

drv_gpu_lib::InputData<void*>
KaufmanFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {
  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  if (data.size() < total)
    throw std::runtime_error("KaufmanFilterROCm::ProcessFromCPU: data too small");

  if (buffer_size != cached_input_size_) {
    if (cached_input_buf_) hipFree(cached_input_buf_);
    hipError_t err = hipMalloc(&cached_input_buf_, buffer_size);
    if (err != hipSuccess)
      throw std::runtime_error("KaufmanFilterROCm: hipMalloc(input) failed");
    cached_input_size_ = buffer_size;
  }

  hipMemcpyHtoD(cached_input_buf_, data.data(), buffer_size);
  return Process(cached_input_buf_, channels, points);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
KaufmanFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) const {
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> output(total);

  const uint32_t N = params_.er_period;
  const float eps = 1e-8f;
  const float sc_diff = fast_sc_ - slow_sc_;

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    // Ring buffer
    std::vector<std::complex<float>> ring(N);

    // Fill ring and passthrough first N samples
    for (uint32_t i = 0; i < N && i < points; ++i) {
      ring[i] = input[base + i];
      output[base + i] = input[base + i];
    }
    if (points <= N) continue;

    auto kama = ring[0];

    // Initial volatility
    float vol_re = 0.0f, vol_im = 0.0f;
    for (uint32_t i = 1; i < N; ++i) {
      vol_re += std::abs(ring[i].real() - ring[i - 1].real());
      vol_im += std::abs(ring[i].imag() - ring[i - 1].imag());
    }

    uint32_t head = 0;

    for (uint32_t n = N; n < points; ++n) {
      auto x = input[base + n];

      uint32_t prev_idx = (head + N - 1) % N;
      auto prev_x = ring[prev_idx];
      auto oldest = ring[head];
      uint32_t before_head = (head == 0) ? N - 1 : head - 1;

      // Direction
      float dir_re = std::abs(x.real() - oldest.real());
      float dir_im = std::abs(x.imag() - oldest.imag());

      // Rolling volatility
      float old_diff_re = std::abs(ring[head].real() - ring[before_head].real());
      float old_diff_im = std::abs(ring[head].imag() - ring[before_head].imag());
      float new_diff_re = std::abs(x.real() - prev_x.real());
      float new_diff_im = std::abs(x.imag() - prev_x.imag());
      vol_re = vol_re - old_diff_re + new_diff_re;
      vol_im = vol_im - old_diff_im + new_diff_im;

      // ER
      float er_re = (vol_re > eps) ? dir_re / vol_re : 0.0f;
      float er_im = (vol_im > eps) ? dir_im / vol_im : 0.0f;

      // SC
      float sc_re = er_re * sc_diff + slow_sc_;
      float sc_im = er_im * sc_diff + slow_sc_;
      sc_re *= sc_re;
      sc_im *= sc_im;

      // Update KAMA
      float kama_re = kama.real() + sc_re * (x.real() - kama.real());
      float kama_im = kama.imag() + sc_im * (x.imag() - kama.imag());
      kama = {kama_re, kama_im};

      // Update ring
      ring[head] = x;
      head = (head + 1) % N;

      output[base + n] = kama;
    }
  }
  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::ReleaseGpuResources() {
  if (cached_input_buf_) { hipFree(cached_input_buf_); cached_input_buf_ = nullptr; }
  cached_input_size_ = 0;
  if (module_) { hipModuleUnload(module_); module_ = nullptr; }
  kernel_ = nullptr;
  kernel_compiled_ = false;
}

}  // namespace filters

#endif  // ENABLE_ROCM
