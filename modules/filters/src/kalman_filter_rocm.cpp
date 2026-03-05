/**
 * @file kalman_filter_rocm.cpp
 * @brief KalmanFilterROCm implementation (1D scalar Kalman)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "filters/kalman_filter_rocm.hpp"
#include "kernels/kalman_kernels_rocm.hpp"
#include "console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cstring>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

KalmanFilterROCm::KalmanFilterROCm(
    drv_gpu_lib::IBackend* backend, unsigned int block_size)
    : backend_(backend), block_size_(block_size) {
  if (!backend_ || !backend_->IsInitialized())
    throw std::runtime_error("KalmanFilterROCm: backend is null or not initialized");

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_)
    throw std::runtime_error("KalmanFilterROCm: failed to get HIP stream");

  CompileKernel();
}

KalmanFilterROCm::~KalmanFilterROCm() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

KalmanFilterROCm::KalmanFilterROCm(KalmanFilterROCm&& other) noexcept
    : backend_(other.backend_), stream_(other.stream_),
      module_(other.module_), kernel_(other.kernel_),
      kernel_compiled_(other.kernel_compiled_),
      params_(other.params_),
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

KalmanFilterROCm& KalmanFilterROCm::operator=(
    KalmanFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    module_ = other.module_;
    kernel_ = other.kernel_;
    kernel_compiled_ = other.kernel_compiled_;
    params_ = other.params_;
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

/**
 * @brief Устанавливает параметры фильтра Калмана
 *
 * Q (process noise) и R (measurement noise) должны быть > 0:
 * при Q=0 фильтр полностью "замирает" (K→0, игнорирует измерения),
 * при R=0 фильтр полностью доверяет измерениям (K→1, нет сглаживания).
 * P0 = начальная неопределённость; обычно P0 = R (мы не знаем начального состояния).
 *
 * Практика настройки: Q/R = 0.01 → сильное сглаживание; Q/R = 1.0 → быстрая реакция.
 *
 * @param params Q, R, x0, P0 — все должны быть > 0
 */
void KalmanFilterROCm::SetParams(const KalmanParams& params) {
  if (params.Q <= 0.0f)
    throw std::invalid_argument("KalmanFilterROCm: Q must be > 0");
  if (params.R <= 0.0f)
    throw std::invalid_argument("KalmanFilterROCm: R must be > 0");
  if (params.P0 <= 0.0f)
    throw std::invalid_argument("KalmanFilterROCm: P0 must be > 0");
  params_ = params;
}

void KalmanFilterROCm::SetParams(float Q, float R, float x0, float P0) {
  SetParams({Q, R, x0, P0});
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation
// ════════════════════════════════════════════════════════════════════════════

void KalmanFilterROCm::CompileKernel() {
  if (kernel_compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const std::string cache_name = "kalman_kernel_rocm";

  // ── Try loading from KernelCacheService (HSACO fast path) ──
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    auto entry = cache.Load(cache_name);
    if (entry.has_binary()) {
      hipError_t hipErr = hipModuleLoadData(
          &module_, entry.binary.data());
      if (hipErr == hipSuccess) {
        hipErr = hipModuleGetFunction(&kernel_, module_, "kalman_kernel");
        if (hipErr == hipSuccess) {
          kernel_compiled_ = true;
          con.Print(0, "Kalman[ROCm]",
              "HIP kernel loaded from cache (kalman_kernel)");
          return;
        }
      }
    }
  } catch (...) {
    // Cache miss — compile from source
  }

  // ── Compile from source (hiprtc) ──
  const char* source = kernels::GetKalmanSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "kalman_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS)
    throw std::runtime_error("KalmanFilterROCm: hiprtcCreateProgram failed");

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

    con.PrintError(0, "Kalman[ROCm]", "Compile log:\n" + log);

    hiprtcDestroyProgram(&prog);
    throw std::runtime_error("KalmanFilterROCm: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess)
    throw std::runtime_error("KalmanFilterROCm: hipModuleLoadData failed");

  hipErr = hipModuleGetFunction(&kernel_, module_, "kalman_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("KalmanFilterROCm: hipModuleGetFunction(kalman_kernel) failed");

  kernel_compiled_ = true;

  // ── Save to cache for next time ──
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    std::vector<uint8_t> binary(code.begin(), code.end());
    cache.Save(cache_name, std::string(source), binary,
               arch_name, "1D scalar Kalman filter");
  } catch (...) {
    // Non-critical: cache save failure
  }

  con.Print(0, "Kalman[ROCm]",
      "HIP kernel compiled (kalman_kernel)" +
      (arch_name.empty() ? "" : " [" + arch_name + "]"));
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
KalmanFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr || channels == 0 || points == 0)
    throw std::runtime_error("KalmanFilterROCm::Process: invalid arguments");
  if (!kernel_compiled_)
    throw std::runtime_error("KalmanFilterROCm::Process: kernel not compiled");

  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("KalmanFilterROCm: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));

  unsigned int ch = channels;
  unsigned int pts = points;
  float Q = params_.Q;
  float R = params_.R;
  float x0 = params_.x0;
  float P0 = params_.P0;

  void* args[] = {
    &input_ptr, &output_ptr,
    &ch, &pts,
    &Q, &R, &x0, &P0
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
    throw std::runtime_error("KalmanFilterROCm: hipModuleLaunchKernel failed: " +
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
KalmanFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {
  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  if (data.size() < total)
    throw std::runtime_error("KalmanFilterROCm::ProcessFromCPU: data too small");

  // Кешируем input-буфер: hipMalloc дорогой (~0.5 мс).
  // При одинаковом размере просто перезаписываем — без повторной аллокации.
  if (buffer_size != cached_input_size_) {
    if (cached_input_buf_) hipFree(cached_input_buf_);
    hipError_t err = hipMalloc(&cached_input_buf_, buffer_size);
    if (err != hipSuccess)
      throw std::runtime_error("KalmanFilterROCm: hipMalloc(input) failed");
    cached_input_size_ = buffer_size;
  }

  hipMemcpyHtoD(cached_input_buf_, data.data(), buffer_size);
  return Process(cached_input_buf_, channels, points);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
KalmanFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) const {
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> output(total);

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    // Re и Im фильтруются независимо (2 отдельных скалярных фильтра).
    // ПОЧЕМУ: предполагаем AWGN-шум — Re и Im не коррелированы.
    // Для коррелированного шума нужен полноценный 2D Kalman, но он в 4× дороже.
    float x_re = params_.x0, x_im = params_.x0;
    float P_re = params_.P0, P_im = params_.P0;

    for (uint32_t n = 0; n < points; ++n) {
      auto z = input[base + n];

      // ── Predict & Update Re ───────────────────────────────────────────
      float P_pred_re = P_re + params_.Q;
      float K_re = P_pred_re / (P_pred_re + params_.R);  // Kalman gain
      x_re = x_re + K_re * (z.real() - x_re);            // update state
      P_re = (1.0f - K_re) * P_pred_re;                  // update covariance

      // ── Predict & Update Im ───────────────────────────────────────────
      float P_pred_im = P_im + params_.Q;
      float K_im = P_pred_im / (P_pred_im + params_.R);
      x_im = x_im + K_im * (z.imag() - x_im);
      P_im = (1.0f - K_im) * P_pred_im;

      output[base + n] = {x_re, x_im};
    }
  }
  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup
// ════════════════════════════════════════════════════════════════════════════

void KalmanFilterROCm::ReleaseGpuResources() {
  if (cached_input_buf_) { hipFree(cached_input_buf_); cached_input_buf_ = nullptr; }
  cached_input_size_ = 0;
  if (module_) { hipModuleUnload(module_); module_ = nullptr; }
  kernel_ = nullptr;
  kernel_compiled_ = false;
}

}  // namespace filters

#endif  // ENABLE_ROCM
