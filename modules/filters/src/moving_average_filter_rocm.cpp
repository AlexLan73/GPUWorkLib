/**
 * @file moving_average_filter_rocm.cpp
 * @brief MovingAverageFilterROCm implementation (SMA, EMA, MMA, DEMA, TEMA)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "filters/moving_average_filter_rocm.hpp"
#include "kernels/moving_average_kernels_rocm.hpp"
#include "console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cstring>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

MovingAverageFilterROCm::MovingAverageFilterROCm(
    drv_gpu_lib::IBackend* backend, unsigned int block_size)
    : backend_(backend), block_size_(block_size) {
  if (!backend_ || !backend_->IsInitialized())
    throw std::runtime_error("MovingAverageFilterROCm: backend is null or not initialized");

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_)
    throw std::runtime_error("MovingAverageFilterROCm: failed to get HIP stream");

  // Kernels compiled lazily in EnsureKernels() — SMA requires window_size from SetParams()
}

MovingAverageFilterROCm::~MovingAverageFilterROCm() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

MovingAverageFilterROCm::MovingAverageFilterROCm(
    MovingAverageFilterROCm&& other) noexcept
    : backend_(other.backend_), stream_(other.stream_),
      module_(other.module_),
      kernel_sma_(other.kernel_sma_), kernel_ema_(other.kernel_ema_),
      kernel_mma_(other.kernel_mma_), kernel_dema_(other.kernel_dema_),
      kernel_tema_(other.kernel_tema_),
      kernel_compiled_(other.kernel_compiled_),
      ma_type_(other.ma_type_), window_size_(other.window_size_),
      alpha_(other.alpha_),
      cached_input_buf_(other.cached_input_buf_),
      cached_input_size_(other.cached_input_size_),
      block_size_(other.block_size_),
      compiled_sma_window_(other.compiled_sma_window_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
  other.kernel_sma_ = nullptr;
  other.kernel_ema_ = nullptr;
  other.kernel_mma_ = nullptr;
  other.kernel_dema_ = nullptr;
  other.kernel_tema_ = nullptr;
  other.kernel_compiled_ = false;
  other.cached_input_buf_ = nullptr;
  other.cached_input_size_ = 0;
  other.compiled_sma_window_ = 0;
}

MovingAverageFilterROCm& MovingAverageFilterROCm::operator=(
    MovingAverageFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    module_ = other.module_;
    kernel_sma_ = other.kernel_sma_;
    kernel_ema_ = other.kernel_ema_;
    kernel_mma_ = other.kernel_mma_;
    kernel_dema_ = other.kernel_dema_;
    kernel_tema_ = other.kernel_tema_;
    kernel_compiled_ = other.kernel_compiled_;
    ma_type_ = other.ma_type_;
    window_size_ = other.window_size_;
    alpha_ = other.alpha_;
    cached_input_buf_ = other.cached_input_buf_;
    cached_input_size_ = other.cached_input_size_;
    block_size_ = other.block_size_;
    compiled_sma_window_ = other.compiled_sma_window_;

    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.kernel_sma_ = nullptr;
    other.kernel_ema_ = nullptr;
    other.kernel_mma_ = nullptr;
    other.kernel_dema_ = nullptr;
    other.kernel_tema_ = nullptr;
    other.kernel_compiled_ = false;
    other.cached_input_buf_ = nullptr;
    other.cached_input_size_ = 0;
    other.compiled_sma_window_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::SetParams(const MovingAverageParams& params) {
  SetParams(params.type, params.window_size);
}

/**
 * @brief Устанавливает тип и размер окна скользящей средней
 *
 * Вычисляет alpha (сглаживающий коэффициент) для передачи в kernel:
 * - EMA/DEMA/TEMA: alpha = 2/(N+1) — классическая формула: при N=10 → alpha=0.182
 * - MMA (Wilder): alpha = 1/N — более медленная реакция чем EMA при том же N
 * - SMA: alpha не используется ядром (работает ring buffer + inv_N)
 *
 * SMA ring buffer размером N хранится в thread-local регистрах. Размер задаётся через
 * hiprtc define -DN_WINDOW=<window_size> при компиляции — нет ограничения 128.
 *
 * @param type Тип скользящей средней (SMA/EMA/MMA/DEMA/TEMA)
 * @param window_size N — размер окна; SMA: max 128
 */
void MovingAverageFilterROCm::SetParams(MAType type, uint32_t window_size) {
  if (window_size == 0)
    throw std::invalid_argument("MovingAverageFilterROCm: window_size must be > 0");

  ma_type_ = type;
  window_size_ = window_size;

  switch (type) {
    case MAType::EMA:
    case MAType::DEMA:
    case MAType::TEMA:
      alpha_ = 2.0f / (static_cast<float>(window_size) + 1.0f);
      break;
    case MAType::MMA:
      alpha_ = 1.0f / static_cast<float>(window_size);
      break;
    case MAType::SMA:
      alpha_ = 1.0f / static_cast<float>(window_size);  // not used by kernel
      break;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::EnsureKernels() {
  // For SMA: recompile when window_size changes (N_WINDOW define changes).
  // For EMA/MMA/DEMA/TEMA: compile once with any N_WINDOW (not used by those kernels).
  bool sma_window_changed = (ma_type_ == MAType::SMA &&
                              compiled_sma_window_ != window_size_);
  if (kernel_compiled_ && !sma_window_changed) return;

  if (kernel_compiled_) {
    hipModuleUnload(module_);
    module_ = nullptr;
    kernel_sma_ = kernel_ema_ = kernel_mma_ = kernel_dema_ = kernel_tema_ = nullptr;
    kernel_compiled_ = false;
  }
  // Pass actual window_size for SMA; for non-SMA pass window_size_ anyway
  // (EMA/MMA/DEMA/TEMA don't use N_WINDOW — it just lives in the compiled module unused)
  CompileKernels(window_size_);
  compiled_sma_window_ = window_size_;
}

void MovingAverageFilterROCm::CompileKernels(uint32_t sma_window) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const std::string cache_name = "moving_average_kernels_rocm_N" + std::to_string(sma_window);

  // ── Try loading from KernelCacheService (HSACO fast path) ──
  {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    auto entry = cache.Load(cache_name);
    if (entry && entry->has_binary()) {
      hipError_t hipErr = hipModuleLoadData(
          &module_, entry->binary.data());
      if (hipErr == hipSuccess) {
        LoadKernelFunctions();
        kernel_compiled_ = true;
        con.Print(0, "MAFilter[ROCm]",
            "HIP kernels loaded from cache (sma/ema/mma/dema/tema)");
        return;
      }
    }
  }

  // ── Compile from source (hiprtc) ──
  const char* source = kernels::GetMovingAverageSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "moving_average_kernels.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS)
    throw std::runtime_error("MovingAverageFilterROCm: hiprtcCreateProgram failed");

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
  std::string n_window_def   = "-DN_WINDOW="   + std::to_string(sma_window);
  std::vector<const char*> opts = {"-O3", "-DWARP_SIZE=32", block_size_def.c_str(), n_window_def.c_str()};
  if (!arch_flag.empty())
    opts.push_back(arch_flag.c_str());

  rtcResult = hiprtcCompileProgram(prog,
      static_cast<int>(opts.size()), opts.data());
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    con.PrintError(0, "MAFilter[ROCm]", "Compile log:\n" + log);

    hiprtcDestroyProgram(&prog);
    throw std::runtime_error("MovingAverageFilterROCm: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleLoadData failed");

  LoadKernelFunctions();
  kernel_compiled_ = true;

  // ── Save to cache for next time ──
  try {
    drv_gpu_lib::KernelCacheService cache(
        FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::ROCm);
    std::vector<uint8_t> binary(code.begin(), code.end());
    cache.Save(cache_name, std::string(source), binary,
               arch_name, "MA filters: SMA/EMA/MMA/DEMA/TEMA");
  } catch (...) {
    // Non-critical: cache save failure
  }

  con.Print(0, "MAFilter[ROCm]",
      "HIP kernels compiled (sma/ema/mma/dema/tema)" +
      (arch_name.empty() ? "" : " [" + arch_name + "]"));
}

void MovingAverageFilterROCm::LoadKernelFunctions() {
  hipError_t hipErr;

  hipErr = hipModuleGetFunction(&kernel_sma_,  module_, "sma_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleGetFunction(sma_kernel) failed");

  hipErr = hipModuleGetFunction(&kernel_ema_,  module_, "ema_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleGetFunction(ema_kernel) failed");

  hipErr = hipModuleGetFunction(&kernel_mma_,  module_, "mma_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleGetFunction(mma_kernel) failed");

  hipErr = hipModuleGetFunction(&kernel_dema_, module_, "dema_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleGetFunction(dema_kernel) failed");

  hipErr = hipModuleGetFunction(&kernel_tema_, module_, "tema_kernel");
  if (hipErr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleGetFunction(tema_kernel) failed");
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
MovingAverageFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr || channels == 0 || points == 0)
    throw std::runtime_error("MovingAverageFilterROCm::Process: invalid arguments");
  EnsureKernels();

  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  // Allocate output (caller must hipFree)
  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));

  unsigned int ch = channels;
  unsigned int pts = points;
  unsigned int N = window_size_;
  float inv_N = 1.0f / static_cast<float>(window_size_);
  float alpha = alpha_;

  // Выбор kernel и набора аргументов по типу MA:
  // SMA: signature (in, out, ch, pts, N, inv_N) — 6 аргументов
  //      N передаём явно для ring buffer, inv_N = 1/N — избегаем деления в kernel
  // EMA/MMA/DEMA/TEMA: signature (in, out, ch, pts, alpha) — 5 аргументов
  //      Alpha уже вычислен в SetParams() и не зависит от N во время работы
  hipFunction_t kernel = nullptr;
  void* args[7];

  if (ma_type_ == MAType::SMA) {
    kernel = kernel_sma_;
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &ch;
    args[3] = &pts;
    args[4] = &N;
    args[5] = &inv_N;  // precomputed 1/N: передаём чтобы избежать деления в kernel
  } else {
    // EMA, MMA, DEMA, TEMA — единая сигнатура с alpha
    switch (ma_type_) {
      case MAType::EMA:  kernel = kernel_ema_;  break;
      case MAType::MMA:  kernel = kernel_mma_;  break;
      case MAType::DEMA: kernel = kernel_dema_; break;
      case MAType::TEMA: kernel = kernel_tema_; break;
      default: break;
    }
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &ch;
    args[3] = &pts;
    args[4] = &alpha;
  }

  // 1D grid: one thread per channel
  unsigned int grid_x = (channels + block_size_ - 1) / block_size_;

  err = hipModuleLaunchKernel(
      kernel,
      grid_x, 1, 1,
      block_size_, 1, 1,
      0, stream_,
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleLaunchKernel failed: " +
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
MovingAverageFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {
  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  if (data.size() < total)
    throw std::runtime_error("MovingAverageFilterROCm::ProcessFromCPU: data too small");

  // Кешируем input-буфер на GPU: hipMalloc/hipFree дорогие операции (~0.5 мс).
  // Если размер совпадает — просто перезаписываем данные без переаллокации.
  // Буфер принадлежит объекту (освобождается в ReleaseGpuResources).
  if (buffer_size != cached_input_size_) {
    if (cached_input_buf_) hipFree(cached_input_buf_);
    hipError_t err = hipMalloc(&cached_input_buf_, buffer_size);
    if (err != hipSuccess)
      throw std::runtime_error("MovingAverageFilterROCm: hipMalloc(input) failed");
    cached_input_size_ = buffer_size;
  }

  hipMemcpyHtoDAsync(cached_input_buf_, data.data(), buffer_size, stream_);
  return Process(cached_input_buf_, channels, points);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference (for testing)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
MovingAverageFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) const {
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> output(total, {0.0f, 0.0f});

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    switch (ma_type_) {
      case MAType::SMA: {
        std::vector<std::complex<float>> ring(window_size_);
        float sum_re = 0.0f, sum_im = 0.0f;
        uint32_t head = 0;
        for (uint32_t n = 0; n < points; ++n) {
          auto x = input[base + n];
          if (n < window_size_) {
            ring[n] = x;
            sum_re += x.real();
            sum_im += x.imag();
            float inv = 1.0f / static_cast<float>(n + 1);
            output[base + n] = {sum_re * inv, sum_im * inv};
          } else {
            auto old_val = ring[head];
            ring[head] = x;
            head = (head + 1) % window_size_;
            sum_re += x.real() - old_val.real();
            sum_im += x.imag() - old_val.imag();
            float inv_N = 1.0f / static_cast<float>(window_size_);
            output[base + n] = {sum_re * inv_N, sum_im * inv_N};
          }
        }
        break;
      }

      case MAType::EMA:
      case MAType::MMA: {
        auto state = input[base];
        output[base] = state;
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          state = {a * x.real() + b * state.real(),
                   a * x.imag() + b * state.imag()};
          output[base + n] = state;
        }
        break;
      }

      case MAType::DEMA: {
        auto ema1 = input[base];
        auto ema2 = input[base];
        output[base] = {2.0f * ema1.real() - ema2.real(),
                        2.0f * ema1.imag() - ema2.imag()};
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          ema1 = {a * x.real()      + b * ema1.real(),
                  a * x.imag()      + b * ema1.imag()};
          ema2 = {a * ema1.real()   + b * ema2.real(),
                  a * ema1.imag()   + b * ema2.imag()};
          output[base + n] = {2.0f * ema1.real() - ema2.real(),
                              2.0f * ema1.imag() - ema2.imag()};
        }
        break;
      }

      case MAType::TEMA: {
        auto ema1 = input[base];
        auto ema2 = input[base];
        auto ema3 = input[base];
        output[base] = {3.0f * ema1.real() - 3.0f * ema2.real() + ema3.real(),
                        3.0f * ema1.imag() - 3.0f * ema2.imag() + ema3.imag()};
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          ema1 = {a * x.real()      + b * ema1.real(),
                  a * x.imag()      + b * ema1.imag()};
          ema2 = {a * ema1.real()   + b * ema2.real(),
                  a * ema1.imag()   + b * ema2.imag()};
          ema3 = {a * ema2.real()   + b * ema3.real(),
                  a * ema2.imag()   + b * ema3.imag()};
          output[base + n] = {3.0f * ema1.real() - 3.0f * ema2.real() + ema3.real(),
                              3.0f * ema1.imag() - 3.0f * ema2.imag() + ema3.imag()};
        }
        break;
      }
    }
  }
  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::ReleaseGpuResources() {
  if (cached_input_buf_) { hipFree(cached_input_buf_); cached_input_buf_ = nullptr; }
  cached_input_size_ = 0;
  if (module_) { hipModuleUnload(module_); module_ = nullptr; }
  kernel_sma_ = nullptr;
  kernel_ema_ = nullptr;
  kernel_mma_ = nullptr;
  kernel_dema_ = nullptr;
  kernel_tema_ = nullptr;
  kernel_compiled_ = false;
}

}  // namespace filters

#endif  // ENABLE_ROCM
