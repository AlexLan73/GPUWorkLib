/**
 * @file fir_filter.cpp
 * @brief FirFilter implementation - GPU FIR convolution (OpenCL)
 *
 * Direct-form convolution with __constant or __global coefficients.
 * 2D NDRange: (channels, points) for multi-channel parallel filtering.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "filters/fir_filter.hpp"
#include "kernels/fir_kernels.hpp"

#include "common/backend_type.hpp"

#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

// Path to filters kernels/ set by CMake (fallback: relative)
#ifndef FILTERS_KERNELS_DIR
#define FILTERS_KERNELS_DIR "modules/filters/kernels"
#endif

/**
 * @brief Конструктор: инициализирует FIR фильтр и компилирует OpenCL-ядро
 *
 * Компиляция ядра выполняется сразу (не lazy) — стоимость первого Process()
 * должна быть предсказуемой, без скачка. KernelCacheService при первом запуске
 * компилирует из source (~50 мс), сохраняет SPIR-V/binary на диск;
 * при повторном запуске загружает бинарник (~1 мс).
 *
 * @param backend Указатель на IBackend (не владеет), должен пережить объект
 */
FirFilter::FirFilter(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "FirFilter: backend is null or not initialized");
  }

  // Cast void* из IBackend в нативные OpenCL-дескрипторы — безопасно,
  // т.к. OpenCL-backend гарантированно возвращает именно эти типы
  context_ = static_cast<cl_context>(backend_->GetNativeContext());
  queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
  device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

  kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
      FILTERS_KERNELS_DIR, drv_gpu_lib::BackendType::OPENCL);

  CompileKernel();
}

FirFilter::~FirFilter() {
  ReleaseGpuResources();
}

FirFilter::FirFilter(FirFilter&& other) noexcept
    : backend_(other.backend_)
    , coefficients_(std::move(other.coefficients_))
    , use_global_coeffs_(other.use_global_coeffs_)
    , kernel_cache_(std::move(other.kernel_cache_))
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_)
    , coeff_buf_(other.coeff_buf_)
    , kernel_constant_(other.kernel_constant_)
    , kernel_global_(other.kernel_global_) {
  other.program_         = nullptr;
  other.coeff_buf_       = nullptr;
  other.kernel_constant_ = nullptr;
  other.kernel_global_   = nullptr;
}

FirFilter& FirFilter::operator=(FirFilter&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_           = other.backend_;
    coefficients_      = std::move(other.coefficients_);
    use_global_coeffs_ = other.use_global_coeffs_;
    kernel_cache_      = std::move(other.kernel_cache_);
    context_           = other.context_;
    queue_             = other.queue_;
    device_            = other.device_;
    program_           = other.program_;
    coeff_buf_         = other.coeff_buf_;
    kernel_constant_   = other.kernel_constant_;
    kernel_global_     = other.kernel_global_;
    other.program_         = nullptr;
    other.coeff_buf_       = nullptr;
    other.kernel_constant_ = nullptr;
    other.kernel_global_   = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void FirFilter::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "fir") {
    throw std::runtime_error(
        "FirFilter::LoadConfig: expected type 'fir', got '" + cfg.type + "'");
  }
  SetCoefficients(cfg.coefficients);
}

/**
 * @brief Устанавливает коэффициенты FIR-фильтра и загружает их на GPU
 *
 * При num_taps > kMaxConstantTaps (16000) автоматически переключается
 * на `__global` буфер вместо `__constant` памяти OpenCL. `__constant` быстрее
 * (cached, broadcast), но ограничена ~64 KB. Для больших фильтров (>16000 тапов)
 * используется `fir_filter_cf32_global` — та же математика, медленнее.
 *
 * @param coeffs h[k] — коэффициенты импульсной характеристики, length = num_taps
 * @throws std::invalid_argument если массив пуст
 */
void FirFilter::SetCoefficients(const std::vector<float>& coeffs) {
  if (coeffs.empty()) {
    throw std::invalid_argument("FirFilter::SetCoefficients: empty coefficients");
  }

  coefficients_ = coeffs;
  // Флаг выбора kernel: constant-memory fast path vs global-memory fallback
  use_global_coeffs_ = (coefficients_.size() > kMaxConstantTaps);

  UploadCoefficients();
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<cl_mem>
FirFilter::Process(cl_mem input_buf, uint32_t channels, uint32_t points,
                   ProfEvents* prof_events) {
  if (channels == 0 || points == 0) {
    throw std::runtime_error("FirFilter::Process: channels or points is 0");
  }
  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilter::Process: no coefficients set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size  = total_points * sizeof(std::complex<float>);

  cl_int err;

  // Output buffer
  cl_mem output_buf = clCreateBuffer(
      context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FirFilter: clCreateBuffer(output) failed: " + std::to_string(err));
  }

  // Выбираем закешированный kernel (создан в CompileKernel, не пересоздаём!)
  cl_kernel k = use_global_coeffs_ ? kernel_global_ : kernel_constant_;
  if (!k) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error("FirFilter: kernel not compiled");
  }

  // Set arguments
  uint32_t num_taps = static_cast<uint32_t>(coefficients_.size());
  uint32_t pts = points;

  err  = clSetKernelArg(k, 0, sizeof(cl_mem),   &input_buf);
  err |= clSetKernelArg(k, 1, sizeof(cl_mem),   &output_buf);
  err |= clSetKernelArg(k, 2, sizeof(cl_mem),   &coeff_buf_);
  err |= clSetKernelArg(k, 3, sizeof(uint32_t), &num_taps);
  err |= clSetKernelArg(k, 4, sizeof(uint32_t), &pts);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(output_buf);
    throw std::runtime_error("FirFilter: clSetKernelArg failed");
  }

  // 2D NDRange: dim0 = channels (один work-item = один канал),
  // dim1 = points (один work-item = один семпл в канале).
  // FIR параллелится по обоим измерениям — нет зависимости между семплами!
  // local_x=1: между каналами общая память не нужна, каждый канал независим.
  // local_y=256: оптимальный размер group для GPU (одна wavefront = 64 на AMD).
  size_t local_x = 1;
  size_t local_y = 256;
  size_t global_x = channels;
  size_t global_y = ((static_cast<size_t>(points) + local_y - 1) / local_y) * local_y;

  size_t global_size[2] = { global_x, global_y };
  size_t local_size[2]  = { local_x,  local_y  };

  cl_event kernel_event = nullptr;
  err = clEnqueueNDRangeKernel(
      queue_, k, 2, nullptr,
      global_size, local_size,
      0, nullptr, &kernel_event);

  if (err != CL_SUCCESS) {
    if (kernel_event) clReleaseEvent(kernel_event);
    clReleaseMemObject(output_buf);
    throw std::runtime_error(
        "FirFilter: enqueue failed: " + std::to_string(err));
  }

  // CollectOrRelease ПЕРЕД clFinish: event должен быть передан профайлеру
  // пока он ещё не завершён — иначе время выполнения окажется 0.
  CollectOrRelease(kernel_event, "Kernel", prof_events);

  clFinish(queue_);

  // Build result
  drv_gpu_lib::InputData<cl_mem> result;
  result.antenna_count   = channels;
  result.n_point         = points;
  result.data            = output_buf;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference Implementation
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
FirFilter::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilter::ProcessCpu: no coefficients set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("FirFilter::ProcessCpu: input too small");
  }

  std::vector<std::complex<float>> output(total, {0.0f, 0.0f});
  uint32_t num_taps = static_cast<uint32_t>(coefficients_.size());

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;
    for (uint32_t n = 0; n < points; ++n) {
      std::complex<float> acc(0.0f, 0.0f);
      for (uint32_t k = 0; k < num_taps; ++k) {
        int idx = static_cast<int>(n) - static_cast<int>(k);
        if (idx >= 0) {
          acc += coefficients_[k] * input[base + idx];
        }
      }
      output[base + n] = acc;
    }
  }

  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Internals
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Компилирует OpenCL-ядро FIR фильтра (или загружает из кеша)
 *
 * Стратегия: cache-first.
 * 1. Пробуем загрузить бинарник из KernelCacheService (~1 мс).
 * 2. При промахе — компилируем из source (~50 мс), сохраняем в cache.
 * Флаг `-cl-fast-relaxed-math` разрешает агрессивные FP оптимизации
 * (reorder, fast reciprocal) — допустимо для float32 ЦОС.
 */
void FirFilter::CompileKernel() {
  const std::string kernel_name = "fir_filter_cf32";

  // Try loading from cache (binary fast path ~1 ms)
  try {
    auto entry = kernel_cache_->Load(kernel_name);
    if (entry.has_binary()) {
      LoadFromBinary(entry.binary);
      return;
    }
  } catch (...) {
    // Cache miss or load error — compile from source
  }

  // Compile from source (~50 ms first time)
  cl_int err;
  const char* source = kernels::GetFirDirectSource_opencl();
  size_t source_len = strlen(source);

  program_ = clCreateProgramWithSource(
      context_, 1, &source, &source_len, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FirFilter: clCreateProgramWithSource failed: " + std::to_string(err));
  }

  err = clBuildProgram(
      program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(
        program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error(
        "FirFilter kernel build failed:\n" + std::string(log.data()));
  }

  // Save to cache for next time
  try {
    auto binary = GetProgramBinary();
    kernel_cache_->Save(kernel_name, std::string(source),
                        binary, "", "FIR direct-form");
  } catch (...) {
    // Non-critical: cache save failed
  }

  CreateKernels();
}

void FirFilter::CreateKernels() {
  cl_int err;

  if (!kernel_constant_) {
    kernel_constant_ = clCreateKernel(program_, "fir_filter_cf32", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "FirFilter::CreateKernels: fir_filter_cf32 failed: " + std::to_string(err));
    }
  }

  if (!kernel_global_) {
    kernel_global_ = clCreateKernel(program_, "fir_filter_cf32_global", &err);
    if (err != CL_SUCCESS) {
      // Non-critical: global-coeff kernel may not exist in all builds
      kernel_global_ = nullptr;
    }
  }
}

std::vector<uint8_t> FirFilter::GetProgramBinary() const {
  if (!program_) {
    throw std::runtime_error("FirFilter::GetProgramBinary: no compiled program");
  }

  // Получаем количество устройств, связанных с программой
  cl_uint num_devices = 0;
  cl_int err = clGetProgramInfo(program_, CL_PROGRAM_NUM_DEVICES,
                                 sizeof(cl_uint), &num_devices, nullptr);
  if (err != CL_SUCCESS || num_devices == 0) {
    throw std::runtime_error("FirFilter::GetProgramBinary: cannot get num devices");
  }

  // CL_PROGRAM_BINARY_SIZES возвращает массив size_t[num_devices]
  std::vector<size_t> binary_sizes(num_devices, 0);
  err = clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                          num_devices * sizeof(size_t), binary_sizes.data(), nullptr);
  if (err != CL_SUCCESS || binary_sizes[0] == 0) {
    throw std::runtime_error("FirFilter::GetProgramBinary: cannot get binary size");
  }

  // Возвращаем бинарник первого устройства
  std::vector<uint8_t> binary(binary_sizes[0]);
  std::vector<uint8_t*> ptrs(num_devices, nullptr);
  ptrs[0] = binary.data();

  err = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                          num_devices * sizeof(uint8_t*), ptrs.data(), nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("FirFilter::GetProgramBinary: cannot get binary");
  }

  return binary;
}

void FirFilter::LoadFromBinary(const std::vector<uint8_t>& binary) {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }

  cl_int err, binary_status;
  size_t bin_size = binary.size();
  const unsigned char* bin_ptr = binary.data();

  program_ = clCreateProgramWithBinary(
      context_, 1, &device_, &bin_size, &bin_ptr, &binary_status, &err);

  if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
    program_ = nullptr;
    throw std::runtime_error(
        "FirFilter::LoadFromBinary: clCreateProgramWithBinary failed");
  }

  err = clBuildProgram(program_, 1, &device_,
                       "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program_);
    program_ = nullptr;
    throw std::runtime_error("FirFilter::LoadFromBinary: clBuildProgram failed");
  }

  CreateKernels();
}

/**
 * @brief Загружает коэффициенты на GPU в OpenCL-буфер
 *
 * Используем CL_MEM_COPY_HOST_PTR — данные копируются сразу при создании
 * буфера, без явного clEnqueueWriteBuffer. После этого host-вектор
 * coefficients_ может быть изменён без влияния на GPU-буфер.
 * При повторном вызове старый буфер освобождается и создаётся новый.
 */
void FirFilter::UploadCoefficients() {
  if (coeff_buf_) {
    clReleaseMemObject(coeff_buf_);
    coeff_buf_ = nullptr;
  }

  cl_int err;
  cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

  coeff_buf_ = clCreateBuffer(
      context_, flags,
      coefficients_.size() * sizeof(float),
      coefficients_.data(), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "FirFilter: upload coefficients failed: " + std::to_string(err));
  }
}

void FirFilter::ReleaseGpuResources() {
  if (kernel_constant_) {
    clReleaseKernel(kernel_constant_);
    kernel_constant_ = nullptr;
  }
  if (kernel_global_) {
    clReleaseKernel(kernel_global_);
    kernel_global_ = nullptr;
  }
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
  if (coeff_buf_) {
    clReleaseMemObject(coeff_buf_);
    coeff_buf_ = nullptr;
  }
}

} // namespace filters
