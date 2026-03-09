/**
 * @file cholesky_inverter_rocm.cpp
 * @brief CholeskyInverterROCm: Core (POTRF+POTRI) + Roundtrip + CholeskyResult
 *
 * Task_11 v2: единый CholeskyResult (void* d_data), два режима симметризации.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#if ENABLE_ROCM

#include "cholesky_inverter_rocm.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ROCm / rocBLAS / rocSOLVER / HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "services/kernel_cache_service.hpp"

// ZeroCopy (Task_08) — только если собирается с OpenCL
#ifdef CL_VERSION_1_0
#include "backends/hybrid/zero_copy_bridge.hpp"
#endif

namespace vector_algebra {

// ════════════════════════════════════════════════════════════════════════════
// CholeskyResult — реализация методов
// ════════════════════════════════════════════════════════════════════════════

CholeskyResult::~CholeskyResult() {
  if (d_data && backend) {
    backend->Free(d_data);
    d_data = nullptr;
  }
}

CholeskyResult::CholeskyResult(CholeskyResult&& other) noexcept
    : d_data(other.d_data),
      backend(other.backend),
      matrix_size(other.matrix_size),
      batch_count(other.batch_count) {
  other.d_data = nullptr;
  other.backend = nullptr;
  other.matrix_size = 0;
  other.batch_count = 0;
}

CholeskyResult& CholeskyResult::operator=(CholeskyResult&& other) noexcept {
  if (this != &other) {
    if (d_data && backend) {
      backend->Free(d_data);
    }
    d_data = other.d_data;
    backend = other.backend;
    matrix_size = other.matrix_size;
    batch_count = other.batch_count;
    other.d_data = nullptr;
    other.backend = nullptr;
    other.matrix_size = 0;
    other.batch_count = 0;
  }
  return *this;
}

std::vector<std::complex<float>> CholeskyResult::AsVector() const {
  if (!d_data || !backend) return {};
  const size_t count =
      static_cast<size_t>(matrix_size) * matrix_size * batch_count;
  std::vector<std::complex<float>> result(count);
  backend->MemcpyDeviceToHost(result.data(), d_data,
                               count * sizeof(std::complex<float>));
  backend->Synchronize();
  return result;
}

std::vector<std::vector<std::complex<float>>> CholeskyResult::matrix() const {
  auto flat = AsVector();
  const int n = matrix_size;
  std::vector<std::vector<std::complex<float>>> result(
      n, std::vector<std::complex<float>>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      result[i][j] = flat[static_cast<size_t>(i) * n + j];
  return result;
}

std::vector<std::vector<std::vector<std::complex<float>>>>
CholeskyResult::matrices() const {
  auto flat = AsVector();
  const int n = matrix_size;
  const int b = batch_count;
  std::vector<std::vector<std::vector<std::complex<float>>>> result(
      b, std::vector<std::vector<std::complex<float>>>(
             n, std::vector<std::complex<float>>(n)));
  for (int k = 0; k < b; ++k)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        result[k][i][j] =
            flat[static_cast<size_t>(k) * n * n +
                 static_cast<size_t>(i) * n + j];
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

CholeskyInverterROCm::CholeskyInverterROCm(drv_gpu_lib::IBackend* backend,
                                             SymmetrizeMode mode)
    : backend_(backend), mode_(mode) {
  rocblas_handle h = nullptr;
  rocblas_status status = rocblas_create_handle(&h);
  if (status != rocblas_status_success) {
    throw std::runtime_error(
        "CholeskyInverterROCm: rocblas_create_handle failed (" +
        std::to_string(static_cast<int>(status)) + ")");
  }
  handle_ = static_cast<void*>(h);

  auto* stream = static_cast<hipStream_t>(backend_->GetNativeQueue());
  rocblas_set_stream(h, stream);

  // Task_12: предаллокация dev_info — 2 слота (potrf + potri)
  rocblas_int* info_ptr = nullptr;
  hipError_t herr = hipMalloc(&info_ptr, 2 * sizeof(rocblas_int));
  if (herr != hipSuccess) {
    throw std::runtime_error(
        "CholeskyInverterROCm: hipMalloc(d_info_) failed");
  }
  d_info_ = static_cast<void*>(info_ptr);

  // Дисковый кеш HSACO (modules/vector_algebra/kernels/bin/)
  kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
      "modules/vector_algebra/kernels", drv_gpu_lib::BackendType::ROCm);

  // Eager compile: предкомпиляция hiprtc kernel при GpuKernel mode
  if (mode_ == SymmetrizeMode::GpuKernel) {
    CompileKernels();
  }
}

void CholeskyInverterROCm::SetSymmetrizeMode(SymmetrizeMode mode) {
  mode_ = mode;
  // Если переключились на GpuKernel — скомпилировать сразу
  if (mode_ == SymmetrizeMode::GpuKernel) {
    CompileKernels();
  }
}

CholeskyInverterROCm::~CholeskyInverterROCm() {
  if (d_info_) {
    (void)hipFree(d_info_);
    d_info_ = nullptr;
  }
  if (handle_) {
    rocblas_destroy_handle(static_cast<rocblas_handle>(handle_));
    handle_ = nullptr;
  }
  if (sym_module_) {
    hipModuleUnload(static_cast<hipModule_t>(sym_module_));
    sym_module_ = nullptr;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ResolveMatrixSize
// ════════════════════════════════════════════════════════════════════════════

int CholeskyInverterROCm::ResolveMatrixSize(uint32_t n_point,
                                             int n_hint) const {
  if (n_hint > 0) return n_hint;
  int n = static_cast<int>(
      std::round(std::sqrt(static_cast<double>(n_point))));
  if (n <= 0 || static_cast<uint32_t>(n * n) != n_point)
    throw std::invalid_argument(
        "ResolveMatrixSize: n_point=" + std::to_string(n_point) +
        " is not a perfect square");
  return n;
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeUpperToFull (CPU) — для Roundtrip mode
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeUpperToFull(std::complex<float>* data,
                                                   int n) {
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      data[static_cast<size_t>(i) * n + j] =
          std::conj(data[static_cast<size_t>(j) * n + i]);
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// CorePotrf / CorePotri (single matrix)
// ════════════════════════════════════════════════════════════════════════════

// Task_12: Оптимизация — предаллоцированный d_info_, без hipMalloc/hipFree,
// без промежуточных hipMemcpy D2H. Проверка info отложена в CheckInfo().

void CholeskyInverterROCm::CorePotrf(void* d_matrix, int n, void* /*stream*/) {
  auto* h = static_cast<rocblas_handle>(handle_);
  auto* A = static_cast<rocblas_float_complex*>(d_matrix);
  auto* info_potrf = static_cast<rocblas_int*>(d_info_);  // slot 0

  rocblas_status rs =
      rocsolver_cpotrf(h, rocblas_fill_lower, n, A, n, info_potrf);
  if (rs != rocblas_status_success) {
    throw std::runtime_error("CorePotrf: rocsolver_cpotrf failed (" +
                             std::to_string(static_cast<int>(rs)) + ")");
  }
  // info проверяется в CheckInfo() после всего pipeline
}

void CholeskyInverterROCm::CorePotri(void* d_matrix, int n, void* /*stream*/) {
  auto* h = static_cast<rocblas_handle>(handle_);
  auto* A = static_cast<rocblas_float_complex*>(d_matrix);
  auto* info_potri = static_cast<rocblas_int*>(d_info_) + 1;  // slot 1

  rocblas_status rs =
      rocsolver_cpotri(h, rocblas_fill_lower, n, A, n, info_potri);
  if (rs != rocblas_status_success) {
    throw std::runtime_error("CorePotri: rocsolver_cpotri failed (" +
                             std::to_string(static_cast<int>(rs)) + ")");
  }
  // info проверяется в CheckInfo() после всего pipeline
}

// Task_12: отложенная проверка — одна синхронизация вместо двух
void CholeskyInverterROCm::CheckInfo(const char* context) {
  rocblas_int host_info[2] = {0, 0};
  hipError_t herr = hipMemcpy(host_info, d_info_, 2 * sizeof(rocblas_int),
                               hipMemcpyDeviceToHost);
  if (herr != hipSuccess) {
    throw std::runtime_error(
        std::string(context) + ": hipMemcpy(d_info_) failed");
  }
  if (host_info[0] != 0) {
    throw std::runtime_error(
        std::string(context) +
        ": матрица не положительно определена (potrf info=" +
        std::to_string(host_info[0]) + ")");
  }
  if (host_info[1] != 0) {
    throw std::runtime_error(
        std::string(context) + ": ошибка инверсии (potri info=" +
        std::to_string(host_info[1]) + ")");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// CorePotrfBatched / CorePotriBatched (sequential per matrix)
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::CorePotrfBatched(void* d_contiguous, int n,
                                              int batch, void* stream) {
  auto* base = static_cast<char*>(d_contiguous);
  const size_t one_bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  for (int k = 0; k < batch; ++k) {
    void* ptr_k = base + static_cast<size_t>(k) * one_bytes;
    CorePotrf(ptr_k, n, stream);
  }
}

void CholeskyInverterROCm::CorePotriBatched(void* d_contiguous, int n,
                                              int batch, void* stream) {
  auto* base = static_cast<char*>(d_contiguous);
  const size_t one_bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  for (int k = 0; k < batch; ++k) {
    void* ptr_k = base + static_cast<size_t>(k) * one_bytes;
    CorePotri(ptr_k, n, stream);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeRoundtrip (download → CPU sym → upload)
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeRoundtrip(void* d_matrix, int n) {
  const size_t bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  std::vector<std::complex<float>> cpu_tmp(static_cast<size_t>(n) * n);
  backend_->MemcpyDeviceToHost(cpu_tmp.data(), d_matrix, bytes);
  backend_->Synchronize();
  SymmetrizeUpperToFull(cpu_tmp.data(), n);
  backend_->MemcpyHostToDevice(d_matrix, cpu_tmp.data(), bytes);
  backend_->Synchronize();
}

void CholeskyInverterROCm::SymmetrizeRoundtripBatched(void* d_contiguous,
                                                        int n, int batch) {
  const size_t one_bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  const size_t total_bytes = static_cast<size_t>(batch) * one_bytes;

  std::vector<std::complex<float>> cpu_tmp(
      static_cast<size_t>(batch) * n * n);
  backend_->MemcpyDeviceToHost(cpu_tmp.data(), d_contiguous, total_bytes);
  backend_->Synchronize();

  for (int k = 0; k < batch; ++k) {
    SymmetrizeUpperToFull(
        cpu_tmp.data() + static_cast<size_t>(k) * n * n, n);
  }

  backend_->MemcpyHostToDevice(d_contiguous, cpu_tmp.data(), total_bytes);
  backend_->Synchronize();
}

// ════════════════════════════════════════════════════════════════════════════
// Symmetrize dispatchers
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::Symmetrize(void* d_matrix, int n, void* stream) {
  if (mode_ == SymmetrizeMode::Roundtrip) {
    SymmetrizeRoundtrip(d_matrix, n);
  } else {
    SymmetrizeGpuKernel(d_matrix, n, stream);
  }
}

void CholeskyInverterROCm::SymmetrizeBatched(void* d_contiguous, int n,
                                               int batch, void* stream) {
  if (mode_ == SymmetrizeMode::Roundtrip) {
    SymmetrizeRoundtripBatched(d_contiguous, n, batch);
  } else {
    SymmetrizeGpuKernelBatched(d_contiguous, n, batch, stream);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Invert — CPU путь (vector → GPU → CholeskyResult)
// ════════════════════════════════════════════════════════════════════════════

CholeskyResult CholeskyInverterROCm::Invert(
    const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
    int n) {
  const int matrix_n = ResolveMatrixSize(input.n_point, n);
  const size_t bytes =
      static_cast<size_t>(matrix_n) * matrix_n * sizeof(std::complex<float>);

  void* d_output = backend_->Allocate(bytes);
  backend_->MemcpyHostToDevice(d_output, input.data.data(), bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrf(d_output, matrix_n, stream);
  CorePotri(d_output, matrix_n, stream);
  // Task_12: убрали backend_->Synchronize() — один stream гарантирует порядок
  Symmetrize(d_output, matrix_n, stream);
  if (check_info_) CheckInfo("Invert(vector)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = matrix_n;
  result.batch_count = 1;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Invert — void* (ROCm GPU pointer)
// ════════════════════════════════════════════════════════════════════════════

CholeskyResult CholeskyInverterROCm::Invert(
    const drv_gpu_lib::InputData<void*>& input, int n) {
  const int matrix_n = ResolveMatrixSize(input.n_point, n);
  const size_t bytes =
      static_cast<size_t>(matrix_n) * matrix_n * sizeof(std::complex<float>);

  void* d_output = backend_->Allocate(bytes);
  backend_->MemcpyDeviceToDevice(d_output, input.data, bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrf(d_output, matrix_n, stream);
  CorePotri(d_output, matrix_n, stream);
  // Task_12: убрали backend_->Synchronize() — один stream гарантирует порядок
  Symmetrize(d_output, matrix_n, stream);
  if (check_info_) CheckInfo("Invert(void*)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = matrix_n;
  result.batch_count = 1;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Invert — cl_mem (ZeroCopy через dma-buf)
// ════════════════════════════════════════════════════════════════════════════

#ifdef CL_VERSION_1_0
CholeskyResult CholeskyInverterROCm::Invert(
    const drv_gpu_lib::InputData<cl_mem>& input, int n) {
  const int matrix_n = ResolveMatrixSize(input.n_point, n);
  const size_t bytes =
      static_cast<size_t>(matrix_n) * matrix_n * sizeof(std::complex<float>);

  int dma_fd = drv_gpu_lib::ExportClBufferToFd(input.data);
  if (dma_fd < 0) {
    throw std::runtime_error(
        "Invert(cl_mem): ZeroCopy export failed (dma_fd < 0)");
  }

  drv_gpu_lib::ZeroCopyBridge bridge;
  bridge.ImportFromOpenCl(dma_fd, bytes);
  void* hip_in = bridge.GetHipPtr();

  void* d_output = backend_->Allocate(bytes);
  backend_->MemcpyDeviceToDevice(d_output, hip_in, bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrf(d_output, matrix_n, stream);
  CorePotri(d_output, matrix_n, stream);
  // Task_12: убрали backend_->Synchronize()
  Symmetrize(d_output, matrix_n, stream);
  if (check_info_) CheckInfo("Invert(cl_mem)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = matrix_n;
  result.batch_count = 1;
  return result;
}
#endif  // CL_VERSION_1_0

// ════════════════════════════════════════════════════════════════════════════
// InvertBatch — CPU путь
// ════════════════════════════════════════════════════════════════════════════

CholeskyResult CholeskyInverterROCm::InvertBatch(
    const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
    int n) {
  if (n <= 0)
    throw std::invalid_argument(
        "InvertBatch: n must be > 0 (got " + std::to_string(n) + ")");
  const int batch = static_cast<int>(input.antenna_count);
  const size_t total_bytes = static_cast<size_t>(batch) * n * n *
                             sizeof(std::complex<float>);

  void* d_output = backend_->Allocate(total_bytes);
  backend_->MemcpyHostToDevice(d_output, input.data.data(), total_bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrfBatched(d_output, n, batch, stream);
  CorePotriBatched(d_output, n, batch, stream);
  // Task_12: убрали backend_->Synchronize()
  SymmetrizeBatched(d_output, n, batch, stream);
  if (check_info_) CheckInfo("InvertBatch(vector)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = n;
  result.batch_count = batch;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// InvertBatch — void* (GPU)
// ════════════════════════════════════════════════════════════════════════════

CholeskyResult CholeskyInverterROCm::InvertBatch(
    const drv_gpu_lib::InputData<void*>& input, int n) {
  if (n <= 0)
    throw std::invalid_argument(
        "InvertBatch(void*): n must be > 0 (got " + std::to_string(n) + ")");
  const int batch = static_cast<int>(input.antenna_count);
  const size_t total_bytes = static_cast<size_t>(batch) * n * n *
                             sizeof(std::complex<float>);

  void* d_output = backend_->Allocate(total_bytes);
  backend_->MemcpyDeviceToDevice(d_output, input.data, total_bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrfBatched(d_output, n, batch, stream);
  CorePotriBatched(d_output, n, batch, stream);
  // Task_12: убрали backend_->Synchronize()
  SymmetrizeBatched(d_output, n, batch, stream);
  if (check_info_) CheckInfo("InvertBatch(void*)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = n;
  result.batch_count = batch;
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// InvertBatch — cl_mem (ZeroCopy)
// ════════════════════════════════════════════════════════════════════════════

#ifdef CL_VERSION_1_0
CholeskyResult CholeskyInverterROCm::InvertBatch(
    const drv_gpu_lib::InputData<cl_mem>& input, int n) {
  if (n <= 0)
    throw std::invalid_argument(
        "InvertBatch(cl_mem): n must be > 0 (got " + std::to_string(n) + ")");
  const int batch = static_cast<int>(input.antenna_count);
  const size_t total_bytes = static_cast<size_t>(batch) * n * n *
                             sizeof(std::complex<float>);

  int dma_fd = drv_gpu_lib::ExportClBufferToFd(input.data);
  if (dma_fd < 0) {
    throw std::runtime_error(
        "InvertBatch(cl_mem): ZeroCopy export failed (dma_fd < 0)");
  }

  drv_gpu_lib::ZeroCopyBridge bridge;
  bridge.ImportFromOpenCl(dma_fd, total_bytes);
  void* hip_in = bridge.GetHipPtr();

  void* d_output = backend_->Allocate(total_bytes);
  backend_->MemcpyDeviceToDevice(d_output, hip_in, total_bytes);

  void* stream = backend_->GetNativeQueue();

  CorePotrfBatched(d_output, n, batch, stream);
  CorePotriBatched(d_output, n, batch, stream);
  // Task_12: убрали backend_->Synchronize()
  SymmetrizeBatched(d_output, n, batch, stream);
  if (check_info_) CheckInfo("InvertBatch(cl_mem)");

  CholeskyResult result;
  result.d_data = d_output;
  result.backend = backend_;
  result.matrix_size = n;
  result.batch_count = batch;
  return result;
}
#endif  // CL_VERSION_1_0

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
