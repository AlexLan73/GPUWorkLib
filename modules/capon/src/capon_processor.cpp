/**
 * @file capon_processor.cpp
 * @brief CaponProcessor — реализация тонкого фасада (Ref03)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 * Вся GPU логика делегируется Op классам (Layer 5).
 *
 * Пайплайн ComputeRelief:
 *   Upload(Y, U) → CovarianceMatrixOp → регуляризация → CaponInvertOp
 *                → ComputeWeightsOp → CaponReliefOp → ReadRelief
 *
 * Пайплайн AdaptiveBeamform:
 *   Upload(Y, U) → CovarianceMatrixOp → регуляризация → CaponInvertOp
 *                → ComputeWeightsOp → AdaptBeamformOp → ReadBeam
 *
 * Инверсия матрицы делегируется в vector_algebra::CholeskyInverterROCm
 * через CaponInvertOp — повторное использование готового модуля.
 *
 * ComputeWeightsOp вычисляет W = R^{-1}*U единожды на шаг и записывает
 * в shared буфер kWeight — исключает дублирование CGEMM в Op'ах.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "capon_processor.hpp"
#include "kernels/capon_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>
#include <complex>

namespace capon {

// Kernel-имена, компилируемые через hiprtc (только Capon-специфичные)
// GEMM-операции выполняются через rocBLAS (не hiprtc)
// Регуляризация (diagonal_load) — через vector_algebra::DiagonalLoadRegularizer
static const std::vector<std::string> kKernelNames = {
  "compute_capon_relief",  // z[m] = 1 / Re(Σ conj(U[p,m]) * W[p,m])
};

// ============================================================================
// Constructor / Destructor / Move
// ============================================================================

CaponProcessor::CaponProcessor(drv_gpu_lib::IBackend* backend)
    : backend_(backend)
    , ctx_(backend, "Capon", "modules/capon/kernels")
    , inv_op_(std::make_unique<CaponInvertOp>(backend))
    , mat_ops_(&ctx_)
    , regularizer_(std::make_unique<vector_algebra::DiagonalLoadRegularizer>(backend)) {
}

CaponProcessor::~CaponProcessor() {
  // GpuKernelOp'ы освобождаем перед ctx_
  cov_op_.Release();
  weights_op_.Release();
  relief_op_.Release();
  beam_op_.Release();
  // inv_op_ — unique_ptr<CaponInvertOp>, освободится автоматически
  // last_inv_ — CholeskyResult с RAII
}

CaponProcessor::CaponProcessor(CaponProcessor&& other) noexcept
    : backend_(other.backend_)
    , ctx_(std::move(other.ctx_))
    , cov_op_(std::move(other.cov_op_))
    , inv_op_(std::move(other.inv_op_))      // unique_ptr — корректный move
    , weights_op_(std::move(other.weights_op_))
    , relief_op_(std::move(other.relief_op_))
    , beam_op_(std::move(other.beam_op_))
    , mat_ops_(std::move(other.mat_ops_))
    , regularizer_(std::move(other.regularizer_))
    , last_inv_(std::move(other.last_inv_))
    , compiled_(other.compiled_) {
  other.backend_  = nullptr;
  other.compiled_ = false;
}

CaponProcessor& CaponProcessor::operator=(CaponProcessor&& other) noexcept {
  if (this != &other) {
    cov_op_.Release();
    weights_op_.Release();
    relief_op_.Release();
    beam_op_.Release();

    backend_     = other.backend_;
    ctx_         = std::move(other.ctx_);
    cov_op_      = std::move(other.cov_op_);   // БЫЛ ПРОПУЩЕН — без этого cov_op_ пуст после move
    inv_op_      = std::move(other.inv_op_);   // unique_ptr — корректный move assignment
    weights_op_  = std::move(other.weights_op_);
    relief_op_   = std::move(other.relief_op_);
    beam_op_     = std::move(other.beam_op_);
    mat_ops_     = std::move(other.mat_ops_);
    regularizer_ = std::move(other.regularizer_);
    last_inv_    = std::move(other.last_inv_);
    compiled_    = other.compiled_;

    other.backend_  = nullptr;
    other.compiled_ = false;
  }
  return *this;
}

// ============================================================================
// EnsureCompiled — ленивая компиляция kernels (один раз)
// ============================================================================

void CaponProcessor::EnsureCompiled() {
  if (compiled_) return;

  std::vector<std::string> defines;

  // Компилировать Capon-специфичные HIP kernels через hiprtc
  ctx_.CompileModule(kernels::GetCaponKernelSource(), kKernelNames, defines);

  // Инициализировать GpuKernelOp'ы (привязать к ctx_)
  cov_op_.Initialize(ctx_);
  weights_op_.Initialize(ctx_);
  relief_op_.Initialize(ctx_);
  beam_op_.Initialize(ctx_);
  // inv_op_ инициализируется в конструкторе (принимает backend напрямую)
  // Warmup hiprtc kernel для симметризации
  inv_op_->CompileKernels();

  compiled_ = true;
}

// ============================================================================
// Upload / Copy helpers
// ============================================================================

void CaponProcessor::UploadSignal(const std::complex<float>* data, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  void* buf = ctx_.RequireShared(shared_buf::kSignal, bytes);
  hipError_t err = hipMemcpyHtoDAsync(buf, const_cast<std::complex<float>*>(data),
                                      bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: signal H2D failed: " +
                              std::string(hipGetErrorString(err)));
}

void CaponProcessor::UploadSteering(const std::complex<float>* data, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  void* buf = ctx_.RequireShared(shared_buf::kSteering, bytes);
  hipError_t err = hipMemcpyHtoDAsync(buf, const_cast<std::complex<float>*>(data),
                                      bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: steering H2D failed: " +
                              std::string(hipGetErrorString(err)));
}

void CaponProcessor::CopySignalGpu(void* src, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  void* buf = ctx_.RequireShared(shared_buf::kSignal, bytes);
  hipError_t err = hipMemcpyDtoDAsync(buf, src, bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: signal D2D failed: " +
                              std::string(hipGetErrorString(err)));
}

void CaponProcessor::CopySteeringGpu(void* src, size_t count) {
  size_t bytes = count * sizeof(std::complex<float>);
  void* buf = ctx_.RequireShared(shared_buf::kSteering, bytes);
  hipError_t err = hipMemcpyDtoDAsync(buf, src, bytes, ctx_.stream());
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: steering D2D failed: " +
                              std::string(hipGetErrorString(err)));
}

// ============================================================================
// Шаги 1-3 пайплайна: ковариационная матрица → инверсия
// ============================================================================

void CaponProcessor::RunCovAndInvert(const CaponParams& params) {
  // Шаг 1: R = (1/N)*Y*Y^H  (только CGEMM, без регуляризации)
  cov_op_.Execute(params.n_channels, params.n_samples, mat_ops_);

  void* R_gpu = ctx_.GetShared(shared_buf::kCovMatrix);

  // Шаг 2: R += mu*I  (Strategy: DiagonalLoadRegularizer по умолчанию)
  // Передаём ctx_.stream() — гарантирует порядок после CGEMM на том же stream.
  // Если mu == 0 — Apply() является no-op внутри регуляризатора.
  regularizer_->Apply(R_gpu, static_cast<int>(params.n_channels), params.mu,
                      ctx_.stream());

  // Шаг 3: R^{-1} через vector_algebra::CholeskyInverterROCm
  last_inv_ = inv_op_->Execute(R_gpu, params.n_channels);
  // last_inv_.AsHipPtr() — указатель R^{-1} на GPU (valid до следующего вызова)
}

// ============================================================================
// Шаг 4: W = R^{-1} * U  (ComputeWeightsOp → kWeight)
// ============================================================================

void CaponProcessor::RunComputeWeights(const CaponParams& params) {
  weights_op_.Execute(params.n_channels, params.n_directions,
                      last_inv_.AsHipPtr(), mat_ops_);
}

// ============================================================================
// Read helpers
// ============================================================================

CaponReliefResult CaponProcessor::ReadReliefResult(uint32_t n_directions) {
  CaponReliefResult result;
  result.relief.resize(n_directions);
  void* buf = ctx_.GetShared(shared_buf::kOutput);
  hipError_t err = hipMemcpyDtoH(result.relief.data(), buf,
                                  n_directions * sizeof(float));
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: relief D2H failed: " +
                              std::string(hipGetErrorString(err)));
  return result;
}

CaponBeamResult CaponProcessor::ReadBeamResult(uint32_t n_directions,
                                               uint32_t n_samples) {
  CaponBeamResult result;
  result.n_directions = n_directions;
  result.n_samples    = n_samples;
  const size_t count  = static_cast<size_t>(n_directions) * n_samples;
  result.output.resize(count);
  void* buf = ctx_.GetShared(shared_buf::kOutput);
  hipError_t err = hipMemcpyDtoH(result.output.data(), buf,
                                  count * sizeof(std::complex<float>));
  if (err != hipSuccess)
    throw std::runtime_error("CaponProcessor: beam D2H failed: " +
                              std::string(hipGetErrorString(err)));
  return result;
}

// ============================================================================
// Валидация входных параметров
// ============================================================================

static void ValidateParams(const CaponParams& params,
                           size_t signal_size,
                           size_t steering_size) {
  if (params.n_channels == 0 || params.n_samples == 0 || params.n_directions == 0) {
    throw std::invalid_argument(
        "CaponProcessor: invalid params — n_channels/n_samples/n_directions must be > 0");
  }
  const size_t expected_signal =
      static_cast<size_t>(params.n_channels) * params.n_samples;
  const size_t expected_steering =
      static_cast<size_t>(params.n_channels) * params.n_directions;
  if (signal_size != expected_signal) {
    throw std::invalid_argument(
        "CaponProcessor: signal size mismatch (got " + std::to_string(signal_size) +
        ", expected " + std::to_string(expected_signal) + ")");
  }
  if (steering_size != expected_steering) {
    throw std::invalid_argument(
        "CaponProcessor: steering size mismatch (got " + std::to_string(steering_size) +
        ", expected " + std::to_string(expected_steering) + ")");
  }
}

// ============================================================================
// Public API — CPU data
// ============================================================================

CaponReliefResult CaponProcessor::ComputeRelief(
    const std::vector<std::complex<float>>& signal,
    const std::vector<std::complex<float>>& steering,
    const CaponParams& params) {
  ValidateParams(params, signal.size(), steering.size());

  EnsureCompiled();
  UploadSignal(signal.data(), signal.size());
  UploadSteering(steering.data(), steering.size());

  RunCovAndInvert(params);
  RunComputeWeights(params);

  relief_op_.Execute(params.n_channels, params.n_directions);

  backend_->Synchronize();
  return ReadReliefResult(params.n_directions);
}

CaponBeamResult CaponProcessor::AdaptiveBeamform(
    const std::vector<std::complex<float>>& signal,
    const std::vector<std::complex<float>>& steering,
    const CaponParams& params) {
  ValidateParams(params, signal.size(), steering.size());

  EnsureCompiled();
  UploadSignal(signal.data(), signal.size());
  UploadSteering(steering.data(), steering.size());

  RunCovAndInvert(params);
  RunComputeWeights(params);

  beam_op_.Execute(params.n_channels, params.n_samples, params.n_directions,
                   mat_ops_);

  backend_->Synchronize();
  return ReadBeamResult(params.n_directions, params.n_samples);
}

// ============================================================================
// Public API — GPU data
// ============================================================================

CaponReliefResult CaponProcessor::ComputeRelief(
    void* gpu_signal,
    void* gpu_steering,
    const CaponParams& params) {
  if (params.n_channels == 0 || params.n_samples == 0 || params.n_directions == 0) {
    throw std::invalid_argument(
        "CaponProcessor: invalid params — n_channels/n_samples/n_directions must be > 0");
  }
  EnsureCompiled();
  CopySignalGpu(gpu_signal,
                static_cast<size_t>(params.n_channels) * params.n_samples);
  CopySteeringGpu(gpu_steering,
                  static_cast<size_t>(params.n_channels) * params.n_directions);

  RunCovAndInvert(params);
  RunComputeWeights(params);
  relief_op_.Execute(params.n_channels, params.n_directions);

  backend_->Synchronize();
  return ReadReliefResult(params.n_directions);
}

CaponBeamResult CaponProcessor::AdaptiveBeamform(
    void* gpu_signal,
    void* gpu_steering,
    const CaponParams& params) {
  if (params.n_channels == 0 || params.n_samples == 0 || params.n_directions == 0) {
    throw std::invalid_argument(
        "CaponProcessor: invalid params — n_channels/n_samples/n_directions must be > 0");
  }
  EnsureCompiled();
  CopySignalGpu(gpu_signal,
                static_cast<size_t>(params.n_channels) * params.n_samples);
  CopySteeringGpu(gpu_steering,
                  static_cast<size_t>(params.n_channels) * params.n_directions);

  RunCovAndInvert(params);
  RunComputeWeights(params);
  beam_op_.Execute(params.n_channels, params.n_samples, params.n_directions,
                   mat_ops_);

  backend_->Synchronize();
  return ReadBeamResult(params.n_directions, params.n_samples);
}

}  // namespace capon

#endif  // ENABLE_ROCM
