#pragma once

/**
 * @file capon_processor.hpp
 * @brief CaponProcessor — фасад для алгоритма Кейпона (MVDR) на GPU (ROCm/HIP)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 *
 * ROCm-only модуль. Реализует:
 *   - Рельеф Кейпона (пространственный спектр MVDR)
 *   - Адаптивное диаграммообразование (adaptive beamforming)
 *
 * Математика:
 *   R = (1/N) * Y * Y^H + μI         — ковариационная матрица с регуляризацией
 *   R^{-1}                            — CholeskyInverterROCm (vector_algebra)
 *   z[m] = 1 / Re(u_m^H * R^{-1} * u_m)  — рельеф Кейпона
 *   Y_out = (R^{-1}*U)^H * Y         — адаптивное ДО
 *
 * Внутренняя структура (Ref03):
 *   GpuContext ctx_               — per-module: stream, compiled kernels, shared buffers
 *   CovarianceMatrixOp cov_op_    — R = (1/N)*Y*Y^H (rocBLAS CGEMM)
 *   CaponInvertOp inv_op_         — R^{-1} через vector_algebra::CholeskyInverterROCm
 *   ComputeWeightsOp weights_op_  — W = R^{-1}*U (rocBLAS CGEMM → kWeight)
 *   CaponReliefOp relief_op_      — z[m] = 1/Re(u^H * W[m]) (HIP kernel)
 *   AdaptBeamformOp beam_op_      — Y_out = W^H * Y (rocBLAS CGEMM)
 *   CholeskyResult last_inv_      — хранит R^{-1} на GPU между шагами пайплайна
 *
 * Прототип: Doc_Addition/Capon/capon_test/ (ArrayFire CPU реализация)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "capon_types.hpp"
#include "interface/gpu_context.hpp"

// Op классы (Layer 5)
#include "operations/covariance_matrix_op.hpp"
#include "operations/capon_invert_op.hpp"      // обёртка над CholeskyInverterROCm
#include "operations/compute_weights_op.hpp"   // W = R^{-1}*U  (единый, без дублирования)
#include "operations/capon_relief_op.hpp"
#include "operations/adapt_beam_op.hpp"

// Тип результата инверсии из vector_algebra
#include "vector_algebra_types.hpp"

// Регуляризация через Strategy (DIP — зависим от интерфейсе)
#include "i_matrix_regularizer.hpp"
#include "diagonal_load_regularizer.hpp"

// GEMM операции через vector_algebra (единый фасад для rocBLAS)
#include "matrix_ops_rocm.hpp"

#include "interface/i_backend.hpp"

#include <complex>
#include <memory>
#include <vector>
#include <cstdint>

namespace capon {

/// @ingroup grp_capon
class CaponProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Конструктор
   * @param backend Указатель на IBackend (не владеющий, должен быть ROCm backend)
   */
  explicit CaponProcessor(drv_gpu_lib::IBackend* backend);

  ~CaponProcessor();

  // No copying
  CaponProcessor(const CaponProcessor&) = delete;
  CaponProcessor& operator=(const CaponProcessor&) = delete;

  // Move semantics
  CaponProcessor(CaponProcessor&& other) noexcept;
  CaponProcessor& operator=(CaponProcessor&& other) noexcept;

  // =========================================================================
  // Public API — CPU data (upload → compute → download)
  // =========================================================================

  /**
   * @brief Вычислить рельеф Кейпона
   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major
   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major
   * @param params    Параметры (n_channels, n_samples, n_directions, mu)
   * @return CaponReliefResult — M вещественных значений пространственного спектра
   */
  CaponReliefResult ComputeRelief(
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const CaponParams& params);

  /**
   * @brief Адаптивное диаграммообразование
   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major
   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major
   * @param params    Параметры
   * @return CaponBeamResult — матрица [n_directions × n_samples]
   */
  CaponBeamResult AdaptiveBeamform(
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const CaponParams& params);

  // =========================================================================
  // Public API — GPU data (данные уже на устройстве)
  // =========================================================================

  /**
   * @brief Рельеф Кейпона (GPU входы)
   * @param gpu_signal   Y на GPU: complex<float>[n_channels × n_samples], column-major
   * @param gpu_steering U на GPU: complex<float>[n_channels × n_directions], column-major
   */
  CaponReliefResult ComputeRelief(
      void* gpu_signal,
      void* gpu_steering,
      const CaponParams& params);

  /**
   * @brief Адаптивное ДО (GPU входы)
   */
  CaponBeamResult AdaptiveBeamform(
      void* gpu_signal,
      void* gpu_steering,
      const CaponParams& params);

private:
  // =========================================================================
  // Internal pipeline
  // =========================================================================

  /// Скомпилировать kernels (ленивая одноразовая инициализация)
  void EnsureCompiled();

  /// Загрузить CPU данные сигнала в shared kSignal буфер
  void UploadSignal(const std::complex<float>* data, size_t count);

  /// Загрузить CPU управляющие векторы в shared kSteering буфер
  void UploadSteering(const std::complex<float>* data, size_t count);

  /// Скопировать GPU данные в shared kSignal (D2D)
  void CopySignalGpu(void* src, size_t count);

  /// Скопировать GPU управляющие векторы в shared kSteering (D2D)
  void CopySteeringGpu(void* src, size_t count);

  /**
   * @brief Шаги 1-3 общего пайплайна:
   *   CovarianceMatrixOp → регуляризация → CaponInvertOp → last_inv_
   */
  void RunCovAndInvert(const CaponParams& params);

  /**
   * @brief Шаг 4: W = R^{-1} * U  (ComputeWeightsOp → kWeight)
   * Вызывается после RunCovAndInvert, до Relief/Beam Op.
   */
  void RunComputeWeights(const CaponParams& params);

  /// Прочитать рельеф из kOutput (float[M])
  CaponReliefResult ReadReliefResult(uint32_t n_directions);

  /// Прочитать адаптивный выход из kOutput (complex<float>[M×N])
  CaponBeamResult ReadBeamResult(uint32_t n_directions, uint32_t n_samples);

  // ── Members ──────────────────────────────────────────────────────────────

  drv_gpu_lib::IBackend* backend_;  ///< Сохраняем для CaponInvertOp

  drv_gpu_lib::GpuContext ctx_;     ///< Per-module контекст (stream, kernels, shared bufs)

  // Op instances (Layer 5)
  CovarianceMatrixOp  cov_op_;     ///< R = (1/N)*Y*Y^H  (rocBLAS CGEMM)
  /// CaponInvertOp хранится в unique_ptr: CholeskyInverterROCm non-copyable,
  /// unique_ptr даёт корректный move constructor И move assignment без дополнительного кода.
  std::unique_ptr<CaponInvertOp> inv_op_;  ///< R^{-1} via CholeskyInverterROCm
  ComputeWeightsOp    weights_op_; ///< W = R^{-1}*U (→ kWeight shared buf)
  CaponReliefOp       relief_op_;  ///< z[m] = 1/Re(u^H * W[m])
  AdaptBeamformOp     beam_op_;    ///< Y_out = W^H * Y

  /// GEMM операции через vector_algebra::MatrixOpsROCm.
  /// Инициализируется с &ctx_ — handle привязан к ctx_.stream() (lazy init).
  vector_algebra::MatrixOpsROCm mat_ops_;

  /// Регуляризатор — Strategy (GoF): DiagonalLoadRegularizer по умолчанию.
  /// DIP: зависим от IMatrixRegularizer, не от конкретной реализации.
  std::unique_ptr<vector_algebra::IMatrixRegularizer> regularizer_;

  /// Результат инверсии — хранит R^{-1} на GPU между шагами пайплайна.
  /// Обновляется каждый раз в RunCovAndInvert().
  vector_algebra::CholeskyResult last_inv_;

  bool compiled_ = false;
};

}  // namespace capon

#else  // !ENABLE_ROCM — Windows stub

#include "capon_types.hpp"
#include "interface/i_backend.hpp"
#include <stdexcept>
#include <complex>
#include <vector>

namespace capon {

class CaponProcessor {
public:
  explicit CaponProcessor(drv_gpu_lib::IBackend*) {}
  ~CaponProcessor() = default;

  CaponProcessor(const CaponProcessor&) = delete;
  CaponProcessor& operator=(const CaponProcessor&) = delete;
  CaponProcessor(CaponProcessor&&) noexcept = default;
  CaponProcessor& operator=(CaponProcessor&&) noexcept = default;

  CaponReliefResult ComputeRelief(const std::vector<std::complex<float>>&,
      const std::vector<std::complex<float>>&, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponBeamResult AdaptiveBeamform(const std::vector<std::complex<float>>&,
      const std::vector<std::complex<float>>&, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponReliefResult ComputeRelief(void*, void*, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponBeamResult AdaptiveBeamform(void*, void*, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
};

}  // namespace capon

#endif  // ENABLE_ROCM
