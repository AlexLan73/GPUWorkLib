#pragma once

/**
 * @file covariance_matrix_op.hpp
 * @brief CovarianceMatrixOp — ковариационная матрица R = (1/N)*Y*Y^H
 *
 * Ref03 Layer 5: Concrete Operation.
 *
 * Шаг:
 *   MatrixOpsROCm::CovarianceMatrix — R = (1/N) * Y * Y^H
 *
 * Регуляризация (R += mu*I) — ответственность IMatrixRegularizer,
 * применяется снаружи в CaponProcessor::RunCovAndInvert().
 *
 * Входные разделяемые буферы: kSignal (Y)
 * Выходные разделяемые буферы: kCovMatrix (R)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "capon_types.hpp"
#include "matrix_ops_rocm.hpp"

#include <rocblas/rocblas.h>

namespace capon {

class CovarianceMatrixOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "CovarianceMatrix"; }

  /**
   * @brief Вычислить R = (1/N) * Y * Y^H
   * @param n_channels P — число каналов
   * @param n_samples  N — число отсчётов
   * @param mat        MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: ctx_->GetShared(kSignal)  [P × N]
   * Пишет:  ctx_->RequireShared(kCovMatrix, P*P*sizeof(complex<float>))  [P × P]
   */
  void Execute(uint32_t n_channels, uint32_t n_samples,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int N = static_cast<int>(n_samples);

    void* Y = ctx_->GetShared(shared_buf::kSignal);
    void* R = ctx_->RequireShared(
        shared_buf::kCovMatrix,
        static_cast<size_t>(P) * P * sizeof(rocblas_float_complex));

    mat.CovarianceMatrix(Y, P, N, R);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
