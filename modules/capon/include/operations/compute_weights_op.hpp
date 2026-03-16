#pragma once

/**
 * @file compute_weights_op.hpp
 * @brief ComputeWeightsOp — весовая матрица W = R^{-1} * U
 *
 * Ref03 Layer 5: Concrete Operation.
 *
 * Вычисляет W = R^{-1} * U через MatrixOpsROCm::Multiply.
 * Результат записывается в shared буфер kWeight.
 *
 * Этот Op выделен отдельно, чтобы исключить дублирование CGEMM
 * в CaponReliefOp и AdaptBeamformOp — оба читают kWeight.
 *
 * Входные:  kSteering (U) [P × M],  R_inv_ptr [P × P] (из CholeskyResult)
 * Выходные: kWeight   (W) [P × M]
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

class ComputeWeightsOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "ComputeWeights"; }

  /**
   * @brief Вычислить W = R^{-1} * U  (MatrixOpsROCm::Multiply)
   * @param n_channels   P — число каналов
   * @param n_directions M — число направлений
   * @param R_inv_ptr    R^{-1} на GPU — из CholeskyResult::AsHipPtr()
   * @param mat          MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: kSteering (U) [P × M]
   * Пишет:  kWeight   (W) [P × M]  (аллоцирует при необходимости)
   */
  void Execute(uint32_t n_channels, uint32_t n_directions, void* R_inv_ptr,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int M = static_cast<int>(n_directions);

    void* U = ctx_->GetShared(shared_buf::kSteering);
    void* W = ctx_->RequireShared(
        shared_buf::kWeight,
        static_cast<size_t>(P) * M * sizeof(rocblas_float_complex));

    // W[P×M] = R^{-1}[P×P] * U[P×M]  (NoTrans × NoTrans)
    mat.Multiply(R_inv_ptr, U, W, P, M, P);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
