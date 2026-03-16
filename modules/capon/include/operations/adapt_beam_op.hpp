#pragma once

/**
 * @file adapt_beam_op.hpp
 * @brief AdaptBeamformOp — адаптивное диаграммообразование Y_out = W^H * Y
 *
 * Ref03 Layer 5: Concrete Operation.
 *
 * Шаг:
 *   MatrixOpsROCm::MultiplyConjTransA — Y_out = W^H * Y  [M × N]
 *
 * W = R^{-1}*U уже вычислена в ComputeWeightsOp и записана в kWeight.
 * Этот Op выполняет только финальный CGEMM Y_out = W^H * Y.
 *
 * Входные разделяемые буферы: kSignal (Y), kWeight (W)
 * Выходные: kOutput (complex<float>[M × N])
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

class AdaptBeamformOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "AdaptBeamform"; }

  /**
   * @brief Адаптивное ДО: Y_out = W^H * Y
   * @param n_channels   P — число каналов
   * @param n_samples    N — число отсчётов
   * @param n_directions M — число направлений (лучей)
   * @param mat          MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: kSignal (Y) [P × N],  kWeight (W = R^{-1}*U) [P × M]
   * Пишет:  kOutput (complex<float>[M × N])
   *
   * Предусловие: ComputeWeightsOp::Execute() уже запущен и записал W в kWeight.
   */
  void Execute(uint32_t n_channels, uint32_t n_samples, uint32_t n_directions,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int N = static_cast<int>(n_samples);
    const int M = static_cast<int>(n_directions);

    void* Y   = ctx_->GetShared(shared_buf::kSignal);
    void* W   = ctx_->GetShared(shared_buf::kWeight);
    void* out = ctx_->RequireShared(shared_buf::kOutput,
                                    static_cast<size_t>(M) * N * sizeof(rocblas_float_complex));

    // Y_out[M×N] = W^H[M×P] * Y[P×N]  (ConjTransA × NoTrans)
    mat.MultiplyConjTransA(W, Y, out, M, N, P);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
