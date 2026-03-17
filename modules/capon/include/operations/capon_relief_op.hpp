#pragma once

/**
 * @file capon_relief_op.hpp
 * @brief CaponReliefOp — рельеф Кейпона z[m] = 1 / Re(u_m^H * W[m])
 *
 * Ref03 Layer 5: Concrete Operation.
 *
 * Шаг:
 *   HIP kernel compute_capon_relief:
 *     z[m] = 1 / Re(Σ_p conj(U[p,m]) * W[p,m])
 *
 * W = R^{-1}*U уже вычислена в ComputeWeightsOp и записана в kWeight.
 * Этот Op только запускает HIP kernel — без CGEMM.
 *
 * Входные разделяемые буферы: kSteering (U), kWeight (W)
 * Выходные: kOutput (float[M] — рельеф)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "capon_types.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace capon {

class CaponReliefOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "CaponRelief"; }

  /**
   * @brief Вычислить рельеф Кейпона
   * @param n_channels   P — число каналов
   * @param n_directions M — число направлений
   *
   * Читает: kSteering (U) [P × M],  kWeight (W = R^{-1}*U) [P × M]
   * Пишет:  kOutput (float[M])
   *
   * Предусловие: ComputeWeightsOp::Execute() уже запущен и записал W в kWeight.
   */
  void Execute(uint32_t n_channels, uint32_t n_directions) {
    uint32_t P = n_channels;
    uint32_t M = n_directions;

    void* U   = ctx_->GetShared(shared_buf::kSteering);
    void* W   = ctx_->GetShared(shared_buf::kWeight);
    void* out = ctx_->RequireShared(shared_buf::kOutput,
                                    static_cast<size_t>(M) * sizeof(float));

    // z[m] = 1 / Re(Σ_p conj(U[p,m]) * W[p,m])
    // W на том же stream (ComputeWeightsOp вызывает rocblas_set_stream(blas, stream()))
    void* args[] = { &U, &W, &out, &P, &M };
    hipError_t err = hipModuleLaunchKernel(
        kernel("compute_capon_relief"),
        (M + 255) / 256, 1, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("CaponReliefOp compute_capon_relief: " +
                               std::string(hipGetErrorString(err)));
    }
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
