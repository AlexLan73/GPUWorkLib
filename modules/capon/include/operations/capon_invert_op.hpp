#pragma once

/**
 * @file capon_invert_op.hpp
 * @brief CaponInvertOp — обёртка над vector_algebra::CholeskyInverterROCm
 *
 * Ref03 Layer 5: Concrete Operation.
 *
 * НЕ реализует инверсию самостоятельно — делегирует в уже готовый модуль
 * vector_algebra::CholeskyInverterROCm (POTRF + POTRI + симметризация).
 *
 * Входные разделяемые буферы: kCovMatrix (R) [P × P]
 * Выходные: CholeskyResult (владеет GPU-памятью R^{-1}) — НЕ в shared_buf,
 *           возвращается напрямую и хранится в CaponProcessor::last_inv_result_.
 *
 * @see vector_algebra::CholeskyInverterROCm
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "interface/input_data.hpp"
#include "capon_types.hpp"

#include "cholesky_inverter_rocm.hpp"   // vector_algebra::CholeskyInverterROCm
#include "vector_algebra_types.hpp"     // vector_algebra::CholeskyResult

#include <hip/hip_runtime.h>
#include <stdexcept>

namespace capon {

/**
 * @brief Обёртка инверсии ковариационной матрицы.
 *
 * Держит экземпляр CholeskyInverterROCm (он не наследует GpuKernelOp,
 * поэтому CaponInvertOp также не наследует — это обычный класс).
 *
 * Использование:
 *   CaponInvertOp inv_op(backend);
 *   void* R_gpu = ctx.GetShared(shared_buf::kCovMatrix);
 *   auto result = inv_op.Execute(R_gpu, n_channels);  // result хранит R^{-1} на GPU
 *   void* R_inv_ptr = result.AsHipPtr();               // передать в CaponReliefOp
 */
class CaponInvertOp {
public:
  /**
   * @param backend IBackend (ROCm). Должен жить дольше объекта.
   */
  explicit CaponInvertOp(drv_gpu_lib::IBackend* backend)
      : inverter_(backend, vector_algebra::SymmetrizeMode::GpuKernel) {}

  ~CaponInvertOp() = default;

  // Не копируемый (CholeskyInverterROCm не копируемый)
  CaponInvertOp(const CaponInvertOp&) = delete;
  CaponInvertOp& operator=(const CaponInvertOp&) = delete;

  /**
   * @brief Обратить матрицу R → R^{-1} через CholeskyInverterROCm
   * @param gpu_R      Указатель на R на GPU (complex<float>[P × P], column-major)
   * @param n_channels P — размер матрицы
   * @return CholeskyResult — владеет GPU-памятью R^{-1}
   *         (caller должен хранить result пока R^{-1} нужен)
   */
  vector_algebra::CholeskyResult Execute(void* gpu_R, uint32_t n_channels) {
    drv_gpu_lib::InputData<void*> input;
    input.data = gpu_R;

    return inverter_.Invert(input, static_cast<int>(n_channels));
  }

  /// Включить/выключить проверку POTRF/POTRI info (для benchmark — false)
  void SetCheckInfo(bool enabled) { inverter_.SetCheckInfo(enabled); }

  /// Явная прекомпиляция hiprtc kernel (warmup)
  void CompileKernels() { inverter_.CompileKernels(); }

private:
  vector_algebra::CholeskyInverterROCm inverter_;
};

}  // namespace capon

#endif  // ENABLE_ROCM
