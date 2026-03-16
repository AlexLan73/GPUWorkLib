#pragma once
#if ENABLE_ROCM

/**
 * @file diagonal_load_regularizer.hpp
 * @brief DiagonalLoadRegularizer — регуляризация A += mu*I на GPU (ROCm/hiprtc)
 *
 * Concrete Strategy (GoF): реализация IMatrixRegularizer для диагональной загрузки.
 *
 * Операция: A[i,i].re += mu  для всех i = 0..n-1
 * Матрица A: квадратная n×n, complex<float>, column-major.
 *
 * Kernel компилируется через hiprtc в конструкторе (один раз на объект).
 * Запуск через hipModuleLaunchKernel — без GpuContext, напрямую через IBackend.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#include "i_matrix_regularizer.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>

namespace vector_algebra {

/**
 * @class DiagonalLoadRegularizer
 * @brief Диагональная загрузка: A += mu * I (GPU, hiprtc kernel).
 *
 * Не копируемый (владеет hipModule_t). Перемещаемый.
 *
 * @code
 * DiagonalLoadRegularizer reg(backend);
 * reg.Apply(d_cov_matrix, P, 0.01f);   // R += 0.01 * I
 * @endcode
 */
class DiagonalLoadRegularizer : public IMatrixRegularizer {
public:
  /**
   * @brief Конструктор — компилирует HIP kernel через hiprtc.
   * @param backend  ROCm backend (должен быть инициализирован)
   * @throws std::runtime_error при ошибке компиляции или backend не ROCm
   */
  explicit DiagonalLoadRegularizer(drv_gpu_lib::IBackend* backend);

  ~DiagonalLoadRegularizer();

  DiagonalLoadRegularizer(const DiagonalLoadRegularizer&)            = delete;
  DiagonalLoadRegularizer& operator=(const DiagonalLoadRegularizer&) = delete;

  DiagonalLoadRegularizer(DiagonalLoadRegularizer&& other) noexcept;
  DiagonalLoadRegularizer& operator=(DiagonalLoadRegularizer&& other) noexcept;

  /**
   * @brief Применить диагональную загрузку: A[i,i] += mu (in-place на GPU).
   * @param d_matrix  GPU-указатель на матрицу n×n (complex<float>, column-major)
   * @param n         Размер матрицы
   * @param mu        Коэффициент регуляризации (mu > 0 рекомендуется)
   * @param stream    HIP stream (nullptr → использовать stream_ из backend).
   *                  Передавайте ctx.stream() для гарантии порядка исполнения
   *                  после предшествующего rocBLAS CGEMM.
   *
   * Если mu == 0.0f — kernel не запускается (no-op).
   */
  void Apply(void* d_matrix, int n, float mu,
             hipStream_t stream = nullptr) override;

private:
  hipStream_t   stream_   = nullptr;  ///< Stream из backend (non-owning)
  void*         module_   = nullptr;  ///< hipModule_t (owning)
  void*         function_ = nullptr;  ///< hipFunction_t (non-owning, из module_)

  void Compile(drv_gpu_lib::IBackend* backend);
  void Release() noexcept;
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
