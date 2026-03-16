#pragma once

/**
 * @file i_matrix_regularizer.hpp
 * @brief IMatrixRegularizer — интерфейс матричной регуляризации
 *
 * Strategy Pattern (GoF): сменный алгоритм регуляризации квадратной матрицы.
 *
 * Операция применяется к матрице in-place на GPU перед инверсией.
 * Конкретный смысл mu зависит от реализации:
 *   - DiagonalLoadRegularizer: A += mu * I
 *   - NoOpRegularizer:         ничего (Null Object)
 *
 * Принципы:
 *   ISP — один метод, минимальный интерфейс
 *   DIP — зависеть от этой абстракции, не от конкретного регуляризатора
 *   OCP — новые алгоритмы добавляются без изменения существующего кода
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#else
using hipStream_t = void*;  ///< Stub для non-ROCm платформ (NoOpRegularizer)
#endif

namespace vector_algebra {

/**
 * @interface IMatrixRegularizer
 * @brief Применить регуляризацию к квадратной комплексной матрице на GPU.
 *
 * Вызывается перед инверсией матрицы для улучшения обусловленности.
 *
 * @code
 * // Использование:
 * std::unique_ptr<IMatrixRegularizer> reg =
 *     std::make_unique<DiagonalLoadRegularizer>(backend);
 * reg->Apply(d_matrix, n, 0.01f);
 * inverter.Invert(input, n);
 * @endcode
 */
class IMatrixRegularizer {
public:
  virtual ~IMatrixRegularizer() = default;

  /**
   * @brief Применить регуляризацию к матрице (GPU, in-place).
   *
   * @param d_matrix  GPU-указатель на квадратную матрицу n×n
   *                  (complex<float>, column-major)
   * @param n         Размер матрицы (строки = столбцы = n)
   * @param mu        Коэффициент регуляризации (семантика — в реализации)
   * @param stream    HIP stream для запуска kernel (nullptr → использовать
   *                  внутренний stream регуляризатора из backend).
   *                  Передавайте ctx.stream() чтобы гарантировать порядок
   *                  выполнения после предшествующих CGEMM/kernel операций
   *                  на том же stream.
   */
  virtual void Apply(void* d_matrix, int n, float mu,
                     hipStream_t stream = nullptr) = 0;
};

}  // namespace vector_algebra
