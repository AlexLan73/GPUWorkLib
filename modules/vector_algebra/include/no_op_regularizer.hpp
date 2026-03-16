#pragma once

/**
 * @file no_op_regularizer.hpp
 * @brief NoOpRegularizer — регуляризатор-заглушка (Null Object Pattern)
 *
 * Null Object (GoF): безопасный no-op вместо nullptr.
 * Apply() ничего не делает — матрица не изменяется.
 *
 * Применение:
 *   - mu = 0 (регуляризация не нужна, матрица гарантированно HPD)
 *   - Тестирование pipeline без реального GPU
 *   - Default-значение там где регуляризатор опционален
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#include "i_matrix_regularizer.hpp"

namespace vector_algebra {

/**
 * @class NoOpRegularizer
 * @brief Ничего не делает. Безопасный заменитель nullptr (Null Object).
 */
class NoOpRegularizer : public IMatrixRegularizer {
public:
  void Apply(void* /*d_matrix*/, int /*n*/, float /*mu*/,
             hipStream_t /*stream*/ = nullptr) override {}
};

}  // namespace vector_algebra
