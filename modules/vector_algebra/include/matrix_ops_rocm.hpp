#pragma once

/**
 * @file matrix_ops_rocm.hpp
 * @brief MatrixOpsROCm — обёртка над rocBLAS CGEMM для complex<float> матриц
 *
 * Централизует rocBLAS CGEMM операции, убирая прямую зависимость от rocblas.h
 * из модулей верхнего уровня (capon, statistics и др.).
 *
 * Использует rocBLAS handle из GpuContext — handle уже привязан к stream_
 * при первом вызове GetRocblasHandleRaw() (lazy init в GpuContext).
 * Дополнительных rocblas_set_stream не требуется.
 *
 * Соглашения:
 *   - Все матрицы complex<float> (rocblas_float_complex = float2)
 *   - Хранение column-major (как в BLAS/LAPACK/rocBLAS)
 *   - Размерности: m=строки C, n=столбцы C, k=внутреннее измерение
 *
 * Поддерживаемые паттерны (capon + общий):
 *   CovarianceMatrix   : R = (1/N) * Y * Y^H          [P×N → P×P]
 *   Multiply           : C = A * B                     [m×k, k×n → m×n]
 *   MultiplyConjTransA : C = A^H * B                   [k×m stored, k×n → m×n]
 *   CGEMM              : C = α*op(A)*op(B) + β*C       (общий вызов)
 *
 * Требования: ROCm 7.2+, rocBLAS
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "interface/gpu_context.hpp"

#include <rocblas/rocblas.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vector_algebra {

/**
 * @class MatrixOpsROCm
 * @brief rocBLAS CGEMM операции, привязанные к GpuContext (stream + handle).
 *
 * Создаётся в конструкторе модуля:
 * @code
 *   MatrixOpsROCm mat_ops(&ctx_);
 *   mat_ops.CovarianceMatrix(Y, P, N, R);
 * @endcode
 *
 * Не копируемый (владеет ссылкой на ctx_). Перемещаемый (меняет указатель).
 */
class MatrixOpsROCm {
public:
  // =========================================================================
  // Constructor
  // =========================================================================

  MatrixOpsROCm() = default;

  /**
   * @param ctx GpuContext модуля (non-owning). Должен жить дольше объекта.
   *            ctx->GetRocblasHandleRaw() возвращает handle, привязанный
   *            к ctx->stream() — синхронизация гарантирована GpuContext.
   */
  explicit MatrixOpsROCm(drv_gpu_lib::GpuContext* ctx) : ctx_(ctx) {}

  // No copy
  MatrixOpsROCm(const MatrixOpsROCm&)            = delete;
  MatrixOpsROCm& operator=(const MatrixOpsROCm&) = delete;

  // Move
  MatrixOpsROCm(MatrixOpsROCm&& o) noexcept : ctx_(o.ctx_) { o.ctx_ = nullptr; }
  MatrixOpsROCm& operator=(MatrixOpsROCm&& o) noexcept {
    ctx_ = o.ctx_; o.ctx_ = nullptr; return *this;
  }

  // =========================================================================
  // Named shortcuts — конкретные паттерны для сигнальной обработки
  // =========================================================================

  /**
   * @brief Ковариационная матрица: R = (1/N) * Y * Y^H
   *
   * @param Y   [P × N] complex<float>, column-major
   * @param P   число строк (каналы)
   * @param N   число столбцов (отсчёты)
   * @param R   [P × P] выход, complex<float>, column-major
   *
   * rocBLAS: C[P×P] = alpha * Y[P×N] * Y^H[N×P]
   *   opA=None, opB=ConjTrans, alpha=1/N, beta=0
   */
  void CovarianceMatrix(const void* Y, int P, int N, void* R);

  /**
   * @brief Матричное умножение: C = A * B  (NoTrans × NoTrans)
   *
   * @param A   [m × k] complex<float>, column-major
   * @param B   [k × n] complex<float>, column-major
   * @param C   [m × n] выход, complex<float>, column-major
   * @param m   строки A и C
   * @param n   столбцы B и C
   * @param k   столбцы A = строки B
   *
   * Применения в capon:
   *   W = R^{-1} * U  →  Multiply(R_inv, U, W, P, M, P)
   */
  void Multiply(const void* A, const void* B, void* C, int m, int n, int k);

  /**
   * @brief Умножение с эрмитовым транспонированием первого аргумента: C = A^H * B
   *
   * A хранится как [k × m] column-major (до транспонирования).
   * После ConjTrans: A^H = [m × k].
   *
   * @param A   [k × m] complex<float>, column-major (хранится транспонированным)
   * @param B   [k × n] complex<float>, column-major
   * @param C   [m × n] выход, complex<float>, column-major
   * @param m   строки C  (= столбцы A до транспонирования)
   * @param n   столбцы C (= столбцы B)
   * @param k   строки A и B (= contraction dimension)
   *
   * Применения в capon:
   *   Y_out = W^H * Y  →  MultiplyConjTransA(W, Y, Y_out, M, N, P)
   *     W[P×M] хранится как [k=P × m=M] → lda=P
   */
  void MultiplyConjTransA(const void* A, const void* B, void* C,
                           int m, int n, int k);

  // =========================================================================
  // General CGEMM — для произвольных транспозиций
  // =========================================================================

  /**
   * @brief Общий CGEMM: C = alpha * op(A) * op(B) + beta * C
   *
   * @param transA  rocblas_operation_none / rocblas_operation_conjugate_transpose
   * @param transB  rocblas_operation_none / rocblas_operation_conjugate_transpose
   * @param m       строки op(A) и C
   * @param n       столбцы op(B) и C
   * @param k       столбцы op(A) = строки op(B)
   * @param alpha   скаляр alpha (complex<float>)
   * @param A       матрица A на GPU
   * @param lda     leading dimension A
   * @param B       матрица B на GPU
   * @param ldb     leading dimension B
   * @param beta    скаляр beta (complex<float>)
   * @param C       матрица C на GPU (вход/выход)
   * @param ldc     leading dimension C
   *
   * @throws std::runtime_error при ошибке rocBLAS
   */
  void CGEMM(rocblas_operation transA, rocblas_operation transB,
             int m, int n, int k,
             const rocblas_float_complex* alpha,
             const void* A, int lda,
             const void* B, int ldb,
             const rocblas_float_complex* beta,
             void* C, int ldc);

private:
  drv_gpu_lib::GpuContext* ctx_ = nullptr;

  /// Получить rocBLAS handle (lazy init в GpuContext, уже привязан к stream)
  rocblas_handle blas() const {
    return static_cast<rocblas_handle>(ctx_->GetRocblasHandleRaw());
  }
};

}  // namespace vector_algebra

#else  // !ENABLE_ROCM — Windows stub

namespace vector_algebra {
class MatrixOpsROCm {
public:
  MatrixOpsROCm() = default;
  void CovarianceMatrix(const void*, int, int, void*) {}
  void Multiply(const void*, const void*, void*, int, int, int) {}
  void MultiplyConjTransA(const void*, const void*, void*, int, int, int) {}
};
}  // namespace vector_algebra

#endif  // ENABLE_ROCM
