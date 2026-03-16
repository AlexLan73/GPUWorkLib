/**
 * @file matrix_ops_rocm.cpp
 * @brief MatrixOpsROCm — реализация rocBLAS CGEMM операций
 *
 * Все методы используют rocBLAS handle из GpuContext.
 * Handle при первом вызове GetRocblasHandleRaw() автоматически привязывается
 * к ctx_->stream() (lazy init в GpuContext) — rocblas_set_stream не требуется.
 *
 * ROCm 7.2+ / rocBLAS
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "matrix_ops_rocm.hpp"

namespace vector_algebra {

// ============================================================================
// Internal helper
// ============================================================================

static void CheckStatus(rocblas_status st, const char* where) {
  if (st != rocblas_status_success) {
    throw std::runtime_error(
        std::string(where) + ": rocblas_cgemm failed (" +
        std::to_string(static_cast<int>(st)) + ")");
  }
}

// ============================================================================
// CovarianceMatrix: R = (1/N) * Y * Y^H   [P×N → P×P]
// ============================================================================

void MatrixOpsROCm::CovarianceMatrix(const void* Y, int P, int N, void* R) {
  const rocblas_float_complex alpha = {1.0f / static_cast<float>(N), 0.0f};
  const rocblas_float_complex beta  = {0.0f, 0.0f};

  // C[P×P] = (1/N) * Y[P×N] * Y^H[N×P]
  //   opA = None,       lda = P
  //   opB = ConjTrans,  ldb = P  (Y^H: B хранится как [P×N])
  CheckStatus(rocblas_cgemm(
      blas(),
      rocblas_operation_none,
      rocblas_operation_conjugate_transpose,
      P, P, N,
      &alpha,
      static_cast<const rocblas_float_complex*>(Y), P,
      static_cast<const rocblas_float_complex*>(Y), P,
      &beta,
      static_cast<rocblas_float_complex*>(R), P),
    "MatrixOpsROCm::CovarianceMatrix");
}

// ============================================================================
// Multiply: C = A * B   NoTrans × NoTrans   [m×k, k×n → m×n]
// ============================================================================

void MatrixOpsROCm::Multiply(const void* A, const void* B, void* C,
                              int m, int n, int k) {
  const rocblas_float_complex alpha = {1.0f, 0.0f};
  const rocblas_float_complex beta  = {0.0f, 0.0f};

  // C[m×n] = A[m×k] * B[k×n]
  //   lda = m,  ldb = k,  ldc = m
  CheckStatus(rocblas_cgemm(
      blas(),
      rocblas_operation_none,
      rocblas_operation_none,
      m, n, k,
      &alpha,
      static_cast<const rocblas_float_complex*>(A), m,
      static_cast<const rocblas_float_complex*>(B), k,
      &beta,
      static_cast<rocblas_float_complex*>(C), m),
    "MatrixOpsROCm::Multiply");
}

// ============================================================================
// MultiplyConjTransA: C = A^H * B   ConjTrans × NoTrans   [k×m stored, k×n → m×n]
// ============================================================================

void MatrixOpsROCm::MultiplyConjTransA(const void* A, const void* B, void* C,
                                        int m, int n, int k) {
  const rocblas_float_complex alpha = {1.0f, 0.0f};
  const rocblas_float_complex beta  = {0.0f, 0.0f};

  // C[m×n] = A^H[m×k] * B[k×n]
  //   opA = ConjTrans: A хранится как [k×m], lda = k
  //   opB = None:      B хранится как [k×n], ldb = k
  //   ldc = m
  CheckStatus(rocblas_cgemm(
      blas(),
      rocblas_operation_conjugate_transpose,
      rocblas_operation_none,
      m, n, k,
      &alpha,
      static_cast<const rocblas_float_complex*>(A), k,
      static_cast<const rocblas_float_complex*>(B), k,
      &beta,
      static_cast<rocblas_float_complex*>(C), m),
    "MatrixOpsROCm::MultiplyConjTransA");
}

// ============================================================================
// CGEMM: C = alpha * op(A) * op(B) + beta * C   (общий)
// ============================================================================

void MatrixOpsROCm::CGEMM(rocblas_operation transA, rocblas_operation transB,
                           int m, int n, int k,
                           const rocblas_float_complex* alpha,
                           const void* A, int lda,
                           const void* B, int ldb,
                           const rocblas_float_complex* beta,
                           void* C, int ldc) {
  CheckStatus(rocblas_cgemm(
      blas(),
      transA, transB,
      m, n, k,
      alpha,
      static_cast<const rocblas_float_complex*>(A), lda,
      static_cast<const rocblas_float_complex*>(B), ldb,
      beta,
      static_cast<rocblas_float_complex*>(C), ldc),
    "MatrixOpsROCm::CGEMM");
}

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
