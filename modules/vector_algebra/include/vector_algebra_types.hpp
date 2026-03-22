#pragma once
#if ENABLE_ROCM

/**
 * @file vector_algebra_types.hpp
 * @brief CholeskyResult (единый не-шаблонный тип) + SymmetrizeMode
 *
 * Task_11: переделка из шаблонного CholeskyResult<T> в единый тип
 * с void* d_data (GPU). Методы AsVector(), AsHipPtr(), matrix(), matrices().
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#include <complex>
#include <vector>
#include "interface/i_backend.hpp"

namespace vector_algebra {

/// Режим симметризации после POTRI
enum class SymmetrizeMode {
  Roundtrip,   ///< Download → CPU sym → Upload (простой, без kernel)
  GpuKernel    ///< HIP kernel in-place на GPU (всё на GPU, без round-trip)
};

/**
 * @brief Результат инверсии матрицы — владеет GPU памятью.
 *
 * Базовый формат — void* d_data (HIP device pointer).
 * Методы AsVector() / matrix() / matrices() скачивают на CPU.
 * AsHipPtr() возвращает raw указатель (caller НЕ владеет — НЕ вызывать Free).
 *
 * Не копируемый. Перемещаемый (move semantics).
 */
struct CholeskyResult {
  void* d_data = nullptr;                   ///< HIP device ptr (базовый формат)
  drv_gpu_lib::IBackend* backend = nullptr; ///< Для Memcpy/Free
  int matrix_size = 0;                      ///< n (одна сторона матрицы)
  int batch_count = 0;                      ///< Количество матриц

  // --- Доступ к данным ---

  /// Download GPU → CPU vector
  std::vector<std::complex<float>> AsVector() const;

  /// Вернуть HIP device ptr (caller НЕ владеет — НЕ вызывать Free!)
  void* AsHipPtr() const { return d_data; }

  /// Одна матрица как 2D vector [n][n] (для batch_count=1)
  std::vector<std::vector<std::complex<float>>> matrix() const;

  /// Batch как 3D vector [batch][n][n]
  std::vector<std::vector<std::vector<std::complex<float>>>> matrices() const;

  // --- Владение памятью ---
  ~CholeskyResult();
  CholeskyResult() = default;
  CholeskyResult(CholeskyResult&& other) noexcept;
  CholeskyResult& operator=(CholeskyResult&& other) noexcept;
  CholeskyResult(const CholeskyResult&) = delete;
  CholeskyResult& operator=(const CholeskyResult&) = delete;
};

}  // namespace vector_algebra

#else  // !ENABLE_ROCM — minimal stub types for Windows compilation

namespace vector_algebra {

enum class SymmetrizeMode { Roundtrip, GpuKernel };

struct CholeskyResult {
  void* d_data = nullptr;
  int matrix_size = 0;
  int batch_count = 0;
  ~CholeskyResult() = default;
  CholeskyResult() = default;
  CholeskyResult(CholeskyResult&&) noexcept = default;
  CholeskyResult& operator=(CholeskyResult&&) noexcept = default;
  CholeskyResult(const CholeskyResult&) = delete;
  CholeskyResult& operator=(const CholeskyResult&) = delete;
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
