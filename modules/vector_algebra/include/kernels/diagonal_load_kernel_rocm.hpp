#pragma once
#if ENABLE_ROCM

/**
 * @file diagonal_load_kernel_rocm.hpp
 * @brief HIP kernel source — диагональная загрузка матрицы (hiprtc)
 *
 * Операция: A[i,i] += mu  (для всех i = 0..n-1)
 *
 * Матрица A — квадратная n×n, комплексная (float2), column-major.
 * mu — вещественный скаляр. Мнимая часть диагонали не меняется.
 *
 * Применение:
 *   - Регуляризация ковариационной матрицы перед инверсией
 *   - Стабилизация плохо обусловленных систем (Тихонов, Ridge)
 *   - Любой pipeline: A += μI
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

namespace vector_algebra {
namespace kernels {

inline const char* GetDiagonalLoadKernelSource() {
  return R"HIP(

// ============================================================================
// diagonal_load — добавить mu к вещественной части диагонали матрицы A
// ============================================================================
// A         — [n × n], column-major, complex<float> хранится как float2
// mu        — вещественный коэффициент регуляризации
// n         — размер матрицы
//
// Диагональный элемент [i,i] в column-major: индекс = i*n + i
// Мнимая часть не меняется (mu вещественное → A[i,i].im не затрагивается)
//
// Launch: grid = (n + 255) / 256, block = 256
// Каждый thread обрабатывает один диагональный элемент.
// ============================================================================

extern "C" __global__ void diagonal_load(
    float2*      A,    // [n × n], column-major, complex<float>
    float        mu,   // коэффициент регуляризации
    unsigned int n)    // размер матрицы
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // column-major: A[i, i] → A[i*n + i]
  A[i * n + i].x += mu;
}

)HIP";
}

}  // namespace kernels
}  // namespace vector_algebra

#endif  // ENABLE_ROCM
