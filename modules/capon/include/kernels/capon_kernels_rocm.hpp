#pragma once

/**
 * @file capon_kernels_rocm.hpp
 * @brief HIP kernel sources для CaponProcessor (ROCm/hiprtc)
 *
 * Содержит:
 *   add_regularization     — R[i,i] += mu (диагональная загрузка)
 *   compute_capon_relief   — z[m] = 1/Re(Σ_p conj(U[p,m]) * W[p,m])
 *
 * Компилируется через hiprtc в GpuContext::CompileModule().
 * GEMM-операции (R=Y*Y^H, W=R^{-1}*U, Y_out=W^H*Y) — через rocBLAS, не здесь.
 *
 * Соглашения:
 *   - complex<float> хранится как float2 (x=re, y=im)
 *   - матрицы column-major (как в rocBLAS/LAPACK): [p, m] → m*P + p
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

namespace capon {
namespace kernels {

inline const char* GetCaponKernelSource() {
  return R"HIP(

// ============================================================================
// Kernel: compute_capon_relief
// ============================================================================
// z[m] = 1 / Re( Σ_{p=0}^{P-1} conj(U[p,m]) * W[p,m] )
//
// W = R^{-1} * U уже вычислена через rocBLAS CGEMM в CaponReliefOp::Execute().
// Каждый thread обрабатывает одно направление m.
//
// Launch: grid = (M + 255) / 256, block = 256
// ============================================================================

extern "C" __global__ void compute_capon_relief(
    const float2* __restrict__ U,   // [P × M], column-major
    const float2* __restrict__ W,   // W = R^{-1}*U [P × M], column-major
    float*                     z,   // выход: рельеф [M], float
    unsigned int               P,   // число каналов
    unsigned int               M)   // число направлений
{
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;

  // Суммируем Re(conj(U[p,m]) * W[p,m]) по всем каналам p
  float acc = 0.0f;

  // column-major: элемент [p, m] = m*P + p
  const float2* U_col = U + m * P;
  const float2* W_col = W + m * P;

  for (unsigned int p = 0; p < P; ++p) {
    float2 u = U_col[p];
    float2 w = W_col[p];
    // Re(conj(u) * w) = u.x*w.x + u.y*w.y
    acc += u.x * w.x + u.y * w.y;
  }

  // z[m] = 1 / Re(u^H * R^{-1} * u)
  // Защита от нуля: при корректной регуляризации acc > 0 всегда
  z[m] = (acc > 0.0f) ? (1.0f / acc) : 0.0f;
}

)HIP";
}

}  // namespace kernels
}  // namespace capon

#endif  // ENABLE_ROCM
