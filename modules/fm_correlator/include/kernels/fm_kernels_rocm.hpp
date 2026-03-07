#pragma once

/**
 * @file fm_kernels_rocm.hpp
 * @brief HIP kernel source strings for FM Correlator (hiprtc)
 *
 * 4 kernels:
 *   1. apply_cyclic_shifts   — float[N] -> float2[K×N] with cyclic shifts
 *   2. multiply_conj_fused   — conj(ref_fft) * inp_fft, half_N = N/2+1
 *   3. extract_magnitudes_real — |corr_time[j]| / N for first n_kg points
 *   4. generate_test_inputs  — utility: circshift(ref, s*shift_step) on GPU
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

namespace drv_gpu_lib {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// All FM kernels combined into a single compilation unit
// ════════════════════════════════════════════════════════════════════════════

inline const char* GetFMCorrelatorKernelSource() {
  return R"HIP(

// Собственный float2_t вместо hipfloat2: hiprtc не имеет доступа к device-заголовкам
// ROCm по умолчанию, и hipfloat2/float2 из <hip/hip_runtime.h> недоступны в строке
// hiprtcCompileProgram. Поэтому объявляем минимальный эквивалент прямо здесь.
// aligned(8) гарантирует 64-bit load/store вместо двух 32-bit транзакций
struct __attribute__((aligned(8))) float2_t {
    float x;
    float y;
};

// ════════════════════════════════════════════════════════════════════════════
// apply_cyclic_shifts: ref[N] float -> out[K×N] float2 with cyclic shifts
//   out[k*N + i] = { ref[(i+k) % N], 0.0f }
//   Grid: ((N+255)/256, K), Block: (256)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void apply_cyclic_shifts(
    const float*  __restrict__ ref,
    float2_t*     __restrict__ out,
    int N, int num_shifts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (i >= N || k >= num_shifts) return;

    int src = (i + k) & (N - 1);  // N — степень 2 (гарантировано FMCorrelatorParams)
    float2_t v;
    v.x = ref[src];
    v.y = 0.0f;
    out[k * N + i] = v;
}

// ════════════════════════════════════════════════════════════════════════════
// multiply_conj_fused: conj(ref_fft[k]) * inp_fft[s]
//   half_N = N/2+1 (hermitian symmetry from R2C)
//   Grid: ((half_N+255)/256, K, S), Block: (256)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void multiply_conj_fused(
    const float2_t* __restrict__ ref_fft,
    const float2_t* __restrict__ inp_fft,
    float2_t*       __restrict__ corr_fft,
    int half_N, int ref_stride, int K, int S)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int s = blockIdx.z;
    if (i >= half_N || k >= K || s >= S) return;

    float2_t a = ref_fft[k * ref_stride + i];
    a.y = -a.y;  // сопряжение: conj(a+jb) = a-jb; нет hipConjf в hiprtc → inline
    float2_t b = inp_fft[s * half_N + i];

    float2_t out;
    out.x = a.x * b.x - a.y * b.y;
    out.y = a.x * b.y + a.y * b.x;
    corr_fft[(s * K + k) * half_N + i] = out;
}

// ════════════════════════════════════════════════════════════════════════════
// extract_magnitudes_real: |corr_time[(s*K+k)*N + j]| / N
//   corr_time is float (C2R output), not float2
//   Grid: ((n_kg+255)/256, K, S), Block: (256)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void extract_magnitudes_real(
    const float* __restrict__ corr_time,
    float*       __restrict__ peaks,
    int N, int n_kg, int K, int S, float inv_N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int s = blockIdx.z;
    if (j >= n_kg || k >= K || s >= S) return;

    int src = (s * K + k) * N + j;
    float val = corr_time[src];
    // bitwise fabsf: убираем знаковый бит без branch divergence (1 инструкция V_AND)
    union { float f; unsigned u; } u = { val };
    u.u &= 0x7FFFFFFFu;
    val = u.f;
    // hipFFT не нормирует IFFT (в отличие от numpy.ifft): результат умножен на N.
    // inv_N = 1/N передан параметром: FMUL (~4 такта) вместо FDIV (~20 тактов).
    peaks[(s * K + k) * n_kg + j] = val * inv_N;
}

// ════════════════════════════════════════════════════════════════════════════
// generate_test_inputs: inp[s*N + i] = ref[(i + s*shift_step) % N]
//   Grid: ((N+255)/256, S), Block: (256)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void generate_test_inputs(
    const float* __restrict__ ref,
    float*       __restrict__ inp,
    int N, int S, int shift_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (i >= N || s >= S) return;

    int src = (i + s * shift_step) & (N - 1);  // N — степень 2
    inp[s * N + i] = ref[src];
}

)HIP";
}

}  // namespace kernels
}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
