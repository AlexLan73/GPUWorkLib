/**
 * @file dechirp_window_kernel.hip
 * @brief Dechirp + Hamming window kernels — offline HIP reference
 *
 * These kernels are compiled at RUNTIME via hiprtc from
 * include/kernels/range_angle_kernels_rocm.hpp.
 * This .hip file is for:
 *   1. Offline syntax checking (hipcc)
 *   2. Documentation / code review
 *
 * Algorithm (dechirp_window_kernel):
 *   out[ant][i] = rx[ant][i] * conj(ref[i]) * hamming[i]    if i < n_samples
 *               = 0                                          if i >= n_samples (zero-pad)
 *
 * Optimizations (RDNA4 gfx1201):
 *   - __launch_bounds__(256) — correct VGPR reservation
 *   - __restrict__ on all pointers — enables register reuse
 *   - 2D grid (blockIdx.y = ant) — no div/mod for antenna index
 *   - FP32 literals everywhere
 *   - __cosf/__sinf in gen_ref_kernel (fast intrinsics)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#include <hip/hip_runtime.h>

// hiprtc does not have hip float2 → use float2 from HIP runtime here (offline only)

/// Fill Hamming window array on GPU.
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
__launch_bounds__(256)
__global__ void hamming_window_kernel(
    float* __restrict__ window,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float phase = 6.28318530f * float(i) / float(n - 1u);
  window[i] = 0.54f - 0.46f * __cosf(phase);
}

/// Generate conjugate LFM chirp signal on GPU.
/// Grid: (ceil(n_samples/256), 1, 1)  Block: (256, 1, 1)
__launch_bounds__(256)
__global__ void gen_ref_kernel(
    float2* __restrict__ ref_conj,
    unsigned int n_samples,
    float chirp_rate,
    float sample_rate)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float t     = float(i) / sample_rate;
  float phase = 3.14159265f * chirp_rate * t * t;
  ref_conj[i] = make_float2(__cosf(phase), -__sinf(phase));  // conj
}

/// Dechirp + window + zero-pad.
/// Grid: (ceil(nfft_r/256), n_ant, 1)  Block: (256, 1, 1)
__launch_bounds__(256)
__global__ void dechirp_window_kernel(
    const float2* __restrict__ rx,
    const float2* __restrict__ ref_conj,
    const float*  __restrict__ window,
    float2*       __restrict__ out,
    unsigned int n_ant,
    unsigned int n_samples,
    unsigned int nfft_r)
{
  unsigned int ant = blockIdx.y;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (ant >= n_ant || i >= nfft_r) return;

  float2 val = make_float2(0.f, 0.f);
  if (i < n_samples) {
    float2 r = rx[ant * n_samples + i];
    float2 c = ref_conj[i];
    float  w = window[i];
    val = make_float2(
        (r.x * c.x - r.y * c.y) * w,
        (r.x * c.y + r.y * c.x) * w);
  }
  out[ant * nfft_r + i] = val;
}
