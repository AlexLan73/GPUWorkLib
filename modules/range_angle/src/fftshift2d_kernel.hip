/**
 * @file fftshift2d_kernel.hip
 * @brief 2-D FFT shift + magnitude squared kernels — offline HIP reference
 *
 * Compiled at RUNTIME via hiprtc from range_angle_kernels_rocm.hpp.
 * This file is for offline syntax checking and code review only.
 *
 * fftshift2d_kernel:
 *   In-place quadrant swap for each range bin.
 *   Swaps Q1↔Q3 and Q2↔Q4 simultaneously.
 *   Grid: (n_range_bins, n_az/2, n_el/2)  Block: (1, 1, 1)
 *
 * magnitude_sq_kernel:
 *   |z|^2 : complex→float power.
 *   Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#include <hip/hip_runtime.h>

/// In-place 2-D fftshift for [n_range_bins × n_az × n_el] data.
__launch_bounds__(1)
__global__ void fftshift2d_kernel(
    float2*      data,
    unsigned int n_range_bins,
    unsigned int n_az,
    unsigned int n_el)
{
  unsigned int r   = blockIdx.x;
  unsigned int az  = blockIdx.y;
  unsigned int el  = blockIdx.z;

  unsigned int half_az = n_az >> 1u;
  unsigned int half_el = n_el >> 1u;

  size_t base = (size_t)r * n_az * n_el;
  size_t i00  = base +  az               * n_el +  el;
  size_t i11  = base + (az + half_az)    * n_el + (el + half_el);
  size_t i01  = base +  az               * n_el + (el + half_el);
  size_t i10  = base + (az + half_az)    * n_el +  el;

  float2 tmp;
  tmp = data[i00]; data[i00] = data[i11]; data[i11] = tmp;
  tmp = data[i01]; data[i01] = data[i10]; data[i10] = tmp;
}

/// |z|^2 complex → float power.
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
__launch_bounds__(256)
__global__ void magnitude_sq_kernel(
    const float2* __restrict__ in,
    float*        __restrict__ out,
    unsigned int  total)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float re = in[i].x;
  float im = in[i].y;
  out[i] = re * re + im * im;
}
