#pragma once

/**
 * @file range_angle_kernels_rocm.hpp
 * @brief HIP kernel sources for range_angle module (hiprtc JIT compilation)
 *
 * Contains:
 *   - hamming_window_kernel    — fill Hamming window array on GPU
 *   - dechirp_window_kernel    — rx * conj(ref) * window + zero-pad
 *   - gen_ref_kernel           — generate complex LFM reference signal on GPU
 *   - transpose_complex_kernel — tiled 32×32 transpose with +1 padding
 *   - fftshift2d_kernel        — swap 4 quadrants of 2-D spatial spectrum
 *   - magnitude_sq_kernel      — |z|^2 complex→float power cube
 *
 * Optimizations (RDNA4 gfx1201):
 *   - __launch_bounds__ on all kernels
 *   - __restrict__ on all pointer args
 *   - 2D grid (blockIdx.y = antenna / range_bin) — no div/mod
 *   - float2_t struct (hiprtc does not support hip float2)
 *   - LDS tile[32][33] +1 padding for bank-conflict-free transpose
 *   - FP32 literals everywhere (no accidental double promotion)
 *   - fast intrinsics: __cosf, __sinf
 *
 * WARP_SIZE passed via -DWARP_SIZE=32 at compile time.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

namespace range_angle {
namespace kernels {

inline const char* GetRangeAngleKernelSource() {
  return R"HIP(

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// hiprtc does not provide float2 — use own struct
struct float2_t {
  float x;
  float y;
};

// ============================================================================
// hamming_window_kernel
// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
// ============================================================================

__launch_bounds__(256)
__global__ void hamming_window_kernel(
    float* __restrict__ window,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // Hamming: 0.54 - 0.46*cos(2*pi*i/(n-1))
  float phase = 6.28318530f * float(i) / float(n - 1u);
  window[i] = 0.54f - 0.46f * __cosf(phase);
}

// ============================================================================
// gen_ref_kernel — generate complex LFM reference (conj) on GPU
// out[i] = conj( exp(j*pi*mu*t^2) ) = exp(-j*pi*mu*t^2)
// mu = bandwidth / duration = (f_end - f_start) / (n_samples / sample_rate)
// Grid: (ceil(n_samples/256), 1, 1)  Block: (256, 1, 1)
// ============================================================================

__launch_bounds__(256)
__global__ void gen_ref_kernel(
    float2_t* __restrict__ ref_conj,
    unsigned int n_samples,
    float chirp_rate,   // mu = B/T  (Hz/s)
    float sample_rate)  // fs (Hz)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float t = float(i) / sample_rate;
  float phase = 3.14159265f * chirp_rate * t * t;
  // conj: negate imaginary part
  ref_conj[i].x =  __cosf(phase);
  ref_conj[i].y = -__sinf(phase);
}

// ============================================================================
// dechirp_window_kernel
// out[ant * nfft_r + i] = rx[ant * n_samples + i] * ref_conj[i] * window[i]
// Zero-pad: if i >= n_samples → out[...] = 0
// Grid: (ceil(nfft_r/256), n_ant, 1)  Block: (256, 1, 1)
// ============================================================================

__launch_bounds__(256)
__global__ void dechirp_window_kernel(
    const float2_t* __restrict__ rx,        // [n_ant * n_samples]
    const float2_t* __restrict__ ref_conj,  // [n_samples]
    const float*    __restrict__ window,    // [n_samples]  Hamming
    float2_t*       __restrict__ out,       // [n_ant * nfft_r]
    unsigned int n_ant,
    unsigned int n_samples,
    unsigned int nfft_r)
{
  unsigned int ant = blockIdx.y;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (ant >= n_ant || i >= nfft_r) return;

  float2_t val;
  if (i < n_samples) {
    float2_t r = rx[ant * n_samples + i];
    float2_t c = ref_conj[i];
    float w = window[i];
    // complex multiply: (r.x + j*r.y) * (c.x + j*c.y)
    val.x = (r.x * c.x - r.y * c.y) * w;
    val.y = (r.x * c.y + r.y * c.x) * w;
  } else {
    val.x = 0.f;
    val.y = 0.f;
  }
  out[ant * nfft_r + i] = val;
}

// ============================================================================
// transpose_complex_kernel
// Tiled 32×32 with +1 padding to avoid LDS bank conflicts.
// in [n_rows × n_cols] → out [n_cols × n_rows]   (n_rows=n_ant, n_cols=n_range_bins)
// Grid: (ceil(n_cols/32), ceil(n_rows/32), 1)  Block: (32, 32, 1)
// ============================================================================

#define TILE_DIM 32

__launch_bounds__(1024)
__global__ void transpose_complex_kernel(
    const float2_t* __restrict__ in,
    float2_t*       __restrict__ out,
    unsigned int n_rows,
    unsigned int n_cols)
{
  // +1 padding eliminates bank conflicts (each row offset by 1 float2_t)
  __shared__ float2_t tile[TILE_DIM][TILE_DIM + 1];

  unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;  // col in 'in'
  unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;  // row in 'in'

  if (x < n_cols && y < n_rows)
    tile[threadIdx.y][threadIdx.x] = in[y * n_cols + x];
  __syncthreads();

  // Write transposed tile
  unsigned int out_x = blockIdx.y * TILE_DIM + threadIdx.x;  // col in 'out' (= row in 'in')
  unsigned int out_y = blockIdx.x * TILE_DIM + threadIdx.y;  // row in 'out' (= col in 'in')
  if (out_x < n_rows && out_y < n_cols)
    out[out_y * n_rows + out_x] = tile[threadIdx.x][threadIdx.y];
}

#undef TILE_DIM

// ============================================================================
// fftshift2d_kernel
// Swap 4 quadrants of 2-D spectrum for each range bin (in-place).
// data [n_range_bins × n_az × n_el]
// Grid: (n_range_bins, n_az/2, n_el/2)  Block: (1, 1, 1)
// ============================================================================

__launch_bounds__(1)
__global__ void fftshift2d_kernel(
    float2_t* data,
    unsigned int n_range_bins,
    unsigned int n_az,
    unsigned int n_el)
{
  unsigned int r   = blockIdx.x;
  unsigned int az  = blockIdx.y;
  unsigned int el  = blockIdx.z;

  unsigned int half_az = n_az >> 1u;
  unsigned int half_el = n_el >> 1u;

  unsigned long base = (unsigned long)r * n_az * n_el;

  // Swap (az, el) <-> (az+half_az, el+half_el)  — Q1↔Q3
  unsigned long i00 = base +  az               * n_el +  el;
  unsigned long i11 = base + (az + half_az)    * n_el + (el + half_el);
  // Swap (az, el+half_el) <-> (az+half_az, el)  — Q2↔Q4
  unsigned long i01 = base +  az               * n_el + (el + half_el);
  unsigned long i10 = base + (az + half_az)    * n_el +  el;

  float2_t tmp;
  tmp = data[i00]; data[i00] = data[i11]; data[i11] = tmp;
  tmp = data[i01]; data[i01] = data[i10]; data[i10] = tmp;
}

// ============================================================================
// apply_beam_window_kernel
// Multiply spatial data by 2D Hamming window (separable: w_az * w_el).
// data layout: [n_range_bins × n_ant]  where n_ant = n_az * n_el
// window layout: [n_ant]  (row-major [n_az × n_el], flattened)
// Each block handles one range bin × up to 256 antennas.
// Grid: (n_range_bins, ceil(n_ant/256), 1)  Block: (256, 1, 1)
// ============================================================================

__launch_bounds__(256)
__global__ void apply_beam_window_kernel(
    float2_t*    __restrict__ data,    // [n_range_bins × n_ant]
    const float* __restrict__ window,  // [n_ant]
    unsigned int n_range_bins,
    unsigned int n_ant)
{
  unsigned int r = blockIdx.x;
  unsigned int k = blockIdx.y * blockDim.x + threadIdx.x;  // antenna index
  if (r >= n_range_bins || k >= n_ant) return;

  float w = window[k];
  unsigned long idx = (unsigned long)r * n_ant + k;
  data[idx].x *= w;
  data[idx].y *= w;
}

// ============================================================================
// magnitude_sq_kernel
// |z|^2 : float2_t → float   (power cube)
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
// ============================================================================

__launch_bounds__(256)
__global__ void magnitude_sq_kernel(
    const float2_t* __restrict__ in,
    float*          __restrict__ out,
    unsigned int total)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float re = in[i].x;
  float im = in[i].y;
  out[i] = re * re + im * im;
}

)HIP";
}

}  // namespace kernels
}  // namespace range_angle

#endif  // ENABLE_ROCM
