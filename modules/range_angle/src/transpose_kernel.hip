/**
 * @file transpose_kernel.hip
 * @brief Tiled matrix transpose kernel — offline HIP reference
 *
 * Compiled at RUNTIME via hiprtc from range_angle_kernels_rocm.hpp.
 * This file is for offline syntax checking and code review only.
 *
 * Optimizations:
 *   - 32×32 tile with +1 LDS padding → bank-conflict-free
 *   - __launch_bounds__(1024) for 32×32 block
 *   - __restrict__ on input/output pointers
 *   - 2D grid: blockIdx.x = col tile, blockIdx.y = row tile
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#include <hip/hip_runtime.h>

#define TILE_DIM 32

/// Tiled complex matrix transpose [n_rows × n_cols] → [n_cols × n_rows]
/// Grid: (ceil(n_cols/32), ceil(n_rows/32), 1)  Block: (32, 32, 1)
__launch_bounds__(1024)
__global__ void transpose_complex_kernel(
    const float2* __restrict__ in,
    float2*       __restrict__ out,
    unsigned int n_rows,
    unsigned int n_cols)
{
  // +1 padding on second dimension: avoids bank conflicts on read-back
  __shared__ float2 tile[TILE_DIM][TILE_DIM + 1];

  unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;  // col in input
  unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;  // row in input

  if (x < n_cols && y < n_rows)
    tile[threadIdx.y][threadIdx.x] = in[y * n_cols + x];
  __syncthreads();

  unsigned int out_x = blockIdx.y * TILE_DIM + threadIdx.x;  // = source row
  unsigned int out_y = blockIdx.x * TILE_DIM + threadIdx.y;  // = source col
  if (out_x < n_rows && out_y < n_cols)
    out[out_y * n_rows + out_x] = tile[threadIdx.x][threadIdx.y];
}

#undef TILE_DIM
