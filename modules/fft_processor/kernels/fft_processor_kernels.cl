

// TASK-5: WARP_SIZE define (for future warp shuffle use)
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct float2_t {
    float x;
    float y;
};

// TASK-2+3: 2D grid (blockIdx.y == beam_id) — eliminates expensive int div/mod.
//           Zeros handled by hipMemsetAsync before launch — no divergent else-branch.
// TASK-5: __launch_bounds__(256) for better occupancy.
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int beam_id = blockIdx.y;                             // NO div/mod!
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;  // Only copy valid points; rest already zero

    output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
}

// TASK-4A: Interleaved {mag, phase} output (one buffer instead of two).
//          Caller does single DtoH instead of two separate transfers.
// TASK-5:  __launch_bounds__(256), __fsqrt_rn, __atan2f fast intrinsics.
__launch_bounds__(256)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float2_t* __restrict__ mag_phase,   // interleaved: .x=mag, .y=phase
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = fft_output[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);  // TASK-5: fast sqrt intrinsic
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

