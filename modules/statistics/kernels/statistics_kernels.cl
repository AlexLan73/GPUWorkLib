

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct float2_t {
    float x;
    float y;
};

// Per-beam Welford result (5 floats)
struct WelfordResult {
    float mean_re;       // complex mean (real)
    float mean_im;       // complex mean (imag)
    float mean_mag;      // mean of magnitudes
    float variance;      // variance of magnitudes
    float std_dev;       // std deviation of magnitudes
};

// =========================================================================
// Kernel 1: compute_magnitudes   [TASK-4.1, 4.5]
// Still used for ComputeMedian path (magnitudes → rocPRIM sort).
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void compute_magnitudes(
    const float2_t* __restrict__ input,
    float* __restrict__ magnitudes,
    unsigned int total_elements)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_elements) return;

    float2_t z = input[gid];
    magnitudes[gid] = __fsqrt_rn(z.x * z.x + z.y * z.y);
}

// =========================================================================
// Kernel 2: mean_reduce_phase1   [TASK-4.1, 4.2, 4.3, 4.4, P2-B]
// Block-level complex sum with double-load + warp shuffle.
// Grid: (blocks_per_beam * beam_count, 1, 1)  where
//   blocks_per_beam = ceil(n_point / (blockDim.x * 2))  -- double-load!
// P2-B: LDS accessed as float[] with +1 padding to avoid bank conflicts.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void mean_reduce_phase1(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ partial_sums,
    unsigned int beam_count,
    unsigned int n_point,
    unsigned int blocks_per_beam)   // TASK-4.2: passed from CPU, avoids intra-kernel div
{
    // P2-B: LDS with +1 padding per row to avoid bank conflicts
    __shared__ float sdata_x[256 + 1];
    __shared__ float sdata_y[256 + 1];

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // Determine beam and block position (2 divs, blocks_per_beam is from CPU)
    unsigned int beam_id       = blockIdx.x / blocks_per_beam;
    unsigned int block_in_beam = blockIdx.x % blocks_per_beam;

    if (beam_id >= beam_count) return;

    unsigned int beam_start = beam_id * n_point;

    // TASK-4.3: Double-load — each thread reads 2 elements
    unsigned int local1 = block_in_beam * (block_size * 2) + tid;
    unsigned int local2 = local1 + block_size;

    float2_t zero; zero.x = 0.0f; zero.y = 0.0f;
    float2_t v1 = (local1 < n_point) ? input[beam_start + local1] : zero;
    float2_t v2 = (local2 < n_point) ? input[beam_start + local2] : zero;

    sdata_x[tid] = v1.x + v2.x;
    sdata_y[tid] = v1.y + v2.y;
    __syncthreads();

    // LDS tree reduction (down to WARP_SIZE elements)
    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata_x[tid] += sdata_x[tid + s];
            sdata_y[tid] += sdata_y[tid + s];
        }
        __syncthreads();
    }

    // TASK-4.4: Warp shuffle — final WARP_SIZE → 1 without __syncthreads
    if (tid < WARP_SIZE) {
        float vx = sdata_x[tid], vy = sdata_y[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            vx += __shfl_down(vx, offset);
            vy += __shfl_down(vy, offset);
        }
        if (tid == 0) {
            partial_sums[blockIdx.x].x = vx;
            partial_sums[blockIdx.x].y = vy;
        }
    }
}

// =========================================================================
// Kernel 3: mean_reduce_final   [TASK-4.1, 4.4, P2-B]
// Reduces partial sums per beam, divides by n_point.
// P2-B: LDS with +1 padding.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void mean_reduce_final(
    const float2_t* __restrict__ partial_sums,
    float2_t* __restrict__ means,
    unsigned int beam_count,
    unsigned int blocks_per_beam,
    unsigned int n_point)
{
    // P2-B: LDS with +1 padding to avoid bank conflicts
    __shared__ float sdata_x[256 + 1];
    __shared__ float sdata_y[256 + 1];

    unsigned int beam_id = blockIdx.x;
    unsigned int tid     = threadIdx.x;

    if (beam_id >= beam_count) return;

    float sum_x = 0.0f, sum_y = 0.0f;
    unsigned int base_offset = beam_id * blocks_per_beam;

    for (unsigned int i = tid; i < blocks_per_beam; i += blockDim.x) {
        float2_t ps = partial_sums[base_offset + i];
        sum_x += ps.x;
        sum_y += ps.y;
    }
    sdata_x[tid] = sum_x;
    sdata_y[tid] = sum_y;
    __syncthreads();

    // LDS tree reduction
    for (unsigned int s = blockDim.x / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata_x[tid] += sdata_x[tid + s];
            sdata_y[tid] += sdata_y[tid + s];
        }
        __syncthreads();
    }

    // Warp shuffle final stage
    if (tid < WARP_SIZE) {
        float vx = sdata_x[tid], vy = sdata_y[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            vx += __shfl_down(vx, offset);
            vy += __shfl_down(vy, offset);
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            float2_t mean_val;
            mean_val.x = vx * inv_n;
            mean_val.y = vy * inv_n;
            means[beam_id] = mean_val;
        }
    }
}

// =========================================================================
// Kernel 4: welford_stats   [TASK-4.1, 4.4, 4.5]
// Optimized version of original kernel (kept for backward compat).
// Reads both input + magnitudes (used internally if needed).
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void welford_stats(
    const float2_t* __restrict__ input,
    const float* __restrict__ magnitudes,
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id   = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ char shared_mem[];
    float* s_sum_re  = (float*)shared_mem;
    float* s_sum_im  = s_sum_re  + block_size;
    float* s_sum_mag = s_sum_im  + block_size;
    float* s_sum_sq  = s_sum_mag + block_size;

    float sum_re = 0.0f, sum_im = 0.0f, sum_mag = 0.0f, sum_sq = 0.0f;
    unsigned int base = beam_id * n_point;

    for (unsigned int i = tid; i < n_point; i += block_size) {
        float2_t z  = input[base + i];
        float mag   = magnitudes[base + i];
        sum_re  += z.x;
        sum_im  += z.y;
        sum_mag += mag;
        sum_sq  += mag * mag;
    }

    s_sum_re[tid]  = sum_re;
    s_sum_im[tid]  = sum_im;
    s_sum_mag[tid] = sum_mag;
    s_sum_sq[tid]  = sum_sq;
    __syncthreads();

    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            s_sum_re[tid]  += s_sum_re[tid + s];
            s_sum_im[tid]  += s_sum_im[tid + s];
            s_sum_mag[tid] += s_sum_mag[tid + s];
            s_sum_sq[tid]  += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float vr = s_sum_re[tid], vi = s_sum_im[tid];
        float vm = s_sum_mag[tid], vs = s_sum_sq[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            vr += __shfl_down(vr, off);
            vi += __shfl_down(vi, off);
            vm += __shfl_down(vm, off);
            vs += __shfl_down(vs, off);
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            WelfordResult r;
            r.mean_re  = vr * inv_n;
            r.mean_im  = vi * inv_n;
            r.mean_mag = vm * inv_n;
            float mean_sq = vs * inv_n;
            r.variance = mean_sq - r.mean_mag * r.mean_mag;
            if (r.variance < 0.0f) r.variance = 0.0f;
            r.std_dev = __fsqrt_rn(r.variance);
            results[beam_id] = r;
        }
    }
}

// =========================================================================
// Kernel 5: welford_fused   [TASK-1, TASK-4.1, 4.4, 4.5]
// Single pass: reads only input[], computes |z| on the fly.
// Eliminates separate compute_magnitudes call for ComputeStatistics path.
// Expected speedup: -40-50% on ComputeStatistics.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void welford_fused(
    const float2_t* __restrict__ input,   // only input buffer — no magnitudes!
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id   = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ char shared_mem[];
    float* s_sum_re  = (float*)shared_mem;
    float* s_sum_im  = s_sum_re  + block_size;
    float* s_sum_mag = s_sum_im  + block_size;
    float* s_sum_sq  = s_sum_mag + block_size;

    float sum_re = 0.0f, sum_im = 0.0f, sum_mag = 0.0f, sum_sq = 0.0f;
    unsigned int base = beam_id * n_point;

    // Grid-stride loop: reads input ONCE, computes |z| on the fly (TASK-1)
    for (unsigned int i = tid; i < n_point; i += block_size) {
        float2_t z   = input[base + i];
        float mag    = __fsqrt_rn(z.x * z.x + z.y * z.y);  // TASK-4.5
        sum_re  += z.x;
        sum_im  += z.y;
        sum_mag += mag;
        sum_sq  += mag * mag;
    }

    s_sum_re[tid]  = sum_re;
    s_sum_im[tid]  = sum_im;
    s_sum_mag[tid] = sum_mag;
    s_sum_sq[tid]  = sum_sq;
    __syncthreads();

    // LDS tree reduction
    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            s_sum_re[tid]  += s_sum_re[tid + s];
            s_sum_im[tid]  += s_sum_im[tid + s];
            s_sum_mag[tid] += s_sum_mag[tid + s];
            s_sum_sq[tid]  += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    // Warp shuffle final stage (TASK-4.4)
    if (tid < WARP_SIZE) {
        float vr = s_sum_re[tid], vi = s_sum_im[tid];
        float vm = s_sum_mag[tid], vs = s_sum_sq[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            vr += __shfl_down(vr, off);
            vi += __shfl_down(vi, off);
            vm += __shfl_down(vm, off);
            vs += __shfl_down(vs, off);
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            WelfordResult r;
            r.mean_re  = vr * inv_n;
            r.mean_im  = vi * inv_n;
            r.mean_mag = vm * inv_n;
            float mean_sq = vs * inv_n;
            r.variance = mean_sq - r.mean_mag * r.mean_mag;
            if (r.variance < 0.0f) r.variance = 0.0f;
            r.std_dev  = __fsqrt_rn(r.variance);
            results[beam_id] = r;
        }
    }
}

// =========================================================================
// Kernel 6: extract_medians   [TASK-2]
// Extracts middle element of each sorted beam into compact float array.
// Eliminates 256 separate hipMemcpyDtoH calls in ComputeMedian.
// Grid: (ceil(beam_count / blockDim.x), 1, 1)  — one thread per beam
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void extract_medians(
    const float* __restrict__ sorted,    // sort_buf_ after rocPRIM segmented sort
    float* __restrict__ medians,         // compact output: beam_count floats
    unsigned int n_point,
    unsigned int beam_count)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= beam_count) return;
    medians[b] = sorted[b * n_point + n_point / 2];
}

