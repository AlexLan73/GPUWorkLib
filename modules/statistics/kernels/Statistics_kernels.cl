

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
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
// Kernel 1: compute_magnitudes   [TASK-4.1, 4.5, P3-D]
// Still used for ComputeMedian path (magnitudes → rocPRIM sort).
// P3-D: double-load — each thread processes 2 elements for better ILP.
// Grid: ceil(total / (blockDim.x * 2))
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void compute_magnitudes(
    const float2_t* __restrict__ input,
    float* __restrict__ magnitudes,
    unsigned int total_elements)
{
    unsigned int gid1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gid2 = gid1 + blockDim.x;

    if (gid1 < total_elements) {
        float2_t z = input[gid1];
        magnitudes[gid1] = __fsqrt_rn(z.x * z.x + z.y * z.y);
    }
    if (gid2 < total_elements) {
        float2_t z = input[gid2];
        magnitudes[gid2] = __fsqrt_rn(z.x * z.x + z.y * z.y);
    }
}

// =========================================================================
// Kernel 2: mean_reduce_phase1   [TASK-4.1, 4.2, 4.3, 4.4, P2-B, P3-A]
// Block-level complex sum with double-load + warp shuffle.
// P3-A: 2D grid — blockIdx.y = beam_id, blockIdx.x = block_in_beam
//   Grid: (blocks_per_beam, beam_count, 1)
//   blocks_per_beam = ceil(n_point / (blockDim.x * 2))  -- double-load!
// P2-B: LDS accessed as float[] with +1 padding to avoid bank conflicts.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void mean_reduce_phase1(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ partial_sums,
    unsigned int beam_count,
    unsigned int n_point)
{
    // P2-B: LDS with +1 padding per row to avoid bank conflicts
    __shared__ float sdata_x[BLOCK_SIZE + 1];
    __shared__ float sdata_y[BLOCK_SIZE + 1];

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // P3-A: 2D grid eliminates div/mod (~40 cycles saved per thread)
    unsigned int beam_id       = blockIdx.y;
    unsigned int block_in_beam = blockIdx.x;

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
            unsigned int out_idx = beam_id * gridDim.x + block_in_beam;
            partial_sums[out_idx].x = vx;
            partial_sums[out_idx].y = vy;
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
    __shared__ float sdata_x[BLOCK_SIZE + 1];
    __shared__ float sdata_y[BLOCK_SIZE + 1];

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
// Kernel 4: welford_stats   [TRUE PARALLEL WELFORD — Bennett et al. 2009]
// Reads complex input + pre-computed magnitudes (backward compat path).
// Uses true Welford merge for numerically stable variance computation.
// LDS: 5 arrays with +1 padding — sum_re, sum_im, mean_mag, M2, count.
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
    float* s_sum_re   = (float*)shared_mem;
    float* s_sum_im   = s_sum_re   + block_size + 1;
    float* s_mean_mag = s_sum_im   + block_size + 1;
    float* s_M2       = s_mean_mag + block_size + 1;
    float* s_count    = s_M2       + block_size + 1;

    float t_sum_re = 0.0f, t_sum_im = 0.0f;
    float t_mean = 0.0f, t_M2 = 0.0f, t_cnt = 0.0f;
    unsigned int base = beam_id * n_point;

    #pragma unroll 4
    for (unsigned int i = tid; i < n_point; i += block_size) {
        float2_t z = input[base + i];
        float mag  = magnitudes[base + i];
        t_sum_re += z.x;
        t_sum_im += z.y;
        t_cnt += 1.0f;
        float delta = mag - t_mean;
        t_mean += delta / t_cnt;
        float delta2 = mag - t_mean;
        t_M2 += delta * delta2;
    }

    s_sum_re[tid]   = t_sum_re;
    s_sum_im[tid]   = t_sum_im;
    s_mean_mag[tid] = t_mean;
    s_M2[tid]       = t_M2;
    s_count[tid]    = t_cnt;
    __syncthreads();

    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            s_sum_re[tid] += s_sum_re[tid + s];
            s_sum_im[tid] += s_sum_im[tid + s];
            float na = s_count[tid], nb = s_count[tid + s];
            float nab = na + nb;
            if (nab > 0.0f) {
                float d = s_mean_mag[tid + s] - s_mean_mag[tid];
                s_mean_mag[tid] = (s_mean_mag[tid] * na + s_mean_mag[tid + s] * nb) / nab;
                s_M2[tid] = s_M2[tid] + s_M2[tid + s] + d * d * na * nb / nab;
                s_count[tid] = nab;
            }
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float vr = s_sum_re[tid], vi = s_sum_im[tid];
        float vm = s_mean_mag[tid], vM2 = s_M2[tid], vc = s_count[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float or_ = __shfl_down(vr, off), oi = __shfl_down(vi, off);
            float om = __shfl_down(vm, off), oM2 = __shfl_down(vM2, off);
            float oc = __shfl_down(vc, off);
            vr += or_; vi += oi;
            float nab = vc + oc;
            if (nab > 0.0f) {
                float d = om - vm;
                vm = (vm * vc + om * oc) / nab;
                vM2 = vM2 + oM2 + d * d * vc * oc / nab;
                vc = nab;
            }
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            WelfordResult r;
            r.mean_re  = vr * inv_n;
            r.mean_im  = vi * inv_n;
            r.mean_mag = vm;
            r.variance = (vc > 1.0f) ? vM2 / vc : 0.0f;
            if (r.variance < 0.0f) r.variance = 0.0f;
            r.std_dev = __fsqrt_rn(r.variance);
            results[beam_id] = r;
        }
    }
}

// =========================================================================
// Kernel 5: welford_fused   [TRUE PARALLEL WELFORD — Bennett et al. 2009]
// Single pass: reads only input[], computes |z| on the fly.
// Numerically stable: uses Welford per-thread accumulator + merge at reduction.
// Eliminates separate compute_magnitudes call for ComputeStatistics path.
// LDS: 5 arrays with +1 padding — sum_re, sum_im, mean_mag, M2, count.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void welford_fused(
    const float2_t* __restrict__ input,
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id   = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ char shared_mem[];
    float* s_sum_re   = (float*)shared_mem;
    float* s_sum_im   = s_sum_re   + block_size + 1;
    float* s_mean_mag = s_sum_im   + block_size + 1;
    float* s_M2       = s_mean_mag + block_size + 1;
    float* s_count    = s_M2       + block_size + 1;

    // Per-thread Welford accumulator for magnitudes + simple sum for complex mean
    float t_sum_re = 0.0f, t_sum_im = 0.0f;
    float t_mean = 0.0f, t_M2 = 0.0f, t_cnt = 0.0f;
    unsigned int base = beam_id * n_point;

    #pragma unroll 4
    for (unsigned int i = tid; i < n_point; i += block_size) {
        float2_t z = input[base + i];
        float mag  = __fsqrt_rn(z.x * z.x + z.y * z.y);
        t_sum_re += z.x;
        t_sum_im += z.y;
        // Welford online update for magnitude statistics
        t_cnt += 1.0f;
        float delta = mag - t_mean;
        t_mean += delta / t_cnt;
        float delta2 = mag - t_mean;
        t_M2 += delta * delta2;
    }

    s_sum_re[tid]   = t_sum_re;
    s_sum_im[tid]   = t_sum_im;
    s_mean_mag[tid] = t_mean;
    s_M2[tid]       = t_M2;
    s_count[tid]    = t_cnt;
    __syncthreads();

    // LDS tree reduction: simple sum for complex, Welford merge for magnitude
    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            s_sum_re[tid] += s_sum_re[tid + s];
            s_sum_im[tid] += s_sum_im[tid + s];
            // Welford merge (Bennett et al. 2009)
            float na = s_count[tid], nb = s_count[tid + s];
            float nab = na + nb;
            if (nab > 0.0f) {
                float d = s_mean_mag[tid + s] - s_mean_mag[tid];
                s_mean_mag[tid] = (s_mean_mag[tid] * na + s_mean_mag[tid + s] * nb) / nab;
                s_M2[tid] = s_M2[tid] + s_M2[tid + s] + d * d * na * nb / nab;
                s_count[tid] = nab;
            }
        }
        __syncthreads();
    }

    // Warp shuffle: simple sum for complex, Welford merge for magnitude
    if (tid < WARP_SIZE) {
        float vr = s_sum_re[tid], vi = s_sum_im[tid];
        float vm = s_mean_mag[tid], vM2 = s_M2[tid], vc = s_count[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float or_ = __shfl_down(vr, off), oi = __shfl_down(vi, off);
            float om = __shfl_down(vm, off), oM2 = __shfl_down(vM2, off);
            float oc = __shfl_down(vc, off);
            vr += or_; vi += oi;
            float nab = vc + oc;
            if (nab > 0.0f) {
                float d = om - vm;
                vm = (vm * vc + om * oc) / nab;
                vM2 = vM2 + oM2 + d * d * vc * oc / nab;
                vc = nab;
            }
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            WelfordResult r;
            r.mean_re  = vr * inv_n;
            r.mean_im  = vi * inv_n;
            r.mean_mag = vm;  // Welford tracks running mean directly
            r.variance = (vc > 1.0f) ? vM2 / vc : 0.0f;
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

    unsigned int mid = n_point / 2;
    if (n_point % 2 == 0 && mid > 0) {
        // Even count: average of two middle elements for correct median
        medians[b] = (sorted[b * n_point + mid - 1] + sorted[b * n_point + mid]) * 0.5f;
    } else {
        medians[b] = sorted[b * n_point + mid];
    }
}

// =========================================================================
// Kernel 7: welford_float   [TRUE PARALLEL WELFORD — Bennett et al. 2009]
// Welford stats for float input (magnitudes already computed).
// Numerically stable: per-thread Welford + merge at reduction.
// LDS: 3 arrays with +1 padding — mean_mag, M2, count.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void welford_float(
    const float* __restrict__ input,
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id   = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ char shared_mem[];
    float* s_mean = (float*)shared_mem;
    float* s_M2   = s_mean + block_size + 1;
    float* s_cnt  = s_M2   + block_size + 1;

    float t_mean = 0.0f, t_M2 = 0.0f, t_cnt = 0.0f;
    unsigned int base = beam_id * n_point;

    #pragma unroll 4
    for (unsigned int i = tid; i < n_point; i += block_size) {
        float val = input[base + i];
        t_cnt += 1.0f;
        float delta = val - t_mean;
        t_mean += delta / t_cnt;
        float delta2 = val - t_mean;
        t_M2 += delta * delta2;
    }

    s_mean[tid] = t_mean;
    s_M2[tid]   = t_M2;
    s_cnt[tid]  = t_cnt;
    __syncthreads();

    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            float na = s_cnt[tid], nb = s_cnt[tid + s];
            float nab = na + nb;
            if (nab > 0.0f) {
                float d = s_mean[tid + s] - s_mean[tid];
                s_mean[tid] = (s_mean[tid] * na + s_mean[tid + s] * nb) / nab;
                s_M2[tid] = s_M2[tid] + s_M2[tid + s] + d * d * na * nb / nab;
                s_cnt[tid] = nab;
            }
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float vm = s_mean[tid], vM2 = s_M2[tid], vc = s_cnt[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float om = __shfl_down(vm, off);
            float oM2 = __shfl_down(vM2, off);
            float oc = __shfl_down(vc, off);
            float nab = vc + oc;
            if (nab > 0.0f) {
                float d = om - vm;
                vm = (vm * vc + om * oc) / nab;
                vM2 = vM2 + oM2 + d * d * vc * oc / nab;
                vc = nab;
            }
        }
        if (tid == 0) {
            WelfordResult r;
            r.mean_re  = 0.0f;
            r.mean_im  = 0.0f;
            r.mean_mag = vm;
            r.variance = (vc > 1.0f) ? vM2 / vc : 0.0f;
            if (r.variance < 0.0f) r.variance = 0.0f;
            r.std_dev  = __fsqrt_rn(r.variance);
            results[beam_id] = r;
        }
    }
}

// =========================================================================
// Kernel 8: histogram_median_pass   [HISTOGRAM MEDIAN]
// Builds a 256-bin histogram of the specified byte of float magnitudes.
// Float >= 0 has order-preserving uint32 representation (just flip sign bit).
// 2D grid: (blocks_per_beam, beam_count), Block: (256)
// Shared: uint32_t local_hist[256] = 1 KB
//
// Parameters:
//   magnitudes   — float[beam_count × n_point] (already computed |z|)
//   histograms   — uint32[beam_count × 256] (output, must be zeroed before call)
//   n_point      — samples per beam
//   beam_count   — number of beams
//   pass         — 0..3 (which byte: 0=MSB bits[31:24], 3=LSB bits[7:0])
//   target_value — uint32[beam_count] accumulated target from previous passes
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void histogram_median_pass(
    const float* __restrict__ magnitudes,
    unsigned int* __restrict__ histograms,
    unsigned int n_point,
    unsigned int beam_count,
    unsigned int pass,
    const unsigned int* __restrict__ target_value)
{
    __shared__ unsigned int local_hist[256];

    unsigned int tid = threadIdx.x;
    unsigned int beam_id = blockIdx.y;
    if (beam_id >= beam_count) return;

    // Clear shared histogram
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();

    unsigned int base = beam_id * n_point;
    unsigned int tv = target_value[beam_id];
    unsigned int shift = (3u - pass) * 8u;
    // Mask for bits above current pass (used for filtering in passes 1-3)
    unsigned int filter_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << ((4u - pass) * 8u));

    // Grid-stride loop
    #pragma unroll 4
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n_point;
         i += gridDim.x * blockDim.x) {
        float val = magnitudes[base + i];
        // Float >= 0: flip sign bit for order-preserving uint representation
        unsigned int u = __float_as_uint(val) ^ 0x80000000u;

        // Filter: passes 1-3 only process elements matching target bytes from prior passes
        if (pass > 0 && (u & filter_mask) != (tv & filter_mask))
            continue;

        unsigned int byte_val = (u >> shift) & 0xFFu;
        atomicAdd(&local_hist[byte_val], 1u);
    }
    __syncthreads();

    // Reduce local histogram → global histogram
    if (tid < 256) {
        atomicAdd(&histograms[beam_id * 256u + tid], local_hist[tid]);
    }
}

// =========================================================================
// Kernel 9: histogram_median_pass_complex   [HISTOGRAM MEDIAN]
// Same as histogram_median_pass but reads complex input and computes |z| on the fly.
// Eliminates need for separate compute_magnitudes call (like welford_fused).
// 2D grid: (blocks_per_beam, beam_count), Block: (256)
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void histogram_median_pass_complex(
    const float2_t* __restrict__ input,
    unsigned int* __restrict__ histograms,
    unsigned int n_point,
    unsigned int beam_count,
    unsigned int pass,
    const unsigned int* __restrict__ target_value)
{
    __shared__ unsigned int local_hist[256];

    unsigned int tid = threadIdx.x;
    unsigned int beam_id = blockIdx.y;
    if (beam_id >= beam_count) return;

    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();

    unsigned int base = beam_id * n_point;
    unsigned int tv = target_value[beam_id];
    unsigned int shift = (3u - pass) * 8u;
    unsigned int filter_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << ((4u - pass) * 8u));

    #pragma unroll 4
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n_point;
         i += gridDim.x * blockDim.x) {
        float2_t z = input[base + i];
        float val = __fsqrt_rn(z.x * z.x + z.y * z.y);
        unsigned int u = __float_as_uint(val) ^ 0x80000000u;

        if (pass > 0 && (u & filter_mask) != (tv & filter_mask))
            continue;

        unsigned int byte_val = (u >> shift) & 0xFFu;
        atomicAdd(&local_hist[byte_val], 1u);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&histograms[beam_id * 256u + tid], local_hist[tid]);
    }
}

// =========================================================================
// Kernel 10: find_median_bucket   [HISTOGRAM MEDIAN]
// Prefix-sum over 256-bin histogram to find the bucket containing the median.
// Updates target_prefix and target_value for the next pass.
// Grid: (beam_count), Block: (1)  — simple sequential scan per beam
// =========================================================================
extern "C" __global__ void find_median_bucket(
    const unsigned int* __restrict__ histograms,
    unsigned int* __restrict__ target_prefix,
    unsigned int* __restrict__ target_value,
    unsigned int median_rank,
    unsigned int pass)
{
    unsigned int beam_id = blockIdx.x;
    unsigned int prefix = target_prefix[beam_id];
    unsigned int shift = (3u - pass) * 8u;
    unsigned int prev_tv = target_value[beam_id];
    // Clear bits at current pass position and below
    unsigned int clear_mask = ~((0xFFu) << shift);
    unsigned int base_tv = prev_tv & clear_mask;

    for (unsigned int b = 0; b < 256u; b++) {
        unsigned int count = histograms[beam_id * 256u + b];
        if (prefix + count > median_rank) {
            // Median is in this bucket
            target_value[beam_id] = base_tv | (b << shift);
            target_prefix[beam_id] = prefix;
            return;
        }
        prefix += count;
    }
    // Fallback (should not happen for valid data)
    target_value[beam_id] = base_tv | (255u << shift);
    target_prefix[beam_id] = prefix;
}



// ═══════════════════════════════════════════════════════════════
// Kernel: gather_decimated      (SNR_03)
// Gather subset [n_ant_out × n_samp_out] from [n_antennas × n_samples]
// with step_antennas / step_samples strides.
// Thread mapping: 1 thread = 1 antenna, sequential loop over samples.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(64)
extern "C" __global__ void gather_decimated(
    const float2_t* __restrict__ src,   // [n_antennas × n_samples], row-major
    float2_t*       __restrict__ dst,   // [n_ant_out × n_samp_out], row-major
    unsigned int n_samples,             // ширина исходной матрицы
    unsigned int n_samp_out,            // ширина выходной матрицы
    unsigned int step_antennas,         // шаг по строкам (antennas)
    unsigned int step_samples,          // шаг по столбцам (samples)
    unsigned int n_ant_out)             // число выходных антенн
{
    unsigned int ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant_out) return;

    // size_t cast для защиты от переполнения при n_antennas * n_samples > 4G.
    const float2_t* src_row =
        src + (size_t)ant * (size_t)step_antennas * (size_t)n_samples;
    float2_t* dst_row =
        dst + (size_t)ant * (size_t)n_samp_out;

    // Sequential loop — L2 prefetcher видит линейный паттерн (stride const).
    for (unsigned int s = 0; s < n_samp_out; ++s) {
        dst_row[s] = src_row[(size_t)s * (size_t)step_samples];
    }
}



#ifndef PEAK_CFAR_BLOCK_SIZE
#define PEAK_CFAR_BLOCK_SIZE 256
#endif

// ═══════════════════════════════════════════════════════════════
// Kernel: peak_cfar      (SNR_05 — CA-CFAR SNR-estimator)
// Работает с |X|² (square-law). Результат: snr_db[ant].
//
// Thread mapping: 1 block = 1 антенна, blockDim.x = 256.
// search_full_spectrum управляется через параметр nFFT.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(PEAK_CFAR_BLOCK_SIZE)
extern "C" __global__ void peak_cfar(
    const float* __restrict__ mag_sq,      // [n_ant × nFFT] |X|²
    float*       __restrict__ snr_db_out,  // [n_ant]
    unsigned int nFFT,                     // search range (full or half spectrum)
    unsigned int guard_bins,
    unsigned int ref_bins)
{
    __shared__ float        s_max_val[PEAK_CFAR_BLOCK_SIZE];
    __shared__ unsigned int s_max_idx[PEAK_CFAR_BLOCK_SIZE];
    __shared__ float        s_ref_sum;
    __shared__ unsigned int s_ref_count;

    unsigned int ant = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float* row = mag_sq + (size_t)ant * (size_t)nFFT;

    // ─── Pass 1: parallel argmax over [0..nFFT) ────────────────────
    float        my_max = -1.0f;
    unsigned int my_idx = 0;
    for (unsigned int k = tid; k < nFFT; k += PEAK_CFAR_BLOCK_SIZE) {
        float v = row[k];
        if (v > my_max) { my_max = v; my_idx = k; }
    }
    s_max_val[tid] = my_max;
    s_max_idx[tid] = my_idx;
    __syncthreads();

    // Tree reduction: argmax in s_max_val[0] / s_max_idx[0]
    for (unsigned int s = PEAK_CFAR_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    unsigned int k_peak = s_max_idx[0];
    float        peak   = s_max_val[0];

    // ─── Pass 2: parallel ref-window sum with wraparound ────────────
    if (tid == 0) { s_ref_sum = 0.0f; s_ref_count = 0u; }
    __syncthreads();

    // Ref индексы: k_peak ± (guard+1 .. guard+ref)  mod nFFT
    // Всего 2*ref_bins точек (по ref_bins с каждой стороны).
    unsigned int total_ref = 2u * ref_bins;
    for (unsigned int i = tid; i < total_ref; i += PEAK_CFAR_BLOCK_SIZE) {
        int offset;
        if (i < ref_bins) {
            // Левая сторона: k_peak - (guard+1+i)
            offset = -(int)(guard_bins + 1u + i);
        } else {
            // Правая сторона: k_peak + (guard+1+(i-ref_bins))
            offset = (int)(guard_bins + 1u + (i - ref_bins));
        }
        // Wraparound: +nFFT защищает от отрицательных при малых k_peak.
        int k_ref = ((int)k_peak + offset + (int)nFFT) % (int)nFFT;
        atomicAdd(&s_ref_sum, row[k_ref]);
        atomicAdd(&s_ref_count, 1u);
    }
    __syncthreads();

    // ─── Pass 3: результат (только поток 0) ─────────────────────────
    if (tid == 0) {
        float noise_mean = (s_ref_count > 0)
            ? (s_ref_sum / (float)s_ref_count)
            : 1.0f;
        // Защита 1: log10(0) и деление на 0
        float ratio = (noise_mean > 1e-30f) ? (peak / noise_mean) : 1.0f;
        // Защита 2: log10(отрицательного) — peak может быть 0 для чисто шумовой антенны
        ratio = fmaxf(ratio, 1e-30f);
        snr_db_out[ant] = 10.0f * __log10f(ratio);
    }
}

