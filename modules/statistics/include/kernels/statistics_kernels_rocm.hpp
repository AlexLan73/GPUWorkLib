#pragma once

/**
 * @file statistics_kernels_rocm.hpp
 * @brief HIP kernel sources for StatisticsProcessor (optimized v2)
 *
 * Contains:
 * - compute_magnitudes:    complex -> |z| per element  (used by ComputeMedian)
 * - mean_reduce_phase1:    block-level sum reduction   (double-load + warp shuffle)
 * - mean_reduce_final:     divide partial sums by n    (warp shuffle)
 * - welford_stats:         single-pass Welford         (kept for compat, optimized)
 * - welford_fused:  TASK-1 one-pass stats, no magnitudes buffer
 * - extract_medians: TASK-2 GPU compact extraction (1 DtoH instead of 256)
 *
 * Optimizations applied (v4):
 *   P0-A: welford_fused   — один pass по данным (нет промежуточного magnitudes buffer)
 *   P0-B: extract_medians — GPU kernel вместо 256 hipMemcpyDtoH
 *   P1-A: Warp shuffle    — финальная стадия reduction без __syncthreads
 *   P1-B: Double-load     — каждый поток читает 2 элемента (вдвое меньше блоков)
 *   P1-C: __launch_bounds__(256) — правильный резерв регистров
 *   P1-D: blocks_per_beam — передаётся как параметр (убран div в ядре)
 *   P2-A: __fsqrt_rn      — fast intrinsic вместо sqrtf
 *   P2-B: LDS +1 padding  — устранение bank conflicts при tree reduction
 *   P3-A: 2D grid          — mean_reduce_phase1 без div/mod (blockIdx.y = beam)
 *   P3-B: LDS +1 padding   — welford ядра: extern __shared__ с +1 padding
 *   P3-C: #pragma unroll 4  — grid-stride loops в welford ядрах
 *   P3-D: double-load       — compute_magnitudes: 2 элемента на поток
 *
 * WARP_SIZE: передаётся через -DWARP_SIZE=N при компиляции (32 RDNA, 64 CDNA)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-26
 */

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

/**
 * @brief Optimized HIP kernel source (v2) for statistics operations
 */
inline const char* GetStatisticsKernelSource() {
    return R"HIP(

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
    __shared__ float sdata_x[256 + 1];
    __shared__ float sdata_y[256 + 1];

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
// Kernel 4: welford_stats   [TASK-4.1, 4.4, 4.5, P3-B, P3-C]
// Optimized version of original kernel (kept for backward compat).
// Reads both input + magnitudes (used internally if needed).
// P3-B: LDS +1 padding to avoid bank conflicts in tree reduction.
// P3-C: #pragma unroll 4 on grid-stride loop for ILP.
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

    // P3-B: +1 padding per array to avoid LDS bank conflicts
    extern __shared__ char shared_mem[];
    float* s_sum_re  = (float*)shared_mem;
    float* s_sum_im  = s_sum_re  + block_size + 1;
    float* s_sum_mag = s_sum_im  + block_size + 1;
    float* s_sum_sq  = s_sum_mag + block_size + 1;

    float sum_re = 0.0f, sum_im = 0.0f, sum_mag = 0.0f, sum_sq = 0.0f;
    unsigned int base = beam_id * n_point;

    #pragma unroll 4
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
// Kernel 5: welford_fused   [TASK-1, TASK-4.1, 4.4, 4.5, P3-B, P3-C]
// Single pass: reads only input[], computes |z| on the fly.
// Eliminates separate compute_magnitudes call for ComputeStatistics path.
// P3-B: LDS +1 padding to avoid bank conflicts in tree reduction.
// P3-C: #pragma unroll 4 on grid-stride loop for ILP.
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

    // P3-B: +1 padding per array to avoid LDS bank conflicts
    extern __shared__ char shared_mem[];
    float* s_sum_re  = (float*)shared_mem;
    float* s_sum_im  = s_sum_re  + block_size + 1;
    float* s_sum_mag = s_sum_im  + block_size + 1;
    float* s_sum_sq  = s_sum_mag + block_size + 1;

    float sum_re = 0.0f, sum_im = 0.0f, sum_mag = 0.0f, sum_sq = 0.0f;
    unsigned int base = beam_id * n_point;

    // P3-C: unroll grid-stride loop for ILP (overlap sqrt with memory latency)
    #pragma unroll 4
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

// =========================================================================
// Kernel 7: welford_float   [P3-B, P3-C]
// Welford stats for float input (magnitudes already computed).
// Used by strategies module for post-FFT stats on |spectrum|.
// Grid: (beam_count, 1, 1), Block: (256, 1, 1)
// P3-B: LDS +1 padding to avoid bank conflicts in tree reduction.
// P3-C: #pragma unroll 4 on grid-stride loop for ILP.
// =========================================================================
__launch_bounds__(256)
extern "C" __global__ void welford_float(
    const float* __restrict__ input,     // float magnitudes [beam_count x n_point]
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id   = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid        = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // P3-B: +1 padding per array to avoid LDS bank conflicts
    extern __shared__ char shared_mem[];
    float* s_sum_mag = (float*)shared_mem;
    float* s_sum_sq  = s_sum_mag + block_size + 1;

    float sum_mag = 0.0f, sum_sq = 0.0f;
    unsigned int base = beam_id * n_point;

    #pragma unroll 4
    for (unsigned int i = tid; i < n_point; i += block_size) {
        float val = input[base + i];
        sum_mag += val;
        sum_sq  += val * val;
    }

    s_sum_mag[tid] = sum_mag;
    s_sum_sq[tid]  = sum_sq;
    __syncthreads();

    for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            s_sum_mag[tid] += s_sum_mag[tid + s];
            s_sum_sq[tid]  += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float vm = s_sum_mag[tid], vs = s_sum_sq[tid];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            vm += __shfl_down(vm, off);
            vs += __shfl_down(vs, off);
        }
        if (tid == 0) {
            float inv_n = 1.0f / (float)n_point;
            WelfordResult r;
            r.mean_re  = 0.0f;
            r.mean_im  = 0.0f;
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

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
