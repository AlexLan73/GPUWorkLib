#pragma once

/**
 * @file strategies_kernels_rocm.hpp
 * @brief Optimized HIP kernel sources for AntennaProcessor pipeline
 *
 * Kernels:
 *   1. hamming_pad_fused  - Hamming window + zero-padding in one kernel (fusion P13+P10)
 *   2. compute_magnitudes - |FFT[i]| via fast intrinsic (P9)
 *   3. global_minmax      - Per-beam min+max with warp shuffle reduction (P7+P8)
 *   4. one_max_no_phase   - Per-beam one max + parabola with warp shuffle (P7+P8)
 *
 * Optimizations applied:
 *   - __launch_bounds__(BLOCK_SIZE) on all kernels (P5)
 *   - 2D grid: blockIdx.y = beam_id, no div/mod (P6)
 *   - Warp shuffle for final reduction stages (P7)
 *   - LDS +1 padding to avoid bank conflicts (P8)
 *   - Fast math intrinsics: __cosf, __fsqrt_rn, __log10f (P9)
 *   - Precomputed Hamming window table passed as argument (P10)
 *   - Fused hamming+pad kernel eliminates intermediate global mem R/W (P13)
 *   - BLOCK_SIZE and WARP_SIZE passed via -D at compile time
 *
 * Compiled at runtime via hiprtc with -O3 --offload-arch=gfxXXXX.
 *
 * @date 2026-03-07
 */

#if ENABLE_ROCM

namespace strategies {
namespace kernels {

inline const char* GetStrategiesHIPKernelSource() {
  return R"HIP(

// ============================================================================
// Common structures
// ============================================================================

struct float2_t {
    float x;
    float y;
};

// OneMaxLite: lightweight result without phase (Step2.1)
struct OneMaxLite_t {
    unsigned int beam_id;
    unsigned int bin_index;
    float magnitude;
    float freq_offset;
    float refined_freq_hz;
};

// MinMaxResult GPU-side (Step2.3)
struct MinMaxResult_t {
    unsigned int beam_id;
    float min_magnitude;
    unsigned int min_bin;
    float min_frequency_hz;
    float max_magnitude;
    unsigned int max_bin;
    float max_frequency_hz;
    float dynamic_range_dB;
    unsigned int pad;
};

// ============================================================================
// 1. hamming_pad_fused - Apply precomputed Hamming window + zero-pad in one pass
// ============================================================================
// 2D grid: blockIdx.y = beam_id, blockIdx.x * blockDim.x + threadIdx.x = position in nFFT
// Reads from input [n_ant x n_samples], writes to output [n_ant x nFFT]
// Positions >= n_samples are written as zero (buffer pre-zeroed via hipMemsetAsync)
// Only threads with pos < n_samples do useful work — no else-branch divergence (P11)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void hamming_pad_fused(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    const float* __restrict__ window,
    unsigned int n_ant,
    unsigned int n_samples,
    unsigned int nFFT)
{
    unsigned int beam_idx = blockIdx.y;
    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (beam_idx >= n_ant || pos >= n_samples) return;

    float w = window[pos];
    float2_t val = input[beam_idx * n_samples + pos];
    val.x *= w;
    val.y *= w;
    output[beam_idx * nFFT + pos] = val;
}

// ============================================================================
// 2. compute_magnitudes - |FFT[i]| = sqrt(re^2 + im^2)
// ============================================================================
// 2D grid: blockIdx.y = beam_id, blockIdx.x covers nFFT positions

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void compute_magnitudes(
    const float2_t* __restrict__ spectrum,
    float* __restrict__ magnitudes,
    unsigned int n_ant,
    unsigned int nFFT)
{
    unsigned int beam_idx = blockIdx.y;
    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (beam_idx >= n_ant || pos >= nFFT) return;

    unsigned int gid = beam_idx * nFFT + pos;
    float2_t val = spectrum[gid];
    magnitudes[gid] = __fsqrt_rn(val.x * val.x + val.y * val.y);
}

// ============================================================================
// 3. global_minmax - Per-beam global MIN + MAX (Step2.3)
// ============================================================================
// Grid: dim3(1, n_ant), Block: dim3(BLOCK_SIZE)
// Uses LDS +1 padding to avoid bank conflicts (P8)
// Uses warp shuffle for final reduction stages (P7)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void global_minmax(
    const float* __restrict__ magnitudes,
    MinMaxResult_t* __restrict__ results,
    unsigned int n_ant,
    unsigned int nFFT,
    float sample_rate)
{
    unsigned int beam_idx = blockIdx.y;
    unsigned int lid = threadIdx.x;

    if (beam_idx >= n_ant) return;

    // LDS with +1 padding to avoid bank conflicts
    __shared__ float s_min_mag[BLOCK_SIZE + 1];
    __shared__ unsigned int s_min_idx[BLOCK_SIZE + 1];
    __shared__ float s_max_mag[BLOCK_SIZE + 1];
    __shared__ unsigned int s_max_idx[BLOCK_SIZE + 1];

    float my_min = 1e30f;
    unsigned int my_min_idx = 0;
    float my_max = -1.0f;
    unsigned int my_max_idx = 0;

    unsigned int base = beam_idx * nFFT;

    // Each thread scans multiple elements (stride loop)
    for (unsigned int i = lid; i < nFFT; i += BLOCK_SIZE) {
        float mag = magnitudes[base + i];
        if (mag < my_min) { my_min = mag; my_min_idx = i; }
        if (mag > my_max) { my_max = mag; my_max_idx = i; }
    }

    s_min_mag[lid] = my_min;
    s_min_idx[lid] = my_min_idx;
    s_max_mag[lid] = my_max;
    s_max_idx[lid] = my_max_idx;
    __syncthreads();

    // Tree reduction in LDS — down to WARP_SIZE
    for (unsigned int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (lid < stride) {
            if (s_min_mag[lid + stride] < s_min_mag[lid]) {
                s_min_mag[lid] = s_min_mag[lid + stride];
                s_min_idx[lid] = s_min_idx[lid + stride];
            }
            if (s_max_mag[lid + stride] > s_max_mag[lid]) {
                s_max_mag[lid] = s_max_mag[lid + stride];
                s_max_idx[lid] = s_max_idx[lid + stride];
            }
        }
        __syncthreads();
    }

    // Warp shuffle for final reduction (no __syncthreads needed!)
    if (lid < WARP_SIZE) {
        float wmin = s_min_mag[lid];
        unsigned int wmin_idx = s_min_idx[lid];
        float wmax = s_max_mag[lid];
        unsigned int wmax_idx = s_max_idx[lid];

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_min = __shfl_down(wmin, offset);
            unsigned int other_min_idx = __shfl_down(wmin_idx, offset);
            float other_max = __shfl_down(wmax, offset);
            unsigned int other_max_idx = __shfl_down(wmax_idx, offset);

            if (other_min < wmin) { wmin = other_min; wmin_idx = other_min_idx; }
            if (other_max > wmax) { wmax = other_max; wmax_idx = other_max_idx; }
        }

        // Thread 0 writes result
        if (lid == 0) {
            float bin_width = sample_rate / (float)nFFT;

            results[beam_idx].beam_id          = beam_idx;
            results[beam_idx].min_magnitude    = wmin;
            results[beam_idx].min_bin          = wmin_idx;
            results[beam_idx].min_frequency_hz = (float)wmin_idx * bin_width;
            results[beam_idx].max_magnitude    = wmax;
            results[beam_idx].max_bin          = wmax_idx;
            results[beam_idx].max_frequency_hz = (float)wmax_idx * bin_width;

            float safe_min = (wmin > 1e-30f) ? wmin : 1e-30f;
            results[beam_idx].dynamic_range_dB = 20.0f * __log10f(wmax / safe_min);
            results[beam_idx].pad = 0;
        }
    }
}

// ============================================================================
// 4. one_max_no_phase - Per-beam ONE max + parabola, NO phase (Step2.1)
// ============================================================================
// Grid: dim3(1, n_ant), Block: dim3(BLOCK_SIZE)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void one_max_no_phase(
    const float* __restrict__ magnitudes,
    const float2_t* __restrict__ spectrum,
    OneMaxLite_t* __restrict__ results,
    unsigned int n_ant,
    unsigned int nFFT,
    float sample_rate)
{
    unsigned int beam_idx = blockIdx.y;
    unsigned int lid = threadIdx.x;

    if (beam_idx >= n_ant) return;

    // LDS with +1 padding
    __shared__ float s_max_mag[BLOCK_SIZE + 1];
    __shared__ unsigned int s_max_idx[BLOCK_SIZE + 1];

    float my_max = -1.0f;
    unsigned int my_max_idx = 0;

    unsigned int base = beam_idx * nFFT;

    for (unsigned int i = lid; i < nFFT; i += BLOCK_SIZE) {
        float mag = magnitudes[base + i];
        if (mag > my_max) { my_max = mag; my_max_idx = i; }
    }

    s_max_mag[lid] = my_max;
    s_max_idx[lid] = my_max_idx;
    __syncthreads();

    // Tree reduction down to WARP_SIZE
    for (unsigned int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (lid < stride) {
            if (s_max_mag[lid + stride] > s_max_mag[lid]) {
                s_max_mag[lid] = s_max_mag[lid + stride];
                s_max_idx[lid] = s_max_idx[lid + stride];
            }
        }
        __syncthreads();
    }

    // Warp shuffle for final stages
    if (lid < WARP_SIZE) {
        float wmax = s_max_mag[lid];
        unsigned int wmax_idx = s_max_idx[lid];

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down(wmax, offset);
            unsigned int other_idx = __shfl_down(wmax_idx, offset);
            if (other > wmax) { wmax = other; wmax_idx = other_idx; }
        }

        // Thread 0: parabolic interpolation
        if (lid == 0) {
            unsigned int center_idx = wmax_idx;
            float y_c = wmax;
            float bin_width = sample_rate / (float)nFFT;

            float fo = 0.0f;
            float rf = (float)center_idx * bin_width;

            // 3-point parabolic interpolation
            if (center_idx > 0 && center_idx < nFFT - 1) {
                float y_l = magnitudes[base + center_idx - 1];
                float y_r = magnitudes[base + center_idx + 1];
                float denom = y_l - 2.0f * y_c + y_r;
                if (fabsf(denom) > 1e-10f) {
                    fo = 0.5f * (y_l - y_r) / denom;
                    if (fo < -0.5f) fo = -0.5f;
                    if (fo >  0.5f) fo =  0.5f;
                    rf = ((float)center_idx + fo) * bin_width;
                }
            }

            results[beam_idx].beam_id         = beam_idx;
            results[beam_idx].bin_index       = center_idx;
            results[beam_idx].magnitude       = y_c;
            results[beam_idx].freq_offset     = fo;
            results[beam_idx].refined_freq_hz = rf;
        }
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace strategies

#endif  // ENABLE_ROCM
