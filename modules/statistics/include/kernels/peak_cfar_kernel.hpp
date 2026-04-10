#pragma once

/**
 * @file peak_cfar_kernel.hpp
 * @brief HIP kernel source: peak_cfar (SNR-estimator, SNR_05)
 *
 * CA-CFAR detector на уже вычисленных |X|² (power spectrum от
 * `complex_to_magnitude_squared`).
 *
 * Thread mapping:
 *   - 1 BLOCK = 1 антенна (blockIdx.x = ant_id, blockDim.x = BLOCK_SIZE = 256)
 *   - Pass 1: parallel argmax через LDS reduction
 *   - Pass 2: parallel ref-window sum через atomicAdd
 *   - Pass 3: thread 0 пишет snr_db_out[ant] = 10·log10(peak² / noise_mean)
 *
 * ref window: [k_peak − (guard+ref)..k_peak − (guard+1)] ∪
 *             [k_peak + (guard+1)..k_peak + (guard+ref)]
 * c wraparound через `(k + offset + nFFT) % nFFT` — безопасно при малых k_peak.
 *
 * search_full_spectrum:
 *   НЕ параметр ядра — caller передаёт nFFT (поиск по всему) или nFFT/2
 *   (только положительные частоты).
 *
 * @note Source concatenates with statistics main source которая определяет
 *       BLOCK_SIZE=256. Используем явное __launch_bounds__(256).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

/**
 * @brief HIP kernel source: peak_cfar (CA-CFAR argmax + ref-sum + log10)
 */
inline const char* GetPeakCfarKernelSource() {
    return R"HIP(

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

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
