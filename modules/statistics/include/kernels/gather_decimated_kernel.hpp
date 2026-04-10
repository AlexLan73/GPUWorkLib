#pragma once

/**
 * @file gather_decimated_kernel.hpp
 * @brief HIP kernel source: gather_decimated (SNR-estimator, SNR_03)
 *
 * Вырезает подвыборку из 2D complex float матрицы [n_antennas × n_samples]
 * с шагами step_antennas и step_samples.
 *
 * Thread mapping (КРИТИЧНО — одно из ключевых решений плана):
 *   - 1 поток = 1 выходная антенна (blockIdx.x * blockDim.x + threadIdx.x)
 *   - sequential loop по samples ВНУТРИ потока
 *
 * Почему НЕ «поток на элемент»:
 *   При step_samples > 8 (stride > 64 байт = cache line) соседние потоки
 *   варпа читают из разных cache line'ов → ×32 amplification memory txns.
 *   Sequential loop внутри потока → L2 prefetcher префетчит вдоль строки.
 *
 * @note Source concatenates with GetStatisticsKernelSource() — он определяет
 *       struct float2_t, поэтому здесь НЕ переопределяем её.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

/**
 * @brief HIP kernel source: gather_decimated
 *
 * Launch config (см. SnrEstimatorOp::ExecuteGather):
 *   grid(ceil(n_ant_out / 64), 1, 1)
 *   block(64, 1, 1)
 *
 * Функция `gather_decimated` (без суффикса `_kernel` — имя совпадает с
 * `kernel("gather_decimated")` в Op).
 *
 * @note NOT standalone — concatenates with statistics main source which
 *       already defines struct float2_t и BLOCK_SIZE. Мы используем явное
 *       `__launch_bounds__(64)` чтобы не зависеть от BLOCK_SIZE из main source.
 */
inline const char* GetGatherDecimatedKernelSource() {
    return R"HIP(

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

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
