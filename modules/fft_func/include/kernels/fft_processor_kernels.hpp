#pragma once

/**
 * @file fft_processor_kernels.hpp
 * @brief OpenCL kernel sources для FFTProcessor
 *
 * Содержит:
 * - Pre-callback (padding n_point → nFFT) — переиспользуется из fft_maxima
 * - Post-processing kernel (complex → magnitude + phase + freq)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

namespace fft_processor {
namespace kernels {

/**
 * @brief Pre-callback для clFFT: padding n_point → nFFT с нулями
 *
 * clFFT вызывает этот callback ВМЕСТО чтения из входного буфера — позволяет
 * реализовать zero-padding внутри самого FFT-pass без отдельного GPU-прохода.
 *
 * Структура userdata (32 bytes header + данные):
 *   header[0]: beam_count
 *   header[1]: count_points (= n_point)
 *   header[2]: nFFT (целевой размер FFT)
 *   header[3..7]: padding до 32 байт
 *   header[8]+:  входные данные (beam_count × count_points × float2)
 *
 * Callback вызывается для каждого элемента FFT-входа (offset = абс. позиция в batch).
 * Если pos < count_points — возвращаем реальные данные, иначе {0,0} (zero-padding).
 */
inline const char* GetPreCallbackSource_opencl() {
    return R"CL(
float2 prepareDataPre(__global void* input,
                      uint offset,
                      __global void* userdata)
{
    __global uint* header = (__global uint*)userdata;
    uint beam_count    = header[0];
    uint count_points  = header[1];
    uint nFFT          = header[2];

    uint beam_id  = offset / nFFT;
    uint pos      = offset % nFFT;

    if (pos < count_points && beam_id < beam_count) {
        // header + 8: пропустить 8 × uint32_t = 32 байта заголовка (PRE_CALLBACK_HEADER_SIZE).
        // header — __global uint*, поэтому +8 сдвигает на 32 байта → начало реальных данных.
        __global float2* data = (__global float2*)(header + 8);
        return data[beam_id * count_points + pos];
    }
    return (float2)(0.0f, 0.0f);  // zero-padding: позиции за пределами n_point
}
)CL";
}

/**
 * @brief Post-processing kernel: complex → magnitude + phase
 *
 * Аргументы:
 *   fft_output     — выход FFT (complex interleaved, beam_count * nFFT)
 *   mag_output     — магнитуды (beam_count * nFFT)
 *   phase_output   — фазы в радианах (beam_count * nFFT)
 *   beam_count     — количество лучей
 *   nFFT           — размер FFT
 */
inline const char* GetMagPhaseKernelSource_opencl() {
    return R"CL(
// reqd_work_group_size for better compiler optimization (like __launch_bounds__)
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void complex_to_mag_phase(
    __global const float2* fft_output,
    __global float* mag_output,
    __global float* phase_output,
    const uint beam_count,
    const uint nFFT)
{
    uint gid = get_global_id(0);
    uint total = beam_count * nFFT;
    if (gid >= total) return;

    float2 z = fft_output[gid];
    mag_output[gid] = native_sqrt(z.x * z.x + z.y * z.y);
#ifdef AMD_GPU
    phase_output[gid] = native_atan2(z.y, z.x);  // AMD: hardware intrinsic
#else
    phase_output[gid] = atan2(z.y, z.x);          // NVIDIA/portable
#endif
}
)CL";
}

} // namespace kernels
} // namespace fft_processor
