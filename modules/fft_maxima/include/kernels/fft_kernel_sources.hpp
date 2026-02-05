#pragma once
// ════════════════════════════════════════════════════════════════════════════
// FFT Kernel Sources для AntennaFFTProcMax
// Автоматически генерируемые строки с OpenCL kernel'ами
// ════════════════════════════════════════════════════════════════════════════

namespace antenna_fft {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// Padding Kernel - подготовка данных: count_points → nFFT с padding нулями
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPaddingKernelSource() {
    return R"CL(
__kernel void padding_kernel(
    __global const float2* input,    // Входные данные: ПОЛНЫЙ буфер (все лучи)
    __global float2* output,         // Выходные данные: batch_beam_count * nFFT
    uint batch_beam_count,           // Количество лучей в батче
    uint count_points,               // Точек на луч
    uint nFFT,                       // Размер FFT
    uint beam_offset                 // Смещение в лучах (для batch processing)
) {
    uint gid = get_global_id(0);
    uint local_beam_idx = gid / nFFT;
    uint pos_in_fft = gid % nFFT;

    if (local_beam_idx >= batch_beam_count) return;

    uint global_beam_idx = local_beam_idx + beam_offset;

    if (pos_in_fft < count_points) {
        uint src_idx = global_beam_idx * count_points + pos_in_fft;
        output[gid] = input[src_idx];
    } else {
        output[gid] = (float2)(0.0f, 0.0f);
    }
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// Post Kernel (UNIFIED) - magnitude + поиск max + фаза + параболическая интерполяция
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPostKernelSource() {
    return R"CL(
// Структура результата (должна совпадать с C++ MaxValue)
typedef struct {
    uint index;
    float real;
    float imag;
    float magnitude;
    float phase;
    float freq_offset;
    float refined_frequency;
    uint pad;
} MaxValue;

__kernel void post_kernel(
    __global const float2* fft_output,     // FFT результат: beam_count * nFFT
    __global MaxValue* maxima_output,      // Результат: beam_count * max_peaks_count
    uint beam_count,
    uint nFFT,
    uint search_range,                     // Сколько точек анализировать
    uint max_peaks_count,                  // Сколько максимумов искать
    float sample_rate                      // Частота дискретизации (12 МГц)
) {
    uint beam_idx = get_group_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    if (beam_idx >= beam_count) return;

    // Local memory для редукции
    __local float local_mag[256];
    __local uint local_idx[256];
    __local float2 local_complex[256];
    __local float found_mags[16];
    __local uint found_indices[16];
    __local float2 found_complex[16];

    // ЭТАП 1: Каждый поток находит свой локальный максимум
    float my_max_mag = -1.0f;
    uint my_max_idx = 0;
    float2 my_max_complex = (float2)(0.0f, 0.0f);

    for (uint i = lid; i < search_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        float mag = sqrt(val.x * val.x + val.y * val.y);

        if (mag > my_max_mag) {
            my_max_mag = mag;
            my_max_idx = i;
            my_max_complex = val;
        }
    }

    local_mag[lid] = my_max_mag;
    local_idx[lid] = my_max_idx;
    local_complex[lid] = my_max_complex;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ЭТАП 2: Поток 0 ищет top-N максимумов
    if (lid == 0) {
        for (uint peak = 0; peak < max_peaks_count && peak < 16; ++peak) {
            float best_mag = -1.0f;
            uint best_idx = 0;
            float2 best_complex = (float2)(0.0f, 0.0f);
            uint best_local_idx = 0;

            for (uint j = 0; j < local_size; ++j) {
                if (local_mag[j] > best_mag) {
                    best_mag = local_mag[j];
                    best_idx = local_idx[j];
                    best_complex = local_complex[j];
                    best_local_idx = j;
                }
            }

            if (best_mag > 0.0f) {
                found_mags[peak] = best_mag;
                found_indices[peak] = best_idx;
                found_complex[peak] = best_complex;
                local_mag[best_local_idx] = -1.0f;
            } else {
                found_mags[peak] = 0.0f;
                found_indices[peak] = 0;
                found_complex[peak] = (float2)(0.0f, 0.0f);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // ЭТАП 3: Записать результаты с Re/Im и параболической интерполяцией
        // ═══════════════════════════════════════════════════════════════

        float bin_width = sample_rate / (float)nFFT;

        for (uint peak = 0; peak < max_peaks_count && peak < 16; ++peak) {
            uint out_idx = beam_idx * max_peaks_count + peak;

            MaxValue mv;
            mv.index = found_indices[peak];

            float2 c = found_complex[peak];
            mv.real = c.x;
            mv.imag = c.y;
            mv.magnitude = found_mags[peak];

            if (found_mags[peak] > 0.0f) {
                float phase_rad = atan2(c.y, c.x);
                mv.phase = phase_rad * 57.29577951f;
            } else {
                mv.phase = 0.0f;
            }

            mv.freq_offset = 0.0f;
            mv.refined_frequency = (float)mv.index * bin_width;

            // Параболическая интерполяция ТОЛЬКО для peak == 0
            if (peak == 0 && found_mags[0] > 0.0f) {
                uint center_idx = found_indices[0];

                if (center_idx > 0 && center_idx < search_range - 1) {
                    uint base_idx = beam_idx * nFFT;

                    float2 left_val = fft_output[base_idx + center_idx - 1];
                    float2 right_val = fft_output[base_idx + center_idx + 1];

                    float y_left = sqrt(left_val.x * left_val.x + left_val.y * left_val.y);
                    float y_center = found_mags[0];
                    float y_right = sqrt(right_val.x * right_val.x + right_val.y * right_val.y);

                    float denom = y_left - 2.0f * y_center + y_right;

                    if (fabs(denom) > 1e-10f) {
                        float offset = 0.5f * (y_left - y_right) / denom;
                        offset = clamp(offset, -0.5f, 0.5f);

                        mv.freq_offset = offset;
                        float refined_index = (float)center_idx + offset;
                        mv.refined_frequency = refined_index * bin_width;
                    }
                }
            }

            mv.pad = 0;
            maxima_output[out_idx] = mv;
        }
    }
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// Debug Post Kernel - fftshift + magnitude (без поиска максимумов)
// Используется в AntennaFFTDebug для пошаговой отладки
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetDebugPostKernelSource() {
    return R"CL(
__kernel void debug_post_kernel(
    __global const float2* fft_output,       // FFT результат: beam_count * nFFT
    __global float2* selected_complex,       // Выход: beam_count * out_count_points_fft
    __global float* selected_magnitude,      // Выход: beam_count * out_count_points_fft
    uint beam_count,
    uint nFFT,
    uint out_count_points_fft
) {
    uint gid = get_global_id(0);
    uint beam_idx = gid / out_count_points_fft;
    uint out_idx = gid % out_count_points_fft;

    if (beam_idx >= beam_count) return;

    // fftshift: переупорядочиваем спектр
    // Выходной диапазон [0, out_count_points_fft) должен содержать:
    // - Первая половина: отрицательные частоты [nFFT - half, nFFT)
    // - Вторая половина: положительные частоты [0, half)
    uint half_size = out_count_points_fft / 2;

    uint fft_idx;
    if (out_idx < half_size) {
        // Отрицательные частоты: из конца FFT буфера
        fft_idx = nFFT - half_size + out_idx;
    } else {
        // Положительные частоты: из начала FFT буфера
        fft_idx = out_idx - half_size;
    }

    uint src_idx = beam_idx * nFFT + fft_idx;
    uint dst_idx = beam_idx * out_count_points_fft + out_idx;

    float2 val = fft_output[src_idx];
    selected_complex[dst_idx] = val;
    selected_magnitude[dst_idx] = sqrt(val.x * val.x + val.y * val.y);
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// Pre-Callback Source для clFFT (16-байт структура)
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPreCallbackSource() {
    return R"(
typedef struct {
    uint beam_count;
    uint count_points;
    uint nFFT;
    uint padding;
} PreCallbackUserData;

float2 prepareDataPre(__global void* input, uint inoffset, __global void* userdata) {
    __global PreCallbackUserData* params = (__global PreCallbackUserData*)userdata;
    __global float2* input_signal = (__global float2*)((__global char*)userdata + sizeof(PreCallbackUserData));

    uint beam_count = params->beam_count;
    uint count_points = params->count_points;
    uint nFFT = params->nFFT;

    // Вычислить индекс луча и позицию в блоке nFFT
    uint beam_idx = inoffset / nFFT;
    uint pos_in_fft = inoffset % nFFT;

    if (beam_idx >= beam_count) {
        return (float2)(0.0f, 0.0f);
    }

    // Если позиция в пределах count_points - копируем данные
    if (pos_in_fft < count_points) {
        uint input_idx = beam_idx * count_points + pos_in_fft;
        return input_signal[input_idx];
    } else {
        // Остальное - padding (нули)
        return (float2)(0.0f, 0.0f);
    }
}
)";
}

// ════════════════════════════════════════════════════════════════════════════
// Pre-Callback Source для clFFT (32-байт структура, для CreateFFTPlanWithPreCallbackOnly)
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPreCallbackSource32() {
    return
        "typedef struct { "
        "    uint beam_count; "
        "    uint count_points; "
        "    uint nFFT; "
        "    uint padding1; "
        "    uint padding2; "
        "    uint padding3; "
        "    uint padding4; "
        "    uint padding5; "
        "} PreCallbackUserData; "
        "float2 prepareDataPre(__global void* input, uint inoffset, __global void* userdata) { "
        "    __global PreCallbackUserData* params = (__global PreCallbackUserData*)userdata; "
        "    __global float2* input_signal = (__global float2*)((__global char*)userdata + 32); "
        "    uint beam_count = params->beam_count; "
        "    uint count_points = params->count_points; "
        "    uint nFFT = params->nFFT; "
        "    uint beam_idx = inoffset / nFFT; "
        "    uint pos_in_fft = inoffset % nFFT; "
        "    if (beam_idx >= beam_count) { "
        "        return (float2)(0.0f, 0.0f); "
        "    } "
        "    if (pos_in_fft < count_points) { "
        "        uint input_idx = beam_idx * count_points + pos_in_fft; "
        "        return input_signal[input_idx]; "
        "    } else { "
        "        return (float2)(0.0f, 0.0f); "
        "    } "
        "}";
}

// ════════════════════════════════════════════════════════════════════════════
// Post-Callback Source для clFFT (fftshift + magnitude + complex write)
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPostCallbackSource() {
    return R"(
typedef struct {
    uint beam_count;
    uint nFFT;
    uint out_count_points_fft;
    uint max_peaks_count;
} PostCallbackUserData;

void processFFTPost(__global void* output, uint outoffset, __global void* userdata, float2 fftoutput) {
    __global PostCallbackUserData* params = (__global PostCallbackUserData*)userdata;

    uint beam_count = params->beam_count;
    uint nFFT = params->nFFT;
    uint out_count_points_fft = params->out_count_points_fft;

    // Вычислить индекс луча и позицию в FFT
    uint beam_idx = outoffset / nFFT;
    uint pos_in_fft = outoffset % nFFT;

    if (beam_idx >= beam_count) {
        return;
    }

    // Диапазоны для fftshift:
    // Диапазон 1 (отрицательные частоты): [nFFT - out_count_points_fft/2, nFFT - 1]
    // Диапазон 2 (положительные частоты): [0, out_count_points_fft/2 - 1]
    uint half_size = out_count_points_fft / 2;
    uint range1_start = nFFT - half_size;

    // Быстрая проверка - 99.9% потоков выходят здесь!
    bool in_range1 = (pos_in_fft >= range1_start);
    bool in_range2 = (pos_in_fft < half_size);

    if (!in_range1 && !in_range2) {
        return;  // Не в диапазоне fftshift - выходим быстро
    }

    // Вычислить индекс в выходном буфере (после fftshift)
    uint output_idx;
    if (in_range1) {
        // Отрицательные частоты → начало выходного буфера
        output_idx = pos_in_fft - range1_start;
    } else {
        // Положительные частоты → после отрицательных
        output_idx = half_size + pos_in_fft;
    }

    // Layout userdata: params | complex_buffer | magnitude_buffer
    __global float2* complex_buffer = (__global float2*)((__global char*)userdata + sizeof(PostCallbackUserData));
    __global float* magnitude_buffer = (__global float*)(complex_buffer + (beam_count * out_count_points_fft));

    uint base_idx = beam_idx * out_count_points_fft + output_idx;

    // Записать комплексный спектр (только для fftshift диапазона)
    complex_buffer[base_idx] = fftoutput;

    // Записать magnitude (без atomic - просто прямая запись)
    magnitude_buffer[base_idx] = length(fftoutput);
}
)";
}

} // namespace kernels
} // namespace antenna_fft
