/**
 * @file antenna_fft.cl
 * @brief OpenCL kernels для Antenna FFT Module
 * 
 * Содержит:
 * 1. padding_kernel - копирование данных с zero padding
 * 2. post_kernel - выбор диапазона + вычисление magnitude/phase
 * 3. reduction_kernel - поиск топ-N максимумов
 */

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: Padding Kernel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Копирование данных count_points → nFFT с zero padding
 * 
 * @param input_signal Входной сигнал [beam_count][count_points] (с offset)
 * @param output_padded Выходной буфер [num_beams][nFFT] (zero-padded)
 * @param beam_offset Смещение луча в input_signal (для batch processing)
 * @param count_points Количество входных точек на луч
 * @param nFFT Размер FFT (больше count_points)
 * @param num_beams Количество лучей в батче
 * 
 * Global work size: [nFFT, num_beams]
 * 
 * Пример:
 * - beam_offset = 0 → полная обработка (все лучи)
 * - beam_offset = 10 → батч начинается с 10-го луча
 */
__kernel void padding_kernel(
    __global const float2* input_signal,   // Входные данные (complex float)
    __global float2* output_padded,        // Выходные данные (zero-padded)
    const uint beam_offset,                // Смещение луча (для batch)
    const uint count_points,               // Входных точек на луч
    const uint nFFT,                       // Размер FFT
    const uint num_beams)                  // Количество лучей в батче
{
    const uint fft_idx = get_global_id(0);  // Индекс в FFT (0..nFFT-1)
    const uint beam_local = get_global_id(1); // Индекс луча в батче (0..num_beams-1)
    
    if (fft_idx >= nFFT || beam_local >= num_beams) return;
    
    // Индекс в выходном буфере
    const uint out_idx = beam_local * nFFT + fft_idx;
    
    if (fft_idx < count_points) {
        // Копировать данные из input
        // beam_global = beam_offset + beam_local
        const uint beam_global = beam_offset + beam_local;
        const uint in_idx = beam_global * count_points + fft_idx;
        output_padded[out_idx] = input_signal[in_idx];
    } else {
        // Zero padding
        output_padded[out_idx] = (float2)(0.0f, 0.0f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: Post Kernel
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Post-processing: fftshift + выбор диапазона + magnitude/phase
 * 
 * После FFT нужно:
 * 1. Выбрать центральный диапазон [nFFT/2 - out_count/2 .. nFFT/2 + out_count/2)
 * 2. Вычислить magnitude = sqrt(re^2 + im^2)
 * 3. Вычислить phase = atan2(im, re)
 * 
 * @param fft_output FFT результат [num_beams][nFFT]
 * @param selected_complex Выбранные комплексные значения [num_beams][out_count]
 * @param selected_magnitude Magnitude для выбранных точек [num_beams][out_count]
 * @param out_count Количество выходных точек (для анализа)
 * @param nFFT Размер FFT
 * @param num_beams Количество лучей
 * 
 * Global work size: [out_count, num_beams]
 */
__kernel void post_kernel(
    __global const float2* fft_output,
    __global float2* selected_complex,
    __global float* selected_magnitude,
    const uint out_count,
    const uint nFFT,
    const uint num_beams)
{
    const uint out_idx = get_global_id(0);  // Индекс в выходном диапазоне
    const uint beam_idx = get_global_id(1); // Индекс луча
    
    if (out_idx >= out_count || beam_idx >= num_beams) return;
    
    // Вычислить индекс в FFT (центрированный диапазон)
    const uint center = nFFT / 2;
    const uint half_out = out_count / 2;
    const uint fft_idx = center - half_out + out_idx;
    
    // Читать из FFT
    const uint fft_offset = beam_idx * nFFT + fft_idx;
    const float2 complex_val = fft_output[fft_offset];
    
    // Записать комплексное значение
    const uint out_offset = beam_idx * out_count + out_idx;
    selected_complex[out_offset] = complex_val;
    
    // Вычислить magnitude
    const float re = complex_val.x;
    const float im = complex_val.y;
    const float magnitude = sqrt(re * re + im * im);
    selected_magnitude[out_offset] = magnitude;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 3: Reduction Kernel (поиск топ-N максимумов)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Найти топ-N максимумов в magnitude для каждого луча
 * 
 * Алгоритм:
 * 1. Для каждого луча сканировать magnitude массив
 * 2. Найти N максимальных значений
 * 3. Записать результаты: index, amplitude, phase, real, imag
 * 
 * @param selected_complex Комплексные значения [num_beams][out_count]
 * @param selected_magnitude Magnitude значения [num_beams][out_count]
 * @param maxima_output Выходной буфер [num_beams][max_peaks]
 *                      Структура: [index, amplitude, phase, real, imag, pad, pad, pad]
 * @param out_count Количество точек для поиска
 * @param num_beams Количество лучей
 * @param max_peaks Количество пиков для поиска
 * 
 * Global work size: [num_beams]
 * 
 * TODO: Оптимизировать через shared memory / parallel reduction
 */
__kernel void reduction_kernel(
    __global const float2* selected_complex,
    __global const float* selected_magnitude,
    __global float* maxima_output,
    const uint out_count,
    const uint num_beams,
    const uint max_peaks)
{
    const uint beam_idx = get_global_id(0);
    if (beam_idx >= num_beams) return;
    
    // Offset для текущего луча
    const uint beam_offset = beam_idx * out_count;
    const uint result_offset = beam_idx * max_peaks * 8; // 8 floats per peak
    
    // Временный массив для хранения топ-N максимумов
    // Структура: [index, magnitude]
    float top_indices[16];  // max 16 peaks (можно увеличить)
    float top_magnitudes[16];
    
    // Инициализация
    for (uint i = 0; i < max_peaks && i < 16; ++i) {
        top_indices[i] = -1.0f;
        top_magnitudes[i] = -1.0f;
    }
    
    // Поиск топ-N максимумов
    for (uint pt = 0; pt < out_count; ++pt) {
        const float mag = selected_magnitude[beam_offset + pt];
        
        // Проверить, попадает ли в топ-N
        for (uint i = 0; i < max_peaks && i < 16; ++i) {
            if (mag > top_magnitudes[i]) {
                // Сдвинуть все элементы вниз
                for (uint j = max_peaks - 1; j > i; --j) {
                    if (j < 16) {
                        top_magnitudes[j] = top_magnitudes[j - 1];
                        top_indices[j] = top_indices[j - 1];
                    }
                }
                // Вставить новый элемент
                top_magnitudes[i] = mag;
                top_indices[i] = (float)pt;
                break;
            }
        }
    }
    
    // Записать результаты
    for (uint i = 0; i < max_peaks && i < 16; ++i) {
        const uint idx = (uint)top_indices[i];
        
        if (idx < out_count) {
            const float2 complex_val = selected_complex[beam_offset + idx];
            const float re = complex_val.x;
            const float im = complex_val.y;
            const float amplitude = top_magnitudes[i];
            const float phase = atan2(im, re);
            
            // Записать: [index, amplitude, phase, real, imag, 0, 0, 0]
            const uint out_idx = result_offset + i * 8;
            maxima_output[out_idx + 0] = (float)idx;
            maxima_output[out_idx + 1] = amplitude;
            maxima_output[out_idx + 2] = phase;
            maxima_output[out_idx + 3] = re;
            maxima_output[out_idx + 4] = im;
            maxima_output[out_idx + 5] = 0.0f; // padding
            maxima_output[out_idx + 6] = 0.0f; // padding
            maxima_output[out_idx + 7] = 0.0f; // padding
        } else {
            // Нет данных - записать нули
            const uint out_idx = result_offset + i * 8;
            for (uint j = 0; j < 8; ++j) {
                maxima_output[out_idx + j] = 0.0f;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// END OF FILE
// ═══════════════════════════════════════════════════════════════════════════
