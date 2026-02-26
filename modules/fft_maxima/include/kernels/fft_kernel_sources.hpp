#pragma once
// ════════════════════════════════════════════════════════════════════════════
// FFT Kernel Sources для AntennaFFTProcMax
// Автоматически генерируемые строки с OpenCL kernel'ами
// ════════════════════════════════════════════════════════════════════════════

namespace antenna_fft {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// GetPaddingKernelSource() — отдельное ядро для пакетной обработки
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Подготовка данных для FFT: копирование count_points → nFFT с дополнением нулями
//   Используется когда нужно обработать БОЛЬШИЕ данные по частям (batch processing)
//
// АРХИТЕКТУРА:
//   - Тип: отдельное ядро OpenCL (вызов через clEnqueueNDRangeKernel)
//   - Буферы: input и output — отдельные объекты cl_mem
//   - Аргументы: 6 параметров через clSetKernelArg()
//
// ОПТИМИЗАЦИИ:
//   - TASK-4 (P1-B): 2D NDRange устраняет div/mod (get_global_id вместо gid/nFFT, gid%nFFT)
//   - TASK-8 (P2-D): __restrict на все pointer params → агрессивное ILP
//   - Нули: output должен быть заполнен clEnqueueFillBuffer(0) ПЕРЕД запуском
//             → нет else-branch → нет divergent execution
//
// DISPATCH (2D NDRange):
//   size_t global[2] = { nFFT, batch_beam_count };
//   size_t local[2]  = { 256, 1 };
//   // Обнулить output заранее:
//   float2 zero = {0.0f, 0.0f};
//   clEnqueueFillBuffer(queue, output, &zero, sizeof(float2), 0,
//       batch_beam_count * nFFT * sizeof(float2), 0, nullptr, nullptr);
//   clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, ...);
//
// КЛЮЧЕВАЯ ФИЧА — beam_offset:
//   Batch 0: offset=0,  лучи 0-9
//   Batch 1: offset=10, лучи 10-19
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPaddingKernelSource_opencl() {
    return R"CL(
__kernel void padding_kernel(
    __global const float2* restrict input,    // Входные данные: ПОЛНЫЙ буфер (все лучи)
    __global float2* restrict output,         // Выходные данные: batch_beam_count * nFFT
    uint batch_beam_count,                    // Количество лучей в батче
    uint count_points,                        // Точек на луч
    uint nFFT,                                // Размер FFT
    uint beam_offset                          // Смещение в лучах (для batch processing)
) {
    // TASK-4: 2D NDRange — нет div/mod
    uint pos_in_fft    = get_global_id(0);    // [0, nFFT)
    uint local_beam_idx = get_global_id(1);   // [0, batch_beam_count)

    if (local_beam_idx >= batch_beam_count) return;

    uint global_beam_idx = local_beam_idx + beam_offset;
    uint out_idx = local_beam_idx * nFFT + pos_in_fft;

    // Нули уже записаны через clEnqueueFillBuffer — только копируем данные
    if (pos_in_fft < count_points) {
        uint src_idx = global_beam_idx * count_points + pos_in_fft;
        output[out_idx] = input[src_idx];
    }
    // Иначе — 0.0 уже в буфере, else-branch не нужен
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// GetPostKernelSource_TwoPeaks_opencl() - Post Kernel с поиском ДВУХ максимумов
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Post-processing kernel: поиск ДВУХ независимых максимумов в краевых диапазонах
//   спектра (левый и правый) + параболическая интерполяция + вывод 3х соседних точек
//
// АЛГОРИТМ:
//   1. Делим search_range пополам → half_range
//   2. Ищем максимум ОТДЕЛЬНО в левом [0, half_range] и правом [nFFT-half_range, nFFT-1]
//   3. Для каждого пика: 3 точки [max_idx-1, max_idx, max_idx+1] + парабола
//   4. Выводим 8 структур MaxValue на каждый луч:
//      - левый диапазон;
//      [0] - результат параболической интерполяции (с freq_offset, refined_frequency)
//      [1] - левая точка (index-1)
//      [2] - центральная точка (главный максимум)
//      [3] - правая точка (index+1)
//      - правый диапазон;
//      [4] - результат параболической интерполяции (с freq_offset, refined_frequency)
//      [5] - левая точка (index-1)
//      [6] - центральная точка (главный максимум)
//      [7] - правая точка (index+1)
//
// ОПТИМИЗАЦИИ:
//   - TASK-1 (P0):   Параллельная tree-reduction 256→1 (O(log₂256)=8 шагов) вместо
//                    последовательного O(256) цикла в thread-0
//   - TASK-2 (P1-C): native_sqrt() вместо sqrt() — 2-4× быстрее, достаточная точность
//   - TASK-5 (P2-A): LDS +1 padding → меньше bank conflicts при reduction
//   - TASK-7 (P2-C): reqd_work_group_size(256,1,1) → компилятор устраняет dead-branches
//   - TASK-8 (P2-D): __restrict → агрессивное ILP
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPostKernelSource_TwoPeaks_opencl() {
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

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void post_kernel(
    __global const float2* restrict fft_output,     // FFT результат: beam_count * nFFT
    __global MaxValue* restrict maxima_output,      // Результат: beam_count * 8 структуры
    uint beam_count,
    uint nFFT,
    uint search_range,                     // Ширина диапазона (делим пополам)
    float sample_rate                      // Частота дискретизации (Гц)
) {
    uint beam_idx = get_group_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    if (beam_idx >= beam_count) return;

    // ═══════════════════════════════════════════════════════════════════════
    // ШАГ 1: Вычисляем half_range (половина search_range)
    // ═══════════════════════════════════════════════════════════════════════
    uint half_range = search_range / 2;

    // ═══════════════════════════════════════════════════════════════════════
    // ШАГ 2: Параллельный поиск максимума в ДВУХ диапазонах
    // Диапазон 1: [0, half_range] - положительные частоты
    // Диапазон 2: [nFFT - half_range, nFFT - 1] - отрицательные частоты
    // ═══════════════════════════════════════════════════════════════════════

    // TASK-5: +1 padding устраняет bank conflicts при reduction
    __local float local_left_mag[257];
    __local uint  local_left_idx[257];
    __local float local_right_mag[257];
    __local uint  local_right_idx[257];

    float my_left_mag = -1.0f;
    uint my_left_idx = 0;
    uint range2_start = nFFT - half_range;
    uint my_right_idx = range2_start;

    // Поиск в диапазоне 1: [0, half_range] — левый
    for (uint i = lid; i < half_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        // TASK-2: native_sqrt — 2-4× быстрее
        float mag = native_sqrt(val.x * val.x + val.y * val.y);
        if (mag > my_left_mag) {
            my_left_mag = mag;
            my_left_idx = i;
        }
    }

    // Поиск в диапазоне 2: [nFFT - half_range, nFFT - 1] — правый
    float my_right_mag = -1.0f;
    for (uint i = range2_start + lid; i < nFFT; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        // TASK-2: native_sqrt
        float mag = native_sqrt(val.x * val.x + val.y * val.y);
        if (mag > my_right_mag) {
            my_right_mag = mag;
            my_right_idx = i;
        }
    }

    local_left_mag[lid] = my_left_mag;
    local_left_idx[lid] = my_left_idx;
    local_right_mag[lid] = my_right_mag;
    local_right_idx[lid] = my_right_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ═══════════════════════════════════════════════════════════════════════
    // ШАГ 3: TASK-1 — Параллельная tree-reduction 256→128→64→...→1
    //        O(log₂(256) = 8 шагов) вместо O(256) последовательного цикла
    // ═══════════════════════════════════════════════════════════════════════
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_left_mag[lid + stride] > local_left_mag[lid]) {
                local_left_mag[lid] = local_left_mag[lid + stride];
                local_left_idx[lid] = local_left_idx[lid + stride];
            }
            if (local_right_mag[lid + stride] > local_right_mag[lid]) {
                local_right_mag[lid] = local_right_mag[lid + stride];
                local_right_idx[lid] = local_right_idx[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // После reduction: local_left_mag[0]/local_left_idx[0] = глобальный максимум левого
    //                  local_right_mag[0]/local_right_idx[0] = глобального максимума правого

    // ═══════════════════════════════════════════════════════════════════════
    // ШАГ 4: Поток 0 — вывод 8 MaxValue
    // ═══════════════════════════════════════════════════════════════════════
    if (lid == 0) {
        uint global_left_idx  = local_left_idx[0];
        uint global_right_idx = local_right_idx[0];

        uint base_fft_idx = beam_idx * nFFT;
        float bin_width = sample_rate / (float)nFFT;
        uint out_base = beam_idx * 8;

        // Макрос: mirror_freq=true для правого диапазона — частота = sample_rate - raw (зеркало 2.75 вместо 997.25)
        #define WRITE_FOUR(base_offset, center_idx, mirror_freq) do { \
            float2 cv = fft_output[base_fft_idx + center_idx]; \
            float y_c = native_sqrt(cv.x * cv.x + cv.y * cv.y); \
            float2 lv = (float2)(0.0f, 0.0f); \
            float2 rv = (float2)(0.0f, 0.0f); \
            float y_l = 0.0f, y_r = 0.0f; \
            bool hl = (center_idx > 0); \
            bool hr = (center_idx < nFFT - 1); \
            if (hl) { lv = fft_output[base_fft_idx + center_idx - 1]; y_l = native_sqrt(lv.x*lv.x + lv.y*lv.y); } \
            if (hr) { rv = fft_output[base_fft_idx + center_idx + 1]; y_r = native_sqrt(rv.x*rv.x + rv.y*rv.y); } \
            float fo = 0.0f; \
            float rf = (float)center_idx * bin_width; \
            if (hl && hr) { \
                float denom = y_l - 2.0f*y_c + y_r; \
                if (fabs(denom) > 1e-10f) { fo = clamp(0.5f*(y_l-y_r)/denom, -0.5f, 0.5f); rf = ((float)center_idx + fo) * bin_width; } \
            } \
            if (mirror_freq) { rf = sample_rate - rf; } \
            maxima_output[out_base + base_offset + 0].index = center_idx; \
            maxima_output[out_base + base_offset + 0].real = cv.x; \
            maxima_output[out_base + base_offset + 0].imag = cv.y; \
            maxima_output[out_base + base_offset + 0].magnitude = y_c; \
            maxima_output[out_base + base_offset + 0].phase = atan2(cv.y, cv.x) * 57.29577951f; \
            maxima_output[out_base + base_offset + 0].freq_offset = fo; \
            maxima_output[out_base + base_offset + 0].refined_frequency = rf; \
            maxima_output[out_base + base_offset + 0].pad = 0; \
            float rf_l = hl ? (float)(center_idx-1)*bin_width : 0.0f; \
            if (mirror_freq && hl) rf_l = sample_rate - rf_l; \
            maxima_output[out_base + base_offset + 1].index = hl ? center_idx-1 : 0; \
            maxima_output[out_base + base_offset + 1].real = lv.x; \
            maxima_output[out_base + base_offset + 1].imag = lv.y; \
            maxima_output[out_base + base_offset + 1].magnitude = y_l; \
            maxima_output[out_base + base_offset + 1].phase = hl ? atan2(lv.y, lv.x)*57.29577951f : 0.0f; \
            maxima_output[out_base + base_offset + 1].freq_offset = 0.0f; \
            maxima_output[out_base + base_offset + 1].refined_frequency = rf_l; \
            maxima_output[out_base + base_offset + 1].pad = 0; \
            float rf_c = (float)center_idx * bin_width; \
            if (mirror_freq) rf_c = sample_rate - rf_c; \
            maxima_output[out_base + base_offset + 2].index = center_idx; \
            maxima_output[out_base + base_offset + 2].real = cv.x; \
            maxima_output[out_base + base_offset + 2].imag = cv.y; \
            maxima_output[out_base + base_offset + 2].magnitude = y_c; \
            maxima_output[out_base + base_offset + 2].phase = atan2(cv.y, cv.x) * 57.29577951f; \
            maxima_output[out_base + base_offset + 2].freq_offset = 0.0f; \
            maxima_output[out_base + base_offset + 2].refined_frequency = rf_c; \
            maxima_output[out_base + base_offset + 2].pad = 0; \
            float rf_r = hr ? (float)(center_idx+1)*bin_width : 0.0f; \
            if (mirror_freq && hr) rf_r = sample_rate - rf_r; \
            maxima_output[out_base + base_offset + 3].index = hr ? center_idx+1 : 0; \
            maxima_output[out_base + base_offset + 3].real = rv.x; \
            maxima_output[out_base + base_offset + 3].imag = rv.y; \
            maxima_output[out_base + base_offset + 3].magnitude = y_r; \
            maxima_output[out_base + base_offset + 3].phase = hr ? atan2(rv.y, rv.x)*57.29577951f : 0.0f; \
            maxima_output[out_base + base_offset + 3].freq_offset = 0.0f; \
            maxima_output[out_base + base_offset + 3].refined_frequency = rf_r; \
            maxima_output[out_base + base_offset + 3].pad = 0; \
        } while(0)

        // Левый диапазон [0..3] — положительные частоты
        WRITE_FOUR(0, global_left_idx, false);
        // Правый диапазон [4..7] — зеркало: sample_rate - raw → 2.75 вместо 997.25
        WRITE_FOUR(4, global_right_idx, true);

        #undef WRITE_FOUR
    }
}
)CL";
}



// ════════════════════════════════════════════════════════════════════════════
// GetPreCallbackSource32() - clFFT Pre-Callback (PRODUCTION)
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Pre-callback для clFFT: автоматическая подготовка данных ДО каждого FFT элемента
//   Выполняет padding: count_points → nFFT с заполнением нулями
//
// АРХИТЕКТУРА:
//   - Тип: clFFT callback функция (вызывается АВТОМАТИЧЕСКИ clFFT)
//   - Вызов: clFFT вызывает prepareDataPre() для каждого элемента ПЕРЕД FFT
//   - Возврат: float2 → clFFT использует это значение как вход для FFT
//
// MEMORY LAYOUT:
//   userdata = [32 байта структуры PreCallbackUserData][данные лучей]
//              ↑                                      ↑
//              Параметры (beam_count, nFFT...)       input_signal
//
// СТРУКТУРА (32 байта):
//   - beam_count, count_points, nFFT (используются)
//   - nFFT_log2 (TASK-3: для bitwise div/mod)
//   - padding2..padding5 (для выравнивания 32 байта = 256 бит)
//
// ОПТИМИЗАЦИИ:
//   - TASK-3 (P1-A): nFFT гарантированно pow2 → bitwise вместо div/mod:
//       beam_idx    = inoffset >> nFFT_log2      (вместо inoffset / nFFT)
//       pos_in_fft  = inoffset & (nFFT - 1)     (вместо inoffset % nFFT)
//
// ОГРАНИЧЕНИЕ - НЕТ beam_offset:
//   ⚠️ Callback ВСЕГДА читает с луча 0!
//   Данные должны быть упакованы в userdata ПОДРЯД с начала
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPreCallbackSource32_opencl() {
    return
        "typedef struct { "
        "    uint beam_count; "
        "    uint count_points; "
        "    uint nFFT; "
        "    uint nFFT_log2; "      // TASK-3: log2(nFFT) для bitwise ops
        "    uint padding2; "
        "    uint padding3; "
        "    uint padding4; "
        "    uint padding5; "
        "} PreCallbackUserData; "
        "float2 prepareDataPre(__global void* input, uint inoffset, __global void* userdata) { "
        "    __global PreCallbackUserData* params = (__global PreCallbackUserData*)userdata; "
        "    __global float2* input_signal = (__global float2*)((__global char*)userdata + 32); "
        "    uint beam_count   = params->beam_count; "
        "    uint count_points = params->count_points; "
        "    uint nFFT         = params->nFFT; "
        "    uint nFFT_log2    = params->nFFT_log2; "  // TASK-3
        "    uint beam_idx    = inoffset >> nFFT_log2; "       // TASK-3: было inoffset / nFFT
        "    uint pos_in_fft  = inoffset & (nFFT - 1); "      // TASK-3: было inoffset % nFFT
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
// GetPostKernelSource_OnePeaks() - Post Kernel с поиском максимума в краевых диапазонах
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Post-processing kernel: поиск ОДНОГО максимума в краевых диапазонах
//   спектра (левый и правый) + параболическая интерполяция + вывод 3х соседних точек
//
// АЛГОРИТМ:
//   1. Делим search_range пополам → half_range
//   2. Ищем максимум ОТДЕЛЬНО в левом [0, half_range] и правом [nFFT-half_range, nFFT-1]
//   3. Выбираем максимальный получаем индекс max_idx
//   4. Для пика: 3 точки [max_idx-1, max_idx, max_idx+1] + парабола
//   5. Выводим 4 структур MaxValue на каждый луч:
//      [0] - результат параболической интерполяции (с freq_offset, refined_frequency)
//      [1] - левая точка (max_idx-1)
//      [2] - центральная точка (главный максимум)
//      [3] - правая точка (max_idx+1)
//
// ОПТИМИЗАЦИИ:
//   - TASK-1 (P0):   Параллельная tree-reduction 256→1 (O(log₂256)=8 шагов)
//   - TASK-2 (P1-C): native_sqrt() вместо sqrt()
//   - TASK-5 (P2-A): LDS +1 padding
//   - TASK-7 (P2-C): reqd_work_group_size(256,1,1)
//   - TASK-8 (P2-D): __restrict на все pointer params
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPostKernelSource_OnePeak_opencl() {
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

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void post_kernel_one_peak(
    __global const float2* restrict fft_output,     // FFT результат: beam_count * nFFT
    __global MaxValue* restrict maxima_output,      // Результат: beam_count * 4 структуры
    uint beam_count,
    uint nFFT,
    uint search_range,                     // Ширина диапазона (делим пополам)
    float sample_rate                      // Частота дискретизации (Гц)
) {
    uint beam_idx = get_group_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    if (beam_idx >= beam_count) return;

    // ШАГ 1: Вычисляем half_range
    uint half_range = search_range / 2;

    // TASK-5: +1 padding устраняет bank conflicts при reduction
    __local float local_left_mag[257];
    __local uint  local_left_idx[257];
    __local float local_right_mag[257];
    __local uint  local_right_idx[257];

    float my_left_mag = -1.0f;
    uint my_left_idx = 0;
    uint range2_start = nFFT - half_range;
    uint my_right_idx = range2_start;

    // Поиск в левом диапазоне [0, half_range]
    for (uint i = lid; i < half_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        // TASK-2: native_sqrt
        float mag = native_sqrt(val.x * val.x + val.y * val.y);
        if (mag > my_left_mag) {
            my_left_mag = mag;
            my_left_idx = i;
        }
    }

    // Поиск в правом диапазоне [nFFT - half_range, nFFT - 1]
    float my_right_mag = -1.0f;
    for (uint i = range2_start + lid; i < nFFT; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        // TASK-2: native_sqrt
        float mag = native_sqrt(val.x * val.x + val.y * val.y);
        if (mag > my_right_mag) {
            my_right_mag = mag;
            my_right_idx = i;
        }
    }

    local_left_mag[lid] = my_left_mag;
    local_left_idx[lid] = my_left_idx;
    local_right_mag[lid] = my_right_mag;
    local_right_idx[lid] = my_right_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // TASK-1: Параллельная tree-reduction 256→128→64→...→1
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_left_mag[lid + stride] > local_left_mag[lid]) {
                local_left_mag[lid] = local_left_mag[lid + stride];
                local_left_idx[lid] = local_left_idx[lid + stride];
            }
            if (local_right_mag[lid + stride] > local_right_mag[lid]) {
                local_right_mag[lid] = local_right_mag[lid + stride];
                local_right_idx[lid] = local_right_idx[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // После reduction: local_*[0] содержит глобальный максимум

    // ШАГ 3: Редукция и выбор БОЛЬШЕГО максимума
    if (lid == 0) {
        uint global_left_idx  = local_left_idx[0];
        float global_left_mag = local_left_mag[0];
        uint global_right_idx = local_right_idx[0];
        float global_right_mag = local_right_mag[0];

        // АЛГОРИТМ ALEX: Выбираем БОЛЬШИЙ из двух максимумов
        uint center_idx;
        bool is_right_side;
        if (global_right_mag > global_left_mag) {
            center_idx = global_right_idx;
            is_right_side = true;
        } else {
            center_idx = global_left_idx;
            is_right_side = false;
        }

        uint base_fft_idx = beam_idx * nFFT;
        float bin_width = sample_rate / (float)nFFT;
        uint out_base = beam_idx * 4;  // 4 структуры на луч!

        // Читаем центральную точку и соседние
        float2 cv = fft_output[base_fft_idx + center_idx];
        // TASK-2: native_sqrt
        float y_c = native_sqrt(cv.x * cv.x + cv.y * cv.y);

        float2 lv = (float2)(0.0f, 0.0f);
        float2 rv = (float2)(0.0f, 0.0f);
        float y_l = 0.0f, y_r = 0.0f;
        bool hl = (center_idx > 0);
        bool hr = (center_idx < nFFT - 1);
        // TASK-2: native_sqrt
        if (hl) { lv = fft_output[base_fft_idx + center_idx - 1]; y_l = native_sqrt(lv.x*lv.x + lv.y*lv.y); }
        if (hr) { rv = fft_output[base_fft_idx + center_idx + 1]; y_r = native_sqrt(rv.x*rv.x + rv.y*rv.y); }

        // Параболическая интерполяция
        float fo = 0.0f;
        float rf = (float)center_idx * bin_width;
        if (hl && hr) {
            float denom = y_l - 2.0f*y_c + y_r;
            if (fabs(denom) > 1e-10f) {
                fo = clamp(0.5f*(y_l-y_r)/denom, -0.5f, 0.5f);
                rf = ((float)center_idx + fo) * bin_width;
            }
        }
        if (is_right_side) { rf = sample_rate - rf; }

        // [0] Параболическая интерполяция
        maxima_output[out_base + 0].index = center_idx;
        maxima_output[out_base + 0].real = cv.x;
        maxima_output[out_base + 0].imag = cv.y;
        maxima_output[out_base + 0].magnitude = y_c;
        maxima_output[out_base + 0].phase = atan2(cv.y, cv.x) * 57.29577951f;
        maxima_output[out_base + 0].freq_offset = fo;
        maxima_output[out_base + 0].refined_frequency = rf;
        maxima_output[out_base + 0].pad = 0;

        // [1] Левая точка
        float rf_l = hl ? (float)(center_idx-1)*bin_width : 0.0f;
        if (is_right_side && hl) rf_l = sample_rate - rf_l;
        maxima_output[out_base + 1].index = hl ? center_idx-1 : 0;
        maxima_output[out_base + 1].real = lv.x;
        maxima_output[out_base + 1].imag = lv.y;
        maxima_output[out_base + 1].magnitude = y_l;
        maxima_output[out_base + 1].phase = hl ? atan2(lv.y, lv.x)*57.29577951f : 0.0f;
        maxima_output[out_base + 1].freq_offset = 0.0f;
        maxima_output[out_base + 1].refined_frequency = rf_l;
        maxima_output[out_base + 1].pad = 0;

        // [2] Центральная точка
        float rf_c = (float)center_idx * bin_width;
        if (is_right_side) rf_c = sample_rate - rf_c;
        maxima_output[out_base + 2].index = center_idx;
        maxima_output[out_base + 2].real = cv.x;
        maxima_output[out_base + 2].imag = cv.y;
        maxima_output[out_base + 2].magnitude = y_c;
        maxima_output[out_base + 2].phase = atan2(cv.y, cv.x) * 57.29577951f;
        maxima_output[out_base + 2].freq_offset = 0.0f;
        maxima_output[out_base + 2].refined_frequency = rf_c;
        maxima_output[out_base + 2].pad = 0;

        // [3] Правая точка
        float rf_r = hr ? (float)(center_idx+1)*bin_width : 0.0f;
        if (is_right_side && hr) rf_r = sample_rate - rf_r;
        maxima_output[out_base + 3].index = hr ? center_idx+1 : 0;
        maxima_output[out_base + 3].real = rv.x;
        maxima_output[out_base + 3].imag = rv.y;
        maxima_output[out_base + 3].magnitude = y_r;
        maxima_output[out_base + 3].phase = hr ? atan2(rv.y, rv.x)*57.29577951f : 0.0f;
        maxima_output[out_base + 3].freq_offset = 0.0f;
        maxima_output[out_base + 3].refined_frequency = rf_r;
        maxima_output[out_base + 3].pad = 0;
    }
}
)CL";
}

} // namespace kernels
} // namespace antenna_fft
