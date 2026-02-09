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
// РАЗМЕЩЕНИЕ В ПАМЯТИ:
//   input  (cl_mem): [луч0][луч1][луч2]...[лучN] — весь массив
//   output (cl_mem): [луч_batch0][луч_batch1]... — результат пакета
//
// КЛЮЧЕВАЯ ФИЧА — beam_offset:
//   Позволяет обрабатывать данные по частям:
//   - Batch 0: offset=0,  обрабатывает лучи 0-9
//   - Batch 1: offset=10, обрабатывает лучи 10-19
//   - Batch 2: offset=20, обрабатывает лучи 20-29
//
// ЛОГИКА:
//   1. gid → определяем local_beam_idx и pos_in_fft
//   2. Вычисляем global_beam_idx = local_beam_idx + beam_offset  ← OFFSET!
//   3. Читаем из input[global_beam_idx * count_points + pos_in_fft]
//   4. Пишем в output[gid]
//   5. Если pos >= count_points → пишем нули (padding)
//
// ПРИМЕР:
//   batch_beam_count=2, beam_offset=3, nFFT=2048, count_points=1024
//   → Обработает лучи 3 и 4 из полного буфера
//   → output[0..2047] = луч3 с padding, output[2048..4095] = луч4 с padding
//
// ИСПОЛЬЗОВАНИЕ:
//   - Когда GPU memory < размер всех данных
//   - Нужна гибкость для обработки по частям
//   - Требуется отладка промежуточных результатов
//
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
// GetPostKernelSource() - Post Kernel с поиском максимума в краевых диапазонах
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

// ВХОДНЫЕ ПАРАМЕТРЫ:
//   fft_output    - результат FFT (beam_count * nFFT комплексных чисел)
//   maxima_output - выходной массив (beam_count * 8 структуры MaxValue)
//   beam_count    - количество лучей
//   nFFT          - размер FFT
//   search_range  - ширина анализируемого диапазона (half_range = search_range/2)
//   sample_rate   - частота дискретизации для вычисления частоты в Гц
//
// ВЫХОДНОЙ ФОРМАТ (8 структуры MaxValue на луч):
//  - Левый диапазон;
//   MaxValue[0]: Параболическая интерполяция центральной точки
//     - index: center_idx_left
//     - real/imag: комплексное значение центра
//     - magnitude: |magnitude| центра
//     - phase: фаза в градусах
//     - freq_offset: параболическая поправка [-0.5, 0.5]
//     - refined_frequency: уточнённая частота (center + offset) * bin_width
//
//   MaxValue[1]: Левая точка (index-1)
//     - index: center_idx_left - 1
//     - real/imag: комплексное значение (или 0.0 если за границей)
//     - magnitude: |magnitude| (или 0.0)
//     - phase: фаза (или 0.0)
//     - freq_offset: 0.0
//     - refined_frequency: (center-1) * bin_width
//
//   MaxValue[2]: Центральная точка (главный максимум)
//     - index: center_idx_left
//     - real/imag: комплексное значение
//     - magnitude: |magnitude|
//     - phase: фаза
//     - freq_offset: 0.0
//     - refined_frequency: center * bin_width
//
//   MaxValue[3]: Правая точка (index+1)
//     - index: center_idx_left + 1
//     - real/imag: комплексное значение (или 0.0 если за границей)
//     - magnitude: |magnitude| (или 0.0)
//     - phase: фаза (или 0.0)
//     - freq_offset: 0.0
//     - refined_frequency: (center+1) * bin_width
//  - Правый диапазон;
//   MaxValue[4]: Параболическая интерполяция центральной точки
//     - index: center_idx_right
//     - real/imag: комплексное значение центра
//     - magnitude: |magnitude| центра
//     - phase: фаза в градусах
//     - freq_offset: параболическая поправка [-0.5, 0.5]
//     - refined_frequency: уточнённая частота (center + offset) * bin_width
//
//   MaxValue[5]: Левая точка (index-1)
//     - index: center_idx_right - 1
//     - real/imag: комплексное значение (или 0.0 если за границей)
//     - magnitude: |magnitude| (или 0.0)
//     - phase: фаза (или 0.0)
//     - freq_offset: 0.0
//     - refined_frequency: (center-1) * bin_width
//
//   MaxValue[6]: Центральная точка (главный максимум)
//     - index: center_idx_right
//     - real/imag: комплексное значение
//     - magnitude: |magnitude|
//     - phase: фаза
//     - freq_offset: 0.0
//     - refined_frequency: center * bin_width
//
//   MaxValue[7]: Правая точка (index+1)
//     - index: center_idx_right + 1
//     - real/imag: комплексное значение (или 0.0 если за границей)
//     - magnitude: |magnitude| (или 0.0)
//     - phase: фаза (или 0.0)
//     - freq_offset: 0.0
//     - refined_frequency: (center+1) * bin_width
//
// ГРАНИЧНЫЕ СЛУЧАИ:
//   - Если max_idx == 0 или за границей диапазона → пишем 0.0 для отсутствующих точек
//
// ПРИМЕР:
//   nFFT = 2048, search_range = 512, half_range = 256
//   Ищем в: [0..255] и [1792..2047]
//   Игнорируем: [256..1791]
//   Найден максимум в индексе 205 → выводим точки 204, 205, 206 + параболу левого диапазона
//   Найден максимум в индексе 1802 → выводим точки 1801, 1802, 1803 + параболу правого диапазона
//
// ════════════════════════════════════════════════════════════════════════════
inline const char*  GetPostKernelSource(){
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
    __global MaxValue* maxima_output,      // Результат: beam_count * 8 структуры
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
    // ШАГ 2: Ищем максимум в ДВУХ диапазонах
    // Диапазон 1: [0, half_range] - положительные частоты
    // Диапазон 2: [nFFT - half_range, nFFT - 1] - отрицательные частоты
    // ═══════════════════════════════════════════════════════════════════════

    // Local memory для параллельной редукции (левый и правый диапазоны отдельно)
    __local float local_left_mag[256];
    __local uint local_left_idx[256];
    __local float local_right_mag[256];
    __local uint local_right_idx[256];

    float my_left_mag = -1.0f;
    uint my_left_idx = 0;
    uint range2_start = nFFT - half_range;
    uint my_right_idx = range2_start;

    // Поиск в диапазоне 1: [0, half_range] — левый
    for (uint i = lid; i < half_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        float mag = sqrt(val.x * val.x + val.y * val.y);
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
        float mag = sqrt(val.x * val.x + val.y * val.y);
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
    // ШАГ 3: Поток 0 — редукция левого и правого, вывод 8 MaxValue
    // ═══════════════════════════════════════════════════════════════════════
    if (lid == 0) {
        float global_left_mag = -1.0f;
        uint global_left_idx = 0;
        float global_right_mag = -1.0f;
        uint global_right_idx = range2_start;

        for (uint j = 0; j < local_size; ++j) {
            if (local_left_mag[j] > global_left_mag) {
                global_left_mag = local_left_mag[j];
                global_left_idx = local_left_idx[j];
            }
            if (local_right_mag[j] > global_right_mag) {
                global_right_mag = local_right_mag[j];
                global_right_idx = local_right_idx[j];
            }
        }

        uint base_fft_idx = beam_idx * nFFT;
        float bin_width = sample_rate / (float)nFFT;
        uint out_base = beam_idx * 8;

        // Макрос: mirror_freq=true для правого диапазона — частота = sample_rate - raw (зеркало 2.75 вместо 997.25)
        #define WRITE_FOUR(base_offset, center_idx, mirror_freq) do { \
            float2 cv = fft_output[base_fft_idx + center_idx]; \
            float y_c = sqrt(cv.x * cv.x + cv.y * cv.y); \
            float2 lv = (float2)(0.0f, 0.0f); \
            float2 rv = (float2)(0.0f, 0.0f); \
            float y_l = 0.0f, y_r = 0.0f; \
            bool hl = (center_idx > 0); \
            bool hr = (center_idx < nFFT - 1); \
            if (hl) { lv = fft_output[base_fft_idx + center_idx - 1]; y_l = sqrt(lv.x*lv.x + lv.y*lv.y); } \
            if (hr) { rv = fft_output[base_fft_idx + center_idx + 1]; y_r = sqrt(rv.x*rv.x + rv.y*rv.y); } \
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
//   - padding1..padding5 (для выравнивания 32 байта = 256 бит)
//   Зачем 32 байта? Выравнивание GPU memory для оптимальной производительности
//
// ЛОГИКА:
//   1. inoffset → определяем beam_idx и pos_in_fft
//   2. Читаем из input_signal[beam_idx * count_points + pos_in_fft]
//   3. ВОЗВРАЩАЕМ значение (clFFT использует для FFT)
//   4. Если pos >= count_points → возвращаем (0, 0) - padding
//
// ОГРАНИЧЕНИЕ - НЕТ beam_offset:
//   ⚠️ Callback ВСЕГДА читает с луча 0!
//   Невозможно "пропустить" первые N лучей, как в GetPaddingKernelSource
//   Данные должны быть упакованы в userdata ПОДРЯД с начала
//
// ПРИМЕР:
//   beam_count=5, nFFT=2048, count_points=1024
//   inoffset=0..2047   → beam_idx=0, читает луч 0
//   inoffset=2048..4095 → beam_idx=1, читает луч 1
//   ...всегда с начала userdata
//
// ИСПОЛЬЗОВАНИЕ:
//   - Production (Release) режим - максимальная скорость
//   - Все данные влезают в один вызов clFFT
//   - Zero-copy: нет промежуточных буферов между padding и FFT
//
// ВЫЗЫВАЕТСЯ ИЗ:
//   antenna_fft_release.cpp:225 → CreateFFTPlanWithCallbacks()
//
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

} // namespace kernels
} // namespace antenna_fft
