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
//   Post-processing kernel: поиск ОДНОГО главного максимума в краевых диапазонах
//   спектра + параболическая интерполяция + вывод 3х соседних точек
//
// АЛГОРИТМ:
//   1. Делим search_range пополам → half_range
//   2. Ищем максимум в ДВУХ диапазонах:
//      - Диапазон 1 (положительные частоты): [0, half_range]
//      - Диапазон 2 (отрицательные частоты): [nFFT - half_range, nFFT - 1]
//   3. Выбираем ОДИН главный максимум (с наибольшей magnitude)
//   4. Берём 3 точки: [max_idx-1, max_idx, max_idx+1]
//   5. Считаем параболическую интерполяцию для уточнения частоты
//   6. Выводим 4 структуры MaxValue на каждый луч:
//      [0] - результат параболической интерполяции (с freq_offset, refined_frequency)
//      [1] - левая точка (index-1)
//      [2] - центральная точка (главный максимум)
//      [3] - правая точка (index+1)
//
// ВХОДНЫЕ ПАРАМЕТРЫ:
//   fft_output    - результат FFT (beam_count * nFFT комплексных чисел)
//   maxima_output - выходной массив (beam_count * 4 структуры MaxValue)
//   beam_count    - количество лучей
//   nFFT          - размер FFT
//   search_range  - ширина анализируемого диапазона (half_range = search_range/2)
//   sample_rate   - частота дискретизации для вычисления частоты в Гц
//
// ВЫХОДНОЙ ФОРМАТ (4 структуры MaxValue на луч):
//   MaxValue[0]: Параболическая интерполяция центральной точки
//     - index: center_idx
//     - real/imag: комплексное значение центра
//     - magnitude: |magnitude| центра
//     - phase: фаза в градусах
//     - freq_offset: параболическая поправка [-0.5, 0.5]
//     - refined_frequency: уточнённая частота (center + offset) * bin_width
//
//   MaxValue[1]: Левая точка (index-1)
//     - index: center_idx - 1
//     - real/imag: комплексное значение (или 0.0 если за границей)
//     - magnitude: |magnitude| (или 0.0)
//     - phase: фаза (или 0.0)
//     - freq_offset: 0.0
//     - refined_frequency: (center-1) * bin_width
//
//   MaxValue[2]: Центральная точка (главный максимум)
//     - index: center_idx
//     - real/imag: комплексное значение
//     - magnitude: |magnitude|
//     - phase: фаза
//     - freq_offset: 0.0
//     - refined_frequency: center * bin_width
//
//   MaxValue[3]: Правая точка (index+1)
//     - index: center_idx + 1
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
//   Найден максимум в индексе 205 → выводим точки 204, 205, 206 + параболу
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
    __global MaxValue* maxima_output,      // Результат: beam_count * 4 структуры
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

    // Local memory для параллельной редукции
    __local float local_mag[256];
    __local uint local_idx[256];

    float my_max_mag = -1.0f;
    uint my_max_idx = 0;

    // Поиск в диапазоне 1: [0, half_range]
    for (uint i = lid; i < half_range; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        float mag = sqrt(val.x * val.x + val.y * val.y);

        if (mag > my_max_mag) {
            my_max_mag = mag;
            my_max_idx = i;
        }
    }

    // Поиск в диапазоне 2: [nFFT - half_range, nFFT - 1]
    uint range2_start = nFFT - half_range;
    for (uint i = range2_start + lid; i < nFFT; i += local_size) {
        uint fft_idx = beam_idx * nFFT + i;
        float2 val = fft_output[fft_idx];
        float mag = sqrt(val.x * val.x + val.y * val.y);

        if (mag > my_max_mag) {
            my_max_mag = mag;
            my_max_idx = i;
        }
    }

    // Сохраняем локальные максимумы в shared memory
    local_mag[lid] = my_max_mag;
    local_idx[lid] = my_max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ═══════════════════════════════════════════════════════════════════════
    // ШАГ 3: Поток 0 находит ОДИН главный максимум из всех локальных
    // ═══════════════════════════════════════════════════════════════════════
    if (lid == 0) {
        float global_max_mag = -1.0f;
        uint global_max_idx = 0;

        for (uint j = 0; j < local_size; ++j) {
            if (local_mag[j] > global_max_mag) {
                global_max_mag = local_mag[j];
                global_max_idx = local_idx[j];
            }
        }

        uint center_idx = global_max_idx;
        uint base_fft_idx = beam_idx * nFFT;

        // ═══════════════════════════════════════════════════════════════════
        // ШАГ 4: Читаем 3 точки: [center-1, center, center+1]
        // ═══════════════════════════════════════════════════════════════════
        float2 left_val = (float2)(0.0f, 0.0f);
        float2 center_val = fft_output[base_fft_idx + center_idx];
        float2 right_val = (float2)(0.0f, 0.0f);

        float y_left = 0.0f;
        float y_center = sqrt(center_val.x * center_val.x + center_val.y * center_val.y);
        float y_right = 0.0f;

        // Проверяем границы для левой точки
        bool has_left = false;
        if (center_idx > 0) {
            // Проверяем что left_idx в одном из диапазонов
            uint left_idx = center_idx - 1;
            if ((left_idx < half_range) || (left_idx >= range2_start)) {
                left_val = fft_output[base_fft_idx + left_idx];
                y_left = sqrt(left_val.x * left_val.x + left_val.y * left_val.y);
                has_left = true;
            }
        }

        // Проверяем границы для правой точки
        bool has_right = false;
        if (center_idx < nFFT - 1) {
            // Проверяем что right_idx в одном из диапазонов
            uint right_idx = center_idx + 1;
            if ((right_idx < half_range) || (right_idx >= range2_start)) {
                right_val = fft_output[base_fft_idx + right_idx];
                y_right = sqrt(right_val.x * right_val.x + right_val.y * right_val.y);
                has_right = true;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // ШАГ 5: Параболическая интерполяция
        // ═══════════════════════════════════════════════════════════════════
        float bin_width = sample_rate / (float)nFFT;
        float freq_offset = 0.0f;
        float refined_frequency = (float)center_idx * bin_width;

        if (has_left && has_right) {
            float denom = y_left - 2.0f * y_center + y_right;
            if (fabs(denom) > 1e-10f) {
                float offset = 0.5f * (y_left - y_right) / denom;
                offset = clamp(offset, -0.5f, 0.5f);
                freq_offset = offset;
                refined_frequency = ((float)center_idx + offset) * bin_width;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // ШАГ 6: Заполняем 4 структуры MaxValue
        // ═══════════════════════════════════════════════════════════════════
        uint out_base = beam_idx * 4;

        // [0] - Результат параболической интерполяции
        maxima_output[out_base + 0].index = center_idx;
        maxima_output[out_base + 0].real = center_val.x;
        maxima_output[out_base + 0].imag = center_val.y;
        maxima_output[out_base + 0].magnitude = y_center;
        maxima_output[out_base + 0].phase = atan2(center_val.y, center_val.x) * 57.29577951f;
        maxima_output[out_base + 0].freq_offset = freq_offset;
        maxima_output[out_base + 0].refined_frequency = refined_frequency;
        maxima_output[out_base + 0].pad = 0;

        // [1] - Левая точка (index-1)
        if (has_left) {
            maxima_output[out_base + 1].index = center_idx - 1;
            maxima_output[out_base + 1].real = left_val.x;
            maxima_output[out_base + 1].imag = left_val.y;
            maxima_output[out_base + 1].magnitude = y_left;
            maxima_output[out_base + 1].phase = atan2(left_val.y, left_val.x) * 57.29577951f;
            maxima_output[out_base + 1].freq_offset = 0.0f;
            maxima_output[out_base + 1].refined_frequency = (float)(center_idx - 1) * bin_width;
        } else {
            // Нет левой точки → нули
            maxima_output[out_base + 1].index = 0;
            maxima_output[out_base + 1].real = 0.0f;
            maxima_output[out_base + 1].imag = 0.0f;
            maxima_output[out_base + 1].magnitude = 0.0f;
            maxima_output[out_base + 1].phase = 0.0f;
            maxima_output[out_base + 1].freq_offset = 0.0f;
            maxima_output[out_base + 1].refined_frequency = 0.0f;
        }
        maxima_output[out_base + 1].pad = 0;

        // [2] - Центральная точка (главный максимум)
        maxima_output[out_base + 2].index = center_idx;
        maxima_output[out_base + 2].real = center_val.x;
        maxima_output[out_base + 2].imag = center_val.y;
        maxima_output[out_base + 2].magnitude = y_center;
        maxima_output[out_base + 2].phase = atan2(center_val.y, center_val.x) * 57.29577951f;
        maxima_output[out_base + 2].freq_offset = 0.0f;
        maxima_output[out_base + 2].refined_frequency = (float)center_idx * bin_width;
        maxima_output[out_base + 2].pad = 0;

        // [3] - Правая точка (index+1)
        if (has_right) {
            maxima_output[out_base + 3].index = center_idx + 1;
            maxima_output[out_base + 3].real = right_val.x;
            maxima_output[out_base + 3].imag = right_val.y;
            maxima_output[out_base + 3].magnitude = y_right;
            maxima_output[out_base + 3].phase = atan2(right_val.y, right_val.x) * 57.29577951f;
            maxima_output[out_base + 3].freq_offset = 0.0f;
            maxima_output[out_base + 3].refined_frequency = (float)(center_idx + 1) * bin_width;
        } else {
            // Нет правой точки → нули
            maxima_output[out_base + 3].index = 0;
            maxima_output[out_base + 3].real = 0.0f;
            maxima_output[out_base + 3].imag = 0.0f;
            maxima_output[out_base + 3].magnitude = 0.0f;
            maxima_output[out_base + 3].phase = 0.0f;
            maxima_output[out_base + 3].freq_offset = 0.0f;
            maxima_output[out_base + 3].refined_frequency = 0.0f;
        }
        maxima_output[out_base + 3].pad = 0;
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
