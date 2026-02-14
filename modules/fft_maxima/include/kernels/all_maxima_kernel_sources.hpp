#pragma once
// ════════════════════════════════════════════════════════════════════════════
// AllMaxima Kernel Sources — детекция всех локальных максимумов
// ════════════════════════════════════════════════════════════════════════════
//
// Pipeline: [FFT + PostCallback OR ComputeMagnitudes] → Detection → Scan → Compaction
//
// 0. post-callback: вычисляет |FFT[i]| прямо в clFFT (при full pipeline)
// 0b. compute_magnitudes: вычисляет |FFT[i]| отдельным кернелом (для raw FFT данных)
// 1. detect_all_maxima — помечает флагами локальные максимумы (читает float*)
// 2. prefix_sum_* — Blelloch work-efficient exclusive scan
// 3. compact_maxima — stream compaction (читает float*)
//
// @author Kodo (AI Assistant)
// @date 2026-02-14
// ════════════════════════════════════════════════════════════════════════════

namespace antenna_fft {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// 0. clFFT Post-Callback — вычисление |FFT[i]| во время FFT
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Post-callback для clFFT: автоматическое вычисление амплитуды ПОСЛЕ FFT.
//   Вызывается clFFT для КАЖДОГО выходного элемента.
//   Записывает: complex FFT → output (без изменений),
//               |FFT[i]| → userdata (magnitudes buffer)
//
// СИГНАТУРА clFFT post-callback:
//   void callback(__global void* output, uint outoffset,
//                 __global void* userdata, float2 fftoutput)
//
// ПАРАМЕТРЫ:
//   output    — выходной буфер FFT (cl_mem fft_output_)
//   outoffset — индекс текущего элемента (0..beam_count*nFFT-1)
//   userdata  — буфер для амплитуд (cl_mem magnitudes_buffer_)
//   fftoutput — результат FFT для текущего элемента (complex float2)
//
// ИСПОЛЬЗОВАНИЕ:
//   Регистрируется через clfftSetPlanCallback(plan, "computeMagnitudePost",
//       source, 0, POSTCALLBACK, &magnitudes_buffer_, 1);
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPostCallbackMagnitudeSource_opencl() {
    return
        "void computeMagnitudePost("
        "    __global void* output,"
        "    uint outoffset,"
        "    __global void* userdata,"
        "    float2 fftoutput"
        ") {"
        "    ((__global float2*)output)[outoffset] = fftoutput;"
        "    ((__global float*)userdata)[outoffset] = length(fftoutput);"
        "}";
}

// ════════════════════════════════════════════════════════════════════════════
// 0b. compute_magnitudes — вычисление |FFT[i]| отдельным кернелом
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Для случая когда FFT данные уже на GPU (FindAllMaxima(cl_mem) — старый API).
//   FFT не выполняется → post-callback не вызывается → нужен отдельный кернел.
//
// ВХОДНЫЕ ПАРАМЕТРЫ:
//   fft_output  — результат FFT (beam_count * nFFT float2)
//   magnitudes  — выходной массив |FFT[i]| (beam_count * nFFT float)
//   total_size  — общее количество элементов (beam_count * nFFT)
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetComputeMagnitudesKernelSource_opencl() {
    return R"CL(
__kernel void compute_magnitudes(
    __global const float2* fft_output,  // FFT: beam_count * nFFT (complex)
    __global float* magnitudes,         // Out: beam_count * nFFT (float)
    const uint total_size
) {
    const uint gid = get_global_id(0);
    if (gid >= total_size) return;

    float2 val = fft_output[gid];
    magnitudes[gid] = sqrt(val.x * val.x + val.y * val.y);
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// 1. detect_all_maxima — детекция локальных максимумов
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Для каждого элемента проверяет, является ли он локальным максимумом
//   по pre-computed амплитуде (mag[i] > mag[i-1] && mag[i] > mag[i+1]).
//   Записывает 1 в flags[i] если максимум, 0 иначе.
//
// ВХОДНЫЕ ПАРАМЕТРЫ:
//   magnitudes   - pre-computed |FFT[i]| (beam_count * nFFT float)
//   flags        - выходной массив (beam_count * nFFT uint32_t: 0 или 1)
//   beam_count   - количество лучей
//   nFFT         - размер FFT
//   search_start - начало диапазона поиска (обычно 1, пропускаем DC bin=0)
//   search_end   - конец диапазона поиска (обычно nFFT/2, Найквист)
//
// ОПТИМИЗАЦИЯ (vs старая версия):
//   - Читает float* вместо float2* — на 50% меньше чтений памяти
//   - НЕ вычисляет sqrt — 0 вместо 3 sqrt на work-item
//   - Амплитуды pre-computed через clFFT post-callback или compute_magnitudes
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetDetectAllMaximaKernelSource_opencl() {
  return R"CL(
__kernel void detect_all_maxima(
    __global const float* magnitudes,   // Pre-computed |FFT[i]| (beam_count * nFFT)
    __global uint* flags,               // Out: beam_count * nFFT (0 or 1)
    const uint beam_count,
    const uint nFFT,
    const uint search_start,            // Начало поиска (обычно 1)
    const uint search_end               // Конец поиска (обычно nFFT/2)
) {
    const uint gid = get_global_id(0);
    const uint beam_idx = gid / nFFT;
    const uint pos = gid % nFFT;

    // Проверка границ
    if (beam_idx >= beam_count) {
        return;
    }

    // Вне диапазона поиска — не максимум
    if (pos < search_start || pos >= search_end) {
        flags[gid] = 0;
        return;
    }

    // Читаем pre-computed амплитуды (без sqrt!)
    float mag_c = magnitudes[gid];
    float mag_l = magnitudes[gid - 1];
    float mag_r = magnitudes[gid + 1];

    // Строгое неравенство: локальный максимум
    flags[gid] = (mag_c > mag_l && mag_c > mag_r) ? 1 : 0;
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Prefix Sum (Blelloch Work-Efficient Exclusive Scan)
// ════════════════════════════════════════════════════════════════════════════
//
// (без изменений — prefix sum не зависит от формата данных)
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetPrefixSumKernelSource_opencl() {
  return R"CL(
// ═══════════════════════════════════════════════
// Beam-aware block-level exclusive scan (Blelloch)
// Один work-group обрабатывает BLOCK_SIZE элементов
// ВСЕ лучи сканируются параллельно!
//
// Layout: [beam0: nFFT элементов][beam1: nFFT элементов][...]
// group_id = beam_idx * blocks_per_beam + block_idx
// ═══════════════════════════════════════════════
__kernel void block_scan(
    __global const uint* input,       // Входные данные [beam_count * n_per_beam]
    __global uint* output,            // Результат scan [beam_count * n_per_beam]
    __global uint* block_sums,        // Суммы блоков [beam_count * blocks_per_beam]
    const uint n_per_beam,            // Элементов на луч (nFFT)
    const uint beam_count,            // Количество лучей
    const uint blocks_per_beam,       // Блоков на луч
    __local uint* temp                // Локальная память: BLOCK_SIZE uint
) {
    const uint lid = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    const uint BLOCK_SIZE = local_size * 2;

    // Beam-aware индексация
    const uint beam_idx = group_id / blocks_per_beam;
    const uint block_idx = group_id % blocks_per_beam;
    const uint block_offset = beam_idx * n_per_beam + block_idx * BLOCK_SIZE;
    const uint beam_end = beam_idx * n_per_beam + n_per_beam;

    const uint ai = lid;
    const uint bi = lid + local_size;

    // Загрузка в локальную память (bounds check per beam)
    temp[ai] = (block_offset + ai < beam_end) ? input[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < beam_end) ? input[block_offset + bi] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // === Up-sweep (reduce) ===
    uint offset = 1;
    for (uint d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset <<= 1;
    }

    // Сохраняем сумму блока и обнуляем последний элемент
    if (lid == 0) {
        if (block_sums != 0) {
            block_sums[group_id] = temp[BLOCK_SIZE - 1];
        }
        temp[BLOCK_SIZE - 1] = 0;
    }

    // === Down-sweep ===
    for (uint d = 1; d < BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            uint t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Записываем результат (bounds check per beam)
    if (block_offset + ai < beam_end) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < beam_end) output[block_offset + bi] = temp[bi];
}

// ═══════════════════════════════════════════════
// Beam-aware добавление суммы предыдущих блоков
// Каждый луч обрабатывается независимо!
// ═══════════════════════════════════════════════
__kernel void block_add(
    __global uint* data,              // Результат scan (in-place) [beam_count * n_per_beam]
    __global const uint* block_sums,  // Scanned block sums [beam_count * blocks_per_beam]
    const uint n_per_beam,            // Элементов на луч
    const uint beam_count,            // Количество лучей
    const uint blocks_per_beam,       // Блоков на луч
    const uint block_size             // BLOCK_SIZE = 2 * local_size
) {
    const uint gid = get_global_id(0);
    const uint total_n = beam_count * n_per_beam;
    if (gid >= total_n) return;

    const uint beam_idx = gid / n_per_beam;
    const uint pos_in_beam = gid % n_per_beam;
    const uint block_idx = pos_in_beam / block_size;

    if (block_idx > 0) {
        data[gid] += block_sums[beam_idx * blocks_per_beam + block_idx];
    }
}
)CL";
}

// ════════════════════════════════════════════════════════════════════════════
// 3. compact_maxima — Stream Compaction
// ════════════════════════════════════════════════════════════════════════════
//
// НАЗНАЧЕНИЕ:
//   Для каждого flags[i] == 1 записывает позицию и амплитуду
//   в компактный выходной массив.
//   Используем scan_output[i] как индекс записи.
//
// ОПТИМИЗАЦИЯ (vs старая версия):
//   - Читает pre-computed float* magnitudes вместо float2* + sqrt
//   - НЕ вычисляет sqrt
//
// ════════════════════════════════════════════════════════════════════════════
inline const char* GetCompactMaximaKernelSource_opencl() {
  return R"CL(
__kernel void compact_maxima(
    __global const float* magnitudes,    // Pre-computed |FFT[i]| (beam_count * nFFT)
    __global const uint* flags,          // Flags: beam_count * nFFT
    __global const uint* scan_output,    // Scan: beam_count * nFFT (per-beam scan)
    __global uint* out_positions,        // Out: beam_count * max_output_per_beam
    __global float* out_magnitudes,      // Out: beam_count * max_output_per_beam
    __global uint* out_beam_counts,      // Out: beam_count (кол-во максимумов на луч)
    const uint beam_count,
    const uint nFFT,
    const float sample_rate,
    const uint max_output_per_beam       // Макс. кол-во максимумов на луч
) {
    const uint gid = get_global_id(0);
    const uint beam_idx = gid / nFFT;
    const uint pos = gid % nFFT;

    if (beam_idx >= beam_count) return;

    // Если это последний элемент в луче — записываем count
    // count = scan[last_element] + flags[last_element]
    if (pos == nFFT - 1) {
        uint base = beam_idx * nFFT;
        uint count = scan_output[base + nFFT - 1] + flags[base + nFFT - 1];
        out_beam_counts[beam_idx] = count;
    }

    // Если это максимум — записываем в компактный буфер
    uint base = beam_idx * nFFT;
    if (flags[base + pos] == 1) {
        uint compact_idx = scan_output[base + pos];  // Позиция в компактном массиве

        // Проверяем лимит
        if (compact_idx < max_output_per_beam) {
            uint out_base = beam_idx * max_output_per_beam;

            // Записываем позицию
            out_positions[out_base + compact_idx] = pos;

            // Амплитуда уже pre-computed — просто читаем
            out_magnitudes[out_base + compact_idx] = magnitudes[gid];
        }
    }
}
)CL";
}

} // namespace kernels
} // namespace antenna_fft
