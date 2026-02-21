---
name: FindAllMaxima Batch Processing
overview: "Добавление batch-обработки в FindAllMaxima() по образцу Process(): замена throw на цикл BatchManager при нехватке GPU-памяти, введение FindAllMaximaBatchFromCPU/GPU. Ограничение kernel: 1000 максимумов на луч."
date: 2026-02-15
---

# План: Batch Processing для FindAllMaxima()

## Цель

Реализовать batch-обработку в `FindAllMaxima()` по принципу `Process()`: при большом объёме данных разбивать обработку на пакеты через `BatchManager` вместо выброса исключения.

## Текущее состояние

- **Process()** — полная batch-поддержка в [spectrum_maxima_finder_process.cpp](modules/fft_maxima/src/spectrum_maxima_finder_process.cpp)
- **FindAllMaxima()** — при `!AllItemsFit` или `optimal_batch < antenna_count` выбрасывает `std::runtime_error`

---

## Архитектура batch-обработки

### Почему нужен batch?

Batch нужен НЕ из-за входных данных (они уже на GPU в cl_mem), а из-за **временных буферов**:

**Временные буферы на batch (переиспользуются между batch'ами):**
- FFT: `3 * batch_size * nFFT * 8` байт (input/output/temp)
- Magnitudes/Flags/Scan: `3 * batch_size * nFFT * 4` байт

**Выходные буферы (аллоцируются ОДИН РАЗ на ВСЕ beam_count лучей):**
- `out_positions`: `beam_count * 1000 * sizeof(uint32_t)`
- `out_magnitudes`: `beam_count * 1000 * sizeof(float)`
- `out_counts`: `beam_count * sizeof(uint32_t)`

**Пример:** 256 лучей × nFFT=4M → ~36 GB временных буферов (не влезет на GPU 30 GB).

### Схема работы для GPU destination

```
Входные данные: один cl_mem (уже на GPU)
                     ↓
         ┌──────────┬──────────┬──────────┐
         │ batch_0  │ batch_1  │ batch_2  │
         │ (0..63)  │(64..127) │(128..255)│
         └──────────┴──────────┴──────────┘
                     ↓
    Временные буферы (на batch_size, переиспользуются)
                     ↓
    Выходные GPU-буферы (на ВСЕ beam_count)
    ┌────────────────────────────────────────┐
    │ out_positions  [256 * 1000 * 4 байт]   │
    │ out_magnitudes [256 * 1000 * 4 байт]   │
    │ out_counts     [256 * 4 байт]          │
    └────────────────────────────────────────┘
         ↑           ↑           ↑
    offset=0    offset=64K  offset=128K
```

**Ключ:** Каждый batch пишет в СВОЙ offset выходного буфера:
- `offset = batch.start * max_output_per_beam`
- Фиксированный `MAX_MAXIMA_PER_BEAM=1000` делает layout предсказуемым

### Мерж результатов

**OutputDestination::CPU:**
```cpp
for (batch : batches) {
    auto batch_result = FindAllMaximaBatch(..., batch.start, batch.count);
    final_result.beams.insert(final_result.beams.end(),
        batch_result.beams.begin(), batch_result.beams.end());
    final_result.total_maxima += batch_result.total_maxima;
}
```

**OutputDestination::GPU:**
- Выходные GPU-буферы аллоцируются ОДИН РАЗ на все `beam_count` лучей
- Каждый batch пишет в свой offset через kernel параметр `beam_offset`
- **Мерж НЕ нужен** — данные не покидают GPU
- На CPU читается только `out_counts` (beam_count * 4 байт) для подсчёта `total_maxima`

**OutputDestination::ALL:**
- GPU-буферы как в `GPU` + заполнение CPU vectors из них

---

## Изменения по файлам

### 1. MAX_MAXIMA_PER_BEAM — параметр с дефолтом 1000

**Файлы:**
- [spectrum_maxima_finder.h](modules/fft_maxima/include/spectrum_maxima_finder.h):562
- [all_maxima_kernel_sources.hpp](modules/fft_maxima/include/kernels/all_maxima_kernel_sources.hpp)
- [spectrum_maxima_finder_all_maxima.cpp](modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp):716

**Изменения:**
1. Добавить параметр в `SpectrumMaximaFinderParams`:
   ```cpp
   struct SpectrumMaximaFinderParams {
       // ...
       size_t max_maxima_per_beam = 1000;  // Дефолт 1000
   };
   ```

2. Использовать `params_.max_maxima_per_beam` вместо константы:
   ```cpp
   size_t max_output_per_beam = std::min(
       (search_end - search_start) / 2,
       params_.max_maxima_per_beam
   );
   ```

3. **Лог-предупреждение при обрезке** (в compact_kernel или после чтения counts):
   ```cpp
   if (actual_count >= params_.max_maxima_per_beam) {
       console_output(gpu_context_.get(),
           "WARNING: Beam {} reached max_maxima limit ({}), results may be truncated",
           beam_id, params_.max_maxima_per_beam);
   }
   ```

**Обоснование:**
- На практике обычно ≤100 максимумов на луч
- 1000 — консервативный верхний предел
- Экономия памяти: 256 лучей × 1M → 2 GB, 256 × 1000 → 2 MB

### 2. CalculateBytesPerAntenna — ветка ALL_MAXIMA

**Файл:** [spectrum_maxima_finder.cpp](modules/fft_maxima/src/spectrum_maxima_finder.cpp)

**Добавить:**
```cpp
if (params_.peak_mode == PeakSearchMode::ALL_MAXIMA) {
    // Временные буферы: magnitudes + flags + scan
    size_t pipeline_bytes = 3 * params_.nFFT * sizeof(uint32_t);

    // Выходные буферы: positions + magnitudes + counts
    size_t output_compact = params_.max_maxima_per_beam *
                           (sizeof(uint32_t) + sizeof(float)) +
                           sizeof(uint32_t);

    return input_bytes + fft_bytes + pipeline_bytes + output_compact;
}
```

**Расчёт:** 4M точек × 256 лучей, GPU 30 GB (80% доступно):
- nFFT = 8M (с repeat_count=2): ~320 MB/луч
- Batch size ≈ 24 GB / 320 MB ≈ 75 лучей (с запасом ~67)

### 3. Kernel compact_maxima — параметр beam_offset

**Файл:** [all_maxima_kernel_sources.hpp](modules/fft_maxima/include/kernels/all_maxima_kernel_sources.hpp)

**Добавить параметр:**
```c
__kernel void compact_maxima(
    __global const uint* flags,
    __global const uint* scan,
    __global const float* magnitudes,
    __global uint* out_positions,
    __global float* out_magnitudes,
    __global uint* out_beam_counts,
    uint beam_count,
    uint nfft,
    uint max_output_per_beam,
    uint beam_offset  // <-- НОВЫЙ ПАРАМЕТР
) {
    // ...
    uint global_beam = beam_offset + beam;  // <-- Использовать вместо beam

    // Запись в выходные буферы:
    out_beam_counts[global_beam] = compact_idx;

    uint out_idx = global_beam * max_output_per_beam + compact_idx;
    out_positions[out_idx] = pos;
    out_magnitudes[out_idx] = mag;
}
```

**Применение:** при batch-обработке передавать `beam_offset = batch.start`.

### 4. Аллокация выходных буферов для batch

**Файл:** [spectrum_maxima_finder_all_maxima.cpp](modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp)

**Изменить:** В `FindAllMaxima(cl_mem fft_data, ...)` строки 700-745:

**Было:**
```cpp
size_t max_output_per_beam = ...;
size_t out_pos_size = beam_count * max_output_per_beam * sizeof(uint32_t);
cl_mem out_positions = clCreateBuffer(..., out_pos_size, ...);
```

**Стало (при batch):**
```cpp
// При batch: аллоцируем на ВСЕ antenna_count, но обрабатываем только beam_count
size_t total_beams = (is_batch_mode ? antenna_count : beam_count);
size_t out_pos_size = total_beams * max_output_per_beam * sizeof(uint32_t);
cl_mem out_positions = clCreateBuffer(..., out_pos_size, ...);
```

**Альтернатива:** Передавать готовые `out_positions/magnitudes/counts` буферы в `FindAllMaxima()` извне (при batch они создаются один раз в FindAllMaximaFromCPU/GPU).

### 5. FindAllMaximaBatchFromCPU

**Файл:** [spectrum_maxima_finder.h](modules/fft_maxima/include/spectrum_maxima_finder.h)

**Добавить метод:**
```cpp
AllMaximaResult FindAllMaximaBatchFromCPU(
    const std::vector<std::complex<float>>& data,
    size_t start_antenna,
    size_t batch_count,
    OutputDestination dest,
    size_t search_start = 0,
    size_t search_end = 0
);
```

**Реализация:** По образцу `ProcessBatch()` (строки 239-360 в spectrum_maxima_finder_process.cpp):
1. `ReallocateBuffersForBatch(batch_count)` — пересоздать FFT-план и временные буферы
2. Upload данных batch'а на GPU
3. Вызов `FindAllMaxima(cl_mem, ..., beam_offset=start_antenna)`
4. Возврат результата (beams с корректированным antenna_id)

### 6. FindAllMaximaBatchFromGPUPipeline

**Файл:** [spectrum_maxima_finder.h](modules/fft_maxima/include/spectrum_maxima_finder.h)

**Добавить метод:**
```cpp
AllMaximaResult FindAllMaximaBatchFromGPUPipeline(
    cl_mem gpu_data,
    size_t gpu_offset,
    size_t start_antenna,
    size_t batch_count,
    OutputDestination dest,
    size_t search_start = 0,
    size_t search_end = 0
);
```

**Реализация:** По образцу `ProcessBatchFromGPU()` (строки 362-427 в spectrum_maxima_finder_process.cpp):
1. `ReallocateBuffersForBatch(batch_count)`
2. GPU→GPU copy в `pre_callback_userdata_` с offset
3. FFT через callback
4. `FindAllMaxima(cl_mem, ..., beam_offset=start_antenna)`

### 7. Рефакторинг FindAllMaximaFromCPU

**Файл:** [spectrum_maxima_finder_all_maxima.cpp](modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp):179

**Заменить throw на batch-цикл:**

**Было (строки 224-227):**
```cpp
if (optimal_batch < antenna_count) {
    throw std::runtime_error(
        "FindAllMaximaFromCPU: data doesn't fit in GPU memory...");
}
```

**Стало:**
```cpp
if (optimal_batch < antenna_count) {
    // Batch processing
    auto batches = batch_manager_->CreateBatches(antenna_count, optimal_batch);

    AllMaximaResult combined;
    // При dest=GPU: аллоцировать выходные буферы на ВСЕ antenna_count
    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        AllocateAllMaximaOutputBuffers(antenna_count, max_maxima_per_beam,
                                       combined.gpu_positions,
                                       combined.gpu_magnitudes,
                                       combined.gpu_counts);
    }

    for (const auto& batch : batches) {
        auto batch_result = FindAllMaximaBatchFromCPU(
            data, batch.start, batch.count, dest, search_start, search_end);

        // Мерж результатов
        if (dest == OutputDestination::CPU || dest == OutputDestination::ALL) {
            combined.beams.insert(combined.beams.end(),
                batch_result.beams.begin(), batch_result.beams.end());
        }
        combined.total_maxima += batch_result.total_maxima;

        // При dest=GPU данные уже в combined.gpu_* (kernel писал с beam_offset)
    }

    return combined;
}
```

### 8. Рефакторинг FindAllMaximaFromGPUPipeline

**Файл:** [spectrum_maxima_finder_all_maxima.cpp](modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp):255

**Аналогично FindAllMaximaFromCPU**, но с использованием `FindAllMaximaBatchFromGPUPipeline()`.

### 9. Профилирование

**Требование:** Суммарное время всех batch'ей с выводом через GPUProfiler → console_output.

**Файлы:**
- [spectrum_maxima_finder_all_maxima.cpp](modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp)

**Реализация:**
```cpp
auto profile_data = gpu_context_->profiler().GetProfilingData();
float total_time_ms = 0.0f;

for (const auto& event : profile_data) {
    total_time_ms += event.duration_ms;
    console_output(gpu_context_.get(),
        "  {} | {:.3f} ms | {} calls",
        event.kernel_name, event.duration_ms, event.call_count);
}

console_output(gpu_context_.get(),
    "FindAllMaxima TOTAL: {:.3f} ms ({} batches, {} beams)",
    total_time_ms, batches.size(), antenna_count);
```

**По образцу:** [test_large_batch.hpp](modules/fft_maxima/tests/test_large_batch.hpp) — там уже есть вывод профилирования.

### 10. Тесты

**Файл:** `modules/fft_maxima/tests/test_batch_all_maxima.hpp` (новый)

**Тесты:**
1. `TestFindAllMaximaBatchCPU` — batch с CPU→GPU, dest=CPU
2. `TestFindAllMaximaBatchGPU` — batch с GPU→GPU, dest=GPU
3. `TestFindAllMaximaBatchLarge` — большой batch (256 лучей × 4M точек)

**По образцу:** [test_large_batch.hpp](modules/fft_maxima/tests/test_large_batch.hpp)

**Проверки:**
- Корректность antenna_id в результатах
- Совпадение total_maxima с суммой counts
- При dest=GPU: проверка GPU-буферов через чтение на CPU

---

## Порядок реализации

1. ✅ **MAX_MAXIMA_PER_BEAM** — параметр с дефолтом 1000 + лог-предупреждение (DONE 2026-02-15)
2. ✅ **CalculateBytesPerAntenna** — ветка ALL_MAXIMA (DONE 2026-02-15)
3. ✅ **compact_maxima kernel** — параметр `beam_offset` (DONE 2026-02-15)
4. ⚠️ **Аллокация выходных буферов** — частично (TODO: передача внешних буферов в FindAllMaxima)
5. ⚠️ **FindAllMaximaBatchFromCPU** — не требуется (batch-цикл внутри FindAllMaximaFromCPU)
6. ⚠️ **FindAllMaximaBatchFromGPUPipeline** — не требуется (batch-цикл внутри FindAllMaximaFromGPUPipeline)
7. ✅ **Рефакторинг FindAllMaximaFromCPU** — замена throw на batch-цикл (DONE 2026-02-15)
8. ✅ **Рефакторинг FindAllMaximaFromGPUPipeline** — замена throw на batch-цикл (DONE 2026-02-15)
9. ✅ **Профилирование** — суммарное время через GPUProfiler + console_output (DONE 2026-02-15)
10. ⏳ **Тесты** — требуют создания test_batch_all_maxima.hpp (TODO)
11. ✅ **Документация** — `Doc/Modules/fft_maxima/Full.md` (batch API, MaxValue)

---

## Вопросы и решения

### 1. AllMaximaFromCPU (без FFT) — нужен ли batch?
**Решение:** Пока **НЕТ**. Метод `AllMaximaFromCPU(cl_mem fft_data, ...)` используется редко (когда FFT уже готов). Оставить как есть с throw.

### 2. Padding последнего batch'а?
**Решение:** НЕТ. BatchManager возвращает реальный размер последнего batch'а. FFT-план пересоздаётся через `ReallocateBuffersForBatch()` — это учтено.

### 3. Рекурсивный prefix sum при batch?
**Решение:** Не требует изменений. `ExecutePrefixSum()` работает с `batch_size * nFFT` элементами, рекурсия автоматически подстраивается.

---

## Ожидаемый результат

✅ FindAllMaxima() корректно обрабатывает большие данные через batch
✅ Экономия памяти: 2 GB → 2 MB на 256 лучей (max_maxima=1000)
✅ GPU→GPU pipeline работает без CPU transfers (только counts для total_maxima)
✅ Профилирование показывает breakdown по batch'ам
✅ Тесты покрывают CPU/GPU/Large сценарии

**Производительность:** Для 256 лучей × 4M точек на GPU 30 GB:
- Batch size ~67 лучей → 4 batch'а
- Overhead на пересоздание FFT-плана: ~100-200 ms на batch (clfftBakePlan)
- Общее время: зависит от GPU, ожидается <1 sec для всех batch'ей

---

## Статус реализации (2026-02-15)

### ✅ Завершено:

1. **MAX_MAXIMA_PER_BEAM** — добавлен параметр `params_.max_maxima_per_beam` (дефолт 1000)
   - Файлы: `spectrum_maxima_types.h`, `spectrum_maxima_finder_all_maxima.cpp`, `all_maxima_pipeline_opencl.cpp`
   - Добавлено лог-предупреждение при обрезке (`console_output`)

2. **CalculateBytesPerAntenna** — добавлена ветка для ALL_MAXIMA
   - Файл: `spectrum_maxima_finder.cpp:600`
   - Учитывает: `pipeline_bytes` (magnitudes+flags+scan) + `output_compact`

3. **compact_maxima kernel** — параметр `beam_offset`
   - Файл: `all_maxima_kernel_sources.hpp:286`
   - Добавлен параметр `beam_offset`, используется `global_beam_idx = beam_offset + beam_idx`
   - Обновлены вызовы в `spectrum_maxima_finder_all_maxima.cpp:820` и `all_maxima_pipeline_opencl.cpp:412`

4. **Batch-циклы в FindAllMaximaFromCPU/GPU**
   - Файл: `spectrum_maxima_finder_all_maxima.cpp`
   - Заменён `throw` на batch-цикл через `BatchManager`
   - Single-batch vs Multi-batch ветки
   - Мерж результатов (CPU-only пока, корректировка `antenna_id`)

5. **Профилирование**
   - Добавлен вывод суммарного времени через `GPUProfiler::GetProfilingData`
   - `console_output` показывает: kernel times, total time, batch count, найденные максимумы

### ⚠️ Ограничения текущей реализации:

1. **Dest=GPU при batch** — пока НЕ поддерживается корректно
   - Выходные буферы создаются на `batch_count`, а не на `antenna_count`
   - Kernel пишет с `beam_offset=0` (не использует глобальный offset)
   - **TODO:** Передача внешних буферов в `FindAllMaxima(cl_mem, ...)`

2. **Beam_offset** — добавлен в kernel, но пока всегда передаётся `0`
   - **TODO:** Использовать `batch.start` при вызове compact_maxima

### ⏳ Осталось:

1. **Полная batch-поддержка для Dest=GPU**
   - Аллокация `combined_out_*` буферов ПЕРЕД циклом
   - Передача их в `FindAllMaxima()` как внешние буферы
   - Использование `beam_offset = batch.start` в compact_maxima

2. **Тесты** — создать `test_batch_all_maxima.hpp`

3. **Документация** — `Doc/Modules/fft_maxima/Full.md`

### 🎯 Что работает СЕЙЧАС:

✅ Batch-обработка для `OutputDestination::CPU`
✅ Автоматический расчёт optimal_batch через `BatchManager`
✅ Корректный мерж результатов (antenna_id корректируется)
✅ Профилирование batch'ей
✅ Лог-предупреждение при max_maxima_per_beam overflow

### 🔧 Следующие шаги:

1. Доработать поддержку `Dest=GPU` при batch (см. п.1 ограничений)
2. Написать тесты
3. Обновить документацию
