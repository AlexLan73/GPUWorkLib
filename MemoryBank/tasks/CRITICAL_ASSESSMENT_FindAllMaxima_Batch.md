# Критическая оценка реализации FindAllMaxima Batch Processing

**Дата:** 2026-02-15  
**План:** [PLAN_FindAllMaxima_Batch_Processing.md](PLAN_FindAllMaxima_Batch_Processing.md)

---

## 1. Единый вызов с шаблоном — как это работает

### Пользователь вызывает ОДИН метод — неважно, где данные

```cpp
// CPU данные (vector) — один вызов
InputData<std::vector<std::complex<float>>> input_cpu{
    .antenna_count = 64,
    .n_point = 512,
    .data = cpu_signal,
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f
};
auto result = finder.FindAllMaxima(input_cpu, OutputDestination::CPU);

// GPU данные (cl_mem) — тот же вызов, другой тип
InputData<cl_mem> input_gpu{
    .antenna_count = 64,
    .n_point = 512,
    .data = gpu_signal,
    .gpu_memory_bytes = 64 * 512 * sizeof(std::complex<float>),
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f
};
auto result = finder.FindAllMaxima(input_gpu, OutputDestination::GPU);
```

### Внутренняя диспетчеризация

**Файл:** `spectrum_maxima_finder.h:623-662`

- `FindAllMaximaFromCPU` и `FindAllMaximaFromGPUPipeline` — внутренние методы
- Пользователь всегда вызывает только `FindAllMaxima<T>(input, dest)`
- Batch-логика встроена внутрь этих методов

---

## 2. Соответствие плану

| Пункт плана | Статус | Файл/строка |
|-------------|--------|--------------|
| MAX_MAXIMA_PER_BEAM | OK | `spectrum_maxima_types.h:59` |
| CalculateBytesPerAntenna ALL_MAXIMA | OK | `spectrum_maxima_finder.cpp:610-620` |
| Kernel compact_maxima beam_offset | OK | `all_maxima_kernel_sources.hpp:286,294` |
| Batch-циклы | OK | FindAllMaximaFromCPU, FindAllMaximaFromGPUPipeline |
| External buffers + beam_offset | OK | FindAllMaxima(cl_mem, ..., beam_offset, external_out_*) |
| CreateBatches | OK | Дефолты min_tail=3, merge_small_tail=true |
| Профилирование | OK | GPUProfiler |
| Лог при overflow | OK | spectrum_maxima_finder_all_maxima.cpp:1113 |

---

## 3. Критические баги

### Баг 1: Dest=GPU/ALL — буферы освобождаются вместо возврата

**Файлы:** `spectrum_maxima_finder_all_maxima.cpp` (строки 354-357 и 555-558)

**Исправление:** Перед return присвоить `combined.gpu_positions = combined_out_positions` и т.д., НЕ вызывать clReleaseMemObject.

### Баг 2: memory_limit в тестах

**Файл:** `test_batch_all_maxima.hpp:84, 134` — `512*1024` неверно, нужна доля `0.01f`.

---

## 4. Таски

### Критические баги (срочно)
- [ ] **TASK-1:** Исправить Dest=GPU в FindAllMaximaFromCPU (строки 354-357)
- [ ] **TASK-2:** Исправить Dest=GPU в FindAllMaximaFromGPUPipeline (строки 555-558)
- [ ] **TASK-3:** Исправить memory_limit в test_batch_all_maxima.hpp (0.01f вместо 512*1024)

### Формат вывода MaxValue
- [ ] **TASK-4:** Перейти на формат MaxValue — kernel compact_maxima пишет MaxValue[] вместо positions+magnitudes
- [ ] **TASK-5:** CPU: `AllMaximaResult.beams` = `vector<vector<MaxValue>>` (beams[beam_idx][peak_idx])
- [ ] **TASK-6:** GPU: один буфер `MaxValue[]` (ray0[...], ray1[...], ...) + gpu_counts; метаданные: beam_count, max_per_beam, total_maxima, gpu_bytes
- [ ] **TASK-7:** Алгоритм: index, real, imag, magnitude, phase из FFT; freq_offset=0; refined_frequency=index*bin_width (без mirror)

### Инфраструктура
- [ ] **TASK-8:** Включить test_batch_all_maxima::run() в main.cpp
- [ ] **TASK-9:** Обновить specs/fft_maxima.md с batch API и форматом MaxValue

### Обязательно
- [ ] **TASK-10:** Тест формата MaxValue — CPU и GPU, проверка структуры, beam_count, counts
- [ ] **TASK-11:** Тест с профилированием — FindAllMaxima + GPUProfiler, вывод времени (Upload, FFT, Detect, Scan, Compact) через console_output

---

## 5. Итоговый формат вывода (согласовано)

### CPU
```cpp
// AllMaximaResult
std::vector<std::vector<MaxValue>> beams;  // beams[beam_idx][peak_idx]
// beams.size() == beam_count
// beams[i].size() == num_maxima для луча i
```

### GPU
- **gpu_maxima:** один буфер `MaxValue[]`, layout: ray0[MaxValue0, MaxValue1, ...], ray1[...], ...
- **gpu_counts:** beam_count × uint32_t (количество максимумов на луч)
- **Метаданные в AllMaximaResult:** beam_count, max_per_beam, total_maxima, gpu_bytes

### MaxValue на пик
- index, real, imag, magnitude, phase из FFT
- freq_offset = 0.0f (без параболы)
- refined_frequency = index * bin_width (реальные значения, без mirror)

### Объём
- max_maxima_per_beam = 1000 (параметр), на практике ~100
- Буфер GPU: beam_count × max_per_beam × 32 байт

---

## 6. Senior-level: принципы и паттерны

### 6.1 Владение GPU-буферами (RAII / Ownership)

**Правило:** Caller владеет буферами, возвращёнными из FindAllMaxima. Модуль не освобождает то, что возвращает.

```cpp
// Плохо: освобождаем то, что отдали
if (combined_out_positions) clReleaseMemObject(combined_out_positions);
return combined;  // combined.gpu_positions == nullptr!

// Хорошо: передаём владение caller'у
combined.gpu_maxima = combined_out_maxima;
combined.gpu_counts = combined_out_counts;
return combined;  // caller вызывает clReleaseMemObject когда готов
```

**Документировать:** В API указать, что при dest=GPU/ALL caller обязан освободить gpu_maxima и gpu_counts.

### 6.2 Kernel: единая структура MaxValue

**OpenCL kernel** — структура должна совпадать с C++ (32 bytes, выравнивание):

```c
typedef struct __attribute__((packed)) {
    uint index;
    float real, imag, magnitude, phase;
    float freq_offset, refined_frequency;
    uint pad;
} MaxValue;
```

**Доступ к FFT:** compact_maxima должен получать `__global const float2* fft_output` для чтения real/imag. Сейчас kernel имеет только magnitudes — добавить fft_output как аргумент.

### 6.3 Профилирование (GPUProfiler)

- Каждый kernel/операция: `profiler.Record(gpu_id, "AllMaxima", "KernelName", pdata)`
- Вывод только через `ConsoleOutput::GetInstance().Print()` — мультиGPU-безопасно
- Тест TASK-11: включить GPUProfiler, вызвать FindAllMaxima, проверить что время записано и выведено

### 6.4 Обработка ошибок

- `clCreateBuffer` / `clEnqueue*` — проверять `err != CL_SUCCESS`
- При ошибке: освободить уже созданные буферы, пробросить `std::runtime_error` с текстом
- Не оставлять утечки при исключениях

### 6.5 Тестируемость

- Тесты изолированы: не зависят от порядка выполнения
- `memory_limit = 0.01f` — детерминированно включает batch
- Проверки: `result.beams.size() == antenna_count`, `total_maxima == sum(counts)`, структура MaxValue заполнена

### 6.6 Зависимости реализации

| Задача | Зависит от |
|--------|------------|
| TASK-4 (kernel MaxValue) | — |
| TASK-5 (CPU beams) | TASK-4 |
| TASK-6 (GPU layout) | TASK-4 |
| TASK-7 (алгоритм) | TASK-4 |
| TASK-10 (тест формата) | TASK-5, TASK-6 |
| TASK-11 (тест профилирования) | TASK-1, TASK-2 |

---

*После прочтения — решим, добавлять ли в PLAN_FindAllMaxima_Batch_Processing.md.*
