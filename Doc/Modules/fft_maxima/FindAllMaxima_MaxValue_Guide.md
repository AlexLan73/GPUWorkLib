# FindAllMaxima: MaxValue Format — Подробное руководство

**Модуль:** `modules/fft_maxima`  
**Дата:** 2026-02-15  
**Автор:** Кодо (AI Assistant)

---

## 1. Обзор

`FindAllMaxima` находит **все локальные максимумы** в FFT-спектре. Результат возвращается в формате **MaxValue** — единая структура на каждый пик с полной информацией (индекс, real/imag, амплитуда, фаза, частота).

### Ключевые особенности
- **Единый вызов** — неважно, откуда данные (CPU или GPU)
- **Batch-обработка** — автоматическая разбивка при нехватке памяти
- **MaxValue** — index, real, imag, magnitude, phase, refined_frequency
- **Dest=CPU или GPU** — результат на хосте или в GPU-буфере

---

## 2. Источники данных: CPU и GPU

### 2.1 CPU данные (`InputData<std::vector<std::complex<float>>>`)

Сырой сигнал (до FFT) в `std::vector`. Pipeline: **Upload → FFT (pre+post callback) → Detect → Scan → Compact**.

```cpp
std::vector<std::complex<float>> signal = GenerateSignal();  // CPU

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 64,
    .n_point = 512,
    .data = signal,
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f   // 80% свободной GPU памяти для batch
};

auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
```

### 2.2 GPU данные (`InputData<cl_mem>`)

Сырой сигнал уже на GPU (например, от `CwGenerator` или после `clCreateBuffer`). Pipeline: **GPU→GPU copy → FFT → Detect → Scan → Compact** (без upload с хоста).

```cpp
cl_mem gpu_signal = generator.GetOutputBuffer();  // или clCreateBuffer + clEnqueueWriteBuffer

InputData<cl_mem> input{
    .antenna_count = 64,
    .n_point = 512,
    .data = gpu_signal,
    .gpu_memory_bytes = 64 * 512 * sizeof(std::complex<float>),
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f
};

auto result = finder.FindAllMaxima(input, OutputDestination::GPU);
```

### 2.3 Сравнение путей

| Параметр        | CPU данные                    | GPU данные                    |
|-----------------|-------------------------------|-------------------------------|
| Вход            | `vector<complex<float>>`      | `cl_mem`                      |
| Upload          | Да (Host→GPU)                | Нет (данные уже на GPU)       |
| FFT             | На GPU                        | На GPU                        |
| Batch           | Да (memory_limit)             | Да (memory_limit)             |
| Dest=CPU        | Результат в `beams[].maxima`  | То же                         |
| Dest=GPU        | `gpu_maxima`, `gpu_counts`    | То же                         |

---

## 3. Формат MaxValue

### 3.1 Структура

```cpp
struct MaxValue {
    uint32_t index;             // Индекс в FFT спектре (bin)
    float real;                 // Re(FFT[index])
    float imag;                 // Im(FFT[index])
    float magnitude;            // |FFT[index]|
    float phase;                // Фаза в градусах
    float freq_offset;          // Параболическая поправка (0 для FindAllMaxima)
    float refined_frequency;    // Частота в Гц = index * sample_rate / nFFT
    uint32_t pad;              // Выравнивание (32 bytes total)
};
```

### 3.2 Результат на CPU

```cpp
// result.beams[beam_idx] — один луч
// result.beams[beam_idx].maxima[peak_idx] — один пик

for (const auto& beam : result.beams) {
    std::cout << "Beam " << beam.antenna_id << ": " << beam.num_maxima << " maxima\n";
    for (const auto& m : beam.maxima) {
        std::cout << "  bin=" << m.index
                  << " freq=" << m.refined_frequency << " Hz"
                  << " mag=" << m.magnitude
                  << " phase=" << m.phase << " deg\n";
    }
}
```

### 3.3 Результат на GPU (Dest=GPU)

- **gpu_maxima** — один буфер `MaxValue[]`, layout: `ray0[MaxValue...], ray1[...], ...`
- **gpu_counts** — `uint32_t[]`, количество максимумов на луч
- **gpu_bytes** — размер `gpu_maxima` в байтах

```cpp
if (result.destination == OutputDestination::GPU) {
    // Caller владеет буферами!
    cl_mem maxima_buf = static_cast<cl_mem>(result.gpu_maxima);
    cl_mem counts_buf = static_cast<cl_mem>(result.gpu_counts);

    // ... использование ...

    clReleaseMemObject(maxima_buf);
    clReleaseMemObject(counts_buf);
}
```

---

## 4. Batch API

### 4.1 Параметр memory_limit

- **0.0–1.0** — доля **свободной** GPU памяти для расчёта размера batch
- **0.80** (по умолчанию) — 80% свободной памяти
- **0.01** — принудительно включить batch (для тестов)

### 4.2 Логика

1. `BatchManager::CreateBatches(antenna_count, batch_size)` разбивает лучи на пакеты
2. Каждый batch: Upload (если CPU) / Copy (если GPU) → FFT → Detect → Scan → Compact
3. При Dest=GPU: буферы `gpu_maxima`, `gpu_counts` аллоцируются один раз на все лучи
4. Каждый batch пишет с `beam_offset` в правильную позицию

---

## 5. Примеры (C++)

### Пример 1: CPU данные → Dest=CPU

```cpp
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,
    .n_point = 1024,
    .data = raw_signal,
    .sample_rate = 1000.0f
};
auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
// result.beams[i].maxima — vector<MaxValue>
```

### Пример 2: GPU данные → Dest=CPU

```cpp
InputData<cl_mem> input{
    .antenna_count = 32,
    .n_point = 512,
    .data = gpu_signal,
    .gpu_memory_bytes = 32 * 512 * sizeof(std::complex<float>),
    .sample_rate = 1000.0f
};
auto result = finder.FindAllMaxima(input, OutputDestination::CPU);
```

### Пример 3: CPU данные → Dest=GPU (batch)

```cpp
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 64,
    .n_point = 512,
    .data = signal,
    .memory_limit = 0.01f
};
auto result = finder.FindAllMaxima(input, OutputDestination::GPU);
// result.gpu_maxima, result.gpu_counts — caller освобождает!
```

### Пример 4: GPU данные → Dest=GPU

```cpp
InputData<cl_mem> input{
    .antenna_count = 32,
    .n_point = 512,
    .data = gpu_signal,
    .gpu_memory_bytes = ...,
    .memory_limit = 0.01f
};
auto result = finder.FindAllMaxima(input, OutputDestination::GPU);
```

---

## 6. Python API

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# fft_data — numpy complex64, 1D или 2D (beams, nFFT)
result = finder.find_all_maxima(fft_data, sample_rate=1000.0)

# 1 луч: dict с positions, magnitudes, frequencies, num_maxima
# Несколько лучей: list[dict]
```

Поля `positions`, `magnitudes`, `frequencies` извлекаются из MaxValue для обратной совместимости.

---

## 7. Тесты

### C++ (test_batch_all_maxima.hpp)
- `TestBatchVectorInput_DestCPU` — CPU данные → Dest=CPU
- `TestBatchVectorInput_DestGPU` — CPU данные → Dest=GPU
- `TestBatchGPUInput_DestCPU` — GPU данные → Dest=CPU
- `TestBatchGPUInput_DestGPU` — GPU данные → Dest=GPU
- `TestBatchWithProfiling` — профилирование

### Python (Python_test/)
- `test_find_all_maxima_maxvalue.py` — CPU vs GPU пути, визуализация
- `test_spectrum_find_all_maxima.py` — базовые тесты

---

## 8. Pipeline (внутренняя реализация)

1. **Upload** (если CPU) или **GPU Copy** (если GPU)
2. **FFT** с pre/post callback (magnitudes в post-callback)
3. **Detect** — `detect_all_maxima` kernel (локальные максимумы)
4. **Scan** — Blelloch prefix sum (beam-aware)
5. **Compact** — `compact_maxima` kernel (вывод MaxValue[])

---

*Документ обновлён: 2026-02-15*
