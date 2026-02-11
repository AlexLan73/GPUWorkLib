# Double Buffering для GPU Pipeline

**Дата**: 2026-02-11
**Статус**: ИССЛЕДОВАНИЕ (не реализовано)
**Модуль**: SpectrumMaximaFinder

---

## Концепция

Double Buffering — техника скрытия задержки передачи данных (Host→GPU) за временем GPU-вычислений.

```
Без Double Buffering:
  Batch 0: [Upload]──[FFT+Post]──[Download]
  Batch 1:            idle      [Upload]──[FFT+Post]──[Download]

С Double Buffering:
  Batch 0: [Upload A]──[FFT+Post A]──[Download A]
  Batch 1:    [Upload B]────────────[FFT+Post B]──[Download B]
                      ↑
              Upload B скрыт за FFT A
```

---

## Текущие метрики (после оптимизаций)

| Операция | Время (мс) | % от GPU |
|----------|-----------|----------|
| Upload (Pinned Memory) | 2.4 | 12% |
| FFT execution | 15.9 | 76% |
| PostKernel | 3.0 | 14% |
| **ИТОГО GPU** | **21.3** | 100% |

**Память на batch (42 антенны):**
- Input userdata: ~435 MB
- FFT buffers (2×): ~2.8 GB
- Temp buffer: ~2.8 GB
- Maxima output: ~5 KB
- **ИТОГО: ~6.0 GB**

---

## Анализ целесообразности

### Потенциальная экономия
- Upload скрыт: 2.4 мс × 7 батчей = **16.8 мс**
- Текущее время: 3342 мс
- Улучшение: **0.5%**

### Почему НЕ реализовывать сейчас

1. **Недостаточно памяти GPU**
   - Нужно 2× буферов = 12+ GB
   - RTX 3060 имеет 12 GB
   - Не поместится!

2. **Минимальный выигрыш**
   - Upload уже оптимизирован Pinned Memory (19× ускорение)
   - Upload занимает только 12% от GPU времени

3. **Высокая сложность**
   - Два набора буферов (A и B)
   - Два FFT плана (callback userdata запекается в план!)
   - Синхронизация через cl_event
   - ~200+ строк кода
   - Сложнее отлаживать

4. **clFFT специфика**
   - Pre-callback userdata указывается при `clfftSetPlanCallback`
   - Указатель запекается при `clfftBakePlan`
   - Нужен ОТДЕЛЬНЫЙ план для каждого буферного набора

---

## Когда Double Buffering НУЖЕН

| Условие | Наш случай | Нужен DB? |
|---------|------------|-----------|
| Upload > 30% GPU времени | 12% | Нет |
| Маленькие batch (есть память на 2×) | 6 GB batch | Нет |
| Real-time streaming | Нет | Нет |
| Multi-GPU с overlap | 1 GPU | Нет |

---

## Альтернативные оптимизации

### 1. Out-of-Order Queue
```cpp
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    0
};
queue = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
```
OpenCL runtime сам оптимизирует порядок операций.

### 2. Multiple Command Queues
Отдельные очереди для upload и compute позволяют overlap без Double Buffering.

### 3. Kernel Fusion
Объединить FFT post-callback с post_kernel — убрать промежуточное чтение/запись.

---

## Архитектура (если реализовывать)

```cpp
class DoubleBufferedProcessor {
    // Буферный набор A
    cl_mem userdata_A_, fft_input_A_, fft_output_A_, maxima_A_;
    clfftPlanHandle plan_A_;

    // Буферный набор B
    cl_mem userdata_B_, fft_input_B_, fft_output_B_, maxima_B_;
    clfftPlanHandle plan_B_;

    // События синхронизации
    cl_event upload_done_[2];
    cl_event compute_done_[2];

    int current_buffer_ = 0;  // 0=A, 1=B

    void ProcessPipeline(batches) {
        // Batch 0: upload A
        StartUpload(0, buffers_A);

        for (int i = 0; i < batches.size(); ++i) {
            WaitUpload(i % 2);
            StartCompute(i % 2);

            if (i + 1 < batches.size()) {
                StartUpload((i + 1) % 2, batches[i + 1]);
            }

            WaitCompute(i % 2);
            ReadResults(i % 2);
        }
    }
};
```

---

## Решение

**НЕ РЕАЛИЗОВЫВАТЬ** в текущем виде.

Причины:
1. Upload уже оптимизирован (Pinned Memory: 44.5ms → 2.4ms)
2. Недостаточно памяти GPU для 2× буферов
3. Выигрыш 0.5% не оправдывает сложность

**Пересмотреть** при:
- Работе с меньшими batch (< 3 GB)
- Появлении GPU с > 24 GB памяти
- Требованиях real-time streaming

---

## Ссылки

- [OpenCL SDK: Buffer Copy](https://github.com/khronosgroup/opencl-sdk/blob/main/samples/core/copybuffer)
- [clFFT Callbacks](https://clmathlibraries.github.io/clFFT/callbacks.html)
- [AMD: Optimizing OpenCL](https://gpuopen.com/learn/amd-opencl-optimization-guide/)

---

*Автор: Кодо (AI Assistant)*
*Дата анализа: 2026-02-11*
