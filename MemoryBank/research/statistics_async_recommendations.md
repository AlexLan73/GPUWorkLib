# Async Statistics — Рекомендации и план изменений

> **Дата**: 2026-03-20
> **Модуль**: `modules/statistics`
> **Автор**: Кодо (AI Assistant)
> **Статус**: Рекомендации (не реализовано)

---

## Текущая проблема

StatisticsProcessor работает **полностью синхронно**:

```
Upload(H2D async) → Execute(Op) → hipStreamSynchronize → ReadResults(D2H sync)
```

**Критичный баг производительности**: `ReadXxxResults` использует **блокирующий** `hipMemcpyDtoH` — это останавливает весь HIP pipeline! `hipMemcpyHtoDAsync` в Upload бесполезен, если D2H блокирующий.

---

## Оценка масштаба данных

| Параметр | Значение |
|----------|----------|
| `beam_count` | 64–256 |
| `n_point` | 4096 – 1 000 000 |
| Размер chunk (100 beams × 64K points) | ~52 MB |
| Размер chunk (100 beams × 1M points) | ~800 MB |
| H2D скорость PCIe4 | ~25 GB/s |
| GPU compute (welford_fused, 52MB) | ~0.1 ms |
| GPU compute (welford_fused, 800MB) | ~5–10 ms |

**Вывод**: для n_point > 100K overlap H2D↔compute даёт значительный выигрыш.

---

## Рекомендации (по приоритету)

---

### ★★★★★ ПРИОРИТЕТ 1 — Pinned memory + async D2H

**Файлы**: `statistics_processor.cpp`, `statistics_types.hpp`
**Сложность**: Минимальная (3–4 функции)

**Что сделать**:
1. Добавить pinned host-буферы для результатов в `StatisticsProcessor`
2. Заменить `hipMemcpyDtoH` → `hipMemcpyDtoHAsync` в `ReadMeanResults`, `ReadStatisticsResults`, `ReadMedianResults`
3. `hipStreamSynchronize` остаётся — но теперь весь pipeline (H2D + compute + D2H) в одной очереди

```cpp
// В StatisticsProcessor (private):
void* pinned_results_ = nullptr;   // hipHostMalloc — pinned buffer для results
size_t pinned_results_bytes_ = 0;

// В ReadStatisticsResults:
hipMemcpyDtoHAsync(pinned_results_, gpu_result_buf, bytes, ctx_.stream()); // async!
hipStreamSynchronize(ctx_.stream()); // один sync в конце
// читаем из pinned_results_
```

**Выигрыш**: GPU scheduler получает возможность overlap compute↔D2H. HIP pipeline видит всю цепочку команд сразу.

---

### ★★★★★ ПРИОРИТЕТ 2 — StatisticsHandle (event-based Begin/End API)

**Новые файлы**: `include/statistics_handle.hpp`
**Изменения**: `statistics_processor.hpp`, `statistics_processor.cpp`
**Сложность**: Средняя

**Что добавить в statistics_types.hpp / statistics_handle.hpp**:

```cpp
/// Async handle — результат BeginCompute*()
struct StatisticsHandle {
  hipEvent_t  event      = nullptr;  // GPU event — сигнал готовности
  void*       host_ptr   = nullptr;  // pinned host buffer с данными
  size_t      host_bytes = 0;
  uint32_t    beam_count = 0;

  enum class Kind { kMean, kMedian, kStatistics } kind = Kind::kStatistics;

  bool IsValid() const { return event != nullptr; }
  void Reset() { event = nullptr; host_ptr = nullptr; host_bytes = 0; }
};
```

**Новые методы в StatisticsProcessor**:

```cpp
// === Begin/End API (event-based async) ===

// Запускает GPU pipeline, возвращает немедленно
StatisticsHandle BeginComputeStatistics(
    void* gpu_data, const StatisticsParams& params);

StatisticsHandle BeginComputeMean(
    void* gpu_data, const StatisticsParams& params);

StatisticsHandle BeginComputeMedian(
    void* gpu_data, const StatisticsParams& params);

// Блокирует только до готовности конкретного handle
std::vector<StatisticsResult> EndComputeStatistics(StatisticsHandle& handle);
std::vector<MeanResult>       EndComputeMean(StatisticsHandle& handle);
std::vector<MedianResult>     EndComputeMedian(StatisticsHandle& handle);

// Проверить без ожидания (polling)
bool IsReady(const StatisticsHandle& handle) const;
```

**Реализация BeginComputeStatistics**:

```cpp
StatisticsHandle StatisticsProcessor::BeginComputeStatistics(
    void* gpu_data, const StatisticsParams& params) {
  EnsureCompiled();
  size_t count = (size_t)params.beam_count * params.n_point;
  CopyComplexGpuData(gpu_data, count);          // async D2D
  welford_fused_op_.Execute(params.beam_count, params.n_point);  // async kernel

  // Async D2H в pinned buffer
  size_t bytes = params.beam_count * 5 * sizeof(float);
  EnsurePinnedResults(bytes);
  hipMemcpyDtoHAsync(pinned_results_,
                     ctx_.GetShared(shared_buf::kResult),
                     bytes, ctx_.stream());

  // Записать event ПОСЛЕ D2H
  StatisticsHandle handle;
  hipEventCreate(&handle.event);
  hipEventRecord(handle.event, ctx_.stream());  // event в stream'е
  handle.host_ptr   = pinned_results_;
  handle.host_bytes = bytes;
  handle.beam_count = params.beam_count;
  handle.kind       = StatisticsHandle::Kind::kStatistics;
  return handle;
}
```

**Реализация EndComputeStatistics**:

```cpp
std::vector<StatisticsResult> StatisticsProcessor::EndComputeStatistics(
    StatisticsHandle& handle) {
  if (!handle.IsValid()) throw std::runtime_error("Invalid handle");

  hipEventSynchronize(handle.event);  // ждём только ЭТОТ event (не весь stream)
  hipEventDestroy(handle.event);

  // Читаем из pinned_results_ (уже в host memory)
  auto results = ParseStatisticsResults(
      static_cast<float*>(handle.host_ptr), handle.beam_count);
  handle.Reset();
  return results;
}
```

**Паттерн использования (10 GPU параллельно)**:

```cpp
// Запустить на всех GPU (неблокирующий)
std::vector<StatisticsHandle> handles;
for (auto& proc : processors) {  // вектор StatisticsProcessor
  handles.push_back(proc.BeginComputeStatistics(gpu_data[i], params));
}

// ... делаем что-то ещё на CPU ...

// Собрать результаты
for (size_t i = 0; i < handles.size(); ++i) {
  auto results = processors[i].EndComputeStatistics(handles[i]);
  // обработка
}
```

---

### ★★★★☆ ПРИОРИТЕТ 3 — std::future wrapper для C++ и Python

**Изменения**: `statistics_processor.hpp`, `statistics_processor.cpp`, Python bindings
**Сложность**: Низкая (обёртка над sync API)

```cpp
// В statistics_processor.hpp
#include <future>

std::future<std::vector<StatisticsResult>> ComputeStatisticsAsync(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);

std::future<std::vector<MeanResult>> ComputeMeanAsync(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);

std::future<std::vector<MedianResult>> ComputeMedianAsync(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);
```

**Реализация (одна строка по сути)**:

```cpp
std::future<std::vector<StatisticsResult>>
StatisticsProcessor::ComputeStatisticsAsync(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params) {
  return std::async(std::launch::async,
      [this, data, params]() {  // data копируется!
        return ComputeStatistics(data, params);
      });
}
```

**Python bindings (py_statistics.hpp)**:

```cpp
// Существующие sync методы остаются
py_class
  .def("compute_statistics_async", [](StatisticsProcessor& self,
       py::array_t<std::complex<float>> data,
       const StatisticsParams& params) {
    // Копируем данные ДО отпускания GIL
    std::vector<std::complex<float>> vec(data.data(), data.data() + data.size());

    // Отпускаем GIL и запускаем async
    py::gil_scoped_release gil_release;
    auto future = self.ComputeStatisticsAsync(vec, params);
    future.wait();
    py::gil_scoped_acquire gil_acquire;
    return future.get();
  });
```

**Важно**: для настоящего async в Python нужны asyncio + concurrent.futures.
Пока достаточно ThreadPoolExecutor:

```python
from concurrent.futures import ThreadPoolExecutor
import gpu_stats

with ThreadPoolExecutor() as pool:
    futures = [pool.submit(proc.compute_statistics, data, params)
               for proc in processors]
    results = [f.result() for f in futures]
```

---

### ★★★☆☆ ПРИОРИТЕТ 4 — Double-stream pipeline (для streaming данных)

**Изменения**: `DrvGPU/interface/gpu_context.hpp`, `statistics_processor.hpp`
**Сложность**: Высокая
**Когда нужно**: непрерывный поток данных (SDR, radar scan)

**Идея**: два HIP stream'а + hipEvent синхронизация

```
stream_io_[0]:      H2D(chunk0) ──────────────── D2H(chunk0)
stream_compute_:               compute(chunk0)  compute(chunk1)
stream_io_[1]:                          H2D(chunk1) ─────────── D2H(chunk1)
                    ↑                   ↑
                  EventA              EventB
```

**Реализация**:
- В `GpuContext`: опциональный `stream_io_` (второй stream, создаётся по запросу)
- `EnableDoubleBuffering()` в GpuContext — создаёт stream_io_ + отдельный буфер kInput2
- `StatisticsProcessor` хранит два набора pinned буферов (ping-pong)

**Выгода при n_point=1M**:
- `pipeline throughput ≈ max(H2D_time, compute_time)` вместо `sum`
- При H2D=32ms, compute=10ms → throughput улучшается с 42ms/chunk до 32ms/chunk (+24%)

---

## Структура изменений (summary)

```
modules/statistics/
├── include/
│   ├── statistics_types.hpp       ← [ИЗМЕНИТЬ] StatisticsHandle struct
│   ├── statistics_handle.hpp      ← [НОВЫЙ] вспомогательные утилиты handle
│   └── statistics_processor.hpp  ← [ИЗМЕНИТЬ] +Begin/End/Async методы
├── src/
│   └── statistics_processor.cpp  ← [ИЗМЕНИТЬ] pinned mem, async D2H, Begin/End impl
└── Python/
    └── py_statistics.hpp         ← [ИЗМЕНИТЬ] async Python bindings
```

---

## Порядок реализации

1. **Сейчас**: Pinned memory + async D2H (Приоритет 1) — quick win
2. **После**: StatisticsHandle + Begin/End API (Приоритет 2) — основа для 10-GPU
3. **После**: std::future + Python async (Приоритет 3) — удобство API
4. **По необходимости**: Double-stream (Приоритет 4) — streaming data

---

## Совместимость с Ref03

- Все **существующие методы остаются неизменными** (sync API)
- Begin/End — **добавляются** рядом (новые перегрузки)
- Python bindings **не ломаются**
- GpuContext API **не меняется** (pinned mem внутри StatisticsProcessor)
