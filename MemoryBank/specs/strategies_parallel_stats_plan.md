# Strategies: параллельная статистика и pipeline

> **Версия**: 1.2
> **Дата**: 2026-03-19
> **Ветка**: main (production C++)  
> **Цель**: Реализовать параллельное выполнение stats и GEMM+FFT для сокращения времени `process()` на ~2.6 мс (≈7.6%)

---

## 1. Текущее состояние (аудит)

### 1.1 Что задумано по диаграммам

Из [AP_C2_Container.md](../Doc/Modules/strategies/AP_C2_Container.md) и [AP_Seq.md](../Doc/Modules/strategies/AP_Seq.md):

```
Stream 1 (stats d_S):     [== welford + sort ==]  parallel with
Stream 2 (main):         [== GEMM ==][== Hamming+FFT ==]
Stream 3 (stats d_X):                    [== stats POST ==]  parallel with Hamming+FFT
Stream 4 (post-FFT):                                        [== stats |spectrum| + OneMax + AllMax + MinMax ==]
```

Ожидаемый порядок:
- Stats PRE (d_S) параллельно с GEMM
- Stats POST (d_X) параллельно с Window+FFT
- Post-FFT сценарии параллельно друг с другом

### 1.2 Что фактически реализовано

**Вердикт: параллельность НЕ реализована.**

Фактический порядок в `antenna_processor_v1.cpp`:

| Шаг | Вызов | Результат |
|-----|-------|-----------|
| 1 | `do_debug_point_21(d_S, result)` | **БЛОКИРУЕТ** до завершения stats |
| 2 | `do_gemm(d_S, d_W)` | Async на stream_main_ |
| 3 | `hipEventRecord(event_gemm_done_)` | — |
| 4 | `hipStreamWaitEvent(stream_debug2_, event_gemm_done_)` | stream_debug2 ждёт GEMM |
| 5 | `do_debug_point_22(result)` | **БЛОКИРУЕТ** до завершения stats |
| 6 | `do_window_fft()` | На stream_main_ |
| 7 | `hipEventRecord(event_fft_done_)` | — |
| 8 | `hipStreamWaitEvent(stream_debug3_, event_fft_done_)` | stream_debug3 ждёт FFT |
| 9 | `do_debug_point_23(result)` | **БЛОКИРУЕТ** до завершения stats |
| 10 | `do_run_post_fft_scenarios(result)` | Последовательно, с sync внутри |

### 1.3 Причина блокировок

`StatisticsProcessor::ComputeStatistics()`, `ComputeMedian()`, `ComputeStatisticsFloat()`, `ComputeMedianFloat()` в конце вызывают:

```cpp
hipStreamSynchronize(ctx_.stream());
```

CPU ждёт завершения работы на stream. Следующий шаг pipeline начинается только после этого.

### 1.4 stream_debug1_ не используется

- `stream_debug1_` создаётся в конструкторе.
- На него ничего не ставится.
- `StatisticsProcessor` использует stream из своего `GpuContext` (backend), а не `stream_debug1_`.
- `event_c1_done_` записывается на пустой stream — событие тут же завершено, синхронизация не несёт смысла.

### 1.5 Временные оценки (256×1.2M, AMD 9070)

| Операция | Время | Сейчас в pipeline |
|----------|-------|--------------------|
| Stats PRE (d_S) | 2.6 мс | До GEMM, блокирует |
| Stats POST (d_X) | 2.6 мс | До Window+FFT, блокирует |
| Stats POST-FFT | ~2.6 мс | До post-FFT, блокирует |
| GEMM | 13 мс | После stats PRE |
| Window + FFT | ~23 мс | После stats POST |
| Post-FFT | ~5 мс | После stats POST-FFT |

Потенциал: при параллелизме Stats PRE с GEMM экономия ≈ 2.6 мс (~7.6% от ~34 мс).

> ⚠️ **HBM contention**: Stats PRE и GEMM оба memory-bound и оба читают `d_S` (2.45 ГБ).
> При одновременном запуске они конкурируют за HBM шину → реальное ускорение может быть < 2.6 мс.
> Необходим micro-benchmark перед реализацией (см. **Шаг 0** в разделе 4).

---

## 2. Целевая архитектура

### 2.1 Желаемый поток выполнения

```
CPU:  Launch stats_21 ─► Launch GEMM ─► Launch stats_22 (после event_gemm) ─► Launch Window+FFT ─► Launch stats_23 + post-FFT ─► Sync all ─► D2H results

GPU Stream1: [== stats d_S ==]───────────────────────────────────────────────────► event_c1_done
GPU Stream2: [== GEMM ==][== Hamming+FFT ==][== magnitude ==]
GPU Stream3:               [== stats d_X (+ медиана d_X) ==]────────────────────► event_c2_done
GPU Stream4:                                                [== stats |spectrum| + медиана |spectrum| + OneMax + AllMaxima + MinMax ==]
```

**Пояснение потоков:**
- **stats d_S** — статистика (mean, variance, std, медиана) по входному сигналу до GEMM.
- **stats d_X** — статистика по результату GEMM (X = W×S); медиана добавляется при `STAT_MEDIAN`.
- **stats |spectrum| + медиана |spectrum|** — статистика и медиана по магнитудам спектра после FFT.
- **OneMax, AllMaxima, MinMax** — обязательные post-FFT сценарии (пики, min/max).

Post-FFT выполняется на Stream4, не на Stream2. После magnitude Stream2 свободен.

Все stats и GEMM+FFT выполняются параллельно на GPU. CPU только ставит задания и в конце синхронизируется.

### 2.2 Сохранение результатов

- `result.pre_input_stats`, `result.post_gemm_stats`, `result.post_fft_stats` нужны для `AntennaResult`.
- Решение: sync в конце pipeline, перед сборкой результата. К этому моменту stats уже закончат работу.
- Результаты stats читаются через `hipMemcpyDtoHAsync` D2H **в pinned-буферы** после `hipEventSynchronize` / `hipStreamSynchronize`.
- ⚠️ `hipMemcpyDtoHAsync` работает асинхронно **только с pinned (page-locked) memory** — обычный `new[]`/`std::vector` молча откатывается к синхронной копии. Буферы pre-аллоцировать через `hipHostMalloc(..., hipHostMallocDefault)` в конструкторе `AntennaProcessor_v1`, освобождать через `hipHostFree` в деструкторе.

---

## 3. Что нужно сделать (техническая проработка)

### 3.1 StatisticsProcessor: асинхронный режим (Вариант B+C)

**Проблема**: Сейчас API только синхронный; внутри `ComputeStatistics`, `ComputeMedian`, `ComputeStatisticsFloat`, `ComputeMedianFloat` в конце вызывается `hipStreamSynchronize(ctx_.stream())`.

**Решение: Вариант B+C** — новый конструктор с внешним stream + три независимых экземпляра.

Вариант A (передача `hipStream_t` в каждый метод) **не применим**: все Op-классы вызывают `ctx_->stream()` через `GpuKernelOp::stream()`, это не переопределяется без cascade изменений в 6+ файлах.

#### Изменения в `GpuContext`

Добавить второй конструктор — stream берётся извне, не из backend:

```cpp
// gpu_context.hpp — новый конструктор
GpuContext(IBackend* backend,
           hipStream_t external_stream,
           const char* module_name,
           const std::string& cache_dir = "");
```

Реализация: `stream_ = external_stream` вместо `backend_->GetNativeQueue()`. Старый конструктор без изменений.

#### Изменения в `StatisticsProcessor`

Добавить второй конструктор:

```cpp
// statistics_processor.hpp — новый конструктор
// stream: внешний HIP stream; auto_sync=false — без hipStreamSynchronize в методах
StatisticsProcessor(IBackend* backend, hipStream_t stream, bool auto_sync = false);
```

Старый конструктор `StatisticsProcessor(IBackend* backend)` → `auto_sync = true` (поведение не меняется). Все Ops автоматически используют переданный stream через `ctx_->stream()` — изменений в Op-классах не требуется.

#### Изменения в `AntennaProcessor_v1`

Три экземпляра вместо одного, каждый привязан к своему stream:

```cpp
// antenna_processor_v1.hpp
std::unique_ptr<statistics::StatisticsProcessor> stats_pre_;   // stream_debug1_
std::unique_ptr<statistics::StatisticsProcessor> stats_post_;  // stream_debug2_
std::unique_ptr<statistics::StatisticsProcessor> stats_fft_;   // stream_debug3_

// конструктор
stats_pre_  = std::make_unique<statistics::StatisticsProcessor>(backend_, stream_debug1_,  false);
stats_post_ = std::make_unique<statistics::StatisticsProcessor>(backend_, stream_debug2_,  false);
stats_fft_  = std::make_unique<statistics::StatisticsProcessor>(backend_, stream_debug3_,  false);
```

Паттерн уже используется в проекте: `AllMaximaPipelineROCm(stream_debug3_, backend_)` — тот же подход.

**Файлы для изменения**: `gpu_context.hpp/.cpp`, `statistics_processor.hpp/.cpp`, `antenna_processor_v1.hpp/.cpp` — **6 файлов**, нет изменений в Op-классах и Python bindings.

### 3.2 Получение результатов stats асинхронно

Сейчас `ComputeStatistics` возвращает `std::vector<StatisticsResult>` — данные читаются с GPU. Для async нужно:

1. Запустить ядра на stream (без `hipStreamSynchronize`).
2. `hipMemcpyDtoHAsync` в **pinned host-буфер** (буфер должен быть валиден до sync).
3. Sync в конце pipeline (`hipEventSynchronize` / `hipStreamSynchronize`).
4. Прочитать результат из pinned-буфера → заполнить `AntennaResult`.

**Требования к pinned-буферам** (аллокация в конструкторе `AntennaProcessor_v1`):
```cpp
// В constructore
HIP_CHECK(hipHostMalloc(&h_stats_pre_,  sizeof(StatisticsResult) * kMaxBeams, hipHostMallocDefault));
HIP_CHECK(hipHostMalloc(&h_stats_post_, sizeof(StatisticsResult) * kMaxBeams, hipHostMallocDefault));
HIP_CHECK(hipHostMalloc(&h_stats_fft_,  sizeof(StatisticsResult) * kMaxBeams, hipHostMallocDefault));

// В деструкторе
if (h_stats_pre_)  hipHostFree(h_stats_pre_);
if (h_stats_post_) hipHostFree(h_stats_post_);
if (h_stats_fft_)  hipHostFree(h_stats_fft_);
```

Практичный путь: stats-методы пишут в переданный pinned-буфер через `hipMemcpyDtoHAsync`; sync делает `AntennaProcessor_v1` в конце `process()`.

### 3.3 AntennaProcessor_v1: новый порядок вызовов

Текущий фрагмент:

```cpp
do_debug_point_21(d_S, result);   // блокирует
do_gemm(d_S, d_W);
// ...
```

Новый порядок:

```cpp
// 1. Запустить stats 2.1 на stream_debug1_ (async)
if (cfg_.pre_input_stats != StatPreset::NONE) {
  LaunchStats21Async(d_S, result);  // без sync
}
HIP_CHECK(hipEventRecord(event_c1_done_, stream_debug1_));

// 2. Сразу запустить GEMM на stream_main_
do_gemm(d_S, d_W);

// 3. Записать event_gemm_done_, stream_debug2_ ждёт GEMM
HIP_CHECK(hipEventRecord(event_gemm_done_, stream_main_));
HIP_CHECK(hipStreamWaitEvent(stream_debug2_, event_gemm_done_, 0));

// 4. Запустить stats 2.2 на stream_debug2_ (async)
if (cfg_.post_gemm_stats != StatPreset::NONE) {
  LaunchStats22Async(result);
}
HIP_CHECK(hipEventRecord(event_c2_done_, stream_debug2_));

// 5. Window + FFT (stream_main_)
do_window_fft();

// 6. Event FFT, stream_debug3_ ждёт
HIP_CHECK(hipEventRecord(event_fft_done_, stream_main_));
HIP_CHECK(hipStreamWaitEvent(stream_debug3_, event_fft_done_, 0));

// 7. Stats 2.3 + post-FFT на stream_debug3_
if (cfg_.post_fft_stats != StatPreset::NONE) {
  LaunchStats23Async(result);
}
// ⚠️ ВАЖНО: заменить do_run_post_fft_scenarios → do_run_post_fft_parallel
// do_run_post_fft_scenarios содержит hipStreamSynchronize внутри (строки 530, 567)
// и будет блокировать CPU до окончания stream_debug3_, нарушая схему sync в шаге 8.
// do_run_post_fft_parallel уже реализован и использует stream_bench3a/b/c.
do_run_post_fft_parallel(result);  // см. секцию 3.5

// 8. Sync всех streams, затем прочитать результаты stats
HIP_CHECK(hipEventSynchronize(event_c1_done_));
HIP_CHECK(hipEventSynchronize(event_c2_done_));
HIP_CHECK(hipStreamSynchronize(stream_debug3_));

// 9. Заполнить result.pre_input_stats, result.post_gemm_stats, result.post_fft_stats
//    из буферов, куда писали async stats
FillStatsResults(result);
```

### 3.4 Checkpoint save

`checkpoint_->save_c1_signal()` и аналог для C2 остаются прямыми вызовами по флагам. Они не входят в оптимизацию и не меняются.

### 3.5 Post-FFT сценарии

`do_run_post_fft_parallel` **уже реализован** в `antenna_processor_v1.cpp` (строки 579–639). Он запускает:
- OneMax → `stream_bench3a_`
- AllMaxima → `stream_debug3_` (AllMaximaPipelineROCm привязан к нему в конструкторе)
- MinMax → `stream_bench3c_`

Параллельно на трёх stream'ах, D2H — после синхронизации каждого.

**Изменение**: в `process()` заменить `do_run_post_fft_scenarios` → `do_run_post_fft_parallel`. `do_run_post_fft_scenarios` содержит `hipStreamSynchronize(stream_debug3_)` внутри цикла (строки 530, 567) — это блокирует CPU и ломает схему общего sync в конце `process()`.

⚠️ **Зависимость**: Stats 2.3 и post-FFT запускаются на одном и том же `stream_debug3_` (stats_fft_) и stream_bench3a/c. Stats 2.3 читает `d_magnitudes_`, post-FFT тоже читает `d_magnitudes_` — это параллельные read-only операции, race condition отсутствует. ✅

### 3.6 Multi-GPU

Каждый `AntennaProcessor_v1` владеет своими stream'ами и `StatisticsProcessor` (создаётся с тем же `backend_`). Дополнительных изменений для multi-GPU не требуется.

---

## 4. План реализации (по шагам)

### Шаг 0: Baseline benchmark (до любых изменений)

1. Запустить `strategies_profiling_benchmark.hpp` или `test_strategies_step_profiling.hpp` и зафиксировать `total_ms`.
2. Отдельно замерить время stats PRE + GEMM при одновременном запуске на двух stream'ах (micro-benchmark HBM contention).
3. Записать baseline в `Results/Profiler/` — без него нельзя измерить реальный выигрыш после.

### Шаг 1: StatisticsProcessor — async конструктор (Вариант B+C)

1. `GpuContext`: добавить второй конструктор с `hipStream_t external_stream` → `stream_ = external_stream`.
2. `StatisticsProcessor`: добавить конструктор `(IBackend*, hipStream_t, bool auto_sync)`. При `auto_sync=false` убрать `hipStreamSynchronize` из конца методов.
3. Старый конструктор `StatisticsProcessor(IBackend*)` — без изменений, `auto_sync=true`.
4. `AntennaProcessor_v1`: заменить `stats_processor_` на три экземпляра `stats_pre_`, `stats_post_`, `stats_fft_`, привязанных к `stream_debug1_/2_/3_`.
5. Pre-аллоцировать pinned host-буферы `h_stats_pre_`, `h_stats_post_`, `h_stats_fft_` через `hipHostMalloc`.

### Шаг 2: AntennaProcessor_v1 — stats 2.1 и GEMM параллельно

1. Создать `LaunchStats21Async(d_S, result)` или интегрировать вызов в `process()`.
2. Запускать stats 2.1 на `stream_debug1_` до `do_gemm`.
3. `event_c1_done_` записывать после постановки stats на `stream_debug1_`.
4. Проверить, что `checkpoint_->save_c1_signal` остаётся на месте (по флагам).

### Шаг 3: Stats 2.2 и Window+FFT параллельно

1. После `hipStreamWaitEvent(stream_debug2_, event_gemm_done_)` запускать stats 2.2 на `stream_debug2_`.
2. Сразу после постановки stats вызывать `do_window_fft()` (без ожидания stats 2.2).
3. `event_c2_done_` записывать после постановки stats 2.2.

### Шаг 4: Stats 2.3 и post-FFT параллельно

1. После `hipStreamWaitEvent(stream_debug3_, event_fft_done_)` запускать stats 2.3 (`stats_fft_`) на `stream_debug3_`.
2. Сразу запускать `do_run_post_fft_parallel(result)` — OneMax на `stream_bench3a_`, AllMaxima на `stream_debug3_`, MinMax на `stream_bench3c_`.
3. Stats 2.3 и post-FFT читают `d_magnitudes_` — read-only, race condition отсутствует ✅.
4. Убедиться что `d_magnitudes_` уже записан в `do_window_fft` перед `event_fft_done_`.

### Шаг 5: Сбор результатов

1. В конце `process()`:
   ```cpp
   HIP_CHECK(hipEventSynchronize(event_c1_done_));   // ждёт stats PRE
   HIP_CHECK(hipEventSynchronize(event_c2_done_));   // ждёт stats POST
   HIP_CHECK(hipStreamSynchronize(stream_debug3_));  // ждёт stats FFT + post-FFT
   // stream_bench3a_, stream_bench3c_ — do_run_post_fft_parallel sync'ит их сам
   ```
2. Заполнить `result.pre_input_stats`, `result.post_gemm_stats`, `result.post_fft_stats` из pinned-буферов `h_stats_pre_/post_/fft_`.
3. `event_c3_done_` не нужен — `hipStreamSynchronize(stream_debug3_)` покрывает и stats 2.3 и AllMaxima.

### Шаг 6: Тесты и верификация

1. Существующие тесты strategies должны проходить.
2. Добавить проверку параллельности (например, сравнение `total_ms` до/после или отдельный benchmark).
3. Проверить корректность `AntennaResult` (сравнение с эталонным последовательным режимом).

---

## 5. Риски и ограничения

| Риск | Мера |
|------|------|
| Изменение API StatisticsProcessor | Добавлять только новые конструктор/методы, старые оставить |
| Pinned memory обязательна для async D2H | `hipHostMalloc` в конструкторе; обычный `new[]` → sync поведение |
| HBM contention: Stats PRE + GEMM замедляют друг друга | Micro-benchmark на шаге 0; реальное ускорение может быть < 2.6 мс |
| `do_run_post_fft_scenarios` блокирует stream_debug3_ | Заменить на `do_run_post_fft_parallel` (уже реализован) |
| Порядок выполнения на stream_debug3_ | Stats 2.3 и post-FFT читают `d_magnitudes_` — read-only, race нет ✅ |
| Регрессия на других модулях | StatisticsProcessor используется не только в strategies; старый API не менять |

---

## 6. Оценка трудозатрат

| Шаг | Оценка |
|-----|--------|
| 0. Baseline benchmark | 0.5 дня |
| 1. GpuContext + StatisticsProcessor (B+C) + pinned buffers | 0.5–1 день |
| 2. Stats 2.1 ‖ GEMM | 0.5 дня |
| 3. Stats 2.2 ‖ Window+FFT | 0.5 дня |
| 4. Stats 2.3 + do_run_post_fft_parallel | 0.25 дня |
| 5. Сбор результатов | 0.25 дня |
| 6. Тесты и верификация | 1 день |
| **Итого** | ~3.5–4 дня |

---

## 7. Ссылки

- [AP_C2_Container.md](../Doc/Modules/strategies/AP_C2_Container.md) — диаграмма потоков
- [AP_Seq.md](../Doc/Modules/strategies/AP_Seq.md) — времена операций
- [antenna_processor_v1.cpp](../modules/strategies/src/antenna_processor_v1.cpp) — текущая реализация
- [statistics_processor.cpp](../modules/statistics/src/statistics_processor.cpp) — StatisticsProcessor
