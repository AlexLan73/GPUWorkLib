# Strategies: параллельная статистика и pipeline

> **Версия**: 1.0  
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
- Результаты stats читаются через `hipMemcpy` D2H после `hipEventSynchronize` / `hipStreamSynchronize`.

---

## 3. Что нужно сделать (техническая проработка)

### 3.1 StatisticsProcessor: асинхронный режим

**Проблема**: Сейчас API только синхронный; внутри везде `hipStreamSynchronize`.

**Варианты решения**:

#### Вариант A: Передача внешнего stream (рекомендуется)

Добавить перегрузки с параметром `hipStream_t stream`:

```cpp
// Существующий (синхронный) — оставить для обратной совместимости
std::vector<StatisticsResult> ComputeStatistics(void* gpu_data, const StatisticsParams& params);

// Новый (асинхронный) — stream передаётся явно, без hipStreamSynchronize
void ComputeStatisticsAsync(void* gpu_data, const StatisticsParams& params, hipStream_t stream,
                             std::vector<StatisticsResult>* out_result);
```

Внутри `ComputeStatisticsAsync`:
- использовать `stream` вместо `ctx_.stream()` для всех вызовов (CopyComplexGpuData, welford_fused_op_.Execute, ReadStatisticsResults);
- **не** вызывать `hipStreamSynchronize`;
- записать результат в `out_result` через асинхронную копию или отложенное чтение.

Сложность: `ReadStatisticsResults` делает D2H — для async нужно либо:
- буфер в shared memory + `hipMemcpyDtoHAsync` + sync позже, либо
- оставить sync только в конце чтения, но выполнять его вызывающей стороной.

#### Вариант B: ComputeStatistics на заданном stream без sync

Расширить `GpuContext` / `StatisticsProcessor`:
- метод `SetStream(hipStream_t)` для временной подстановки stream;
- или `ComputeStatisticsOnStream(void*, params, stream)` — внутри использовать `stream`, без sync.

Результат: вызывающий код сам решает, когда синхронизироваться.

**Рекомендация**: Вариант A или B — добавить API, принимающий внешний stream и не вызывающий sync. Детали чтения результата — см. п. 3.2.

### 3.2 Получение результатов stats асинхронно

Сейчас `ComputeStatistics` возвращает `std::vector<StatisticsResult>` — данные читаются с GPU. Для async нужно:

1. Запустить ядра на stream.
2. `hipMemcpyDtoHAsync` в host-буфер (буфер должен быть валиден до sync).
3. Sync в конце pipeline.
4. Преобразовать host-буфер в `std::vector<StatisticsResult>`.

Альтернатива: `ComputeStatisticsAsync` возвращает структуру с указателем на GPU-буфер и метаданными; чтение выполняется вызывающим кодом после sync. Это потребует изменений в `StatisticsProcessor` (внутренние буферы, layout результата).

Практичный путь: `ComputeStatisticsAsync` пишет в переданный host-буфер через `hipMemcpyDtoHAsync`; sync делает `AntennaProcessor_v1` в конце `process()`.

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
do_run_post_fft_scenarios(result);

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

OneMax, AllMaxima, MinMax уже используют `stream_debug3_`. В `do_run_post_fft_scenarios` есть `hipStreamSynchronize` перед D2H — это нормально, т.к. данные нужны в `result` до возврата. Можно рассмотреть отдельные stream'ы для параллельного выполнения (stream_bench3a_, stream_bench3b_, stream_bench3c_), но это отдельная задача; в первом приближении оставляем текущую схему.

### 3.6 Multi-GPU

Каждый `AntennaProcessor_v1` владеет своими stream'ами и `StatisticsProcessor` (создаётся с тем же `backend_`). Дополнительных изменений для multi-GPU не требуется.

---

## 4. План реализации (по шагам)

### Шаг 1: StatisticsProcessor — async API

1. Добавить `ComputeStatisticsAsync(void* gpu_data, const StatisticsParams& params, hipStream_t stream, ...)`.
2. Реализовать без `hipStreamSynchronize`, с записью результата в предоставленный host-буфер через `hipMemcpyDtoHAsync` или аналогично.
3. Аналогично для `ComputeMedianAsync`, `ComputeStatisticsFloatAsync`, `ComputeMedianFloatAsync` — по необходимости.
4. Сохранить старый синхронный API без изменений (для тестов и других модулей).

### Шаг 2: AntennaProcessor_v1 — stats 2.1 и GEMM параллельно

1. Создать `LaunchStats21Async(d_S, result)` или интегрировать вызов в `process()`.
2. Запускать stats 2.1 на `stream_debug1_` до `do_gemm`.
3. `event_c1_done_` записывать после постановки stats на `stream_debug1_`.
4. Проверить, что `checkpoint_->save_c1_signal` остаётся на месте (по флагам).

### Шаг 3: Stats 2.2 и Window+FFT параллельно

1. После `hipStreamWaitEvent(stream_debug2_, event_gemm_done_)` запускать stats 2.2 на `stream_debug2_`.
2. Сразу после постановки stats вызывать `do_window_fft()` (без ожидания stats 2.2).
3. `event_c2_done_` записывать после постановки stats 2.2.

### Шаг 4: Stats 2.3 и post-FFT

1. После `hipStreamWaitEvent(stream_debug3_, event_fft_done_)` запускать stats 2.3 и post-FFT на `stream_debug3_`.
2. Убедиться, что порядок kernel'ов корректен (magnitude уже посчитан в `do_window_fft`).

### Шаг 5: Сбор результатов

1. В конце `process()`: `hipEventSynchronize(event_c1_done_)`, `hipEventSynchronize(event_c2_done_)`, `hipStreamSynchronize(stream_debug3_)`.
2. Заполнить `result.pre_input_stats`, `result.post_gemm_stats`, `result.post_fft_stats` из буферов, заполненных async stats.

### Шаг 6: Тесты и верификация

1. Существующие тесты strategies должны проходить.
2. Добавить проверку параллельности (например, сравнение `total_ms` до/после или отдельный benchmark).
3. Проверить корректность `AntennaResult` (сравнение с эталонным последовательным режимом).

---

## 5. Риски и ограничения

| Риск | Мера |
|------|------|
| Изменение API StatisticsProcessor | Добавлять только новые методы, старые оставить |
| Буфер для async-результата | Pre-аллоцировать в AntennaProcessor или передавать в LaunchStats*Async |
| Порядок выполнения на stream_debug3_ | Stats 2.3 и post-FFT должны идти после magnitude; проверить зависимости |
| Регрессия на других модулях | StatisticsProcessor используется не только в strategies; старый API не менять |

---

## 6. Оценка трудозатрат

| Шаг | Оценка |
|-----|--------|
| 1. StatisticsProcessor async API | 1–2 дня |
| 2. Stats 2.1 || GEMM | 0.5 дня |
| 3. Stats 2.2 || Window+FFT | 0.5 дня |
| 4. Stats 2.3 + post-FFT | 0.5 дня |
| 5. Сбор результатов | 0.5 дня |
| 6. Тесты и верификация | 1 день |
| **Итого** | ~4–5 дней |

---

## 7. Ссылки

- [AP_C2_Container.md](../Doc/Modules/strategies/AP_C2_Container.md) — диаграмма потоков
- [AP_Seq.md](../Doc/Modules/strategies/AP_Seq.md) — времена операций
- [antenna_processor_v1.cpp](../modules/strategies/src/antenna_processor_v1.cpp) — текущая реализация
- [statistics_processor.cpp](../modules/statistics/src/statistics_processor.cpp) — StatisticsProcessor
