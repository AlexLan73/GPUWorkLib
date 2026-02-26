# 🚀 ROCm Pipeline Optimization — Шпаргалка

> **Основано на**: Task_12 (CholeskyInverterROCm 341×341: 1.482 → **0.941 мс**, -36.5%)
> **GPU**: AMD Radeon RX 9070 (gfx1201), ROCm 7.2.0
> **Дата**: 2026-02-26

---

## 📋 Содержание

1. [Главные правила](#-главные-правила)
2. [Антипаттерны и их стоимость](#-антипаттерны-и-их-стоимость)
3. [Методика профилирования](#-методика-профилирования)
4. [Чеклист оптимизации модуля](#-чеклист-оптимизации-модуля)
5. [Аудит модулей проекта](#-аудит-модулей-проекта)
6. [Паттерны кода: ДО и ПОСЛЕ](#-паттерны-кода-до-и-после)
7. [Продвинутые техники](#-продвинутые-техники)

---

## 🔑 Главные правила

### Правило №1: hipMalloc/hipFree — ДОРОГО!

```
hipMalloc(4 байта)  ≈ 10–50 мкс
hipFree(4 байта)    ≈ 10–50 мкс
hipMalloc(930 КБ)   ≈ 20–100 мкс
```

**НИКОГДА** не вызывать hipMalloc/hipFree в hot path (на каждый вызов Process/Invert).
Аллоцировать в конструкторе, переиспользовать, освобождать в деструкторе.

### Правило №2: hipMemcpy D2H — СИНХРОНИЗАЦИЯ!

```
hipMemcpy D2H (даже 8 байт) ≈ 0.5–0.8 мс (!)
```

Причина: `hipMemcpy` (без Async) — **блокирующий** вызов. CPU ждёт завершения
**ВСЕХ** GPU операций на device, даже если копируем 1 байт. Это implicit
`hipDeviceSynchronize()`.

**Минимизировать** hipMemcpy D2H в hot path. Если нужна проверка — делать
отложенно или async.

### Правило №3: backend->Synchronize() — только когда НЕОБХОДИМО

```
backend->Synchronize()  ≈ 5–20 мкс  (+ stall GPU pipeline)
```

Если все операции на **одном stream** — порядок гарантирован аппаратно.
Synchronize нужен ТОЛЬКО:
- Перед чтением результата на CPU
- При переключении между stream'ами
- Перед hipMemcpy D2H / H2D (если через backend API)

### Правило №4: Один stream = автоматический порядок

GPU stream — это очередь FIFO. Kernel'ы на одном stream выполняются строго
последовательно. НЕ нужно ставить Synchronize между последовательными
операциями на одном stream.

```cpp
// ❌ ПЛОХО — лишняя синхронизация
KernelA<<<grid, block, 0, stream>>>();
backend->Synchronize();           // НЕ НУЖНО!
KernelB<<<grid, block, 0, stream>>>();

// ✅ ХОРОШО — stream гарантирует порядок
KernelA<<<grid, block, 0, stream>>>();
KernelB<<<grid, block, 0, stream>>>();
```

---

## 💀 Антипаттерны и их стоимость

### AP-1: hipMalloc/hipFree на каждый вызов

**Стоимость**: ~20–100 мкс на пару malloc+free

```cpp
// ❌ АНТИПАТТЕРН: аллокация в hot path
void Process(void* input, size_t size) {
    void* output = nullptr;
    hipMalloc(&output, size);        // 20-100 мкс!
    LaunchKernel(input, output);
    hipMemcpy(host, output, size, D2H);
    hipFree(output);                 // 20-100 мкс!
}

// ✅ ПАТТЕРН: предаллокация + переиспользование
class Processor {
    void* d_work_ = nullptr;
    size_t d_work_size_ = 0;

    void EnsureBuffer(size_t needed) {
        if (d_work_size_ >= needed) return;  // переиспользуем!
        if (d_work_) hipFree(d_work_);
        hipMalloc(&d_work_, needed);
        d_work_size_ = needed;
    }

    void Process(void* input, size_t size) {
        EnsureBuffer(size);
        LaunchKernel(input, d_work_);
    }
};
```

### AP-2: hipMemcpy D2H для проверки ошибок в hot path

**Стоимость**: ~0.5–0.8 мс на КАЖДЫЙ вызов!

```cpp
// ❌ АНТИПАТТЕРН: синхронная проверка info
void CorePotrf(void* A, int n) {
    rocblas_int* dev_info;
    hipMalloc(&dev_info, 4);              // 30 мкс
    rocsolver_cpotrf(h, fill, n, A, n, dev_info);
    rocblas_int host_info;
    hipMemcpy(&host_info, dev_info, 4, D2H);  // 500-800 мкс!!!
    hipFree(dev_info);                     // 30 мкс
}

// ✅ ПАТТЕРН: предаллокация + отложенная проверка
class Solver {
    void* d_info_ = nullptr;   // предаллоцирован в конструкторе
    bool check_info_ = true;

    void CorePotrf(void* A, int n) {
        auto* info = static_cast<rocblas_int*>(d_info_);
        rocsolver_cpotrf(h, fill, n, A, n, info);
        // Без проверки! Проверим позже в CheckInfo()
    }

    void CheckInfo() {  // вызывать только когда нужно
        if (!check_info_) return;
        rocblas_int host[2];
        hipMemcpy(host, d_info_, 8, D2H);  // одна синхронизация
        if (host[0] != 0) throw ...;
    }
};
```

### AP-3: Лишний Synchronize между операциями на одном stream

**Стоимость**: ~5–20 мкс + stall GPU pipeline

```cpp
// ❌ АНТИПАТТЕРН
CorePotrf(d_matrix, n, stream);
CorePotri(d_matrix, n, stream);
backend->Synchronize();              // ЛИШНИЙ! Один stream!
SymmetrizeKernel<<<grid,block,0,stream>>>();

// ✅ ПАТТЕРН
CorePotrf(d_matrix, n, stream);
CorePotri(d_matrix, n, stream);
SymmetrizeKernel<<<grid,block,0,stream>>>();
// Synchronize — только перед чтением результата на CPU
```

### AP-4: Повторная аллокация при неизменном размере

**Стоимость**: ~40–200 мкс (free + malloc)

```cpp
// ❌ АНТИПАТТЕРН: каждый раз пересоздаём
void Reconfigure(int new_size) {
    hipFree(buffer_);
    hipMalloc(&buffer_, new_size);
}

// ✅ ПАТТЕРН: аллокация только при увеличении размера
void EnsureCapacity(size_t needed) {
    if (capacity_ >= needed) return;
    if (buffer_) { (void)hipFree(buffer_); buffer_ = nullptr; }
    hipError_t err = hipMalloc(&buffer_, needed);
    if (err != hipSuccess) throw ...;
    capacity_ = needed;
}
```

---

## 📊 Методика профилирования

### Шаг 1: Stage-level profiling с hipEvent

Вставить `hipEvent` между каждым этапом pipeline:

```cpp
hipEvent_t e[N];
for (int i = 0; i < N; ++i) hipEventCreate(&e[i]);

hipDeviceSynchronize();  // чистое начальное состояние

hipEventRecord(e[0], stream);
  /* Stage 1: Alloc */
hipEventRecord(e[1], stream);
  /* Stage 2: Upload/Copy */
hipEventRecord(e[2], stream);
  /* Stage 3: Kernel A */
hipEventRecord(e[3], stream);
  /* Stage 4: Kernel B */
hipEventRecord(e[4], stream);
  /* Stage 5: Download */
hipEventRecord(e[5], stream);

hipEventSynchronize(e[5]);

float ms;
hipEventElapsedTime(&ms, e[0], e[1]);  // Stage 1 time
hipEventElapsedTime(&ms, e[1], e[2]);  // Stage 2 time
// ...
```

### Шаг 2: Warmup + Multiple runs

```
Warmup:  3 итерации   (прогрев GPU, hiprtc compile, JIT)
Measure: 10-20 итераций (среднее / min / max)
```

### Шаг 3: Сравнение "до" и "после"

Замерять **два варианта** в одном тесте:
1. Текущий код (с overhead'ами)
2. Оптимизированный (чистые kernel'ы)

Это показывает **потенциал** оптимизации до изменения production кода.

### Пример: test_stage_profiling.hpp (Task_12)

```
Текущий код:                    Оптимизированный:
  Alloc         0.004 ms (0.3%)   Alloc         0.004 ms (0.4%)
  D2D copy      0.029 ms (1.9%)   D2D copy      0.029 ms (3.0%)
  POTRF+info    0.784 ms (52.0%)  POTRF         0.527 ms (54.5%)  ← -33%!
  POTRI+info    0.638 ms (42.3%)  POTRI         0.362 ms (37.5%)  ← -43%!
  Synchronize   0.017 ms (1.1%)   Synchronize   0.004 ms (0.4%)
  Symmetrize    0.017 ms (1.1%)   Symmetrize    0.023 ms (2.4%)
  Free          0.018 ms (1.2%)   Free          0.018 ms (1.9%)
  TOTAL         1.508 ms          TOTAL         0.967 ms          ← -36%!
```

---

## ✅ Чеклист оптимизации модуля

Для каждого ROCm модуля пройти по чеклисту:

### Аллокации

- [ ] **Нет hipMalloc в hot path** (Process/Invert/Generate)?
- [ ] Все рабочие буферы предаллоцированы (конструктор или Configure)?
- [ ] Используется EnsureCapacity вместо free+malloc?
- [ ] hipFree только в деструкторе или при реконфигурации?

### Синхронизация

- [ ] **Нет лишних** `backend->Synchronize()` между операциями на одном stream?
- [ ] **Нет hipMemcpy D2H** в hot path для проверки ошибок?
- [ ] Проверка ошибок отложена или опциональна (`SetCheckInfo(false)`)?
- [ ] `hipDeviceSynchronize()` — только в тестах, не в production коде?

### Паттерны

- [ ] Данные остаются на GPU максимально долго (GPU→GPU pipe)?
- [ ] Минимум CPU↔GPU трансферов в pipeline?
- [ ] hiprtc kernel'ы кешируются на диске (KernelCacheService)?
- [ ] Batch операции используют один kernel launch (не цикл)?

### Benchmark

- [ ] Есть hipEvent benchmark (не CPU chrono)?
- [ ] Warmup перед замерами (3+ итераций)?
- [ ] MD отчёт с результатами в `Results/Profiler/`?

---

## 🔍 Аудит модулей проекта

### Легенда

| Символ | Значение |
|--------|----------|
| ✅ | Хорошо — предаллокация, правильный паттерн |
| ⚠️ | Проблема — per-call аллокация, нужна оптимизация |
| 🔴 | Критично — hipMemcpy D2H в hot path, hipMalloc в цикле |

---

### 1. ✅ vector_algebra (CholeskyInverterROCm) — ОПТИМИЗИРОВАН (Task_12)

**Файлы**: `modules/vector_algebra/src/cholesky_inverter_rocm.cpp`

**Что сделано**:
- `d_info_` предаллоцирован в конструкторе (2 × rocblas_int)
- `CheckInfo()` отложенная, `SetCheckInfo(false)` для benchmark
- Убран лишний `Synchronize()` между POTRI и Symmetrize

**Результат**: 1.482 → **0.941 мс** (-36.5%)

**Что ещё можно**: предаллокация d_output buffer (EnsureCapacity)

---

### 2. ✅ fft_processor (FFTProcessorROCm) — ХОРОШО

**Файлы**: `modules/fft_processor/src/fft_processor_rocm.cpp`

**Паттерн**: Буферы (`input_buffer_`, `fft_input_`, `fft_output_`, `mag_output_`,
`phase_output_`) предаллоцированы в `AllocateBuffers()`. Free+Malloc только при
изменении конфигурации. **Правильный паттерн!**

**Рекомендации**: Минимальные — проверить нет ли лишних Synchronize в Process().

---

### 3. ✅ heterodyne (HeterodyneProcessorROCm) — ХОРОШО

**Файлы**: `modules/heterodyne/src/heterodyne_processor_rocm.cpp`

**Паттерн**: Все буферы (`buf_rx_`, `buf_dc_`, `buf_corr_`, `buf_ref_`, `buf_freq_`)
предаллоцированы. Free+Malloc только при реконфигурации. **Правильный паттерн!**

**Рекомендации**: Минимальные — только benchmark hipEvent.

---

### 4. ✅ statistics (StatisticsProcessor) — ХОРОШО

**Файлы**: `modules/statistics/src/statistics_processor.cpp`

**Паттерн**: 7 буферов (`input_buffer_`, `magnitudes_buf_`, `sort_buf_`,
`sort_temp_buf_`, `offsets_buf_`, `reduce_buf_`, `result_buf_`) предаллоцированы
в `AllocateBuffers()`. **Правильный паттерн!**

**Рекомендации**: Минимальные — benchmark hipEvent.

---

### 5. ⚠️ filters — FIR (FirFilterROCm) — НУЖНА ОПТИМИЗАЦИЯ

**Файлы**: `modules/filters/src/fir_filter_rocm.cpp`

**Проблемы**:
- **Строка 129**: `hipMalloc(&output_ptr, buffer_size)` — **на каждый `Process()`!**
- **Строка 162**: `hipFree(output_ptr)` — на каждый `Process()`
- **Строка 193**: `hipMalloc(&input_ptr, data_size)` — на каждый `ProcessFromCPU()`
- **Строка 211**: `hipFree(input_ptr)` — на каждый `ProcessFromCPU()`

**Решение**: Добавить `d_output_` и `d_input_tmp_` в класс, предаллокация в
конструкторе или lazy (EnsureCapacity). Коэффициенты (`coeff_buf_`) уже
предаллоцированы — хорошо.

**Ожидаемый эффект**: ~40–200 мкс экономия на вызов.

---

### 6. ⚠️ filters — IIR (IirFilterROCm) — НУЖНА ОПТИМИЗАЦИЯ

**Файлы**: `modules/filters/src/iir_filter_rocm.cpp`

**Проблемы**: Идентично FIR:
- **Строка 129**: `hipMalloc(&output_ptr)` на каждый `Process()`
- **Строка 162**: `hipFree(output_ptr)` на каждый `Process()`
- **Строка 193/211**: `hipMalloc/hipFree(input_ptr)` на каждый `ProcessFromCPU()`

**Решение**: Аналогично FIR — предаллокация `d_output_`, `d_input_tmp_`.

---

### 7. ⚠️ lch_farrow (LchFarrowROCm) — НУЖНА ОПТИМИЗАЦИЯ

**Файлы**: `modules/lch_farrow/src/lch_farrow_rocm.cpp`

**Проблемы**:
- **Строка 327**: `hipMalloc(&output_ptr)` — на каждый `Process()`
- **Строка 337**: `hipMalloc(&delay_buf)` — на каждый `Process()`
- **Строки 390-400**: `hipFree(output_ptr)`, `hipFree(delay_buf)` — на каждый `Process()`
- **Строка 427**: `hipMalloc(&input_ptr)` — на каждый `ProcessFromCPU()`
- **Строка 447**: `hipFree(input_ptr)` — на каждый `ProcessFromCPU()`

**3 аллокации + 3 free на каждый вызов** = ~120–600 мкс overhead!

**Решение**: Предаллокация `d_output_`, `d_delay_`, `d_input_tmp_` с EnsureCapacity.
Матрица (`matrix_buf_`) уже предаллоцирована — хорошо.

---

### 8. 🔴 fft_maxima — AllMaximaPipeline — КРИТИЧНО!

**Файлы**: `modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp`

**Проблемы** (функция `PrefixSumBlocks`):
- **Строка 163**: `hipMalloc(&block_sums)` — на каждый вызов
- **Строка 167**: `hipMalloc(&block_sums_scanned)` — на каждый вызов
- **Строки 228-229**: `hipFree` обоих — на каждый вызов

**Проблемы** (функция `FindAllMaxima` / `RunPipeline`):
- **Строка 272**: `hipMalloc(&flags_buf)` — на каждый вызов
- **Строка 276**: `hipMalloc(&scan_buf)` — на каждый вызов
- **Строка 287**: `hipMalloc(&out_maxima)` — на каждый вызов
- **Строка 295**: `hipMalloc(&out_beam_counts)` — на каждый вызов
- **Строки 442-445**: `hipFree` всех 4 — на каждый вызов

**6 аллокаций + 6 free на каждый вызов** = ~240–1200 мкс overhead!

**Решение**: Предаллокация всех 6 буферов с EnsureCapacity. Размеры зависят от
beam_count и fft_size — кешировать при Configure.

---

### 9. ⚠️ signal_generators — FormSignal — НУЖНА ОПТИМИЗАЦИЯ

**Файлы**: `modules/signal_generators/src/form_signal_generator_rocm.cpp`

**Проблемы**:
- **Строка 95**: `hipMalloc(&output_ptr)` — на каждый `GenerateInputData()`
- **Строка 162/194**: `hipFree(output_ptr)` / `hipFree(gpu_buf)` — на каждый вызов

**Решение**: Предаллокация output buffer, EnsureCapacity.

---

## 💡 Паттерны кода: ДО и ПОСЛЕ

### Паттерн A: EnsureCapacity — универсальный

```cpp
// В заголовке класса:
private:
    void* d_work_ = nullptr;
    size_t d_work_capacity_ = 0;

    void EnsureWorkBuffer(size_t needed_bytes) {
        if (d_work_capacity_ >= needed_bytes) return;
        if (d_work_) { (void)hipFree(d_work_); d_work_ = nullptr; }
        hipError_t err = hipMalloc(&d_work_, needed_bytes);
        if (err != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
        d_work_capacity_ = needed_bytes;
    }

// В деструкторе:
    ~MyProcessor() {
        if (d_work_) { (void)hipFree(d_work_); d_work_ = nullptr; }
    }

// В hot path:
    void Process(size_t size) {
        EnsureWorkBuffer(size);  // 0 мкс если размер не вырос!
        LaunchKernel(d_work_, size);
    }
```

### Паттерн B: SetCheckInfo — опциональная проверка

```cpp
public:
    void SetCheckInfo(bool enabled) { check_info_ = enabled; }

private:
    bool check_info_ = true;  // safe по умолчанию

    void RunPipeline() {
        KernelA();
        KernelB();
        if (check_info_) CheckErrors();  // sync только если нужно
    }
```

### Паттерн C: Stage Profiling Test — шаблон

```cpp
// Для каждого модуля создать test_stage_profiling.hpp:
//
// 1. hipEvent между каждым этапом pipeline
// 2. Два варианта: "текущий код" vs "оптимизированный"
// 3. Warmup + N замеров
// 4. Вывод через ConsoleOutput (con.Print)
// 5. Расчёт ЭКОНОМИЯ = current - optimized
//
// См. образец: modules/vector_algebra/tests/test_stage_profiling.hpp
```

### Паттерн D: Предаллокация dev_info для rocSOLVER

```cpp
// Если модуль использует rocSOLVER (potrf, potri, getrf, etc.):
private:
    void* d_info_ = nullptr;  // rocblas_int на GPU

// Конструктор:
    rocblas_int* ptr = nullptr;
    hipMalloc(&ptr, N_SLOTS * sizeof(rocblas_int));
    d_info_ = ptr;

// Деструктор:
    if (d_info_) { (void)hipFree(d_info_); d_info_ = nullptr; }

// Использование:
    auto* info = static_cast<rocblas_int*>(d_info_) + slot;
    rocsolver_cpotrf(h, fill, n, A, n, info);
    // БЕЗ hipMemcpy D2H здесь! Проверять в CheckInfo() после pipeline.
```

---

## 🧪 Продвинутые техники

### 1. HIP Graph Capture (экспериментально)

Для повторяющихся pipeline с фиксированными размерами:

```cpp
hipGraph_t graph;
hipGraphExec_t exec;

// Записать один раз:
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    hipMemcpyAsync(d_out, d_in, size, hipMemcpyDeviceToDevice, stream);
    KernelA<<<grid, block, 0, stream>>>();
    KernelB<<<grid, block, 0, stream>>>();
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// Запускать многократно (минимальный launch overhead):
hipGraphLaunch(exec, stream);
```

⚠️ rocSOLVER может не поддерживать Graph Capture (внутренние sync'и).

### 2. Strided Batched API (rocSOLVER)

Вместо цикла `for k: rocsolver_cpotrf(A[k])` — один вызов:

```cpp
rocsolver_cpotrf_strided_batched(
    handle, fill, n,
    A,           // указатель на первую матрицу
    lda,
    stride_A,    // n*n * sizeof(element) — шаг между матрицами
    info,
    batch_count);
```

Это позволяет rocSOLVER оптимизировать запуск kernel'ов для batch.

### 3. hipMallocAsync (ROCm 5.6+)

Pool-based аллокатор с минимальным overhead:

```cpp
hipMemPool_t pool;
hipDeviceGetDefaultMemPool(&pool, device);

// Async аллокация из пула (~1 мкс vs ~30 мкс hipMalloc):
hipMallocAsync(&ptr, size, stream);
// ... use ptr ...
hipFreeAsync(ptr, stream);
```

⚠️ Проверить поддержку на gfx1201 (RDNA4).

### 4. Pinned Host Memory для async D2H

```cpp
// Предаллокация pinned memory (один раз):
void* h_pinned;
hipHostMalloc(&h_pinned, size, hipHostMallocDefault);

// Async copy (не блокирует CPU!):
hipMemcpyAsync(h_pinned, d_data, size, hipMemcpyDeviceToHost, stream);

// Проверить когда нужно:
hipStreamSynchronize(stream);
auto* result = static_cast<int*>(h_pinned);
```

---

## 📋 Приоритет оптимизации модулей

| # | Модуль | Проблема | Overhead/call | Приоритет |
|---|--------|----------|--------------|-----------|
| 1 | **fft_maxima (AllMaxima)** | 6×malloc+6×free | ~240–1200 мкс | 🔴 HIGH |
| 2 | **lch_farrow** | 3×malloc+3×free | ~120–600 мкс | ⚠️ MEDIUM |
| 3 | **filters (FIR)** | 2×malloc+2×free | ~80–400 мкс | ⚠️ MEDIUM |
| 4 | **filters (IIR)** | 2×malloc+2×free | ~80–400 мкс | ⚠️ MEDIUM |
| 5 | **signal_generators** | 1×malloc+1×free | ~40–200 мкс | ⚠️ MEDIUM |
| 6 | **vector_algebra** | ✅ Оптимизирован | — | ✅ DONE |
| 7 | **fft_processor** | ✅ Предаллокация | — | ✅ OK |
| 8 | **heterodyne** | ✅ Предаллокация | — | ✅ OK |
| 9 | **statistics** | ✅ Предаллокация | — | ✅ OK |

---

## 🔗 Ссылки

| Ресурс | Путь |
|--------|------|
| **Образец оптимизации** | `modules/vector_algebra/src/cholesky_inverter_rocm.cpp` |
| **Stage profiling пример** | `modules/vector_algebra/tests/test_stage_profiling.hpp` |
| **Benchmark пример** | `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp` |
| **Результаты Task_12** | `Results/Profiler/cholesky_benchmark_2026-02-26.md` |
| **Сессия Task_12** | `MemoryBank/sessions/2026-02-26_Task12_Cholesky_Optimize.md` |

---

*Создано: 2026-02-26 | Кодо (AI Assistant)*
*На основе опыта Task_12: Cholesky <1.0ms Optimization*
