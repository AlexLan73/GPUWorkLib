# FFT Maxima (fft_maxima) — Plan оптимизации OpenCL Kernels

> **Модуль**: `modules/fft_maxima`
> **Backend**: OpenCL (AMD Radeon 9070, RDNA4)
> **Основа**: `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`
> **Дата создания**: 2026-02-26
> **Статус**: ✅ COMPLETED (2026-02-26)

---

## Найденные проблемы (по убыванию важности)

### P0 — Критично: Последовательная редукция в post_kernel

**Файл**: `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`
**Функции**: `GetPostKernelSource_TwoPeaks_opencl()`, `GetPostKernelSource_OnePeaks_opencl()`

**Проблема**: Thread-0 выполняет последовательный цикл по всем 256 LDS-элементам:
```c
if (lid == 0) {
    for (int i = 1; i < 256; i++) {
        if (lds_mag[i] > lds_mag[max_idx]) max_idx = i;
    }
}
```
Это O(256) операций на одном треде. Остальные 255 тредов простаивают.

**Ожидаемое ускорение**: 30–60× для этого kernel (O(256) → O(log₂(256) = 8 шагов).

---

### P1-A — Высокий: div/mod в pre-callback (каждый элемент FFT)

**Файл**: `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`
**Функция**: `GetPreCallbackSource32_opencl()`

**Проблема**:
```c
uint beam_id = callbackData / nFFT;   // дорогой div
uint freq    = callbackData % nFFT;   // дорогой mod
```
Вызывается для каждого из `beam_count * nFFT` элементов (например, 256 × 8192 = 2.1M вызовов).

**Исправление**: `nFFT` гарантированно pow2 → заменить на bitwise:
```c
uint nFFT_log2 = nFFT_log2_param;   // передаём как параметр kernel
uint beam_id   = callbackData >> nFFT_log2;
uint freq      = callbackData & (nFFT - 1);
```
**Альтернатива**: Убрать pre-callback, сделать отдельный `pad_kernel` (как в ROCm версии), тогда callback не нужен вовсе.

**Ожидаемое ускорение**: ~10–15% throughput pre-callback.

---

### P1-B — Высокий: div/mod в padding_kernel

**Файл**: `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`
**Функция**: `GetPaddingKernelSource_opencl()`

**Проблема**:
```c
uint beam_id = gid / nFFT;
uint pos     = gid % nFFT;
```
Дорогие операции для каждого треда.

**Исправление**: 2D NDRange + `clEnqueueFillBuffer` для зануления:
```c
// Dispatch: (nFFT, beam_count) 2D
uint beam_id = get_global_id(1);
uint pos     = get_global_id(0);
// Zeros: clEnqueueFillBuffer перед запуском (нет divergent else-branch)
if (pos < n_point)
    output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
```
**Ожидаемое ускорение**: ~5–10% в padding kernel.

---

### P1-C — Высокий: sqrt() → native_sqrt() во всех kernels

**Файлы**: `fft_kernel_sources.hpp`, `all_maxima_kernel_sources.hpp`

**Проблема**: Используется `sqrt()` (IEEE 754 точность) в hot loops вместо `native_sqrt()`.

**Где встречается**:
- `GetPostKernelSource_TwoPeaks_opencl()` — в цикле поиска максимума
- `GetPostKernelSource_OnePeaks_opencl()` — аналогично
- `GetComputeMagnitudesKernelSource_opencl()` — для каждого из beam_count * nFFT элементов
- `GetPostCallbackMagnitudeSource_opencl()` — в callback

**Исправление**: `sqrt(` → `native_sqrt(`

**Ожидаемое ускорение**: 2–4× для операции sqrt. Точность: достаточна для magnitude (single-precision).

---

### P2-A — Средний: LDS bank conflicts в post_kernel

**Файл**: `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`

**Проблема**: `__local float lds_mag[256]` — без padding.

При parallel tree reduction шаги с stride=1,2,4,... попадают в банки с конфликтами.

**Исправление**:
```c
__local float lds_mag[256 + 1];  // +1 padding устраняет bank conflicts
```
**Ожидаемое ускорение**: ~10–20% для reduction шагов.

---

### P2-B — Средний: div/mod в detect_all_maxima

**Файл**: `modules/fft_maxima/include/kernels/all_maxima_kernel_sources.hpp`
**Функция**: `GetDetectAllMaximaKernelSource_opencl()`

**Проблема**:
```c
uint beam_id = gid / nFFT;
uint freq    = gid % nFFT;
```
**Исправление**: 2D NDRange:
```c
// Dispatch: (nFFT, beam_count)
uint beam_id = get_global_id(1);
uint freq    = get_global_id(0);
```
**Ожидаемое ускорение**: ~5–10%.

---

### P2-C — Средний: Атрибут reqd_work_group_size

**Все kernels** с фиксированным WG=256:
```c
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void post_kernel(...)
```
Позволяет компилятору устранить dead-code branches и оптимизировать register allocation.

**Ожидаемое ускорение**: Косвенное (~2–5%), бесплатно.

---

### P2-D — Средний: __restrict на все pointer параметры

Все `__global float* output` → `__global float* restrict output`.

Позволяет компилятору делать более агрессивное ILP (instruction-level parallelism).

**Ожидаемое ускорение**: ~1–3%, бесплатно.

---

### P3-A — Низкий: LDS padding в prefix_sum_kernel

**Файл**: `modules/fft_maxima/include/kernels/all_maxima_kernel_sources.hpp`
**Функция**: `GetPrefixSumKernelSource_opencl()`

**Проблема**: `__local uint lds[512]` без padding при Blelloch scan.

**Исправление**: `__local uint lds[513]`

**Ожидаемое ускорение**: ~5–10% в prefix sum.

---

## Порядок реализации

| # | Задача | Приоритет | Файл | Ожидаемый эффект |
|---|--------|-----------|------|------------------|
| TASK-1 | Параллельная tree-reduction в post_kernel (OnePeak + TwoPeaks) | **P0** | `fft_kernel_sources.hpp` | **30–60×** для post_kernel |
| TASK-2 | native_sqrt во всех kernels | **P1-C** | оба файла | 2–4× sqrt ops |
| TASK-3 | div/mod в pre-callback → bitwise (или убрать callback) | **P1-A** | `fft_kernel_sources.hpp` | 10–15% |
| TASK-4 | padding_kernel → 2D NDRange + clEnqueueFillBuffer | **P1-B** | `fft_kernel_sources.hpp` | 5–10% |
| TASK-5 | LDS padding +1 в post_kernel | **P2-A** | `fft_kernel_sources.hpp` | 10–20% reduction |
| TASK-6 | detect_all_maxima → 2D NDRange | **P2-B** | `all_maxima_kernel_sources.hpp` | 5–10% |
| TASK-7 | reqd_work_group_size(256,1,1) везде | **P2-C** | оба файла | косвенное |
| TASK-8 | __restrict на все pointer params | **P2-D** | оба файла | 1–3% |
| TASK-9 | LDS padding +1 в prefix_sum_kernel | **P3-A** | `all_maxima_kernel_sources.hpp` | 5–10% |

---

## Файлы для изменения

```
modules/fft_maxima/include/kernels/
├── fft_kernel_sources.hpp           ← TASK-1, 3, 4, 5, 7, 8 (+ TASK-2 sqrt)
└── all_maxima_kernel_sources.hpp    ← TASK-2 sqrt, 6, 7, 8, 9
```

---

## Суммарные ожидаемые ускорения

| Этап | Ускорение pipeline | Что улучшилось |
|------|-------------------|----------------|
| После TASK-1 | **~2–5× общий** | post_kernel 30-60× (доминирующий bottleneck) |
| + TASK-2,3,4 | + ~25–35% | sqrt, div/mod |
| + TASK-5–8 | + ~15–20% | bank conflicts, compiler hints |
| Итого | **~3–8× общий** | относительно исходного |

---

## Ключевые паттерны реализации

### Parallel tree reduction (TASK-1)

```c
// Parallel reduction: 256 → 128 → 64 → ... → 1
for (uint stride = 128; stride > 0; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < stride) {
        if (lds_mag[lid + stride] > lds_mag[lid]) {
            lds_mag[lid] = lds_mag[lid + stride];
            lds_idx[lid] = lds_idx[lid + stride];
        }
    }
}
barrier(CLK_LOCAL_MEM_FENCE);
// thread-0 читает lds_mag[0], lds_idx[0] — готов результат
```

### 2D NDRange dispatch (C++ сторона)

```cpp
// Вместо 1D: (beam_count * nFFT_, 1, 1)
// Использовать 2D: (nFFT_, beam_count, 1)
size_t global[2] = { nFFT_, beam_count };
size_t local[2]  = { 256, 1 };
clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, ...);
```

---

*Создан: 2026-02-26*
*Автор: Кодо (AI Assistant)*
*Источник: sequential-thinking анализ + Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md*
