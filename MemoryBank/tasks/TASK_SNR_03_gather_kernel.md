# TASK SNR_03: HIP kernel `gather_decimated_kernel`

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/kernels/`
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: — (параллельно с SNR_00, SNR_01, SNR_02)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.2** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Написать HIP kernel `gather_decimated_kernel` — вырезает подвыборку `[n_ant_out × n_samp_out]` из исходной `[n_antennas × n_samples]` матрицы с шагами `step_antennas` и `step_samples`.

**Thread mapping: один поток = одна антенна, sequential loop по samples** (не «один поток на элемент»!). Это ключевое решение — даёт L2 prefetcher эффективное чтение вдоль строки.

---

## 📁 Файл (создать)

```
modules/statistics/kernels/gather_decimated_kernel.hpp
```

> ⚠️ **Имя функции в kernel source: `gather_decimated`** (без суффикса `_kernel`).
> SNR_05 должен вызывать `kernel("gather_decimated")` — имена должны совпадать!
> В плане (раздел 2.2) может быть написано `gather_decimated_kernel` — это описательное название файла,
> но сама GPU-функция называется `gather_decimated` (как `peak_cfar`, не `peak_cfar_kernel`).

---

## 📝 Реализация (точный код из плана раздел 2.2)

```cpp
#pragma once

/**
 * @file gather_decimated_kernel.hpp
 * @brief HIP kernel source для gather_decimated (SNR-estimator)
 *
 * Вырезает подвыборку из 2D complex float матрицы с шагами step_antennas
 * и step_samples. Thread mapping: один поток = одна антенна, sequential loop
 * по samples — L2 prefetcher видит линейный паттерн.
 *
 * @author ...
 * @date 2026-04-09
 */

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

/**
 * @brief HIP kernel source: gather_decimated
 *
 * Launch: grid(ceil(n_ant_out/64), 1), block(64, 1)
 *
 * Thread mapping (КРИТИЧНО!):
 *   - 1 thread = 1 output antenna
 *   - Sequential loop inside thread: for (s = 0; s < n_samp_out; ++s)
 *   - Reads: src[ant*step_ant*n_samples + s*step_samples]
 *   - Writes: dst[ant*n_samp_out + s]
 *
 * Почему НЕ «поток на элемент»:
 *   При step_samples > 8 (stride > 64 байт = cache line) соседние потоки
 *   варпа читают из разных cache line'ов → ×32 amplification memory txns.
 *   Sequential loop внутри потока → L2 prefetcher префетчит вдоль строки.
 */
inline const char* GetGatherDecimatedKernelSource() {
    return R"HIP(

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

struct float2_t {
    float x;
    float y;
};

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void gather_decimated(
    const float2_t* __restrict__ src,   // [n_antennas × n_samples], row-major
    float2_t* __restrict__ dst,          // [n_ant_out × n_samp_out], row-major
    unsigned int n_samples,               // ширина исходной матрицы
    unsigned int n_samp_out,              // ширина выходной матрицы
    unsigned int step_antennas,           // шаг по строкам
    unsigned int step_samples,            // шаг по столбцам
    unsigned int n_ant_out)               // число выходных антенн
{
    unsigned int ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant_out) return;

    const float2_t* src_row =
        src + (size_t)ant * step_antennas * n_samples;
    float2_t* dst_row =
        dst + (size_t)ant * n_samp_out;

    // Sequential loop — L2 prefetcher видит линейный паттерн
    for (unsigned int s = 0; s < n_samp_out; ++s) {
        dst_row[s] = src_row[(size_t)s * step_samples];
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
```

---

## ✅ Definition of Done

- [ ] Файл `gather_decimated_kernel.hpp` создан в `modules/statistics/kernels/`
- [ ] Содержит inline функцию `GetGatherDecimatedKernelSource()` возвращающую kernel source как HIP raw string
- [ ] Kernel функция `gather_decimated` с сигнатурой из плана
- [ ] Thread mapping: **один поток на антенну, sequential loop** (не один поток на элемент)
- [ ] `size_t` cast для индексов (чтобы не переполнилось при больших n_samples × n_antennas)
- [ ] `__launch_bounds__(64)` + `__restrict__`
- [ ] Код встроенного kernel компилируется на Debian (в понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Thread mapping **один поток на антенну** (НЕ один поток на выходной элемент)
- ✅ `for (unsigned int s = 0; s < n_samp_out; ++s)` — sequential loop внутри потока
- ✅ `size_t` cast для offset: `(size_t)ant * step_antennas * n_samples` — защита от переполнения uint32 при `n_antennas × n_samples > 4G`
- ✅ `float2_t` struct определён внутри source (как в соседних kernel'ах)
- ✅ `extern "C"` — для hiprtc lookup по имени: `kernel("gather_decimated")` (без `_kernel`!)
- ✅ `__launch_bounds__(BLOCK_SIZE)` + `#define BLOCK_SIZE 64`
- ✅ Нет `if (blockIdx.y ... )` — это 1D grid, не 2D!
- ✅ Нет глобальных переменных в kernel

---

## 🚫 Запреты

- ❌ **НЕ** использовать 2D thread mapping (`grid(ceil(n_samp_out/32), n_ant_out)`) — это приведёт к non-coalesced чтению
- ❌ НЕ писать kernel в .hip файл — используем inline string source для hiprtc (как в соседних модулях statistics/capon/fft_func)
- ❌ НЕ вызывать `hipfftExecC2C` из этого kernel (это только gather!)

---

## 📝 Заметки

**Почему sequential loop эффективный для сценария B:**
- n_samples = 1.3M, step_samples = 635
- 43 антенны × 2047 итераций = ~88K чтений
- Каждое чтение — 8 байт из разных мест (random access)
- НО: внутри одной антенны — читаем вдоль строки, stride постоянный (635 × 8 = 5080 байт)
- L2 prefetcher может распознать arithmetic progression и префетчить

**Оценка времени:**
- Сценарий B: ~1-5 ms (memory-bound, 2.8 MB / 700 GB/s + latency)
- Сценарий C: ~2-8 ms (9000 → 180 × 2000 samples)

Замер — в [TASK_SNR_09](TASK_SNR_09_benchmark.md).

---

## 🔗 Связанные таски

- **Блокирует:** [TASK_SNR_05](TASK_SNR_05_snr_estimator_op.md) — `SnrEstimatorOp` использует этот kernel
- **Параллельно:** SNR_00, SNR_01, SNR_02, SNR_04

---

*Created 2026-04-09 | Кодо*
