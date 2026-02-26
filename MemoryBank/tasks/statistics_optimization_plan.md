# 📐 План оптимизации модуля `statistics`

> **Статус**: ✅ РЕАЛИЗОВАНО (2026-02-26)
> **Дата**: 2026-02-26
> **Источники**: ROCm Performance Guidelines, HIP Reduction Tutorial, AMD Lab Notes,
> `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`, анализ кода `modules/statistics`

---

## 🔍 Диагностика: найденные проблемы

### ❌ КРИТИЧЕСКИЕ (P0)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P0-A | **Двойное чтение `input_buffer_`** в `ComputeStatistics`: `compute_magnitudes` читает input → `welford_stats` читает input СНОВА + magnitudes. Итого 3× pass по данным вместо 1×. | `statistics_kernels_rocm.hpp` | 72-283 |
| P0-B | **256 отдельных `hipMemcpyDtoH` в `ComputeMedian`**: цикл по лучам, каждый — 4 байта = 256 PCIe транзакций. Катастрофически медленно на 256 лучах. | `statistics_processor.cpp` | 247-262 |
| P0-C | **`hiprtcCompileProgram` без `--offload-arch`**: нет кеширования HSACO. Каждый запуск программы = ~100-200 мс перекомпиляция 4 ядер. | `statistics_processor.cpp` | 411 |

### ⚠️ ВЫСОКИЕ (P1)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P1-A | **Нет warp shuffle** в финальных стадиях tree reduction. Последние 5 шагов (s≤32) идут через LDS + `__syncthreads` — медленнее, чем `__shfl_down`. | `statistics_kernels_rocm.hpp` | 124-131, 253-260 |
| P1-B | **Нет double-load** в `mean_reduce_phase1`: каждый поток обрабатывает 1 элемент. Простое изменение → вдвое меньше блоков → вдвое меньше partial sums. | `statistics_kernels_rocm.hpp` | 110-121 |
| P1-C | **Нет `__launch_bounds__`** ни в одном из 4 ядер. Компилятор резервирует регистры для blockSize=1024 вместо реальных 256 → меньше occupancy. | `statistics_kernels_rocm.hpp` | 72, 91, 144, 207 |
| P1-D | **`div + mod` в `mean_reduce_phase1` внутри ядра**: `blocks_per_beam = (n_point + block_size - 1) / block_size` + `beam_id = blockIdx.x / blocks_per_beam` вычисляется КАЖДЫМ потоком. Integer division на GPU — дорогая операция. | `statistics_kernels_rocm.hpp` | 104-106 |

### ℹ️ СРЕДНИЕ (P2)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P2-A | `sqrtf` → `__fsqrt_rn` (fast intrinsic, ~2 ULP) в `compute_magnitudes` и `welford_fused` | `statistics_kernels_rocm.hpp` | 81, 279 |
| P2-B | Синхронный `hipMemcpy` для заполнения `offsets_buf_` в `AllocateBuffers` → нужен `hipMemcpyAsync` | `statistics_processor.cpp` | 499-505 |

---

## 📋 ПЛАН — 5 задач по приоритету

---

### TASK-1 🔴 КРИТИЧЕСКАЯ — Kernel Fusion: `welford_fused`

**Проблема**: `ComputeStatistics` выполняет 2 kernel passes по одним данным:
```
compute_magnitudes:  reads input[i]  → writes magnitudes[i]   (1× input)
welford_stats:       reads input[i]  → re/im sum              (2× input!)
                     reads magnitudes[i] → mag/sq sum
```

**Решение**: Новое ядро `welford_fused` — **один pass**, вычисляет |z| на лету:

```cpp
// НОВОЕ ЯДРО: welford_fused
// Один проход: читает только input[], всё вычисляет сам
// compute_magnitudes для ComputeStatistics пути УДАЛЯЕТСЯ

__launch_bounds__(256)
extern "C" __global__ void welford_fused(
    const float2_t* __restrict__ input,   // только входной буфер!
    WelfordResult* __restrict__ results,
    unsigned int beam_count,
    unsigned int n_point)
{
    unsigned int beam_id = blockIdx.x;
    if (beam_id >= beam_count) return;

    unsigned int tid = threadIdx.x;
    unsigned int base = beam_id * n_point;

    float sum_re = 0.0f, sum_im = 0.0f, sum_mag = 0.0f, sum_sq = 0.0f;

    // Grid-stride loop — каждый поток читает input ОДИН РАЗ
    for (unsigned int i = tid; i < n_point; i += blockDim.x) {
        float2_t z = input[base + i];
        // |z| вычисляется здесь — НЕТ промежуточного буфера!
        float mag = __fsqrt_rn(z.x * z.x + z.y * z.y);
        sum_re  += z.x;
        sum_im  += z.y;
        sum_mag += mag;
        sum_sq  += mag * mag;
    }

    // ... LDS reduction + warp shuffle (см. TASK-4) ...
}
```

**Изменения в `statistics_processor.cpp`**:
- `ExecuteWelfordKernel()` → `ExecuteWelfordFusedKernel()` (аргумент `magnitudes_buf_` убирается)
- `ComputeStatistics` pipeline: `UploadData → ExecuteWelfordFusedKernel → sync → DtoH`
- `magnitudes_buf_` остаётся (нужен для `ComputeMedian` path)

**Ожидаемый эффект**: **-40–50%** времени `ComputeStatistics`

---

### TASK-2 🔴 КРИТИЧЕСКАЯ — Исправить `ComputeMedian`: 1 DtoH вместо 256

**Проблема** в `statistics_processor.cpp:247-262`:
```cpp
// ТЕКУЩИЙ КОД — 256 отдельных PCIe транзакций:
for (uint32_t b = 0; b < params.beam_count; ++b) {
    float median_val = 0.0f;
    size_t mid_idx = b * params.n_point + params.n_point / 2;
    hipMemcpyDtoH(&median_val,                           // ← 4 байта каждый!
        static_cast<char*>(sort_buf_) + mid_idx * sizeof(float),
        sizeof(float));
    ...
}
```

**Решение A — GPU kernel `extract_medians`** (предпочтительнее):
```cpp
// Новое ядро: извлекает средний элемент каждого луча в компактный массив
__launch_bounds__(256)
extern "C" __global__ void extract_medians(
    const float* __restrict__ sorted,    // sort_buf_ после rocPRIM sort
    float* __restrict__ medians,         // compact output: beam_count floats
    unsigned int n_point,
    unsigned int beam_count)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= beam_count) return;
    medians[b] = sorted[b * n_point + n_point / 2];
}
```

Добавляем `medians_compact_buf_` (beam_count floats) в буфера.

**Новый pipeline `ComputeMedian`**:
```
UploadData → ExecuteMagnitudesKernel → ExecuteMedianSort →
  extract_medians<<<ceil(bc/256), 256>>> →   ← НОВОЕ ядро (1 блок)
  hipStreamSynchronize →
  hipMemcpyDtoH(medians_compact, beam_count * sizeof(float))  ← 1 копия!
```

**Ожидаемый эффект**: **-90%+** времени D2H в `ComputeMedian` (256→1 транзакций)

---

### TASK-3 🟡 ВЫСОКАЯ — hiprtc: arch flag + HSACO disk cache

**Проблема**: `CompileKernels()` строка 411:
```cpp
rtcResult = hiprtcCompileProgram(prog, 0, nullptr);  // НЕТ флагов!
```

**Решение**:

```cpp
void StatisticsProcessor::CompileKernels() {
    if (kernels_compiled_) return;

    // Шаг 1: Попытка загрузки из кеша (как в vector_algebra)
    if (TryLoadFromCache()) {
        kernels_compiled_ = true;
        return;
    }

    // Шаг 2: JIT компиляция с явным arch
    // Определяем arch динамически из backend
    std::string arch_flag = "--offload-arch=" + backend_->GetTargetArch();
    // Например: "--offload-arch=gfx1201" для Radeon 9070

    const char* opts[] = {
        "-O3",
        arch_flag.c_str(),
        "-std=c++17",
        // WARP_SIZE через define (для warp shuffle в ядрах)
        "-DWARP_SIZE=32"    // или 64 для CDNA
    };
    rtcResult = hiprtcCompileProgram(prog, 4, opts);

    // Шаг 3: Сохранить HSACO в cache
    SaveToCache(binary);
}
```

**Использовать `KernelCacheService`** (уже есть в проекте — см. `vector_algebra`):
- Кеш: `modules/statistics/kernels/bin/statistics_kernels.hsaco`
- Ключ: arch + source hash
- Холодный старт: ~100-200мс; горячий: ~1-5мс

**Ожидаемый эффект**: **-100-200мс** на первый вызов (только startup latency)

---

### TASK-4 🟡 ВЫСОКАЯ — Оптимизация tree reduction ядер

Применяем к `mean_reduce_phase1`, `mean_reduce_final`, `welford_fused` (из TASK-1):

#### 4.1 — `__launch_bounds__(256)` — ко всем ядрам

```cpp
// Было:
extern "C" __global__ void mean_reduce_phase1(...)

// Стало:
__launch_bounds__(256)
extern "C" __global__ void mean_reduce_phase1(...)
```
Эффект: компилятор знает реальный blockSize → правильный резерв регистров → выше occupancy.

#### 4.2 — Убрать div/mod из `mean_reduce_phase1` — передать как параметр

```cpp
// Было (в ядре — дорогой div+mod):
unsigned int blocks_per_beam = (n_point + block_size - 1) / block_size;  // DIVISION!
unsigned int beam_id = blockIdx.x / blocks_per_beam;                       // DIVISION!
unsigned int block_in_beam = blockIdx.x % blocks_per_beam;                 // MOD!

// Стало (параметр передаётся из CPU):
extern "C" __global__ void mean_reduce_phase1(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ partial_sums,
    unsigned int beam_count,
    unsigned int n_point,
    unsigned int blocks_per_beam)   // ← новый параметр!
{
    unsigned int beam_id      = blockIdx.x / blocks_per_beam;  // всё ещё div...
    // ЛУЧШЕ: blocks_per_beam = pow2 → компилятор заменяет на shift:
    // → передавать log2(blocks_per_beam) и beam_id = blockIdx.x >> log2_bpb
}
```

#### 4.3 — Double-load trick в `mean_reduce_phase1`

```cpp
// Было: 1 поток = 1 элемент (256 threads → 256 элементов/блок)
unsigned int local_idx = block_in_beam * block_size + tid;
if (local_idx < n_point) val = input[global_idx];
sdata[tid] = val;

// Стало: 1 поток = 2 элемента (256 threads → 512 элементов/блок!)
unsigned int gid1 = beam_start + block_in_beam * (block_size * 2) + tid;
unsigned int gid2 = gid1 + block_size;
float2_t v1 = (gid1 - beam_start < n_point) ? input[gid1] : zero;
float2_t v2 = (gid2 - beam_start < n_point) ? input[gid2] : zero;
sdata[tid] = {v1.x + v2.x, v1.y + v2.y};  // сразу суммируем!
```
Эффект: вдвое меньше блоков → вдвое меньше partial sums → phase2 быстрее.

#### 4.4 — Warp Shuffle в финальной стадии (RDNA4: warpSize=32)

```cpp
// В kernel source добавить define:
// #define WARP_SIZE 32   (через -DWARP_SIZE=32 в opts)

// Было (все шаги через LDS + __syncthreads):
for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) { sdata[tid].x += sdata[tid+s].x; ... }
    __syncthreads();
}

// Стало (межвейвфронтовые шаги через LDS, последний вейвфронт через shuffle):
for (unsigned int s = block_size / 2; s > WARP_SIZE; s >>= 1) {
    if (tid < s) { sdata[tid].x += sdata[tid+s].x; ... }
    __syncthreads();
}
// Финальный вейвфронт — через warp shuffle (НЕТ __syncthreads!):
if (tid < WARP_SIZE) {
    float vx = sdata[tid].x, vy = sdata[tid].y;
    vx += __shfl_down(vx, 16); vy += __shfl_down(vy, 16);
    vx += __shfl_down(vx, 8);  vy += __shfl_down(vy, 8);
    vx += __shfl_down(vx, 4);  vy += __shfl_down(vy, 4);
    vx += __shfl_down(vx, 2);  vy += __shfl_down(vy, 2);
    vx += __shfl_down(vx, 1);  vy += __shfl_down(vy, 1);
    if (tid == 0) { sdata[0].x = vx; sdata[0].y = vy; }
}
// Для welford_fused: аналогично для sum_re, sum_im, sum_mag, sum_sq
```

#### 4.5 — `__fsqrt_rn` вместо `sqrtf`

```cpp
// Было:
magnitudes[gid] = sqrtf(z.x * z.x + z.y * z.y);
r.std_dev = sqrtf(r.variance);

// Стало (~4 ULP, достаточно для статистики):
magnitudes[gid] = __fsqrt_rn(z.x * z.x + z.y * z.y);
r.std_dev = __fsqrt_rn(r.variance);
```

**Суммарный ожидаемый эффект TASK-4**: **~15–25%** ускорение mean/welford ядер

---

### TASK-5 🟢 СРЕДНЯЯ — Мелкие улучшения

#### 5.1 — `hipMemcpyAsync` в `AllocateBuffers`

```cpp
// Было (СИНХРОННАЯ копия!):
err = hipMemcpy(offsets_buf_, host_offsets.data(), ..., hipMemcpyHostToDevice);

// Стало (асинхронная в stream_):
err = hipMemcpyAsync(offsets_buf_, host_offsets.data(), ...,
                     hipMemcpyHostToDevice, stream_);
// AllocateBuffers вызывается до ExecuteXxx — stream порядок гарантирован
```

#### 5.2 — Проверка float литералов в ядрах

Текущий код: все литералы уже `0.0f`, `1.0f` — **OK**. Нет скрытых double.

---

## 📊 Ожидаемый суммарный эффект

| Операция | Текущее | После оптимизации | Ускорение |
|----------|---------|-------------------|-----------|
| `ComputeStatistics` | baseline | TASK-1 + TASK-4 | **~2× быстрее** |
| `ComputeMedian` D2H | 256 транзакций | 1 транзакция | **~90–95% быстрее** |
| `ComputeMean` | baseline | TASK-4 | **~20–25% быстрее** |
| Первый запуск | +100-200ms | cold: +5ms | **-95%** startup |

---

## 📁 Затронутые файлы

| Файл | Задача | Тип изменений |
|------|--------|---------------|
| `modules/statistics/include/kernels/statistics_kernels_rocm.hpp` | TASK-1, TASK-2, TASK-4 | Новые ядра, оптимизация существующих |
| `modules/statistics/src/statistics_processor.cpp` | TASK-1, TASK-2, TASK-3, TASK-5 | Новые вызовы, кеширование, исправление DtoH |

---

## ✅ Порядок выполнения (рекомендуемый)

```
TASK-2 (ComputeMedian DtoH) → TASK-1 (welford_fused) →
  TASK-4 (tree reduction) → TASK-3 (hiprtc cache) → TASK-5 (mелочи)
```

Начать с TASK-2 — самое простое и максимальный эффект на 256 лучах.

---

*Составил: Кодо | Источники: ROCm HIP Tutorial, AMD Lab Notes, Context7, sequential-thinking*
