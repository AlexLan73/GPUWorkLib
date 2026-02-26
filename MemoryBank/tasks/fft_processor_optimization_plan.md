# 📐 План оптимизации модуля `fft_processor`

> **Статус**: ✅ РЕАЛИЗОВАНО (2026-02-26)
> **Дата**: 2026-02-26
> **Источники**: `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`, анализ кода `modules/fft_processor`, паттерны из `modules/statistics` и `modules/vector_algebra`

---

## 🔍 Диагностика: найденные проблемы

### ❌ КРИТИЧЕСКИЕ (P0)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P0-A | **`hiprtcCompileProgram(prog, 0, nullptr)` — нет флагов**: нет `-O3`, нет `--offload-arch`, нет `-DWARP_SIZE`. Каждый запуск приложения = ~50-150 мс перекомпиляция 2 ядер без оптимизаций. | `fft_processor_rocm.cpp` | 481 |
| P0-B | **Нет HSACO disk cache** (`KernelCacheService`): в отличие от `statistics` и `vector_algebra`, нет кеширования скомпилированного бинаря. Каждый запуск JIT-компилирует заново. | `fft_processor_rocm.cpp` | 466-530 |

### ⚠️ ВЫСОКИЕ (P1)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P1-A | **Expensive integer div/mod в `pad_data`**: `beam_id = gid / nFFT` и `pos = gid % nFFT` вычисляются каждым потоком (~20 тактов каждый). При 256 лучей × 65536 точек = 16M потоков, каждый делает 2 целочисленных деления. | `fft_processor_kernels_rocm.hpp` | 53-54 |
| P1-B | **Thread divergence в `pad_data`** на границе `n_point`: ветка `else { output[gid] = zero }` выполняется для `(nFFT - n_point)` потоков в каждом луче. Лучший паттерн: `hipMemsetAsync` всего `fft_input_`, затем копировать только n_point. | `fft_processor_kernels_rocm.hpp` | 56-63 |
| P1-C | **Два отдельных `hipMemcpyDtoH` в `ReadMagPhaseResults`**: сначала mag (beam_count × nFFT × 4 байт), потом phase (ещё столько же). Два синхронных DtoH вместо одного interleaved копирования. | `fft_processor_rocm.cpp` | 703-714 |
| P1-D | **`hipfftPlan1d` уничтожается/пересоздаётся** при изменении `batch_count`: последний батч часто меньше, что вызывает Destroy + Create. Создание плана hipFFT — дорогая операция (~мс). | `fft_processor_rocm.cpp` | 433-464 |

### ℹ️ СРЕДНИЕ (P2)

| # | Проблема | Файл | Строки |
|---|---------|------|--------|
| P2-A | **Нет `__launch_bounds__(256)`** ни на `pad_data`, ни на `complex_to_mag_phase`. Компилятор резервирует регистры на худший случай → меньше occupancy. | `fft_processor_kernels_rocm.hpp` | 42, 66 |
| P2-B | **`sqrtf` → `__fsqrt_rn`** в `complex_to_mag_phase`. Fast HIP intrinsic, ~4 ULP точность, заметно быстрее на RDNA4. | `fft_processor_kernels_rocm.hpp` | 78 |
| P2-C | **`atan2f` → `__atan2f`** в `complex_to_mag_phase`. Аналогично — fast intrinsic версия трансцендентной функции. | `fft_processor_kernels_rocm.hpp` | 79 |
| P2-D | **Нет double buffering** для multi-batch: пока GPU работает над батчем N, CPU мог бы заливать батч N+1. Сейчас — строго последовательно: upload → pad → FFT → download → ... | `fft_processor_rocm.cpp` | 563-660 |

---

## 📋 ПЛАН — 5 задач по приоритету

---

### TASK-1 🔴 КРИТИЧЕСКАЯ — Compile flags + HSACO cache

**Проблема**: `CompileKernels()` компилирует без каких-либо флагов (`0, nullptr`) и не кеширует результат:
```cpp
// СЕЙЧАС: строка 481
rtcResult = hiprtcCompileProgram(prog, 0, nullptr);  // NO -O3, no arch!
// + нет KernelCacheService
```

**Решение**:

**Шаг 1**: Добавить в заголовок [fft_processor_rocm.hpp](modules/fft_processor/include/fft_processor_rocm.hpp) forward declaration и поле:
```cpp
// Forward declaration
namespace drv_gpu_lib { class KernelCacheService; }

// В private секции класса:
std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;
```

**Шаг 2**: Добавить include'ы в [fft_processor_rocm.cpp](modules/fft_processor/src/fft_processor_rocm.cpp):
```cpp
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#include "services/kernel_cache_service.hpp"
```

**Шаг 3**: Инициализировать в конструкторе:
```cpp
kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
    "modules/fft_processor/kernels");
```

**Шаг 4**: Перестроить `CompileKernels()` по паттерну `vector_algebra`:
```cpp
void FFTProcessorROCm::CompileKernels() {
    if (kernels_compiled_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    constexpr const char* kCacheName = "fft_processor_kernels";

    // ─── Попытка загрузить из HSACO cache ──────────────────────────────
    if (kernel_cache_) {
        try {
            auto entry = kernel_cache_->Load(kCacheName);
            if (entry.has_binary()) {
                LoadModuleAndFunctions(entry.binary.data(), entry.binary.size());
                kernels_compiled_ = true;
                con.Print(0, "FFTProc", "kernels loaded from HSACO cache");
                return;
            }
        } catch (...) { /* cache miss — компилируем */ }
    }

    // ─── Получить arch name ─────────────────────────────────────────────
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    std::string arch = rocm_backend->GetCore().GetArchName();  // "gfx1201"
    std::string arch_flag = "--offload-arch=" + arch;

    // ─── Компиляция с флагами ───────────────────────────────────────────
    const char* options[] = {
        "-O3",
        arch_flag.c_str(),
        "-DWARP_SIZE=32"
    };
    rtcResult = hiprtcCompileProgram(prog, 3, options);
    // ... (error handling) ...

    // ─── Сохранить в cache ──────────────────────────────────────────────
    if (kernel_cache_) {
        try {
            std::vector<uint8_t> binary(code.begin(), code.end());
            kernel_cache_->Save(kCacheName, src, binary,
                                "", "fft_processor hiprtc kernels");
            con.Print(0, "FFTProc", "kernels saved to HSACO cache");
        } catch (const std::exception& e) {
            con.Print(0, "FFTProc",
                      "warning: cache save failed: " + std::string(e.what()));
        }
    }
}
```

**Ожидаемый эффект**:
- Первый запуск: компиляция с `-O3 --offload-arch=gfx1201 -DWARP_SIZE=32`
- Повторные запуски: ~0 мс (load from disk) вместо ~50-150 мс

---

### TASK-2 🔴 КРИТИЧЕСКАЯ — Устранение div/mod в `pad_data` (2D grid)

**Проблема**: каждый поток выполняет 2 целочисленных деления:
```c
// СЕЙЧАС: строки 53-54 в kernels
unsigned int beam_id = gid / nFFT;   // ~20 тактов на GPU
unsigned int pos     = gid % nFFT;   // ~20 тактов на GPU
```

**Решение**: перейти на 2D grid, где `blockIdx.y == beam_id`:
```c
// В kernels (fft_processor_kernels_rocm.hpp):
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int beam_id = blockIdx.y;                    // NO div!
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= nFFT) return;

    unsigned int out_idx = beam_id * nFFT + pos;
    if (pos < n_point) {
        output[out_idx] = input[beam_id * n_point + pos];
    } else {
        float2_t zero; zero.x = 0.0f; zero.y = 0.0f;
        output[out_idx] = zero;
    }
}
```

**В `ExecutePadKernel()` в cpp**:
```cpp
// beam_count по оси Y, nFFT по оси X:
unsigned int block_x = 256;
unsigned int grid_x = (nFFT_ + block_x - 1) / block_x;
unsigned int grid_y = static_cast<unsigned int>(beam_count);

void* args[] = { &input_buffer_, &fft_input_, &np, &nfft };

hipModuleLaunchKernel(pad_kernel_,
    grid_x, grid_y, 1,    // grid: X=nFFT blocks, Y=beam_count
    block_x, 1, 1,        // block
    0, stream_, args, nullptr);
```

> **Примечание**: убрать `beam_count` из сигнатуры ядра (теперь он в `blockIdx.y`).

**Ожидаемый эффект**: ~20-30% ускорение `pad_data` (устранение 2 int div/thread).

---

### TASK-3 🟠 ВЫСОКАЯ — hipMemsetAsync trick: устранение thread divergence

**Проблема**: ~`(nFFT - n_point)` потоков на луч выполняют ветку `else { zero }`.
При `nFFT=65536, n_point=50000` — 15536 потоков/луч в divergent branch.

**Решение**: `hipMemsetAsync` всего `fft_input_` нулями перед копированием данных.
Убрать zero-branch из ядра совсем:

**В cpp — `ExecutePadKernel()`**:
```cpp
// Шаг 1: обнулить весь fft_input_ асинхронно (быстро на GPU)
hipMemsetAsync(fft_input_,
               0,
               beam_count * nFFT_ * sizeof(std::complex<float>),
               stream_);

// Шаг 2: скопировать только n_point элементов на луч (strided D2D)
// Вариант A: оставить pad_data ядро без zero-ветки:
//   if (pos < n_point) output[out_idx] = input[...];
//   else return;  // fft_input_ уже нулевой!
```

**В ядре** (упрощённая версия после TASK-2 + TASK-3):
```c
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point)                    // nFFT убран — не нужен для guard
{
    unsigned int beam_id = blockIdx.y;
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;              // только valid points

    output[beam_id * /*nFFT*/ + pos] = input[beam_id * n_point + pos];
    // zeros уже расставлены hipMemsetAsync
}
```

> **Внимание**: TASK-3 зависит от TASK-2 (нужен 2D grid). Реализовывать вместе.

**Ожидаемый эффект**: устранение divergence + ядро стало читать/писать только `n_point` элементов (не `nFFT`), при `n_point << nFFT` — значительное ускорение.

---

### TASK-4 🟠 ВЫСОКАЯ — Объединённый DtoH для MagPhase + fix hipfftPlan

**Проблема A** — два DtoH в `ReadMagPhaseResults()` (строки 703-714):
```cpp
// СЕЙЧАС: 2 синхронных DtoH транзакции
hipMemcpyDtoH(raw_mag.data(),   mag_output_,   total * sizeof(float));
hipMemcpyDtoH(raw_phase.data(), phase_output_, total * sizeof(float));
```

**Решение A**: создать interleaved GPU буфер `mag_phase_interleaved_` (`float2_t` × `beam_count × nFFT`),
заполнять его в `complex_to_mag_phase`, затем один DtoH:
```cpp
// В ядре: output float2_t {mag, phase} вместо двух отдельных массивов
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float2_t* __restrict__ mag_phase,        // interleaved!
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    float2_t z = fft_output[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);  // mag (P2-B)
    mp.y = __atan2f(z.y, z.x);                   // phase (P2-C)
    mag_phase[gid] = mp;
}
```

```cpp
// В ReadMagPhaseResults: ОДИН DtoH
std::vector<float2_t_cpu> raw(total);
hipMemcpyDtoH(raw.data(), mag_phase_interleaved_, total * sizeof(float2_t_cpu));
// затем разделить на mag/phase в цикле на CPU (trivial)
```

**Проблема B** — пересоздание плана hipFFT (строки 438-443):
```cpp
// СЕЙЧАС: при последнем батче меньшего размера — Destroy + Create
if (plan_created_) { hipfftDestroy(plan_); ... }
hipfftPlan1d(&plan_, nFFT_, HIPFFT_C2C, batch_beam_count);
```

**Решение B**: создать план с `max_batch_count` (максимальный батч) один раз,
использовать `hipfftPlanMany` с `istride`/`ostride` для управления batch-размером:
```cpp
// Создаём один раз с max_batch_count в AllocateBuffers()
hipfftPlanMany(&plan_,
    1, &nFFT_int,     // rank=1, n=nFFT
    nullptr, 1, nFFT_int,  // inembed, istride, idist
    nullptr, 1, nFFT_int,  // onembed, ostride, odist
    HIPFFT_C2C,
    max_batch_count_);     // максимальный батч — фиксирован
// При меньшем батче: просто обработать меньше данных через padding
```

> Или более простой вариант B2: хранить два плана — `plan_full_` и `plan_last_`
> (создаются при первом вызове с каждым размером батча).

**Ожидаемый эффект**:
- DtoH: один transferт вместо двух → +~5-10% на ReadMagPhase
- hipfftPlan: устранение Destroy/Create при последнем батче → -несколько мс на вызов

---

### TASK-5 🟡 СРЕДНЯЯ — Kernel optimizations: `__launch_bounds__`, intrinsics

**Описание**: набор небольших улучшений в ядрах без изменения архитектуры.

**В `fft_processor_kernels_rocm.hpp`**:

```c
// 1. __launch_bounds__(256) на оба ядра (P2-A)
__launch_bounds__(256)
extern "C" __global__ void pad_data(...) { ... }

__launch_bounds__(256)
extern "C" __global__ void complex_to_mag_phase(...) { ... }

// 2. __fsqrt_rn вместо sqrtf (P2-B)
mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);

// 3. __atan2f вместо atan2f (P2-C)
mp.y = __atan2f(z.y, z.x);

// 4. WARP_SIZE через define (если понадобится warp shuffle позже)
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
```

**Ожидаемый эффект**: `complex_to_mag_phase` — ~15-20% ускорение за счёт intrinsics.
`__launch_bounds__` — лучший occupancy, меньше регистровое давление.

---

## 📊 Ожидаемые результаты

| Задача | Компонент | До | После | Выигрыш |
|--------|-----------|-----|-------|---------|
| TASK-1 | Startup (kernel compile) | ~100-150 мс | ~0 мс | **-95%** |
| TASK-2 | `pad_data` (div/mod) | baseline | -20-30% | **-25%** |
| TASK-3 | `pad_data` (divergence) | baseline | -10-20% | **-15%** |
| TASK-4A | DtoH MagPhase | 2× DtoH | 1× DtoH | **-40%** DtoH latency |
| TASK-4B | hipfftPlan recreate | ~мс/вызов | 0 | **-100%** plan overhead |
| TASK-5 | `complex_to_mag_phase` | sqrtf/atan2f | intrinsics | **-15-20%** |

**Суммарный ожидаемый прирост** (без TASK-1, он только для startup):
- `pad_data` + `complex_to_mag_phase`: **-30-40%** kernel time
- `ReadMagPhaseResults`: **-40%** DtoH latency
- Startup latency: **-95%** (HSACO cache)

---

## 🗂️ Порядок реализации

```
TASK-1  → TASK-5  → TASK-2+TASK-3 (вместе)  → TASK-4
```

1. **TASK-1** (cache + флаги) — независима, даёт максимальный ROI сразу
2. **TASK-5** (intrinsics + launch_bounds) — минимальные изменения, низкий риск
3. **TASK-2 + TASK-3** (2D grid + hipMemset) — реализовывать вместе, зависят друг от друга
4. **TASK-4** (interleaved DtoH + hipfftPlan fix) — более сложные изменения, требуют тестирования

---

## 📁 Затрагиваемые файлы

| Файл | Изменения |
|------|-----------|
| `modules/fft_processor/include/kernels/fft_processor_kernels_rocm.hpp` | TASK-2, TASK-3, TASK-4A, TASK-5 — переписать оба ядра |
| `modules/fft_processor/src/fft_processor_rocm.cpp` | TASK-1, TASK-2, TASK-3, TASK-4A, TASK-4B — CompileKernels, ExecutePadKernel, ReadMagPhaseResults, CreateFFTPlan |
| `modules/fft_processor/include/fft_processor_rocm.hpp` | TASK-1 — forward decl + поле `kernel_cache_`, TASK-4A — новый буфер `mag_phase_interleaved_` |

---

*Создано: 2026-02-26*
*Автор: Кодо (AI Assistant)*
*Источники анализа: `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`, анализ кода `modules/fft_processor`, паттерны из `modules/statistics` и `modules/vector_algebra`*
