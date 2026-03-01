# ROCm 7.2+ / HIP — Пошаговая инструкция оптимизации GPU-ядер

> **Версия**: 1.0 | **Дата**: 2026-02-26
> **Целевые архитектуры**: RDNA4 (Radeon 9070, gfx1201), CDNA3 (MI300, gfx942), CDNA2 (MI200, gfx90a)
> **Источники**: [ROCm Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html), [AMD Lab Notes: Register Pressure](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html), [HLRS HIP Optimization 2025](https://fs.hlrs.de/projects/par/events/2025/GPU-AMD/day2/08.HIP_Optimization.pdf), код модуля `modules/vector_algebra`

---

## ⚡ Правило №1: СНАЧАЛА ПРОФИЛИРУЙ, ПОТОМ ОПТИМИЗИРУЙ

Без данных профайлера ты оптимизируешь вслепую. Узкое место может быть там, где не ждёшь.

---

## 📋 ЧАСТЬ 1: Профилирование — обязательный первый шаг

### 1.1 Определить тип ядра

| Тип | Признак | Решение |
|-----|---------|---------|
| **Memory-bound** | Bandwidth utilization > 70%, низкий Arithmetic Intensity | Coalescing, LDS, векторные типы |
| **Compute-bound** | Высокая утилизация ALU, много FP операций | Intrinsics, FP32 литералы, reduce divides |
| **Latency-bound** | Низкий occupancy, pipeline stalls | Больше wavefronts, увеличить блок |

**Формула Arithmetic Intensity** = Кол-во операций / Байт прочитано/записано
Модель **Roofline** — определяет, к какому типу относится ядро.

### 1.2 Запуск rocprofv3

```bash
# Базовая статистика по всем ядрам
rocprofv3 --stats -- ./your_application

# С трейсингом для Perfetto
rocprofv3 --stats --hip-trace -- ./your_application

# Счётчики производительности
rocprofv3 --pmc SQ_INSTS_VALU SQ_WAVES -- ./your_application
```

### 1.3 Проверка расхода ресурсов при компиляции

```bash
# Показывает VGPRs, SGPRs, occupancy, spill-count для каждого ядра
hipcc --offload-arch=gfx1201 kernel.hip \
      -Rpass-analysis=kernel-resource-usage -c

# Быстрая проверка
hipcc --resource-usage kernel.hip
```

**Что смотреть в выводе:**
```
Kernel: my_kernel
  VGPRs: 96        ← критично для occupancy
  SGPRs: 32
  Occupancy: 5 waves/SIMD
  Scratch: 0       ← 0 = хорошо! Spilling = очень плохо!
  LDS: 4096 bytes
```

### 1.4 GPUProfiler (наш проект)

В проекте используем **только** GPUProfiler через DrvGPU:

```cpp
auto& profiler = backend->GetProfiler();
profiler.SetGPUInfo(backend->GetDeviceName(), backend->GetDriverVersion());
profiler.Start("MyKernel");

// ... запуск ядра ...

profiler.Stop("MyKernel");
profiler.PrintReport();       // только через PrintReport()!
// profiler.ExportMarkdown(); // или ExportMarkdown()
```

> ⚠️ **ЗАПРЕЩЕНО**: `GetStats()` + цикл + `con.Print` или `std::cout`

---

## 📋 ЧАСТЬ 2: Оптимизация памяти

### 2.1 Coalesced Memory Access — САМАЯ ВАЖНАЯ ОПТИМИЗАЦИЯ

**Правило**: потоки внутри одного wavefront должны обращаться к **последовательным** адресам памяти.

```cpp
// ❌ ПЛОХО — stride access, каждый поток пропускает N элементов
__global__ void bad_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid * N];  // stride = N → не coalescable!
}

// ✅ ХОРОШО — sequential access
__global__ void good_kernel(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid];  // consecutive addresses → одна транзакция!
}
```

**Пример из vector_algebra** ([symmetrize_kernel_sources_rocm.hpp](../modules/vector_algebra/include/kernels/symmetrize_kernel_sources_rocm.hpp)):
```cpp
// 16×16 блок: thread(col, row) читает/пишет row-major
// Warp читает data[row*n + 0], [row*n + 1], ..., [row*n + 15] — идеальный coalescing
unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;  // быстрый индекс
unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
float2 v = data[row * n + col];  // ✅ col = threadIdx.x → последовательно
```

### 2.2 Выравнивание 2D-массивов

```cpp
// Паддинг строк до кратного warp-size (32 для RDNA, 64 для CDNA)
const int warp_size = 32;  // RDNA4 (Radeon 9070)
int padded_width = ((width + warp_size - 1) / warp_size) * warp_size;
// Теперь каждая строка начинается с выровненного адреса
```

### 2.3 Векторные типы — увеличиваем ширину транзакций

```cpp
// ❌ МЕДЛЕННО — 4 отдельных load
float a = data[4*tid+0];
float b = data[4*tid+1];
float c = data[4*tid+2];
float d = data[4*tid+3];

// ✅ БЫСТРО — один 128-битный load
float4 v = reinterpret_cast<float4*>(data)[tid];
// v.x, v.y, v.z, v.w — все данные загружены за одну инструкцию

// Для комплексных чисел — float2 (как в нашем проекте)
float2 z = reinterpret_cast<float2*>(complex_data)[tid];
float2 result = {z.x * ref.x - z.y * ref.y,   // Re
                 z.x * ref.y + z.y * ref.x};   // Im
reinterpret_cast<float2*>(output)[tid] = result;
```

**Требование**: указатель должен быть выровнен по sizeof(float4)=16 байт.

### 2.4 Shared Memory (LDS) — тайлинг для повторного использования

```cpp
__global__ void tiled_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    // Паддинг +1 предотвращает bank conflicts (32 bank, каждый 4 байта)
    __shared__ float tile_A[16][17];  // 17 вместо 16!
    __shared__ float tile_B[16][17];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;

    for (int k = 0; k < N; k += 16) {
        // Загружаем тайл в LDS
        tile_A[ty][tx] = A[row * N + (k + tx)];
        tile_B[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();

        // Используем LDS — в 100× быстрее global mem!
        for (int i = 0; i < 16; i++)
            sum += tile_A[ty][i] * tile_B[i][tx];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

**Лимиты LDS**:
- MI200/CDNA2: 64 KiB / CU — если workgroup берёт 48 KiB, на CU помещается 1 WG!
- RDNA4: 128 KiB / CU (Radeon 9070)

### 2.5 Pinned Memory для быстрой передачи

```cpp
// Pinned (page-locked) memory — прямой DMA, значительно быстрее обычного malloc
float* h_pinned;
hipHostMalloc(&h_pinned, size, hipHostMallocDefault);
// ... работа с данными ...
hipHostFree(h_pinned);

// Zero-copy (integrated GPU / APU)
void* h_mapped;
hipHostMalloc(&h_mapped, size, hipHostMallocMapped);
void* d_mapped;
hipHostGetDevicePointer(&d_mapped, h_mapped, 0);
// Ядро использует d_mapped — данные на CPU, но GPU читает без копирования
```

### 2.6 Deferred Synchronization — паттерн из vector_algebra (Task_12)

```cpp
// ❌ МЕДЛЕННО — 3 синхронизации
CorePotrf(d_data, n, stream);
backend_->Synchronize();  // WAIT
CorePotri(d_data, n, stream);
backend_->Synchronize();  // WAIT
Symmetrize(d_data, n, stream);
backend_->Synchronize();  // WAIT

// ✅ БЫСТРО — одна stream, порядок гарантирован, 0 синхронизаций
CorePotrf(d_data, n, stream);    // queue
CorePotri(d_data, n, stream);    // queue (после предыдущего в том же stream)
Symmetrize(d_data, n, stream);   // queue
CheckInfo("context");             // ОДИН D2H memcpy — один implicit sync
```

**Правило**: операции в одном stream выполняются в порядке постановки. Синхронизация нужна только если CPU должен читать результат GPU.

---

## 📋 ЧАСТЬ 3: Occupancy и Register Pressure

### 3.1 Таблица VGPR → Occupancy

**CDNA2 (MI200, gfx90a) — wavefront = 64:**

| VGPRs | Max waves/SIMD | Max waves/CU | Рекомендация |
|-------|---------------|--------------|--------------|
| ≤ 64  | 8 | 32 | 🟢 Отлично |
| ≤ 96  | 5 | 20 | 🟡 Хорошо |
| ≤ 128 | 4 | 16 | 🟡 Приемлемо |
| ≤ 160 | 3 | 12 | 🔴 Плохо |
| ≤ 192 | 2 | 8  | 🔴 Критично |
| > 192 | 1 | 4  | ⛔ Spilling скоро |

**RDNA4 (Radeon 9070, gfx1201) — wavefront = 32:**

| VGPRs | Max waves/SIMD |
|-------|---------------|
| ≤ 64  | 8 |
| ≤ 96  | 5 |
| ≤ 128 | 4 |
| ≤ 256 | 2 |

> **RDNA4 особенность**: физически 64 KiB VGPRs / WGP = 2 CU. Wavefront size = 32 (половина от CDNA).

### 3.2 `__launch_bounds__` — подсказка компилятору

```cpp
// Без __launch_bounds__: компилятор предполагает blockSize=1024
// → избыточный резерв регистров → меньше occupancy

// ✅ ПРАВИЛЬНО: указываем реальный blockSize и желаемое кол-во blocks/CU
__launch_bounds__(256, 4)   // max 256 threads/block, min 4 blocks/CU
__global__ void my_kernel(float* data, int N) {
    // Компилятор теперь оптимизирует для 256 потоков
}

// Для RDNA4 с wavefront=32, blockSize=128:
__launch_bounds__(128)
__global__ void rdna4_kernel(float* data) { ... }
```

### 3.3 Семь техник снижения VGPR

#### Техника 1: `__restrict__` на указателях

```cpp
// ❌ БЕЗ restrict: компилятор не знает, перекрываются ли массивы
__global__ void kernel(float* a, float* b, float* c, int n);

// ✅ С restrict: компилятор агрессивно переиспользует регистры
__global__ void kernel(float* __restrict__ a,
                       float* __restrict__ b,
                       float* __restrict__ c,
                       int n);
// Эффект: SGPRs 98→78, VGPRs могут снизиться на 4-6
```

#### Техника 2: Определять переменные БЛИЖЕ к использованию

```cpp
// ❌ ПЛОХО — все переменные живут с начала ядра
__global__ void bad(float* data) {
    float tmp_a, tmp_b, tmp_c;  // Регистры заняты с самого начала!
    // ... 100 строк кода ...
    tmp_a = data[tid];
    tmp_b = tmp_a * 2.0f;
    tmp_c = tmp_b + 1.0f;
}

// ✅ ХОРОШО — переменные живут только там, где нужны
__global__ void good(float* data) {
    // ... 100 строк кода ...
    float tmp_a = data[tid];    // живёт только здесь
    float tmp_b = tmp_a * 2.0f;
    float tmp_c = tmp_b + 1.0f;
    data[tid] = tmp_c;
    // tmp_a, tmp_b, tmp_c освобождены!
}
```

#### Техника 3: `pow(x, N)` → ручное умножение

```cpp
float x = data[tid];
// ❌ МЕДЛЕННО: pow использует exp(N*log(x)) — много регистров и операций
float r = pow(x, 3.0f);

// ✅ БЫСТРО: две инструкции VMUL
float r = x * x * x;

// ❌ МЕДЛЕННО: pow(x, 2.0) → неявный double! (FP64 инструкция!)
float r = pow(x, 2.0);

// ✅ БЫСТРО: FP32 везде
float r = x * x;
```

#### Техника 4: Избегать стековых массивов

```cpp
// ❌ ПЛОХО — стек → scratch memory → slow!
float local_arr[16];  // Может попасть в scratch (VRAM)

// ✅ ХОРОШО — для небольших массивов: LDS
extern __shared__ float shared_arr[];
// или разворачиваем вручную через регистры
```

#### Техника 5: Контролировать unroll

```cpp
// ❌ ПЛОХО — unroll 16 итераций → 16× больше живых переменных
#pragma unroll 16
for (int i = 0; i < N; i++) { ... }

// ✅ ЛУЧШЕ — ограничить или отключить unroll
#pragma unroll 4        // умеренный unroll
#pragma unroll 1        // запретить unroll (если register pressure критичен)
for (int i = 0; i < N; i++) { ... }
```

#### Техника 6: Избегать избыточных вызовов функций

```cpp
// Каждый вызов device-функции → инлайнинг → больше регистров
// ❌ ПЛОХО внутри горячего пути:
for (int i = 0; i < N; i++) {
    result += complex_helper_function(data[i]);  // inline → +10 VGPRs?
}

// ✅ ЛУЧШЕ: вынести out of loop или упростить вычисления
```

#### Техника 7: Ручной spill в LDS (последнее средство)

```cpp
// Если регистров катастрофически не хватает
__global__ void kernel(...) {
    __shared__ float long_lived_var[256];  // spill to LDS
    long_lived_var[threadIdx.x] = compute_something();

    // много вычислений не использующих long_lived_var ...

    float result = use_value(long_lived_var[threadIdx.x]);
}
```

### 3.4 Замер occupancy из кода

```cpp
// Программное получение occupancy
int num_blocks;
hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, my_kernel, block_size, shared_mem_size);
float occupancy = (float)num_blocks * block_size / max_threads_per_cu;
printf("Occupancy: %.1f%%\n", occupancy * 100.0f);

// Автоматический выбор оптимального размера блока
int min_grid, block_sz;
hipOccupancyMaxPotentialBlockSize(&min_grid, &block_sz, my_kernel, 0, 0);
```

---

## 📋 ЧАСТЬ 4: Compute Optimizations

### 4.1 FP32 — критические правила

```cpp
// ❌ КРИТИЧЕСКАЯ ОШИБКА: двойные литералы → FP64 инструкции!
float result = a * 0.3 + 1.0;  // 0.3 и 1.0 — это double!
// Компилятор: float→double, mul64, add64, double→float

// ✅ ПРАВИЛЬНО: суффикс 'f' обязателен
float result = a * 0.3f + 1.0f;  // всё FP32, ~4× быстрее на Radeon!

// Особо опасные места:
float arr[] = {0.1, 0.2, 0.5};    // ❌ double literals в массиве
float arr[] = {0.1f, 0.2f, 0.5f}; // ✅
```

### 4.2 Быстрые интринсики

```cpp
// Стандартные функции: полная точность, медленнее
float r1 = sinf(x);   // ULP-точный
float r2 = cosf(x);
float r3 = expf(x);
float r4 = logf(x);
float r5 = sqrtf(x);

// Fast intrinsics: ~4 ULP погрешность, значительно быстрее
float r1 = __sinf(x);    // ~4 ULP
float r2 = __cosf(x);    // ~4 ULP
float r3 = __expf(x);    // ~4 ULP
float r4 = __logf(x);    // ~4 ULP
float r5 = __fsqrt_rn(x); // round to nearest

// -cl-fast-relaxed-math (компилятор): автоматически использует intrinsics
// ⚠️ Внимание: даёт ~1e-4 погрешность (наш проект: tolerance 1e-3f для LFM)
```

### 4.3 Замена деления и pow

```cpp
// ❌ МЕДЛЕННО
float r = a / b;           // div: ~20 тактов
float r = pow(a, 3.0f);    // exp + log + mul

// ✅ БЫСТРО
float inv_b = 1.0f / b;    // один раз вне loop
float r = a * inv_b;       // mul: 4 такта
float r = a * a * a;       // 2× VMUL

// Битовые операции для степеней 2
int n = threadIdx.x * 4;   // ❌ IMUL
int n = threadIdx.x << 2;  // ✅ SHIFT (один такт)
int n = threadIdx.x % 8;   // ❌ IDIV
int n = threadIdx.x & 7;   // ✅ AND (один такт)
```

### 4.4 Минимизация Thread Divergence

```cpp
// ❌ ПЛОХО — половина wavefront ждёт другую
if (threadIdx.x % 2 == 0) {
    // Только чётные потоки
    result = heavy_computation_A(data[tid]);
} else {
    // Только нечётные потоки
    result = heavy_computation_B(data[tid]);
}
// GPU сериализует: сначала чётные, потом нечётные

// ✅ ХОРОШО — весь wavefront идёт по одному пути
// Предикация (branchless):
float a = heavy_computation_A(data[tid]);
float b = heavy_computation_B(data[tid]);
float result = (threadIdx.x % 2 == 0) ? a : b;  // select instruction

// ✅ ХОРОШО — переструктурировать данные так,
// чтобы один wavefront обрабатывал однородные данные
```

---

## 📋 ЧАСТЬ 5: Архитектура ядер и запуск

### 5.1 Оптимальные размеры блоков

```cpp
// RDNA4 (Radeon 9070): wavefront = 32
// Блок должен быть кратен 32
dim3 block_rdna4(256, 1, 1);    // 8 wavefronts/block ✅
dim3 block_rdna4(128, 1, 1);    // 4 wavefronts/block ✅
dim3 block_2d(16, 16, 1);       // 256 = 8×32 ✅ (как в vector_algebra)

// CDNA (MI серия): wavefront = 64
// Блок должен быть кратен 64
dim3 block_cdna(256, 1, 1);     // 4 wavefronts/block ✅
dim3 block_cdna(64, 1, 1);      // 1 wavefront/block ✅

// Grid: покрываем все элементы
int N = 1024 * 1024;
dim3 grid((N + 255) / 256);    // ceiling division
```

### 5.2 Kernel Fusion — слияние ядер

```cpp
// ❌ 3 отдельных ядра: 3× overhead запуска, 3× global memory roundtrip
kernel_scale<<<grid, block>>>(data, scale);
kernel_add<<<grid, block>>>(data, offset);
kernel_normalize<<<grid, block>>>(data, norm_factor);

// ✅ 1 ядро: данные остаются в регистрах!
__global__ void fused_scale_add_normalize(
    float* __restrict__ data,
    float scale, float offset, float inv_norm,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float v = data[tid];
    v = (v * scale + offset) * inv_norm;  // всё в регистрах!
    data[tid] = v;
}
```

### 5.3 Streams для перекрытия Copy+Compute

```cpp
hipStream_t stream_compute, stream_copy;
hipStreamCreate(&stream_compute);
hipStreamCreate(&stream_copy);

// Пока GPU считает batch[i], CPU копирует batch[i+1]
for (int i = 0; i < num_batches; i++) {
    // Копируем следующий батч в stream_copy
    hipMemcpyAsync(d_input[next], h_input[next], size,
                   hipMemcpyHostToDevice, stream_copy);

    // Считаем текущий батч в stream_compute
    my_kernel<<<grid, block, 0, stream_compute>>>(d_input[curr], d_output[curr]);

    hipStreamSynchronize(stream_compute);  // Ждём только compute
    swap(curr, next);
}
```

### 5.4 hiprtc — JIT компиляция и кеширование (паттерн из vector_algebra)

Модуль [vector_algebra](../modules/vector_algebra/src/symmetrize_gpu_rocm.cpp) показывает правильный паттерн:

```cpp
// Шаг 1: Попытка загрузить из кеша
auto cached = kernel_cache_->Load(kKernelName);
if (cached.has_binary()) {
    hipModuleLoadData(&hip_module_, cached.binary.data());
    hipModuleGetFunction(&hip_function_, hip_module_, kKernelName);
    return;  // Кеш hit! Экономим 50-100ms на компиляции
}

// Шаг 2: JIT компиляция
hiprtcProgram prog;
hiprtcCreateProgram(&prog, kernel_source, kKernelName, 0, nullptr, nullptr);
const char* opts[] = {
    "-O3",                              // Полная оптимизация
    "--offload-arch=gfx1201",          // Явно указываем архитектуру (RDNA4)
    "-std=c++17"
};
hiprtcResult r = hiprtcCompileProgram(prog, 3, opts);

// Шаг 3: Сохранить в кеш
size_t binary_size;
hiprtcGetCodeSize(prog, &binary_size);
std::vector<char> binary(binary_size);
hiprtcGetCode(prog, binary.data());

kernel_cache_->Save(kKernelName, kernel_source, binary, /*target=*/"gfx1201");

// Шаг 4: Загрузить модуль
hipModuleLoadData(&hip_module_, binary.data());
hipModuleGetFunction(&hip_function_, hip_module_, kKernelName);
```

**Запуск hiprtc-скомпилированного ядра:**

```cpp
// Параметры передаются через void* массив
void* args[] = {
    &d_data,          // float2* __restrict__
    &matrix_n         // unsigned int
};

// Grid/Block для 2D задачи
int tiles = (matrix_n + 15) / 16;
hipModuleLaunchKernel(
    hip_function_,
    tiles, tiles, 1,   // gridDim
    16, 16, 1,         // blockDim
    0, stream,         // sharedMem, stream
    args, nullptr
);
```

---

## 📋 ЧАСТЬ 6: AMD-специфичные особенности

### 6.1 RDNA4 (Radeon 9070, gfx1201) — особенности

| Характеристика | RDNA4 (9070) | CDNA2 (MI200) |
|---------------|--------------|----------------|
| Wavefront size | **32** | 64 |
| L1 cacheline | 128 байт | 64 байт |
| LDS/CU | 128 KiB/WGP | 64 KiB/CU |
| FP32 peak | ~86 TFLOPS | ~383 TFLOPS |
| FP64 peak | ~54 TFLOPS | ~383 TFLOPS |
| VRAM | 16 GB GDDR6 | 128 GB HBM2e |
| Bandwidth | ~640 GB/s | ~3276 GB/s |

```cpp
// RDNA4: wavefront = 32, поэтому float4 особенно эффективен
// 32 threads × 4 floats = 128 floats/wavefront за 1 транзакцию

// Правильный block size для RDNA4:
// 1D задача: 128 или 256 threads (кратно 32)
// 2D задача: 16×16=256 или 8×32=256 (кратно 32 в одном измерении)
```

### 6.2 Важные флаги компиляции

```cmake
# CMakeLists.txt
target_compile_options(my_target PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
        -O3
        --offload-arch=gfx1201     # Radeon 9070 (RDNA4)
        # --offload-arch=gfx942    # MI300 (CDNA3)
        # --offload-arch=gfx90a    # MI200 (CDNA2)
        -std=c++17
        # НЕ добавляй -cl-fast-relaxed-math если нужна точность!
        # Даёт ~1e-4 погрешность в sin/cos
    >
)
```

### 6.3 Auto-tuning — важная особенность AMD

> Исследование 2024-2025: на AMD GPU тюнинг block size / tile size даёт **до 10×** прирост,
> тогда как на NVIDIA — лишь ~2×. Конфигурации, оптимальные для NVIDIA, **плохо** работают на AMD.

```cpp
// Для production: добавить auto-tuning параметров
// Block sizes, tile sizes, unroll factors — всё нужно профилировать
// на целевой AMD архитектуре, не переносить настройки с NVIDIA!
struct TuneParams {
    int block_x = 256;
    int block_y = 1;
    int tile_size = 16;
    int unroll_factor = 4;
};
```

---

## 📋 ЧАСТЬ 7: Проверенные паттерны GPUWorkLib (Best Practices)

> **Источник**: Реальный опыт оптимизации модулей `vector_algebra`, `statistics`, `fft_processor`, `fft_maxima`, `heterodyne`, `filters`
>
> При оптимизации нового модуля — **используй эти паттерны как чеклист**.

### 7.1 hiprtc Kernel Source — `float2_t` struct

**Проблема**: hiprtc не поддерживает `float2` из HIP runtime.

**Решение**: Определяем свой struct в начале kernel source:

```c
struct float2_t {
    float x;
    float y;
};
```

**Где используется**: `statistics`, `fft_processor`, `vector_algebra`, `fft_maxima`
**Файлы**: `modules/*/include/kernels/*_kernels_rocm.hpp`

### 7.2 KernelCacheService — HSACO disk cache

**Проблема**: hiprtcCompileProgram = ~100-200мс на каждый запуск.

**Решение**: Кешируем скомпилированный бинарь на диск, при повторном запуске загружаем за ~1-5мс.

```cpp
#include "services/kernel_cache_service.hpp"

// Конструктор:
kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
    "modules/<module>/kernels");

// CompileKernels():
void Module::CompileKernels() {
    if (kernels_compiled_) return;

    // Шаг 1: Попытка загрузить из HSACO cache
    if (kernel_cache_) {
        try {
            auto entry = kernel_cache_->Load(kCacheName);
            if (entry.has_binary()) {
                LoadModuleAndFunctions(entry.binary.data(), entry.binary.size());
                kernels_compiled_ = true;
                return;
            }
        } catch (...) { /* cache miss — компилируем */ }
    }

    // Шаг 2: Получить arch из backend
    auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    std::string arch = rocm_backend->GetCore().GetArchName();  // "gfx1201"
    std::string arch_flag = "--offload-arch=" + arch;

    // Шаг 3: Компиляция с флагами
    const char* opts[] = { "-O3", arch_flag.c_str(), "-DWARP_SIZE=32" };
    hiprtcCompileProgram(prog, 3, opts);

    // Шаг 4: Сохранить в cache
    kernel_cache_->Save(kCacheName, src, binary, "", "description");
}
```

**Где используется**: `vector_algebra`, `statistics`, `fft_processor`
**Эффект**: cold start ~100-200мс → ~1-5мс (hot cache)

### 7.3 2D Grid — устранение div/mod

**Проблема**: Целочисленное деление на GPU — ~20 тактов per thread.

```c
// ❌ ПЛОХО: каждый поток делает 2 дорогих деления
unsigned int beam_id = gid / nFFT;   // ~20 тактов
unsigned int pos     = gid % nFFT;   // ~20 тактов
```

**Решение**: Использовать 2D grid, `blockIdx.y = beam_id`:

```c
// ✅ ХОРОШО: beam_id из индекса блока — бесплатно
unsigned int beam_id = blockIdx.y;                             // 0 тактов
unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;  // 1 инструкция
```

```cpp
// C++ dispatch:
unsigned int grid_x = (nFFT + 255) / 256;
unsigned int grid_y = beam_count;
hipModuleLaunchKernel(kernel,
    grid_x, grid_y, 1,    // grid: X=позиции, Y=лучи
    256, 1, 1,             // block
    0, stream_, args, nullptr);
```

**Для OpenCL**:
```c
uint beam_id = get_global_id(1);
uint pos     = get_global_id(0);
// dispatch: size_t global[2] = { nFFT, beam_count };
```

**Где используется**: `fft_processor` (pad_data), `fft_maxima` (padding, detect_all_maxima)
**Эффект**: ~20-30% ускорение при 256 лучах × 65536 точек

### 7.4 hipMemsetAsync + simplified kernel

**Проблема**: Thread divergence на границе данных (else branch пишет нули).

**Решение**: Обнулить весь буфер асинхронно, затем ядро копирует только valid данные:

```cpp
// C++: обнулить весь буфер
hipMemsetAsync(fft_input_, 0, total_size, stream_);

// Ядро: только копирование (нет else-branch!)
if (pos >= n_point) return;
output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
```

**Для OpenCL**: `clEnqueueFillBuffer(queue, buffer, &zero, sizeof(float2), 0, total_size, ...)`

**Где используется**: `fft_processor` (pad_data), `fft_maxima` (padding_kernel)
**Эффект**: устранение divergence + ядро обрабатывает только n_point, а не nFFT элементов

### 7.5 Interleaved Output — 1 DtoH вместо 2+

**Проблема**: Несколько отдельных `hipMemcpyDtoH` = несколько PCIe транзакций.

**Решение**: Объединить данные в один interleaved буфер:

```c
// ✅ Один буфер float2_t: {mag, phase}
float2_t mp;
mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);  // magnitude
mp.y = __atan2f(z.y, z.x);                   // phase
mag_phase[gid] = mp;                          // 1 write

// C++: один DtoH
hipMemcpyDtoH(raw.data(), mag_phase_interleaved_, total * sizeof(float2_t));
// Разделить на CPU — trivial
```

**Где используется**: `fft_processor` (complex_to_mag_phase)
**Эффект**: -40% DtoH latency (1 transfer вместо 2)

Аналогичный паттерн для массовых извлечений:
```c
// ✅ extract_medians — 1 DtoH вместо 256
medians[b] = sorted[b * n_point + n_point / 2];  // GPU kernel
// Затем 1 hipMemcpyDtoH для beam_count floats
```
**Где используется**: `statistics` (extract_medians) | **Эффект**: -90-95% DtoH latency

### 7.6 Warp Shuffle в финальных стадиях reduction

**Проблема**: `__syncthreads()` в последних шагах tree reduction — дорого.

**Решение**: Переключиться на `__shfl_down` когда остаётся ≤ WARP_SIZE элементов:

```c
// LDS tree reduction (до WARP_SIZE)
for (unsigned int s = block_size / 2; s >= WARP_SIZE; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
// Warp shuffle (без __syncthreads!)
if (tid < WARP_SIZE) {
    float val = sdata[tid];
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    if (tid == 0) result = val;
}
```

**WARP_SIZE**: передавать через `-DWARP_SIZE=32` (RDNA4) или `=64` (CDNA2)
**Где используется**: `statistics` (mean_reduce, welford_fused)
**Эффект**: ~15-25% ускорение reduction

### 7.7 Double-Load Trick

**Проблема**: 1 thread = 1 элемент → много блоков → много partial sums.

**Решение**: Каждый поток читает 2 элемента, суммирует в регистрах:

```c
// block покрывает block_size * 2 элементов
unsigned int gid1 = base + block_in_beam * (block_size * 2) + tid;
unsigned int gid2 = gid1 + block_size;
float v1 = (gid1 < n_point) ? input[gid1] : 0.0f;
float v2 = (gid2 < n_point) ? input[gid2] : 0.0f;
sdata[tid] = v1 + v2;  // 2 элемента → 1 LDS entry
```

**Где используется**: `statistics` (mean_reduce_phase1)
**Эффект**: вдвое меньше блоков → вдвое меньше partial sums → быстрее phase2

### 7.8 LDS +1 Padding (Bank Conflicts)

**Проблема**: При tree reduction потоки попадают в одинаковые LDS банки.

**Решение**: Добавить +1 к размеру LDS массива:

```c
// ❌ ПЛОХО: bank conflicts при stride reduction
__local float lds_mag[256];

// ✅ ХОРОШО: +1 padding устраняет conflicts
__local float lds_mag[256 + 1];
```

**Для OpenCL**: то же самое через `__local`.
**Где используется**: `fft_maxima` (post_kernel, prefix_sum)
**Эффект**: ~10-20% ускорение reduction шагов

### 7.9 Cached FFT Plan

**Проблема**: `hipfftDestroy` + `hipfftPlan1d` при изменении batch size — дорого (~мс).

**Решение**: Хранить 2 плана (`plan_` и `plan_last_`), swap при повторном размере:

```cpp
if (plan_last_ != 0 && plan_last_batch_ == batch_beam_count) {
    std::swap(plan_, plan_last_);              // мгновенно!
    std::swap(plan_batch_size_, plan_last_batch_);
} else {
    if (plan_last_ != 0) hipfftDestroy(plan_last_);
    plan_last_ = plan_;                        // сохраняем старый
    plan_last_batch_ = plan_batch_size_;
    hipfftPlan1d(&plan_, nFFT_, HIPFFT_C2C, batch_beam_count);
}
```

**Где используется**: `fft_processor`
**Эффект**: устранение Destroy/Create при чередовании batch sizes

### 7.10 C++ паттерны (vector_algebra)

#### Move-only результат с автоматическим освобождением GPU-памяти

```cpp
struct CholeskyResult {
    void* d_data = nullptr;
    IBackend* backend = nullptr;
    CholeskyResult(CholeskyResult&& other) noexcept
        : d_data(other.d_data), backend(other.backend) {
        other.d_data = nullptr;
    }
    CholeskyResult(const CholeskyResult&) = delete;
    ~CholeskyResult() {
        if (d_data && backend) backend->Free(d_data);
    }
};
```

#### Предварительное выделение служебных буферов

```cpp
// В конструкторе — выделяем один раз, reuse в горячем пути
hipMalloc(&d_info_, 2 * sizeof(rocblas_int));
// В Process(): переиспользуем d_info_ без аллокаций
```

#### Deferred Synchronization

```cpp
// ❌ МЕДЛЕННО: sync после каждого шага
CorePotrf(d_data, n, stream); backend_->Synchronize();
CorePotri(d_data, n, stream); backend_->Synchronize();

// ✅ БЫСТРО: одна stream, порядок гарантирован, 0 лишних sync
CorePotrf(d_data, n, stream);
CorePotri(d_data, n, stream);
Symmetrize(d_data, n, stream);
CheckInfo("context");  // ОДИН D2H — один implicit sync
```

---

## 📋 ЧАСТЬ 8: Быстрый чеклист оптимизации

### Перед началом
- [ ] Запустить profiler → определить тип (memory/compute/latency-bound)
- [ ] Проверить VGPRs: `hipcc -Rpass-analysis=kernel-resource-usage`
- [ ] Убедиться, что scratch = 0 (нет spilling)

### Память
- [ ] Consecutive threads → consecutive addresses (coalescing)
- [ ] 2D массивы: ширина кратна warp-size (32/64)
- [ ] Векторные типы float4/float2 для bulk data
- [ ] LDS для данных, которые используются > 1 раза в блоке
- [ ] Паддинг LDS: `[N+1]` вместо `[N]` для tree reduction (bank conflicts) — Паттерн 7.8
- [ ] Deferred sync: один sync в конце, не после каждого ядра
- [ ] `hipHostMalloc` для частых H2D/D2H копий
- [ ] Interleaved output: 1 DtoH вместо нескольких — Паттерн 7.5
- [ ] `hipMemsetAsync` + simplified kernel (без else-branch) — Паттерн 7.4

### Регистры
- [ ] Добавить `__launch_bounds__(BLOCK_SIZE)` ко всем ядрам (через `-DBLOCK_SIZE=`)
- [ ] Добавить `__restrict__` ко всем указателям ядра
- [ ] Переместить объявления переменных ближе к использованию
- [ ] Заменить `pow(x, N)` → `x*x*...*x`
- [ ] Убедиться, что нет `double` литералов (всё `0.3f` not `0.3`)
- [ ] Ограничить `#pragma unroll` если VGPRs > целевого значения

### Вычисления
- [ ] Все литералы с суффиксом `f` (0.3f, 1.0f, etc.)
- [ ] `__sinf/__cosf/__expf` вместо `sinf/cosf/expf` там, где точность не критична
- [ ] Деление заменить на умножение reciprocal вне цикла
- [ ] Битовые операции для `%2, /2, *4, /8` и т.д.
- [ ] Warp shuffle в финальных стадиях reduction — Паттерн 7.6
- [ ] Double-load trick (1 thread → 2 элемента) для reduction — Паттерн 7.7

### Архитектура
- [ ] Block size кратен wavefront (32 для RDNA4, 64 для CDNA)
- [ ] hiprtc: `float2_t` struct вместо `float2` — Паттерн 7.1
- [ ] hiprtc: `-O3 --offload-arch=gfxXXXX -DWARP_SIZE=32 -DBLOCK_SIZE=256`
- [ ] `KernelCacheService` + HSACO disk cache — Паттерн 7.2
- [ ] 2D Grid вместо div/mod для beam_id — Паттерн 7.3
- [ ] Мелкие ядра: рассмотреть kernel fusion
- [ ] Параллельные stream для copy+compute
- [ ] Cached FFT Plan (2 плана + swap) — Паттерн 7.9

### Инфраструктура GPUWorkLib
- [ ] Консольный вывод **только** через `console_output`
- [ ] Профилирование **только** через `GPUProfiler` (PrintReport/ExportMarkdown/ExportJSON)
- [ ] Нет хардкоженных констант в `__launch_bounds__` / `__local` / `__shared__` — всё через `-D`

---

## 📋 ЧАСТЬ 9: Как правильно формулировать задачи для Claude

> Перенесено из `Doc_Addition/Roc hip kernel оптимизация.md`

### Структура промпта (XML-теги)

```xml
<context>
Проект: GPUWorkLib, модуль [название].
GPU: AMD Radeon 9070 (RDNA4, gfx1201), ROCm 7.2.
Wavefront size: 32. LDS: 128 KiB/WGP. Max VGPRs: 512/WGP.
</context>

<current_kernel>
// ... текущий код ядра ...
</current_kernel>

<profiling_data>
VGPRs: 112, occupancy: 4 waves/SIMD,
bandwidth: 45% от пика, scratch: 0
</profiling_data>

<task>
Оптимизируй ядро для снижения VGPR с 112 до ≤96.
Добавь __launch_bounds__, __restrict__, перенеси определения переменных.
Покажи изменённый код с комментариями к каждому изменению.
</task>

<constraints>
- Только FP32, без double-литералов
- Совместимость с ROCm 7.2+, gfx1201
- Не рефакторить код за пределами запрошенных изменений
</constraints>
```

### Принципы
- ✅ **Конкретно**: «снизить VGPR с 112 до ≤96 через __launch_bounds__ и перестановку переменных»
- ✅ **С данными профайлера**: дай VGPRs, occupancy, bandwidth %
- ✅ **С ограничениями**: архитектура, версия ROCm, точность
- ❌ Не пиши «сделай быстрее» — слишком размыто
- ❌ Не пиши «КРИТИЧЕСКИ ВАЖНО!!!» — это не помогает
- ✅ Объясняй **почему**: «нужно ≤96 VGPR чтобы получить 5 waves/SIMD»

---

## 📚 Источники

| Документ | Содержание |
|---------|-----------|
| [ROCm Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) | Официальные рекомендации HIP |
| [AMD Lab Notes: Register Pressure](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html) | 7 техник снижения VGPR |
| [HLRS HIP Optimization 2025 (PDF)](https://fs.hlrs.de/projects/par/events/2025/GPU-AMD/day2/08.HIP_Optimization.pdf) | Полный курс оптимизации |
| [GPUOpen: Register Pressure](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/) | Практические примеры |
| [Auto-tuning AMD vs NVIDIA](https://arxiv.org/abs/2407.11488v1) | AMD даёт 10× vs NVIDIA 2× |
| [GPUOpen: Register Pressure](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/) | Практические примеры |
| `modules/vector_algebra/` | Наш эталонный ROCm-модуль |
| `modules/statistics/` | Reduction, Welford, warp shuffle паттерны |
| `modules/fft_processor/` | FFT Plan caching, interleaved output |
| `modules/fft_maxima/` | LDS padding, 2D grid, parallel reduction |

---

*Обновлено: 2026-03-01 | Автор: Кодо | Версия: 2.0*
*Объединено из Info_ROCm_HIP_Optimization_Guide.md + Roc hip kernel оптимизация.md*
