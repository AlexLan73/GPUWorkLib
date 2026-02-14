# Исследование: Оптимальные методы вычисления среднего значения на GPU

> **Дата**: 2026-02-14
> **Автор**: Кодо (AI Assistant)
> **Задача**: Найти оптимальный алгоритм вычисления среднего для 256 лучей × 4M точек

---

## 📊 Контекст задачи

**Входные данные:**
- 256 лучей (rays/beams)
- 4 миллиона точек в каждом луче
- Общий размер: 256 × 4,000,000 = 1,024,000,000 элементов

**Выходные данные:**
- 256 значений (среднее для каждого луча отдельно)

**Платформы:**
- OpenCL (основной backend)
- ROCm/HIP (планируется для AMD GPU)

**Критерий оптимизации:**
- Минимальное время выполнения
- Максимальная утилизация GPU

---

## 🎯 Методы параллельной редукции

### 1. Tree Reduction (Древовидная редукция)

**Описание:**
Параллельная редукция представляет собой древовидную структуру, где каждый уровень дерева — это шаг редукции. Вместо O(n) операций выполняется O(log n) шагов.

**Алгоритм:**
```
Уровень 0: [a0, a1, a2, a3, a4, a5, a6, a7]
Уровень 1: [a0+a1, a2+a3, a4+a5, a6+a7]
Уровень 2: [a0+a1+a2+a3, a4+a5+a6+a7]
Уровень 3: [a0+a1+a2+a3+a4+a5+a6+a7]
```

**Временная сложность:** O(log n)
**Пространственная сложность:** O(n)

**Особенности для 4M элементов:**
- Требуется log₂(4M) ≈ 22 шага редукции
- Эффективно использует параллелизм GPU
- Минимизирует количество итераций

**Оптимизация:**
- Использовать shared memory для внутриблоковой редукции
- Избегать bank conflicts через sequential addressing
- Thread coarsening: каждый поток обрабатывает несколько элементов

**Источники:**
- [GPU MODE Lecture 9: Reductions](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/)
- [Optimizing Parallel Reduction in CUDA (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

### 2. Two-Level Hierarchical Reduction (Двухуровневая иерархическая редукция)

**Описание:**
Разделение редукции на два этапа: внутри блоков (используя shared memory) и между блоками (используя global memory или атомики).

**Алгоритм:**
```
Stage 1: Редукция внутри каждого workgroup/block
  - Input: 4M элементов
  - Workgroup size: 256 threads
  - Output: 4M/256 = ~15,625 partial results

Stage 2: Редукция частичных результатов
  - Input: 15,625 partial results
  - Output: 1 final result (для каждого луча)
```

**Временная сложность:** O(n/p + log p), где p — число процессоров
**Пространственная сложность:** O(n)

**Особенности для 4M элементов:**
- Первый проход: 256 workgroups × 256 threads = 65,536 threads обрабатывают ~61 элементов каждый
- Второй проход: редуцирует ~15,625 частичных результатов до 1
- Может потребоваться третий проход для очень больших массивов

**Оптимизация:**
- Использовать Thread Coarsening: каждый поток обрабатывает несколько элементов последовательно перед началом параллельной редукции
- Применять векторизацию (float4, float8) для загрузки данных из global memory
- Использовать Global Data Share (GDS) на AMD GPU для обмена данными между workgroups без записи в global memory

**Источники:**
- [HIP Reduction Tutorial](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html)
- [CUDA Parallel Reduction Optimization](https://enccs.github.io/cuda/3.01_ParallelReduction/)

---

### 3. Per-Row Reduction (Редукция по строкам матрицы)

**Описание:**
Специализированный подход для матричных данных: каждый луч (строка) обрабатывается независимо параллельными workgroups.

**Алгоритм:**
```
Workgroup assignment:
  - 256 workgroups (по одному на луч)
  - Каждый workgroup обрабатывает 4M элементов своего луча

Per-workgroup reduction:
  - Workgroup size: 256 threads
  - Каждый thread обрабатывает 4M/256 = ~15,625 элементов
  - Tree reduction внутри workgroup для финального результата
```

**Временная сложность:** O(n/p + log p) на луч
**Пространственная сложность:** O(n)

**Особенности для 256 лучей × 4M точек:**
- Идеально подходит для нашей задачи: естественная декомпозиция
- 256 независимых задач = отличная загрузка современных GPU
- Нет необходимости в синхронизации между лучами

**Оптимизация:**
- Coalesced memory access: упорядочить данные как массив структур (AoS) или структуру массивов (SoA)
- Для AoS layout: `data[ray_idx][point_idx]` — каждый workgroup читает последовательно
- Для SoA layout: `data[point_idx * 256 + ray_idx]` — соседние threads читают соседние адреса (лучше для coalescing)
- Due to workgroup size constraints (max 1024), may need two-stage reduction per row

**Источники:**
- [Desktop Supercomputing – MapTube](http://maptube.blogweb.casa.ucl.ac.uk/2014/11/20/desktop-supercomputing/)

---

### 4. Warp/Wavefront Shuffle Operations

**Описание:**
Использование аппаратных инструкций для обмена данными между потоками внутри warp (NVIDIA) или wavefront (AMD) без использования shared memory.

**Алгоритм:**
```
// AMD GCN Wavefront (64 threads)
for (int offset = 32; offset > 0; offset >>= 1) {
  value += __shfl_down(value, offset);  // Или DPP на AMD
}
if (lane_id == 0) {
  // value содержит сумму 64 элементов
}
```

**Временная сложность:** O(log w), где w — размер warp/wavefront
**Пространственная сложность:** O(1) — не требует shared memory

**Особенности для GPU:**
- **NVIDIA**: warp = 32 threads, инструкции `__shfl_down`, `__shfl_xor`
- **AMD GCN**: wavefront = 64 threads, инструкции DPP (Data Parallel Processing), permute, swizzle
- Очень низкая латентность (обмен через регистры)
- Не требует барьеров синхронизации

**Оптимизация:**
- Использовать для финальной редукции внутри warp/wavefront перед записью в shared memory
- Комбинировать с tree reduction: wavefront shuffle → shared memory → global memory
- Pre-combining с одинаковыми ключами на уровне warp для минимизации атомарных операций

**Особенность для 4M элементов:**
- Wavefront shuffle обрабатывает только 32-64 элемента за раз
- Должен комбинироваться с другими методами для больших массивов
- Отлично подходит для финальной стадии редукции

**Источники:**
- [AMD GCN Assembly Cross-Lane Operations](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)
- [Warp Shuffle Instructions (CSE 599 I)](https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2018.pdf)
- [HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)

---

### 5. Atomic Operations (Атомарные операции)

**Описание:**
Использование атомарных операций для накопления результата в общей переменной, избегая явной синхронизации между потоками.

**Алгоритм:**
```opencl
__kernel void atomic_sum(__global float* input, __global float* result, int n) {
  int gid = get_global_id(0);
  if (gid < n) {
    atomic_add_global(&result[0], input[gid]);
  }
}
```

**Временная сложность:** O(1) на поток (теоретически), но с высокой contention
**Пространственная сложность:** O(1)

**Особенности:**
- Простая реализация
- Высокая contention при большом числе потоков
- Performance bottleneck из-за последовательных атомарных операций

**Оптимизация:**
- **Shared memory atomics**: использовать `atomic_add` на shared memory вместо global (значительно быстрее)
- **Two-stage approach**:
  - Stage 1: Atomic add в shared memory внутри workgroup
  - Stage 2: Один поток на workgroup делает atomic add в global memory
- **Pre-aggregation**: каждый поток сначала суммирует несколько элементов, затем делает один atomic add
- **Hardware support**: современные GPU (NVIDIA Hopper, AMD RDNA3) имеют улучшенную поддержку атомарных операций

**Performance данные:**
- Shared memory atomics: до 28x быстрее чем global memory atomics (Intel GPU)
- Software-based atomics на AMD: до 67x speedup над системными атомарными операциями
- Atomic.add vs atomic.fetch_add на NVIDIA H100: 1.2x improvement

**Особенность для 4M элементов:**
- Прямой atomic add из 4M потоков = катастрофическая contention
- ОБЯЗАТЕЛЬНО требуется иерархический подход с pre-aggregation
- Может быть медленнее tree reduction, но проще в реализации

**Источники:**
- [Performance Characterization of Atomic Operations on AMD GPUs](https://synergy.cs.vt.edu/pubs/papers/elteir-ieeecluster11-atomic-operations.pdf)
- [Voting and Shuffling to Optimize Atomic Operations](https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/)
- [Study on Atomics-based Integer Sum Reduction in HIP](https://dl.acm.org/doi/fullHtml/10.1145/3547276.3548627)
- [Intel GPU Atomics with SLM](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/atomics-with-slm.html)

---

### 6. Sequential Addressing (Последовательная адресация)

**Описание:**
Оптимизация паттерна доступа к памяти в tree reduction для избежания bank conflicts и thread divergence.

**Проблема с Interleaved Addressing:**
```opencl
// BAD: Interleaved addressing (много bank conflicts)
for (int stride = 1; stride < blockDim.x; stride *= 2) {
  if (tid % (2*stride) == 0) {  // Divergence!
    sdata[tid] += sdata[tid + stride];  // Bank conflicts!
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
```

**Решение: Sequential Addressing:**
```opencl
// GOOD: Sequential addressing (нет bank conflicts)
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (tid < stride) {
    sdata[tid] += sdata[tid + stride];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
```

**Преимущества:**
- Нет bank conflicts в shared memory
- Нет thread divergence (активные потоки идут последовательно)
- Coalesced memory access при загрузке из global memory

**Performance improvement:**
- 2.01x step speedup
- 4.68x cumulative speedup (в комбинации с другими оптимизациями)

**Временная сложность:** O(log n)
**Пространственная сложность:** O(n) в shared memory

**Дополнительные оптимизации:**
- **Thread Coarsening**: каждый поток обрабатывает несколько элементов последовательно
  ```opencl
  float sum = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    sum += input[i];
  }
  sdata[tid] = sum;
  ```
- **Unrolling последних шагов**: когда stride < warp size, барьеры не нужны
- **Vectorized loads**: использовать float4/float8 для загрузки данных

**Особенность для 4M элементов:**
- Sequential addressing критически важен для избежания bottlenecks
- Комбинируется с thread coarsening для обработки больших массивов
- Базовый building block для эффективной tree reduction

**Источники:**
- [7 Step Optimization of Parallel Reduction with CUDA](https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8)
- [OpenCL Best Practices Guide](https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf)
- [NVIDIA OpenCL Examples: oclReduction](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction_kernel.cl)

---

## 🔧 Оптимизация памяти

### Coalesced Memory Access (Выровненный доступ к памяти)

**Принцип:**
Соседние потоки (work-items) читают соседние адреса памяти. Это позволяет GPU объединить несколько запросов в одну транзакцию.

**Best practice:**
```opencl
// GOOD: Coalesced access
float value = input[get_global_id(0)];

// BAD: Strided access
float value = input[get_global_id(0) * stride];
```

**Для матрицы 256 лучей × 4M точек:**

**Layout 1: Row-major (SoA - Structure of Arrays):**
```
data[point_idx * 256 + ray_idx]
```
- Соседние потоки в workgroup читают соседние адреса
- Идеально для per-ray reduction
- Coalesced access

**Layout 2: Column-major (AoS - Array of Structures):**
```
data[ray_idx * 4M + point_idx]
```
- Каждый workgroup читает последовательную память
- Но соседние потоки читают с большим stride
- Не coalesced

**Рекомендация:** SoA layout для максимальной производительности

---

### Bank Conflicts (Конфликты банков shared memory)

**Проблема:**
Shared memory разделена на банки (обычно 32). Если несколько потоков обращаются к разным адресам одного банка — serialization.

**Как избежать:**
- Использовать sequential addressing (см. выше)
- Padding массивов в shared memory: `__local float sdata[256 + 1]`
- Читать один и тот же адрес (broadcast) — OK

**Источники:**
- [ROCm OpenCL Optimization Guide](https://rocmdoc.readthedocs.io/en/latest/Programming_Guides/Opencl-optimization.html)
- [Why aren't there bank conflicts in global memory](https://saturncloud.io/blog/why-arent-there-bank-conflicts-in-global-memory-for-cudaopencl/)

---

### Occupancy (Загрузка GPU)

**Определение:**
Соотношение активных wavefronts к максимально возможному числу wavefronts на compute unit.

**Факторы:**
- Workgroup size
- Использование регистров
- Использование shared memory (LDS)

**Рекомендации:**
- Workgroup size: кратно размеру wavefront (64 для AMD, 32 для NVIDIA)
- Типичные размеры: 64, 128, 256, 512
- Для нашей задачи: 256 потоков = оптимальный баланс

**Источники:**
- [Intel OpenCL Kernel Memory Access Optimization](https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-guide-processor-graphics/2019-4/kernel-memory-access-optimization-summary.html)

---

## 📚 Библиотечные решения

### 1. rocPRIM (AMD ROCm Platform)

**Описание:**
Header-only библиотека HIP parallel primitives, оптимизированная для AMD GPU.

**API уровни:**

**Device-level reduction:**
```cpp
#include <rocprim/rocprim.hpp>

// Device-level reduce
rocprim::reduce(
  d_input,           // Input device pointer
  d_output,          // Output device pointer
  input_size,        // Number of elements
  rocprim::plus<>(), // Binary operator
  stream             // HIP stream
);
```

**Block-level reduction:**
```cpp
using block_reduce_f = rocprim::block_reduce<float, 256>;
__shared__ typename block_reduce_f::storage_type storage;

block_reduce_f().reduce(input, output, valid_items, storage, rocprim::minimum<float>());
```

**Warp-level reduction:**
```cpp
using warp_reduce_i = rocprim::warp_reduce<int, 16>;
warp_reduce_i().reduce(value, output, rocprim::minimum<int>());
```

**Преимущества:**
- Оптимизировано для AMD GPU (RDNA, CDNA архитектуры)
- Header-only (не требует линковки)
- Поддержка custom operators
- Аналог CUB/Thrust для CUDA

**Использование для нашей задачи:**
```cpp
// Per-ray reduction с rocPRIM
for (int ray = 0; ray < 256; ++ray) {
  rocprim::reduce(
    d_input + ray * 4000000,  // Start of ray data
    d_output + ray,            // Output for this ray
    4000000,                   // Elements per ray
    rocprim::plus<float>(),
    stream
  );
}
// Затем разделить на 4M для получения среднего
```

**Источники:**
- [rocPRIM Device Reduce Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.1.1/device_ops/reduce.html)
- [rocPRIM Block Reduce](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/block_ops/ops_classes/reduce.html)
- [rocPRIM Warp Reduce](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.6.0/warp_ops/reduce.html)

---

### 2. clBLAS (OpenCL BLAS)

**Описание:**
OpenCL реализация BLAS уровней 1, 2, 3, оптимизированная для AMD GPU.

**Релевантные функции:**

**ASUM (Sum of absolute values):**
```cpp
#include <clBLAS.h>

clblasStatus clblasSasum(
  size_t N,                    // Number of elements
  cl_mem asum,                 // Output: sum of absolute values
  size_t offAsum,              // Offset in asum buffer
  const cl_mem X,              // Input vector
  size_t offx,                 // Offset in X
  int incx,                    // Stride in X
  cl_mem scratchBuff,          // Scratch buffer
  cl_uint numCommandQueues,    // Number of queues
  cl_command_queue *commandQueues,
  cl_uint numEventsInWaitList,
  const cl_event *eventWaitList,
  cl_event *events
);
```

**Ограничения:**
- ASUM возвращает сумму абсолютных значений, а не просто сумму
- Для среднего нужно дополнительно разделить на N
- Нет прямой функции "mean" в clBLAS

**Альтернатива: CLBlast**
CLBlast — современная замена clBLAS с лучшей производительностью и поддержкой auto-tuning.

**Использование для нашей задачи:**
```cpp
// Per-ray summation с clBLAS
for (int ray = 0; ray < 256; ++ray) {
  clblasSasum(
    4000000,                      // N elements
    d_sums + ray,                 // Output
    0,                            // Offset
    d_input + ray * 4000000,      // Input for this ray
    0,                            // Offset
    1,                            // Stride
    d_scratch,                    // Scratch buffer
    1, &queue,
    0, nullptr, nullptr
  );
}
// Host-side: compute mean = sum / 4M для каждого луча
```

**Источники:**
- [clBLAS Overview](http://clmathlibraries.github.io/clBLAS/group__OVERVIEW.html)
- [clBLAS ASUM Documentation](http://clmathlibraries.github.io/clBLAS/group__ASUM.html)
- [CLBlast GitHub](https://github.com/CNugteren/CLBlast)

---

### 3. OpenCL Reduction Examples (GitHub)

**1. NVIDIA OpenCL Examples**
- Repo: [nvidia-opencl-examples](https://github.com/sschaetz/nvidia-opencl-examples)
- Файлы:
  - Kernel: [oclReduction_kernel.cl](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction_kernel.cl)
  - Host: [oclReduction.cpp](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction.cpp)
- Содержит несколько вариантов kernel: reduce4, reduce5, reduce6
- Демонстрирует оптимизации: sequential addressing, loop unrolling, multiple elements per thread

**2. OpenCL Reduction Sum**
- Repo: [OpenCL-reduction-sum](https://github.com/maoshouse/OpenCL-reduction-sum)
- Простой пример с подробными комментариями
- Подходит для начального изучения

**3. KernelTuner OpenCL Reduction**
- Repo: [kernel_tuner](https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/reduction.py)
- Python framework для auto-tuning параметров kernel
- Параметры: block size, vectorization, loop unrolling

**4. Apple Cocoa Sample**
- Repo: [CocoaSampleCode](https://github.com/HelmutJ/CocoaSampleCode/blob/master/OpenCL_Parallel_Reduction_Example/reduce_int2_kernel.cl)
- Примеры для int и float типов
- Efficient parallel reduction implementation

**Источники:**
- [OpenCL Parallel Reduction Guide](https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf)
- [OpenCL Reduction Sum Blog](https://dean-shaff.github.io/blog/c++/opencl/2020/03/29/opencl-reduction-sum.html)

---

## 🏆 Рекомендации для реализации

### Для задачи 256 лучей × 4M точек

**Оптимальный подход: Per-Row Two-Level Hierarchical Reduction**

#### Почему этот метод?

1. **Естественная декомпозиция**: 256 независимых лучей = 256 независимых задач
2. **Отличная загрузка GPU**: современные GPU легко обрабатывают 256 параллельных workgroups
3. **Нет необходимости в синхронизации между лучами**: каждый workgroup работает независимо
4. **Масштабируемость**: легко адаптируется к разному числу лучей

---

### Архитектура решения

**Layout данных: SoA (Structure of Arrays)**
```cpp
// Memory layout (coalesced access)
// data[point_idx * 256 + ray_idx]
// Размер: 4,000,000 * 256 * sizeof(float) = 4GB
```

**Kernel architecture:**

**Stage 1: Per-ray reduction (256 workgroups, 256 threads each)**
```opencl
__kernel void reduce_per_ray_stage1(
  __global const float* input,   // [4M * 256] SoA layout
  __global float* partial_sums,  // [256 * num_blocks_per_ray]
  int n_points                    // 4M
) {
  int ray_idx = get_group_id(0);         // 0..255
  int local_id = get_local_id(0);        // 0..255
  int block_id = get_group_id(1);        // 0..num_blocks_per_ray-1
  int threads_per_block = get_local_size(0); // 256

  __local float sdata[256];

  // Thread coarsening: каждый поток обрабатывает несколько элементов
  int elements_per_thread = n_points / (threads_per_block * num_blocks_per_ray);
  int start_idx = (block_id * threads_per_block + local_id) * elements_per_thread;

  float sum = 0.0f;
  for (int i = 0; i < elements_per_thread; i++) {
    int point_idx = start_idx + i;
    if (point_idx < n_points) {
      sum += input[point_idx * 256 + ray_idx];  // Coalesced access
    }
  }

  sdata[local_id] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction with sequential addressing (избегаем bank conflicts)
  for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      sdata[local_id] += sdata[local_id + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Один поток записывает результат
  if (local_id == 0) {
    partial_sums[ray_idx * num_blocks_per_ray + block_id] = sdata[0];
  }
}
```

**Stage 2: Final reduction per ray (256 workgroups, небольшой размер)**
```opencl
__kernel void reduce_per_ray_stage2(
  __global const float* partial_sums,  // [256 * num_blocks_per_ray]
  __global float* final_means,         // [256] output
  int num_partial_results,             // num_blocks_per_ray
  int n_points                         // 4M (для деления)
) {
  int ray_idx = get_global_id(0);  // 0..255

  if (ray_idx < 256) {
    float sum = 0.0f;
    for (int i = 0; i < num_partial_results; i++) {
      sum += partial_sums[ray_idx * num_partial_results + i];
    }
    final_means[ray_idx] = sum / (float)n_points;  // Среднее
  }
}
```

---

### Параметры запуска

**Для 4M элементов на луч:**

**Stage 1 parameters:**
```cpp
int threads_per_block = 256;
int elements_per_thread = 64;  // Thread coarsening
int blocks_per_ray = (4000000 + (threads_per_block * elements_per_thread) - 1)
                     / (threads_per_block * elements_per_thread);
// blocks_per_ray ≈ 245

size_t global_work_size[2] = {256, blocks_per_ray};  // [rays, blocks_per_ray]
size_t local_work_size[2] = {1, 256};  // [1, threads]

clEnqueueNDRangeKernel(queue, kernel_stage1, 2, nullptr,
                       global_work_size, local_work_size, ...);
```

**Stage 2 parameters:**
```cpp
size_t global_work_size = 256;  // One thread per ray
size_t local_work_size = 64;    // Arbitrary workgroup size

clEnqueueNDRangeKernel(queue, kernel_stage2, 1, nullptr,
                       &global_work_size, &local_work_size, ...);
```

---

### Оптимизации

**1. Vectorized loads (float4):**
```opencl
// Вместо:
sum += input[point_idx * 256 + ray_idx];

// Использовать:
float4 values = vload4(0, &input[point_idx * 256 + ray_idx]);
sum += values.x + values.y + values.z + values.w;
```

**2. Loop unrolling:**
```opencl
#pragma unroll 4
for (int i = 0; i < elements_per_thread; i++) {
  // ...
}
```

**3. Wavefront-level shuffle для финальной редукции:**
```opencl
// Для последних 64 элементов (AMD wavefront size)
if (stride <= 32) {
  // Использовать DPP или __shfl_down вместо shared memory
}
```

**4. Использование rocPRIM для ROCm backend:**
```cpp
// В HIP коде
for (int ray = 0; ray < 256; ++ray) {
  rocprim::reduce(
    d_input + ray,        // Start (stride 256)
    d_output + ray,       // Output
    4000000,              // Elements
    rocprim::plus<float>(),
    256,                  // Input stride
    stream
  );
}
```

---

### Performance ожидания

**Theoretical analysis:**

**Bandwidth-bound:**
- Чтение: 4M × 256 × 4 bytes = 4 GB
- Запись: 256 × 4 bytes = 1 KB (negligible)
- Для GPU с bandwidth 500 GB/s: минимум ~8 ms

**Compute-bound:**
- Operations: 4M × 256 сложений ≈ 1B ops
- Для GPU с 10 TFLOPs: ~0.1 ms (negligible)

**Conclusion:** Операция bandwidth-bound, ожидается ~10-20 ms на современных GPU

**Практические результаты из литературы:**
- 4M элементов, optimized kernel: 0.04 ms, 384 GB/s bandwidth (single reduction)
- Для 256 независимых лучей с хорошим параллелизмом: ~1-5 ms

---

### Альтернативный подход: Single-pass с atomic operations

**Если память SoA layout недоступна:**

```opencl
__kernel void reduce_atomic(
  __global const float* input,  // [256][4M] AoS layout
  __global float* output,       // [256]
  int n_points
) {
  int ray_idx = get_group_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);

  __local float scratch[256];
  scratch[local_id] = 0.0f;

  // Each thread processes multiple elements
  for (int i = local_id; i < n_points; i += local_size) {
    float value = input[ray_idx * n_points + i];
    scratch[local_id] += value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction in shared memory
  for (int stride = local_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      scratch[local_id] += scratch[local_id + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    atomic_add_global(&output[ray_idx], scratch[0]);
  }
}
```

Но этот подход медленнее из-за non-coalesced access.

---

## 📖 Дополнительные источники

### Статьи и туториалы
- [Optimizing Parallel Reduction in CUDA (Mark Harris, NVIDIA)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [GPU MODE Lecture 9: Reductions](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/)
- [Parallel Reduction Blog Post](https://dournac.org/info/gpu_sum_reduction)
- [7 Step Optimization Guide](https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8)

### Документация
- [ROCm HIP Reduction Tutorial](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html)
- [rocPRIM Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/)
- [OpenCL Optimization Guide (ROCm)](https://rocmdoc.readthedocs.io/en/latest/Programming_Guides/Opencl-optimization.html)
- [OpenCL Best Practices Guide (NVIDIA)](https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf)

### GitHub примеры
- [NVIDIA OpenCL Reduction Examples](https://github.com/sschaetz/nvidia-opencl-examples/tree/master/OpenCL/src/oclReduction)
- [OpenCL Reduction Sum](https://github.com/maoshouse/OpenCL-reduction-sum)
- [KernelTuner Reduction Example](https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/reduction.py)
- [rocPRIM Block Reduce](https://github.com/ROCm/rocPRIM/blob/develop/rocprim/include/rocprim/block/block_reduce.hpp)

### Научные статьи
- [Fast and Generic GPU-Based Parallel Reduction (arXiv:1710.07358)](https://arxiv.org/pdf/1710.07358)
- [Performance Characterization of Atomic Operations on AMD GPUs](https://synergy.cs.vt.edu/pubs/papers/elteir-ieeecluster11-atomic-operations.pdf)
- [Atomics-based Integer Sum Reduction in HIP](https://dl.acm.org/doi/fullHtml/10.1145/3547276.3548627)

---

## 🎯 Итоговые рекомендации

### Для проекта GPUWorkLib

**1. Основной метод: Per-Ray Two-Level Hierarchical Reduction**
- Оптимален для 256 лучей × 4M точек
- SoA memory layout для coalesced access
- Sequential addressing для избежания bank conflicts
- Thread coarsening (64 элемента на поток)

**2. Бэкенды:**

**OpenCL:**
- Собственная реализация kernel (см. выше)
- Опционально: использовать clBLAS (если доступна)

**ROCm/HIP:**
- rocPRIM device reduce (рекомендуется)
- Собственная HIP kernel implementation

**3. Оптимизации:**
- Vectorized loads (float4)
- Loop unrolling
- Wavefront shuffle для финального уровня (AMD DPP)
- GPUProfiler для измерения bandwidth utilization

**4. API Design:**
```cpp
// Statistics module API
class GPUMeanCalculator {
public:
  GPUMeanCalculator(std::shared_ptr<DrvGPU> driver);

  // Per-ray mean calculation
  std::vector<float> CalculateMean(
    const float* input_data,     // [n_rays * n_points]
    size_t n_rays,               // 256
    size_t n_points,             // 4M
    MemoryLayout layout = SoA    // SoA или AoS
  );
};
```

**5. Тестирование:**
- Python тесты с NumPy для проверки корректности
- Сравнение с CPU реализацией (np.mean)
- Benchmark для измерения производительности
- Профилирование через GPUProfiler

---

*Исследование завершено: 2026-02-14*
*Автор: Кодо (AI Assistant)*
