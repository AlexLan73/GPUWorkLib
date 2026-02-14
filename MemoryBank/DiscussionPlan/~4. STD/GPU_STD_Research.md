# 📊 GPU Standard Deviation Research
## Оптимальные методы вычисления STD на GPU (OpenCL/ROCm)

**Дата**: 2026-02-14
**Контекст**: 256 лучей × 4M точек каждый
**Цель**: Минимальное время выполнения при приемлемой точности
**Платформы**: OpenCL, ROCm (HIP)

---

## 📐 Математические основы

### Определение STD
```
STD = sqrt(Variance)
Variance = E[X²] - (E[X])²
         = (1/N) * Σ(xᵢ²) - ((1/N) * Σ(xᵢ))²
```

### Альтернативные формулы
```
Variance = (1/N) * Σ((xᵢ - μ)²)    # Two-pass (классическая)
         = E[X²] - (E[X])²          # One-pass naive
```

---

## 🔬 Методы вычисления

### 1️⃣ Two-Pass Algorithm (Классический)

#### Описание
Двухпроходный алгоритм — самый простой и интуитивный подход.

#### Математика
```
Pass 1: μ = (1/N) * Σ(xᵢ)
Pass 2: σ² = (1/N) * Σ((xᵢ - μ)²)
        σ = sqrt(σ²)
```

#### Псевдокод
```cpp
// Pass 1: Compute mean
float sum = parallel_reduce_sum(data, N);
float mean = sum / N;

// Pass 2: Compute variance
float sum_squared_diff = parallel_reduce_sum_squared_diff(data, mean, N);
float variance = sum_squared_diff / N;
float std = sqrt(variance);
```

#### Численная устойчивость
- ✅ **Высокая**: Вычитание (xᵢ - μ) создает малые числа, уменьшая риск переполнения
- ✅ **Нет катастрофической отмены**: Разности всегда возводятся в квадрат
- ⚠️ **Требует 2 прохода по данным**: Увеличивает memory bandwidth usage

#### Сложность
- **Время**: O(N) × 2 (два прохода)
- **Память**: O(1) для результата + temp для mean
- **Bandwidth**: 2 × N × sizeof(float) чтений

#### Преимущества
- Простота реализации
- Численная устойчивость
- Понятная логика

#### Недостатки
- **Два прохода по данным** — основная проблема для 4M элементов
- Необходимость синхронизации между проходами (kernel launches)
- Удвоенное использование memory bandwidth

---

### 2️⃣ One-Pass Naive Algorithm

#### Описание
Вычисление за один проход с использованием E[X²] - (E[X])².

#### Математика
```
σ² = (1/N) * Σ(xᵢ²) - ((1/N) * Σ(xᵢ))²
   = E[X²] - (E[X])²
```

#### Псевдокод
```cpp
// Single pass: Compute sum and sum of squares
struct SumPair { float sum; float sum_sq; };

SumPair result = parallel_reduce_sum_and_sumsq(data, N);
float mean = result.sum / N;
float variance = (result.sum_sq / N) - (mean * mean);
float std = sqrt(variance);
```

#### Численная устойчивость
- ❌ **НИЗКАЯ**: Подвержена катастрофической отмене (catastrophic cancellation)
- ❌ **Проблема**: Когда E[X²] ≈ (E[X])², вычитание уничтожает значащие цифры
- ❌ **Критично для float32**: При больших значениях и малой дисперсии

#### Сложность
- **Время**: O(N) × 1 (один проход)
- **Память**: O(1) для результата (sum + sum_sq)
- **Bandwidth**: 1 × N × sizeof(float) чтений

#### Преимущества
- **Один проход** — быстрее для больших данных
- Простая реализация
- Эффективное использование bandwidth

#### Недостатки
- **Катастрофическая отмена** при малых дисперсиях
- Проблемы с точностью для float32
- Риск overflow при больших N и значениях

#### Пример катастрофы
```
N = 4,000,000
X = [1000000.0, 1000000.1, 1000000.2, ...]  # малая дисперсия, большие значения

E[X²] = 1.000000200000e12   (огромное число)
(E[X])² = 1.000000199999e12 (почти такое же)

Variance = E[X²] - (E[X])²
         = 1.000000200000e12 - 1.000000199999e12
         = 0.000000001e12    # потеря точности!
```

---

### 3️⃣ Welford's Online Algorithm (Рекомендуемый)

#### Описание
Численно устойчивый однопроходный алгоритм, разработанный Б.П. Велфордом (1962).

#### Математика
```
Для каждого xᵢ (i = 1..N):
  n = i
  δ = xᵢ - μₙ₋₁
  μₙ = μₙ₋₁ + δ/n
  M2ₙ = M2ₙ₋₁ + δ * (xᵢ - μₙ)

Variance = M2ₙ / N
STD = sqrt(Variance)
```

#### Псевдокод (Sequential)
```cpp
float mean = 0.0f;
float M2 = 0.0f;
int count = 0;

for (int i = 0; i < N; i++) {
  count++;
  float delta = data[i] - mean;
  mean += delta / count;
  float delta2 = data[i] - mean;
  M2 += delta * delta2;
}

float variance = M2 / count;
float std = sqrt(variance);
```

#### Parallel Welford (Merge)
Welford можно распараллелить через merge операцию:

```cpp
// Merge two Welford states
struct WelfordState {
  int count;
  float mean;
  float M2;
};

WelfordState merge(WelfordState a, WelfordState b) {
  WelfordState result;
  result.count = a.count + b.count;

  float delta = b.mean - a.mean;
  result.mean = a.mean + delta * b.count / result.count;

  result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / result.count;

  return result;
}
```

#### GPU Реализация
```
1. Каждый thread вычисляет локальный Welford для своих элементов
2. Block-level reduction: merge локальных Welford через shared memory
3. Device-level reduction: merge block результатов (второй kernel или atomic)
```

#### Численная устойчивость
- ✅ **ВЫСОКАЯ**: Специально разработан против catastrophic cancellation
- ✅ **Устойчив к overflow**: Обновления инкрементальные, малые значения
- ✅ **Точность**: Даже для float32 при больших N
- ⚠️ **Trade-off**: Больше операций на элемент (division внутри цикла)

#### Сложность
- **Время**: O(N) × 1 проход + O(log N) для merge tree
- **Память**: O(P) для P параллельных workers (count, mean, M2)
- **Bandwidth**: 1 × N × sizeof(float) чтений
- **Арифметика**: ~6-7 операций на элемент (vs 2-3 для naive)

#### Преимущества
- **Численная устойчивость** — лучший из однопроходных
- **Один проход** — эффективен для больших данных
- **Параллелизуется** через merge операцию
- **Универсальность** — работает для любых значений и дисперсий

#### Недостатки
- Сложнее реализация (merge logic)
- Больше арифметики на элемент (division)
- Требует careful merge implementation для GPU

---

### 4️⃣ Compensated Summation (Kahan + Variance)

#### Описание
Использование алгоритма Кахана для повышения точности суммирования в naive one-pass.

#### Kahan Summation
```cpp
float sum = 0.0f;
float c = 0.0f;  // running compensation

for (int i = 0; i < N; i++) {
  float y = data[i] - c;
  float t = sum + y;
  c = (t - sum) - y;
  sum = t;
}
```

#### Применение к Variance
```cpp
// Kahan для Σ(xᵢ) и Σ(xᵢ²)
float sum = 0.0f, c_sum = 0.0f;
float sum_sq = 0.0f, c_sq = 0.0f;

for (int i = 0; i < N; i++) {
  // Kahan for sum
  float y1 = data[i] - c_sum;
  float t1 = sum + y1;
  c_sum = (t1 - sum) - y1;
  sum = t1;

  // Kahan for sum_sq
  float sq = data[i] * data[i];
  float y2 = sq - c_sq;
  float t2 = sum_sq + y2;
  c_sq = (t2 - sum_sq) - y2;
  sum_sq = t2;
}

float variance = (sum_sq / N) - (sum / N) * (sum / N);
```

#### Численная устойчивость
- ✅ **Улучшенная точность**: Kahan компенсирует ошибки округления
- ⚠️ **Не решает catastrophic cancellation**: Проблема остается в вычитании E[X²] - (E[X])²
- ✅ **Хорошо для суммирования**: Особенно при большом N

#### Сложность
- **Время**: O(N) × 1, но 4× больше арифметики (Kahan overhead)
- **Память**: O(1) + compensation variables
- **Bandwidth**: 1 × N × sizeof(float)
- **Арифметика**: ~12-15 операций на элемент

#### GPU Challenges
- ⚠️ **Плохо параллелится**: Kahan требует sequential operations
- ⚠️ **Pairwise summation лучше для GPU**: При merge в GPU block получается pairwise, которое уже numerically stable
- ❌ **Overhead без выигрыша**: GPU block reduction даёт pairwise stability бесплатно

#### Преимущества
- Повышает точность суммирования
- Один проход по данным

#### Недостатки
- **Не подходит для GPU parallel reduction** (pairwise лучше)
- Высокая арифметическая сложность
- Не решает главную проблему (catastrophic cancellation)
- 4× latency vs simple summation

---

## 🚀 GPU Реализации

### Parallel Reduction Pattern

Основа всех GPU variance алгоритмов — parallel reduction.

#### Block-Level Reduction (Shared Memory)
```cpp
__kernel void reduce_sum(__global float* input,
                         __global float* output,
                         __local float* shared,
                         int N) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int block_size = get_local_size(0);

  // Load to shared memory
  shared[tid] = (gid < N) ? input[gid] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Reduction tree
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write block result
  if (tid == 0) {
    output[get_group_id(0)] = shared[0];
  }
}
```

#### Two-Pass Variance Kernel (OpenCL)
```cpp
// Kernel 1: Compute mean
__kernel void compute_mean(__global float* input,
                           __global float* block_sums,
                           __local float* shared,
                           int N) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int block_size = get_local_size(0);

  // Load and reduce
  shared[tid] = (gid < N) ? input[gid] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) block_sums[get_group_id(0)] = shared[0];
}

// Kernel 2: Compute variance given mean
__kernel void compute_variance(__global float* input,
                               __global float* block_vars,
                               __local float* shared,
                               float mean,
                               int N) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int block_size = get_local_size(0);

  // Compute (x - mean)^2 and reduce
  float diff = (gid < N) ? (input[gid] - mean) : 0.0f;
  shared[tid] = diff * diff;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) block_vars[get_group_id(0)] = shared[0];
}
```

#### One-Pass Naive Kernel (OpenCL)
```cpp
struct SumPair {
  float sum;
  float sum_sq;
};

__kernel void compute_variance_onepass(__global float* input,
                                       __global struct SumPair* block_results,
                                       __local struct SumPair* shared,
                                       int N) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int block_size = get_local_size(0);

  // Load and compute sum + sum_sq
  float val = (gid < N) ? input[gid] : 0.0f;
  shared[tid].sum = val;
  shared[tid].sum_sq = val * val;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Reduction
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid].sum += shared[tid + s].sum;
      shared[tid].sum_sq += shared[tid + s].sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) block_results[get_group_id(0)] = shared[0];
}
```

#### Welford Parallel Kernel (OpenCL)
```cpp
struct WelfordState {
  int count;
  float mean;
  float M2;
};

// Helper: Merge two Welford states
inline struct WelfordState welford_merge(struct WelfordState a,
                                         struct WelfordState b) {
  struct WelfordState result;
  result.count = a.count + b.count;

  float delta = b.mean - a.mean;
  result.mean = a.mean + delta * (float)b.count / result.count;
  result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / result.count;

  return result;
}

__kernel void welford_variance(__global float* input,
                               __global struct WelfordState* block_results,
                               __local struct WelfordState* shared,
                               int N) {
  int tid = get_local_id(0);
  int gid = get_global_id(0);
  int block_size = get_local_size(0);

  // Each thread computes local Welford
  struct WelfordState local;
  local.count = 0;
  local.mean = 0.0f;
  local.M2 = 0.0f;

  // Process elements assigned to this thread
  for (int i = gid; i < N; i += get_global_size(0)) {
    local.count++;
    float delta = input[i] - local.mean;
    local.mean += delta / local.count;
    float delta2 = input[i] - local.mean;
    local.M2 += delta * delta2;
  }

  shared[tid] = local;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Block reduction with merge
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = welford_merge(shared[tid], shared[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) block_results[get_group_id(0)] = shared[0];
}
```

---

### ROCm (HIP) Реализации

#### rocPRIM Device Reduce
```cpp
#include <rocprim/rocprim.hpp>

// Custom reduction operator for (sum, sum_sq)
struct variance_op {
  struct result_type {
    float sum;
    float sum_sq;
  };

  __device__ result_type operator()(const result_type& a,
                                    const result_type& b) const {
    return {a.sum + b.sum, a.sum_sq + b.sum_sq};
  }
};

// Transform input to (x, x²)
struct to_sum_pair {
  __device__ variance_op::result_type operator()(float x) const {
    return {x, x * x};
  }
};

// Usage
void compute_variance_rocprim(float* d_input, int N,
                              float* h_variance, hipStream_t stream) {
  // Allocate temp storage
  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  variance_op::result_type* d_output;
  hipMalloc(&d_output, sizeof(variance_op::result_type));

  // Get temp storage size
  rocprim::transform_reduce(
    d_temp, temp_bytes,
    d_input, d_output, N,
    to_sum_pair{},
    variance_op::result_type{0.0f, 0.0f},
    variance_op{},
    stream
  );

  hipMalloc(&d_temp, temp_bytes);

  // Execute reduction
  rocprim::transform_reduce(
    d_temp, temp_bytes,
    d_input, d_output, N,
    to_sum_pair{},
    variance_op::result_type{0.0f, 0.0f},
    variance_op{},
    stream
  );

  // Copy result and compute variance
  variance_op::result_type h_result;
  hipMemcpy(&h_result, d_output, sizeof(h_result), hipMemcpyDeviceToHost);

  float mean = h_result.sum / N;
  *h_variance = (h_result.sum_sq / N) - (mean * mean);

  hipFree(d_temp);
  hipFree(d_output);
}
```

#### HIP Welford Kernel
```cpp
__global__ void welford_kernel(const float* __restrict__ input,
                               WelfordState* block_results,
                               int N) {
  __shared__ WelfordState shared[256];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Local accumulation
  WelfordState local = {0, 0.0f, 0.0f};

  for (int i = gid; i < N; i += stride) {
    local.count++;
    float delta = input[i] - local.mean;
    local.mean += delta / local.count;
    local.M2 += delta * (input[i] - local.mean);
  }

  shared[tid] = local;
  __syncthreads();

  // Block reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = welford_merge(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_results[blockIdx.x] = shared[0];
  }
}
```

---

## 📊 Сравнительная таблица

| Метод | Проходов | Numerical Stability | GPU Efficiency | Precision (float32) | Complexity (арифм.) | Рекомендация |
|-------|----------|---------------------|----------------|---------------------|---------------------|--------------|
| **Two-Pass** | 2 | ✅ Высокая | ⚠️ Средняя (2× BW) | ✅ Отлично | ~4 ops/elem | Для малых N, высокая точность |
| **One-Pass Naive** | 1 | ❌ Низкая | ✅ Высокая | ❌ Плохо для малой σ² | ~3 ops/elem | ❌ Избегать |
| **Welford** | 1 | ✅ Очень высокая | ✅ Хорошая (merge) | ✅ Отлично | ~7 ops/elem | ✅ **Рекомендуется** |
| **Kahan + Naive** | 1 | ⚠️ Средняя | ❌ Низкая (sequential) | ⚠️ Лучше naive | ~15 ops/elem | ❌ Не для GPU |

### Ключевые метрики для 256 × 4M задачи

| Метод | Memory Reads | Kernel Launches | Expected Time (relative) | Precision Loss Risk |
|-------|--------------|-----------------|--------------------------|---------------------|
| Two-Pass | 2 × 4M × 256 = 2048M | 2 × 256 = 512 | 1.0× (baseline) | Низкий |
| One-Pass Naive | 1 × 4M × 256 = 1024M | 1 × 256 = 256 | **0.5×** (fastest) | **Высокий** ⚠️ |
| Welford | 1 × 4M × 256 = 1024M | 1 × 256 = 256 | **0.6×** (очень быстро) | Очень низкий ✅ |
| Kahan | 1 × 4M × 256 = 1024M | 1 × 256 = 256 | 0.8× (overhead) | Средний |

---

## 🎯 Рекомендации для задачи 256 × 4M

### Оптимальный выбор: **Welford's Algorithm**

#### Обоснование
1. **Один проход** — критично для 4M элементов (минимизация bandwidth)
2. **Численная устойчивость** — защита от catastrophic cancellation
3. **Параллелизуется** — хорошо ложится на GPU через merge
4. **Float32 безопасен** — даже при больших N и малых дисперсиях
5. **Batch 256 лучей** — каждый луч независимо обрабатывается

#### Архитектура реализации
```
Input: 256 лучей × 4M точек = 1024M элементов

Стратегия 1: Независимые kernels (256 запусков)
  - Каждый луч — отдельный kernel (4M reduction)
  - Параллелизм на уровне лучей

Стратегия 2: Unified kernel (1 запуск)
  - Kernel обрабатывает все 256 лучей одновременно
  - 256 work-groups, каждая для своего луча
  - Оптимальная загрузка GPU

Рекомендуемая: Стратегия 2 (unified kernel)
  - Минимизация kernel launch overhead
  - Лучшая утилизация GPU
  - Простая синхронизация
```

#### OpenCL Implementation Plan
```cpp
// 1. Kernel на весь batch (256 лучей)
__kernel void batch_welford_variance(
  __global float* input,        // [256][4M] данные
  __global WelfordState* output, // [256] результаты
  __local WelfordState* shared,  // [BLOCK_SIZE] shared memory
  int points_per_beam            // 4M
) {
  int beam_id = get_group_id(0);  // 0..255
  int tid = get_local_id(0);
  int block_size = get_local_size(0);

  // Offset для текущего луча
  __global float* beam_data = input + beam_id * points_per_beam;

  // Local Welford accumulation
  WelfordState local = {0, 0.0f, 0.0f};

  for (int i = tid; i < points_per_beam; i += block_size) {
    welford_update(&local, beam_data[i]);
  }

  shared[tid] = local;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Block reduction
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = welford_merge(shared[tid], shared[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write result for this beam
  if (tid == 0) {
    output[beam_id] = shared[0];
  }
}

// 2. Host code
void compute_batch_std(GPUContext& ctx,
                       cl_mem d_input,      // 256 × 4M
                       float* h_std,        // [256] результаты
                       int num_beams,       // 256
                       int points_per_beam) // 4M
{
  const int BLOCK_SIZE = 256;

  // Allocate output
  cl_mem d_output = ctx.CreateBuffer(
    num_beams * sizeof(WelfordState),
    CL_MEM_READ_WRITE
  );

  // Configure kernel
  cl_kernel kernel = ctx.GetKernel("batch_welford_variance");
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
  clSetKernelArg(kernel, 2, BLOCK_SIZE * sizeof(WelfordState), NULL); // local
  clSetKernelArg(kernel, 3, sizeof(int), &points_per_beam);

  // Launch: 256 work-groups × 256 work-items
  size_t global_size = num_beams * BLOCK_SIZE;
  size_t local_size = BLOCK_SIZE;

  clEnqueueNDRangeKernel(ctx.GetQueue(), kernel, 1,
                         NULL, &global_size, &local_size,
                         0, NULL, NULL);

  // Read results and compute STD
  std::vector<WelfordState> h_states(num_beams);
  clEnqueueReadBuffer(ctx.GetQueue(), d_output, CL_TRUE,
                      0, num_beams * sizeof(WelfordState),
                      h_states.data(), 0, NULL, NULL);

  for (int i = 0; i < num_beams; i++) {
    float variance = h_states[i].M2 / h_states[i].count;
    h_std[i] = std::sqrt(variance);
  }

  clReleaseMemObject(d_output);
}
```

---

### Alternative: Two-Pass для максимальной точности

Если точность критична и bandwidth не проблема:

```cpp
// Pass 1: Compute means for all beams
__kernel void batch_compute_means(
  __global float* input,     // [256][4M]
  __global float* means,     // [256] output
  __local float* shared,     // [BLOCK_SIZE]
  int points_per_beam
) {
  int beam_id = get_group_id(0);
  int tid = get_local_id(0);
  int block_size = get_local_size(0);

  __global float* beam_data = input + beam_id * points_per_beam;

  // Reduction sum
  float sum = 0.0f;
  for (int i = tid; i < points_per_beam; i += block_size) {
    sum += beam_data[i];
  }

  shared[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    means[beam_id] = shared[0] / points_per_beam;
  }
}

// Pass 2: Compute variances using means
__kernel void batch_compute_variances(
  __global float* input,       // [256][4M]
  __global float* means,       // [256] from pass 1
  __global float* variances,   // [256] output
  __local float* shared,       // [BLOCK_SIZE]
  int points_per_beam
) {
  int beam_id = get_group_id(0);
  int tid = get_local_id(0);
  int block_size = get_local_size(0);

  __global float* beam_data = input + beam_id * points_per_beam;
  float mean = means[beam_id];

  // Reduction sum of squared differences
  float sum_sq_diff = 0.0f;
  for (int i = tid; i < points_per_beam; i += block_size) {
    float diff = beam_data[i] - mean;
    sum_sq_diff += diff * diff;
  }

  shared[tid] = sum_sq_diff;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    variances[beam_id] = shared[0] / points_per_beam;
  }
}
```

---

## 🔍 Специфика для 4M элементов

### Precision Considerations

#### Float32 vs Float64
```
float32 (IEEE 754):
  - Mantissa: 23 bits (~7 decimal digits)
  - Exponent: 8 bits
  - Range: ±3.4e38
  - Precision: ~1e-7 relative

float64 (IEEE 754):
  - Mantissa: 52 bits (~16 decimal digits)
  - Exponent: 11 bits
  - Range: ±1.8e308
  - Precision: ~1e-16 relative
```

#### Overflow Risk для N=4M
```
Worst case (One-Pass Naive):
  Max float32 value: ~3.4e38
  Sum of N=4M values at max: 4e6 × 3.4e38 = 1.36e45 > overflow! ⚠️

Realistic scenario:
  Typical signal range: [-1.0, 1.0]
  Sum range: [-4e6, 4e6] ✅ Safe
  Sum² range: [0, 16e12] ✅ Safe

Welford protection:
  Mean updates incrementally: no large sums
  M2 accumulates small terms: numerical stability
```

#### Recommendations
- ✅ **Float32 достаточно** для большинства сигналов
- ⚠️ **Float64 если**: Экстремальные значения или требуется максимальная точность
- ✅ **Welford**: Безопасен даже для float32 при N=4M
- ❌ **Naive one-pass**: Риск catastrophic cancellation, избегать

---

### Memory Bandwidth Optimization

#### Bandwidth Analysis
```
GPU Memory Bandwidth: ~500 GB/s (typical modern GPU)
Data size: 256 лучей × 4M точек × 4 bytes = 4096 MB = 4 GB

Theoretical time:
  One-pass: 4 GB / 500 GB/s = 8 ms
  Two-pass: 8 GB / 500 GB/s = 16 ms

Actual time (with overhead):
  One-pass Welford: ~12-15 ms (estimate)
  Two-pass: ~20-25 ms (estimate)
```

#### Optimization Strategies
1. **Coalesced memory access** — ensure work-items access contiguous memory
2. **Maximize occupancy** — balance registers/shared memory vs active threads
3. **Vector loads** — use float4/float8 loads where possible
4. **Batch processing** — 256 лучей одновременно для amortization launch overhead

---

### Avoiding Numerical Issues

#### Catastrophic Cancellation Example
```cpp
// BAD: One-pass naive with large values
float data[4M] = {1e6, 1e6 + 0.1, 1e6 + 0.2, ...};

float sum = 0.0f, sum_sq = 0.0f;
for (int i = 0; i < 4M; i++) {
  sum += data[i];           // sum ≈ 4e12
  sum_sq += data[i] * data[i]; // sum_sq ≈ 4e18
}

float mean = sum / 4M;         // mean ≈ 1e6
float var = sum_sq / 4M - mean * mean;
// sum_sq/N ≈ 1e12
// mean² ≈ 1e12
// Subtraction loses precision! ❌
```

#### Welford Protection
```cpp
// GOOD: Welford handles large values gracefully
WelfordState state = {0, 0.0f, 0.0f};

for (int i = 0; i < 4M; i++) {
  state.count++;
  float delta = data[i] - state.mean;  // delta is small!
  state.mean += delta / state.count;   // incremental update
  state.M2 += delta * (data[i] - state.mean); // always small terms
}

// M2 accumulates small differences, no catastrophic cancellation ✅
```

---

## 📚 Библиотечные решения

### rocPRIM (ROCm)
**Официальная документация**: [rocPRIM Reduce](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.6.0/device_ops/reduce.html)

**Преимущества**:
- Device-level primitives оптимизированы для AMD GPU
- Header-only, easy integration
- Support для custom operators

**Ограничения**:
- Нет готового variance/std, нужно через custom reduce
- Требует HIP/ROCm (не OpenCL)

**Пример использования**: См. секцию ROCm выше

---

### Thrust (CUDA/ROCm)
**Документация**: [Thrust Reductions](https://docs.nvidia.com/cuda/thrust/)

**ROCm port**: rocThrust доступен для AMD GPU

```cpp
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>

struct variance_functor {
  float mean;

  __host__ __device__
  float operator()(float x) const {
    float diff = x - mean;
    return diff * diff;
  }
};

// Two-pass with Thrust
float compute_std_thrust(thrust::device_vector<float>& data) {
  // Pass 1: mean
  float sum = thrust::reduce(data.begin(), data.end(), 0.0f);
  float mean = sum / data.size();

  // Pass 2: variance
  float variance = thrust::transform_reduce(
    data.begin(), data.end(),
    variance_functor{mean},
    0.0f,
    thrust::plus<float>()
  ) / data.size();

  return std::sqrt(variance);
}
```

**Преимущества**:
- High-level API
- Переносимость (CUDA ↔ ROCm)
- Оптимизированные примитивы

**Недостатки**:
- Two-pass для variance (нет встроенного Welford)
- Overhead для малых задач

---

### Custom Kernels vs Libraries

| Подход | Pros | Cons | Рекомендация |
|--------|------|------|--------------|
| **Custom OpenCL** | Полный контроль, портативность | Больше кода, optimization burden | ✅ Для production |
| **rocPRIM** | Оптимизирован для AMD, low-level control | ROCm-only, custom ops needed | ✅ Для ROCm backend |
| **Thrust/rocThrust** | High-level, easy to use | Two-pass для variance, overhead | ⚠️ Для прототипов |

---

## 🧪 Тестирование и валидация

### Эталонные значения (Python/NumPy)
```python
import numpy as np

# Generate test data
N = 4_000_000
beams = 256
data = np.random.randn(beams, N).astype(np.float32)

# Reference computation
std_ref = np.std(data, axis=1)  # [256] results

# Numerical edge cases
# 1. Large values, small variance
data_large = np.ones((beams, N), dtype=np.float32) * 1e6
data_large += np.random.randn(beams, N).astype(np.float32) * 0.1

# 2. Small values
data_small = np.random.randn(beams, N).astype(np.float32) * 1e-6

# 3. Wide range
data_wide = np.random.randn(beams, N).astype(np.float32) * 1e8
```

### Accuracy Metrics
```python
def test_accuracy(gpu_std, ref_std):
  # Absolute error
  abs_error = np.abs(gpu_std - ref_std)

  # Relative error
  rel_error = abs_error / (np.abs(ref_std) + 1e-10)

  print(f"Max absolute error: {np.max(abs_error):.6e}")
  print(f"Mean absolute error: {np.mean(abs_error):.6e}")
  print(f"Max relative error: {np.max(rel_error):.6e}")
  print(f"Mean relative error: {np.mean(rel_error):.6e}")

  # Acceptance criteria
  assert np.max(rel_error) < 1e-5, "Accuracy test failed!"
```

### Performance Benchmarking
```python
import time

def benchmark_gpu_std(gpu_func, data, iterations=100):
  # Warmup
  for _ in range(10):
    gpu_func(data)

  # Benchmark
  start = time.perf_counter()
  for _ in range(iterations):
    result = gpu_func(data)
  end = time.perf_counter()

  avg_time = (end - start) / iterations * 1000  # ms

  # Bandwidth calculation
  data_size_gb = data.nbytes / 1e9
  bandwidth_gb_s = data_size_gb / (avg_time / 1000)

  print(f"Average time: {avg_time:.3f} ms")
  print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")

  return avg_time
```

---

## 🎓 Научные источники и примеры

### Основные статьи
1. **Welford, B.P. (1962)** - "Note on a method for calculating corrected sums of squares and products"
   - Оригинальная публикация Welford's algorithm

2. **Chan, T.F., Golub, G.H., LeVeque, R.J. (1983)** - "Algorithms for Computing the Sample Variance: Analysis and Recommendations"
   - Comprehensive анализ variance algorithms

3. **Higham, N.J. (2002)** - "Accuracy and Stability of Numerical Algorithms"
   - Детальный анализ numerical stability

### GitHub Примеры
- [Welford C++ Implementation](https://github.com/patrickmineault/welford)
- [PyTorch GroupNorm Welford](https://github.com/pytorch/pytorch/issues/54293) - GPU discussion
- [rocPRIM Examples](https://github.com/ROCm/rocPRIM)

### Документация
- [OpenCL Reduction Examples](https://github.com/rsnemmen/OpenCL-examples)
- [rocPRIM Device Operations](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.6.0/device_ops/)
- [Algorithms for Variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)

---

## 💡 Финальные рекомендации

### Для задачи 256 лучей × 4M точек

#### Primary: Welford's Algorithm (OpenCL)
```cpp
✅ Использовать Welford's parallel algorithm
✅ Unified kernel для всех 256 лучей
✅ Block size: 256 threads
✅ Float32 precision (достаточно для большинства случаев)
✅ Merge-based reduction в shared memory
```

**Ожидаемая производительность**: ~12-15 ms (зависит от GPU)

#### Fallback: Two-Pass для критичной точности
```cpp
⚠️ Два прохода по данным
✅ Максимальная numerical stability
✅ Проще реализация и отладка
⏱️ ~20-25 ms (2× bandwidth)
```

#### Avoid: One-Pass Naive
```cpp
❌ Не использовать из-за catastrophic cancellation
❌ Проблемы с точностью для float32
❌ Риск потери точности при малых дисперсиях
```

---

### Implementation Roadmap

**Phase 1: Prototype (Python)**
- Валидация Welford vs NumPy на CPU
- Тестирование edge cases (large/small values)
- Определение accuracy requirements

**Phase 2: GPU Implementation (OpenCL)**
- Welford kernel для single beam
- Batch kernel для 256 beams
- Unit tests против NumPy reference

**Phase 3: Optimization**
- Профилирование через GPUProfiler
- Memory access optimization
- Occupancy tuning

**Phase 4: ROCm Backend (Optional)**
- HIP port using rocPRIM
- Performance comparison OpenCL vs ROCm
- AMD GPU specific optimizations

---

## 📖 Sources

### Web Search Results
- [Fast and Generic GPU Parallel Reduction](https://arxiv.org/pdf/1710.07358)
- [ROCm OpenCL Programming Guide](https://cgmb-rocm-docs.readthedocs.io/en/latest/Programming_Guides/Opencl-programming-guide.html)
- [Welford's Algorithm Discussion (NVIDIA)](https://forums.developer.nvidia.com/t/welfords-algorithm/325669)
- [Numerically Stable Welford Algorithm](https://nullbuffer.com/articles/welford_algorithm.html)
- [Algorithms for Calculating Variance - Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [rocPRIM Reduce Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.6.0/device_ops/reduce.html)
- [Catastrophic Cancellation - Wikipedia](https://en.wikipedia.org/wiki/Catastrophic_cancellation)
- [Kahan Summation Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
- [Parallel Welford Merge](https://github.com/a-mitani/welford)
- [OpenCL Examples GitHub](https://github.com/rsnemmen/OpenCL-examples)

---

**Автор**: Кодо (AI Assistant)
**Дата создания**: 2026-02-14
**Версия**: 1.0
