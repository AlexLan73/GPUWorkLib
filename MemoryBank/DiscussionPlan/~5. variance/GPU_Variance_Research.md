# GPU Variance Calculation Research

> **Автор**: Кодо (AI Assistant)
> **Дата**: 2026-02-14
> **Контекст**: Модуль Statistics для GPUWorkLib
> **Задача**: 256 лучей × 4 млн точек каждый → 256 variance значений

---

## 📋 Executive Summary

Исследование оптимальных методов вычисления дисперсии (variance) на GPU для OpenCL и ROCm платформ.

**Ключевые выводы:**
- **Для точности**: Welford's algorithm или Compensated two-pass
- **Для скорости**: One-pass с shifted mean (если данные не экстремальные)
- **Универсально**: Pairwise algorithm (баланс скорость/точность)
- **Для 256×4M**: Batched reduction с Welford + SoA layout

---

## 🎯 Математические основы

### Variance Definition

```
Variance σ² = E[(X - μ)²]
            = E[X²] - (E[X])²
            = (1/N) Σ(xᵢ - μ)²
```

Где:
- μ = mean (среднее значение)
- N = количество точек
- xᵢ = отдельные точки данных

### Standard Deviation

```
σ = sqrt(σ²)
```

---

## 🔬 Алгоритмы вычисления Variance

### 1. Two-Pass Algorithm (Наивный)

#### Описание
Классический двухпроходный алгоритм: сначала считаем mean, потом variance.

#### Математика
```
Pass 1: μ = (1/N) Σ xᵢ
Pass 2: σ² = (1/N) Σ(xᵢ - μ)²
```

#### Псевдокод
```cpp
// Pass 1: Compute mean
float sum = 0.0f;
for (i = 0; i < N; i++) {
    sum += x[i];
}
float mean = sum / N;

// Pass 2: Compute variance
float var_sum = 0.0f;
for (i = 0; i < N; i++) {
    float diff = x[i] - mean;
    var_sum += diff * diff;
}
float variance = var_sum / N;
```

#### Численная устойчивость
- ✅ **Хорошая**: результаты стабильны
- ⚠️ **Проблема**: Требует два прохода по данным → 2× memory bandwidth

#### Сложность
- **Passes**: 2
- **Bandwidth**: 2N reads + 1 write
- **FLOPs**: N + 3N = 4N operations
- **Parallel steps**: O(log N) для каждого прохода

#### Trade-offs
- ✅ Простота реализации
- ✅ Численная устойчивость
- ❌ Двойное время выполнения
- ❌ Двойной memory bandwidth

---

### 2. One-Pass Algorithm (Naïve)

#### Описание
Вычисление variance за один проход используя E[X²] - (E[X])².

#### Математика
```
Одновременно накапливаем:
S₁ = Σ xᵢ      (sum)
S₂ = Σ xᵢ²     (sum of squares)

μ = S₁ / N
σ² = S₂/N - (S₁/N)²
```

#### Псевдокод
```cpp
// Single pass: accumulate sum and sum of squares
float sum = 0.0f;
float sum_sq = 0.0f;

for (i = 0; i < N; i++) {
    float x_val = x[i];
    sum += x_val;
    sum_sq += x_val * x_val;
}

float mean = sum / N;
float variance = (sum_sq / N) - (mean * mean);
```

#### Численная устойчивость
- ❌ **ПЛОХАЯ**: Catastrophic cancellation!
- **Проблема**: S₂/N и (S₁/N)² могут быть очень близки
- **Когда fails**:
  - Данные с малой дисперсией (σ² << μ²)
  - Большие значения с малыми отклонениями
  - float32 precision недостаточна
- **Может получиться**: Отрицательная variance!

#### Пример катастрофической отмены
```
Данные: [10000.0, 10000.1, 10000.2]
μ ≈ 10000.1
σ² ≈ 0.0067

S₁/N = 10000.1
S₂/N = 100002000.14
(S₁/N)² = 100002000.01

В float32:
S₂/N ≈ 1.00002e8
(S₁/N)² ≈ 1.00002e8
Результат: 0.0 или даже отрицательное число!
```

#### Сложность
- **Passes**: 1
- **Bandwidth**: N reads + 1 write
- **FLOPs**: 3N operations
- **Parallel steps**: O(log N)

#### Trade-offs
- ✅ Быстрый (один проход)
- ✅ Минимальный memory bandwidth
- ❌ Численная нестабильность
- ❌ Может давать отрицательные результаты
- ⚠️ **НЕ РЕКОМЕНДУЕТСЯ** для production

---

### 3. Welford's Online Algorithm ⭐

#### Описание
Численно устойчивый онлайн-алгоритм, обновляющий mean и variance инкрементально.

#### Математика
```
Инициализация:
M₀ = 0, S₀ = 0

Для каждого xᵢ (i = 1..N):
  Mᵢ = Mᵢ₋₁ + (xᵢ - Mᵢ₋₁) / i
  Sᵢ = Sᵢ₋₁ + (xᵢ - Mᵢ₋₁)(xᵢ - Mᵢ)

Результат:
  mean = Mₙ
  variance = Sₙ / N  (population)
  variance = Sₙ / (N-1)  (sample)
```

#### Псевдокод (Sequential)
```cpp
float M = 0.0f;  // Running mean
float S = 0.0f;  // Running sum of squared differences

for (i = 0; i < N; i++) {
    float delta = x[i] - M;
    M += delta / (i + 1);
    float delta2 = x[i] - M;
    S += delta * delta2;
}

float mean = M;
float variance = S / N;
```

#### Псевдокод (Parallel - Combining)
```cpp
// Каждый thread вычисляет локальный Welford
struct WelfordData {
    int count;
    float mean;
    float M2;  // Sum of squared differences
};

// Combine two Welford states
WelfordData combine(WelfordData a, WelfordData b) {
    WelfordData result;
    result.count = a.count + b.count;

    float delta = b.mean - a.mean;
    result.mean = (a.count * a.mean + b.count * b.mean) / result.count;
    result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / result.count;

    return result;
}

// Final variance
float variance = final.M2 / final.count;
```

#### Численная устойчивость
- ✅ **ОТЛИЧНО**: Численно устойчив
- ✅ Избегает catastrophic cancellation
- ✅ Работает с float32 и float64
- ✅ Никогда не даёт отрицательных результатов

#### Сложность
- **Passes**: 1 (sequential) или log(N) parallel steps для combine
- **Bandwidth**: N reads + 1 write
- **FLOPs**: ~5N operations
- **Parallel steps**: O(log N) для reduction

#### Trade-offs
- ✅ Численная устойчивость
- ✅ Один проход по данным
- ✅ Параллелизуется (через combine)
- ✅ Вычисляет mean и variance одновременно
- ⚠️ Чуть больше операций чем one-pass
- ⚠️ Нужна функция combine для параллелизации

#### GPU Parallelization
Welford's algorithm можно разделить на блоки и распараллелить:
1. Каждый thread/warp обрабатывает свой кусок данных
2. Вычисляет локальный (count, mean, M2)
3. Reduction через combine() в shared memory
4. Final result

---

### 4. Compensated Two-Pass (Chan-Golub-LeVeque)

#### Описание
Улучшенный two-pass алгоритм с компенсацией ошибок округления.

#### Математика
```
Pass 1:
  μ = (1/N) Σ xᵢ

Pass 2:
  Compute residuals and compensated sum
  σ² = (1/N) [Σ(xᵢ - μ)² - (1/N)(Σ(xᵢ - μ))²]

Второй член компенсирует ошибку округления в μ
```

#### Псевдокод
```cpp
// Pass 1: Compute mean
float sum = 0.0f;
for (i = 0; i < N; i++) {
    sum += x[i];
}
float mean = sum / N;

// Pass 2: Compensated variance
float var_sum = 0.0f;
float compensation = 0.0f;

for (i = 0; i < N; i++) {
    float diff = x[i] - mean;
    var_sum += diff * diff;
    compensation += diff;
}

// Apply correction
float variance = (var_sum - compensation * compensation / N) / N;
```

#### Численная устойчивость
- ✅ **ОЧЕНЬ ХОРОШАЯ**: Лучше чем простой two-pass
- ✅ Компенсирует ошибки округления в mean
- ✅ Работает с float32

#### Сложность
- **Passes**: 2
- **Bandwidth**: 2N reads + 1 write
- **FLOPs**: N + 4N = 5N operations
- **Parallel steps**: O(log N) × 2

#### Trade-offs
- ✅ Высокая точность
- ✅ Простота реализации
- ❌ Два прохода (как two-pass)
- ❌ Больше операций

---

### 5. Pairwise Algorithm (Chan-Golub-LeVeque) ⭐

#### Описание
Divide-and-conquer подход: рекурсивно разделяем данные пополам, вычисляем для каждой половины, потом комбинируем.

#### Математика
```
Для двух наборов A и B:
nₐ, nᵦ - количество элементов
μₐ, μᵦ - mean
σ²ₐ, σ²ᵦ - variance

Combined:
n = nₐ + nᵦ
μ = (nₐμₐ + nᵦμᵦ) / n
δ = μᵦ - μₐ

σ² = (nₐσ²ₐ + nᵦσ²ᵦ) / n + (nₐnᵦδ²) / (n²)
```

#### Псевдокод
```cpp
struct Stats {
    int count;
    float mean;
    float variance;
};

Stats pairwise_variance(float* x, int start, int end) {
    int n = end - start;

    // Base case
    if (n == 1) {
        return {1, x[start], 0.0f};
    }

    // Divide
    int mid = start + n / 2;
    Stats left = pairwise_variance(x, start, mid);
    Stats right = pairwise_variance(x, mid, end);

    // Conquer (combine)
    Stats result;
    result.count = left.count + right.count;
    result.mean = (left.count * left.mean + right.count * right.mean) / result.count;

    float delta = right.mean - left.mean;
    float correction = (left.count * right.count * delta * delta) / (result.count * result.count);

    result.variance = (left.count * left.variance + right.count * right.variance) / result.count + correction;

    return result;
}
```

#### Численная устойчивость
- ✅ **ХОРОШАЯ**: Численно стабильна
- ✅ Избегает больших промежуточных сумм
- ✅ Подходит для float32 и float64

#### Сложность
- **Passes**: 1
- **Bandwidth**: N reads + 1 write
- **FLOPs**: ~6N operations
- **Parallel steps**: O(log N) — идеально для GPU!

#### Trade-offs
- ✅ Численная устойчивость
- ✅ Естественная параллелизация (tree reduction)
- ✅ Один проход по данным
- ✅ Оптимально для GPU архитектуры
- ⚠️ Чуть сложнее реализация
- ⚠️ Немного больше операций чем two-pass

#### GPU Parallelization
Идеально подходит для GPU reduction pattern:
- Каждый warp обрабатывает свой блок
- Tree reduction в shared memory
- Minimal synchronization

---

### 6. Shifted Data Algorithm

#### Описание
Вычитаем константу K из всех данных для улучшения численной устойчивости.

#### Математика
```
Выбираем shift K (например, K = x₀ или K ≈ mean)

yᵢ = xᵢ - K

Вычисляем variance для y:
σ²ᵧ = E[y²] - (E[y])²

Variance инвариантна к сдвигу:
σ²ₓ = σ²ᵧ
```

#### Псевдокод
```cpp
// Choose shift constant (e.g., first value or approximate mean)
float K = x[0];  // or K = (x_min + x_max) / 2

// One-pass with shifted data
float sum = 0.0f;
float sum_sq = 0.0f;

for (i = 0; i < N; i++) {
    float y = x[i] - K;
    sum += y;
    sum_sq += y * y;
}

float mean_y = sum / N;
float variance = (sum_sq / N) - (mean_y * mean_y);

// Original mean (if needed)
float mean_x = mean_y + K;
```

#### Численная устойчивость
- ✅ **ХОРОШАЯ**: Улучшает one-pass алгоритм
- ✅ Уменьшает catastrophic cancellation
- ⚠️ Зависит от выбора K

#### Выбор K
- `K = x[0]` — простой вариант
- `K = (x_min + x_max) / 2` — центрируем диапазон
- `K ≈ mean` — идеально, но требует знания mean

#### Сложность
- **Passes**: 1 (+ возможно 1 для выбора K)
- **Bandwidth**: N reads + 1 write
- **FLOPs**: 4N operations
- **Parallel steps**: O(log N)

#### Trade-offs
- ✅ Улучшает one-pass стабильность
- ✅ Один проход
- ⚠️ Не так стабильно как Welford
- ⚠️ Нужно выбирать K

---

### 7. Kahan/Neumaier Compensated Summation

#### Описание
Применяем компенсированную суммацию к one-pass алгоритму для улучшения точности.

#### Математика (Kahan)
```
Для каждой суммации S += x:
  y = x - c        (compensated value)
  t = S + y        (new sum)
  c = (t - S) - y  (new compensation)
  S = t

Где c — накопленная ошибка округления
```

#### Псевдокод (Neumaier variant)
```cpp
float sum = 0.0f;
float sum_comp = 0.0f;  // Compensation for sum
float sum_sq = 0.0f;
float sq_comp = 0.0f;   // Compensation for sum_sq

for (i = 0; i < N; i++) {
    float x_val = x[i];
    float x_sq = x_val * x_val;

    // Compensated sum
    float t = sum + x_val;
    if (abs(sum) >= abs(x_val)) {
        sum_comp += (sum - t) + x_val;
    } else {
        sum_comp += (x_val - t) + sum;
    }
    sum = t;

    // Compensated sum of squares
    t = sum_sq + x_sq;
    if (abs(sum_sq) >= abs(x_sq)) {
        sq_comp += (sum_sq - t) + x_sq;
    } else {
        sq_comp += (x_sq - t) + sum_sq;
    }
    sum_sq = t;
}

sum += sum_comp;
sum_sq += sq_comp;

float mean = sum / N;
float variance = (sum_sq / N) - (mean * mean);
```

#### Численная устойчивость
- ✅ **ХОРОШАЯ**: Значительно лучше чем наивный one-pass
- ✅ Работает с float32
- ⚠️ Все ещё может иметь проблемы с E[X²] - (E[X])²

#### Сложность
- **Passes**: 1
- **Bandwidth**: N reads + 1 write
- **FLOPs**: ~10N operations (много!)
- **Parallel steps**: O(N) — плохо параллелизуется!

#### Trade-offs
- ✅ Улучшает точность суммации
- ✅ Один проход
- ❌ Много операций
- ❌ ПЛОХАЯ параллелизация (sequential по природе)
- ⚠️ **НЕ РЕКОМЕНДУЕТСЯ для GPU**

---

## 📊 Сравнительная таблица методов

| Метод | Passes | Numerical Stability | Parallelization | FLOPs | Complexity | Recommended |
|-------|--------|---------------------|-----------------|-------|------------|-------------|
| **Two-Pass** | 2 | ✅ Good | ✅ Easy | 4N | Low | ⚠️ OK |
| **One-Pass Naïve** | 1 | ❌ Poor | ✅ Easy | 3N | Low | ❌ No |
| **Welford** | 1 | ✅ Excellent | ✅ Good | 5N | Medium | ⭐ Yes |
| **Compensated Two-Pass** | 2 | ✅ Excellent | ✅ Easy | 5N | Medium | ⭐ Yes |
| **Pairwise** | 1 | ✅ Good | ✅ Excellent | 6N | Medium | ⭐ Yes |
| **Shifted Data** | 1 | ⚠️ OK | ✅ Easy | 4N | Low | ⚠️ Maybe |
| **Kahan Compensated** | 1 | ⚠️ Good | ❌ Poor | 10N | High | ❌ No (GPU) |

### Рекомендации по выбору

**Для максимальной точности:**
- **Welford's algorithm** — универсально хорош
- **Compensated two-pass** — если можно два прохода

**Для максимальной скорости (float32):**
- **Pairwise** — оптимальный баланс
- **Shifted data** — если диапазон данных известен

**Для GPU (OpenCL/ROCm):**
- **Pairwise algorithm** — естественная tree reduction
- **Welford + parallel combine** — отлично для batched operations

**Для 256 лучей × 4M точек:**
- **Pairwise** или **Welford** с batched reduction
- SoA memory layout (Structure of Arrays)
- Per-ray parallel processing

---

## 💻 GPU Implementation Considerations

### Memory Access Patterns

#### Coalesced Memory Access
```
Правильно (coalesced):
thread_0 → data[0]
thread_1 → data[1]
thread_2 → data[2]
...
thread_31 → data[31]

Неправильно (strided):
thread_0 → data[0]
thread_1 → data[256]  // stride = 256
thread_2 → data[512]
```

#### Memory Layout: AoS vs SoA

**Array of Structures (AoS)** — ПЛОХО для GPU:
```cpp
struct Ray {
    float data[4000000];
};
Ray rays[256];

// Access pattern (non-coalesced):
thread_0 reads rays[0].data[i]
thread_1 reads rays[1].data[i]
// Memory addresses далеко друг от друга!
```

**Structure of Arrays (SoA)** — ХОРОШО для GPU:
```cpp
float ray_data[256][4000000];
// or better:
float* ray_data;  // size = 256 * 4000000

// Access pattern (coalesced):
thread_0 reads ray_data[ray * 4000000 + i]
thread_1 reads ray_data[ray * 4000000 + i + 1]
// Memory addresses consecutive!
```

**Для 256 rays:**
```cpp
// Вариант 1: Каждый ray — отдельный batch
for (int ray = 0; ray < 256; ray++) {
    variance[ray] = compute_variance_gpu(&data[ray * 4000000], 4000000);
}

// Вариант 2: Все rays одновременно (лучше!)
compute_variance_batched_gpu(data, 256, 4000000, variance);
```

---

### Reduction Pattern

Стандартный GPU reduction pattern:

```
Input: [x0, x1, x2, x3, x4, x5, x6, x7]

Step 1: [x0+x1, x2+x3, x4+x5, x6+x7]
Step 2: [x0+x1+x2+x3, x4+x5+x6+x7]
Step 3: [x0+x1+x2+x3+x4+x5+x6+x7]

Parallel steps: log₂(N)
```

**Применение к Welford:**
```
Step 1: Combine pairs (Welford combine)
Step 2: Combine pairs of pairs
...
Step log₂(N): Final result
```

---

### Workgroup Size Optimization

**NVIDIA GPU (warp = 32 threads):**
- Optimal workgroup: 256, 512 (multiples of 32)
- Shared memory per block: 48 KB

**AMD GPU (wavefront = 64 threads):**
- Optimal workgroup: 256, 512 (multiples of 64)
- LDS (local data share) per workgroup: 64 KB

**Для reduction:**
```cpp
// Optimal для 4M точек
#define WORKGROUP_SIZE 256
#define ITEMS_PER_THREAD 16

// Каждый thread обрабатывает 16 элементов локально
// Потом reduction в shared memory
```

---

### Precision Requirements: float32 vs float64

#### Когда float32 достаточно
- Данные в "нормальном" диапазоне (1e-6 ... 1e6)
- Относительная дисперсия σ²/μ² > 1e-7
- Используем численно устойчивые алгоритмы (Welford, Pairwise)

#### Когда нужен float64
- Очень малая дисперсия относительно mean (σ² << μ²)
- Данные с большими абсолютными значениями
- Накопление большого количества значений (N > 10⁷)
- Критическая точность

#### Performance Impact
```
GPU performance:
- Consumer GPU: FP64 = 1/32 × FP32 (RTX 4090: 1/64)
- Professional GPU: FP64 = 1/2 × FP32 (A100, MI250X)

Для 256 × 4M:
- float32: ~10-20 ms
- float64: ~320-640 ms (consumer) или ~20-40 ms (pro)
```

**Рекомендация для GPUWorkLib:**
- По умолчанию float32 с Welford/Pairwise
- Опционально float64 для критических случаев
- Template или runtime выбор precision

---

## 🔧 OpenCL Implementation Examples

### Example 1: Two-Pass Variance (Basic)

```c
// Kernel 1: Compute mean
__kernel void compute_mean(
    __global const float* data,
    __global float* partial_sums,
    __local float* local_sum,
    int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Each thread accumulates local sum
    float sum = 0.0f;
    for (int i = gid; i < N; i += get_global_size(0)) {
        sum += data[i];
    }

    // Store to local memory
    local_sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sum[lid] += local_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        partial_sums[get_group_id(0)] = local_sum[0];
    }
}

// Kernel 2: Compute variance
__kernel void compute_variance(
    __global const float* data,
    __global float* partial_vars,
    __local float* local_var,
    float mean,
    int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Each thread accumulates local variance
    float var_sum = 0.0f;
    for (int i = gid; i < N; i += get_global_size(0)) {
        float diff = data[i] - mean;
        var_sum += diff * diff;
    }

    // Store to local memory
    local_var[lid] = var_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_var[lid] += local_var[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        partial_vars[get_group_id(0)] = local_var[0];
    }
}
```

---

### Example 2: Welford's Algorithm (Parallel)

```c
// Welford state structure
typedef struct {
    int count;
    float mean;
    float M2;  // Sum of squared differences
} WelfordState;

// Combine two Welford states
WelfordState welford_combine(WelfordState a, WelfordState b) {
    WelfordState result;
    result.count = a.count + b.count;

    float delta = b.mean - a.mean;
    result.mean = (a.count * a.mean + b.count * b.mean) / (float)result.count;
    result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / (float)result.count;

    return result;
}

// Kernel: Welford with local reduction
__kernel void welford_variance(
    __global const float* data,
    __global WelfordState* partial_states,
    __local WelfordState* local_states,
    int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Each thread computes local Welford
    WelfordState state = {0, 0.0f, 0.0f};

    for (int i = gid; i < N; i += get_global_size(0)) {
        float x = data[i];
        state.count++;
        float delta = x - state.mean;
        state.mean += delta / state.count;
        float delta2 = x - state.mean;
        state.M2 += delta * delta2;
    }

    // Store to local memory
    local_states[lid] = state;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory (combine Welford states)
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_states[lid] = welford_combine(local_states[lid], local_states[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        partial_states[get_group_id(0)] = local_states[0];
    }
}
```

**Host code:**
```cpp
// 1. Run kernel to get partial Welford states
clEnqueueNDRangeKernel(..., welford_variance, ...);

// 2. CPU-side final combine
WelfordState final_state = {0, 0.0f, 0.0f};
for (int i = 0; i < num_workgroups; i++) {
    final_state = welford_combine(final_state, partial_states[i]);
}

float variance = final_state.M2 / final_state.count;
float mean = final_state.mean;
```

---

### Example 3: Pairwise Algorithm (Tree Reduction)

```c
// Kernel: Pairwise variance with tree reduction
__kernel void pairwise_variance(
    __global const float* data,
    __global float* result_mean,
    __global float* result_var,
    __local float* local_mean,
    __local float* local_var,
    __local int* local_count,
    int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Each thread processes multiple elements
    int count = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = gid; i < N; i += get_global_size(0)) {
        float x = data[i];
        sum += x;
        sum_sq += x * x;
        count++;
    }

    // Compute local stats
    local_count[lid] = count;
    local_mean[lid] = (count > 0) ? (sum / count) : 0.0f;
    local_var[lid] = (count > 0) ? (sum_sq / count - local_mean[lid] * local_mean[lid]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pairwise tree reduction
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            int na = local_count[lid];
            int nb = local_count[lid + s];
            int n = na + nb;

            if (n > 0) {
                float mean_a = local_mean[lid];
                float mean_b = local_mean[lid + s];
                float var_a = local_var[lid];
                float var_b = local_var[lid + s];

                // Combine means
                float mean = (na * mean_a + nb * mean_b) / (float)n;

                // Combine variances
                float delta = mean_b - mean_a;
                float var = (na * var_a + nb * var_b) / (float)n
                          + (na * nb * delta * delta) / (float)(n * n);

                local_count[lid] = n;
                local_mean[lid] = mean;
                local_var[lid] = var;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        result_mean[get_group_id(0)] = local_mean[0];
        result_var[get_group_id(0)] = local_var[0];
    }
}
```

---

### Example 4: Batched Variance (256 rays)

```c
// Kernel: Compute variance for multiple rays simultaneously
__kernel void batched_variance_welford(
    __global const float* data,        // [num_rays * points_per_ray]
    __global float* output_mean,       // [num_rays]
    __global float* output_variance,   // [num_rays]
    int points_per_ray,
    int num_rays
) {
    // Each workgroup processes one ray
    int ray_id = get_group_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);

    if (ray_id >= num_rays) return;

    __global const float* ray_data = data + ray_id * points_per_ray;

    // Local Welford state per thread
    int count = 0;
    float mean = 0.0f;
    float M2 = 0.0f;

    // Each thread processes subset of points
    for (int i = lid; i < points_per_ray; i += wg_size) {
        float x = ray_data[i];
        count++;
        float delta = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    // Shared memory for reduction
    __local int local_counts[256];
    __local float local_means[256];
    __local float local_M2s[256];

    local_counts[lid] = count;
    local_means[lid] = mean;
    local_M2s[lid] = M2;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Welford reduction in shared memory
    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            int na = local_counts[lid];
            int nb = local_counts[lid + s];
            int n = na + nb;

            if (n > 0) {
                float mean_a = local_means[lid];
                float mean_b = local_means[lid + s];
                float M2_a = local_M2s[lid];
                float M2_b = local_M2s[lid + s];

                float delta = mean_b - mean_a;
                float new_mean = (na * mean_a + nb * mean_b) / (float)n;
                float new_M2 = M2_a + M2_b + delta * delta * na * nb / (float)n;

                local_counts[lid] = n;
                local_means[lid] = new_mean;
                local_M2s[lid] = new_M2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Thread 0 writes final result for this ray
    if (lid == 0) {
        output_mean[ray_id] = local_means[0];
        output_variance[ray_id] = local_M2s[0] / local_counts[0];
    }
}
```

**Host code:**
```cpp
// Dispatch kernel: 256 workgroups (one per ray), 256 threads per workgroup
size_t global_size = 256 * 256;  // 256 rays × 256 threads
size_t local_size = 256;

clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &mean_buffer);
clSetKernelArg(kernel, 2, sizeof(cl_mem), &variance_buffer);
clSetKernelArg(kernel, 3, sizeof(int), &points_per_ray);  // 4000000
clSetKernelArg(kernel, 4, sizeof(int), &num_rays);        // 256

clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
```

---

## 🚀 ROCm/HIP Implementation

ROCm использует HIP — очень похож на CUDA.

### Example: Welford Variance (HIP)

```cpp
#include <hip/hip_runtime.h>

// Welford state
struct WelfordState {
    int count;
    float mean;
    float M2;
};

// Device function: combine Welford states
__device__ WelfordState welford_combine(WelfordState a, WelfordState b) {
    WelfordState result;
    result.count = a.count + b.count;

    float delta = b.mean - a.mean;
    result.mean = (a.count * a.mean + b.count * b.mean) / (float)result.count;
    result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / (float)result.count;

    return result;
}

// HIP Kernel: Welford variance
__global__ void welford_variance_hip(
    const float* __restrict__ data,
    WelfordState* partial_states,
    int N
) {
    __shared__ WelfordState shared_states[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // Local Welford computation
    WelfordState state = {0, 0.0f, 0.0f};

    for (int i = gid; i < N; i += grid_size) {
        float x = data[i];
        state.count++;
        float delta = x - state.mean;
        state.mean += delta / state.count;
        float delta2 = x - state.mean;
        state.M2 += delta * delta2;
    }

    shared_states[tid] = state;
    __syncthreads();

    // Warp-level reduction (AMD wavefront = 64)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_states[tid] = welford_combine(shared_states[tid], shared_states[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_states[blockIdx.x] = shared_states[0];
    }
}

// Host code
void compute_variance_rocm(float* d_data, int N, float* mean, float* variance) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    WelfordState* d_partial;
    hipMalloc(&d_partial, num_blocks * sizeof(WelfordState));

    // Launch kernel
    hipLaunchKernelGGL(
        welford_variance_hip,
        dim3(num_blocks),
        dim3(block_size),
        0, 0,
        d_data, d_partial, N
    );

    // Copy partial results to host
    WelfordState* h_partial = new WelfordState[num_blocks];
    hipMemcpy(h_partial, d_partial, num_blocks * sizeof(WelfordState), hipMemcpyDeviceToHost);

    // Final combine on CPU
    WelfordState final = {0, 0.0f, 0.0f};
    for (int i = 0; i < num_blocks; i++) {
        final = welford_combine(final, h_partial[i]);
    }

    *mean = final.mean;
    *variance = final.M2 / final.count;

    delete[] h_partial;
    hipFree(d_partial);
}
```

---

## 📚 Library Solutions

### 1. rocPRIM (AMD ROCm)

rocPRIM — библиотека параллельных примитивов для ROCm.

**Reduce operation:**
```cpp
#include <rocprim/rocprim.hpp>

// Custom reduce operation for variance
struct variance_op {
    __device__ WelfordState operator()(const WelfordState& a, const WelfordState& b) const {
        return welford_combine(a, b);
    }
};

void compute_variance_rocprim(float* d_data, int N) {
    // 1. Transform data to WelfordState (single element states)
    WelfordState* d_states;
    hipMalloc(&d_states, N * sizeof(WelfordState));

    // Transform kernel (not shown)

    // 2. Reduce using rocPRIM
    WelfordState* d_result;
    hipMalloc(&d_result, sizeof(WelfordState));

    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;

    // Get required temp storage
    rocprim::reduce(
        d_temp_storage, temp_storage_bytes,
        d_states, d_result, WelfordState{0, 0.0f, 0.0f},
        N, variance_op()
    );

    hipMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform reduction
    rocprim::reduce(
        d_temp_storage, temp_storage_bytes,
        d_states, d_result, WelfordState{0, 0.0f, 0.0f},
        N, variance_op()
    );

    // Get result
    WelfordState h_result;
    hipMemcpy(&h_result, d_result, sizeof(WelfordState), hipMemcpyDeviceToHost);

    float variance = h_result.M2 / h_result.count;
}
```

**Примечание:** rocPRIM не имеет встроенных статистических функций, но предоставляет reduce/scan примитивы.

---

### 2. Thrust (NVIDIA/AMD)

Thrust доступен для CUDA и ROCm (через rocThrust).

**Example:**
```cpp
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

// Functor for variance
struct variance_functor {
    float mean;

    variance_functor(float m) : mean(m) {}

    __host__ __device__
    float operator()(float x) const {
        float diff = x - mean;
        return diff * diff;
    }
};

void compute_variance_thrust(thrust::device_vector<float>& data) {
    int N = data.size();

    // 1. Compute mean
    float sum = thrust::reduce(data.begin(), data.end(), 0.0f, thrust::plus<float>());
    float mean = sum / N;

    // 2. Compute variance
    float var_sum = thrust::transform_reduce(
        data.begin(), data.end(),
        variance_functor(mean),
        0.0f,
        thrust::plus<float>()
    );

    float variance = var_sum / N;
}
```

---

### 3. CuPy (Python GPU arrays)

CuPy — NumPy-like API для GPU.

```python
import cupy as cp

# GPU array
data = cp.random.randn(4000000, dtype=cp.float32)

# Compute variance (uses optimized CUDA kernels)
variance = cp.var(data)
mean = cp.mean(data)
std = cp.std(data)

# Batch variance (256 rays)
ray_data = cp.random.randn(256, 4000000, dtype=cp.float32)
variances = cp.var(ray_data, axis=1)  # Per-ray variance
```

**Under the hood:**
- CuPy uses Welford-like algorithm для numerical stability
- Optimized CUDA kernels
- Supports float32/float64

---

### 4. PyTorch

PyTorch также имеет GPU variance:

```python
import torch

# GPU tensor
data = torch.randn(4000000, device='cuda', dtype=torch.float32)

# Compute variance
variance = torch.var(data, unbiased=False)  # Population variance
variance_sample = torch.var(data, unbiased=True)  # Sample variance

# Batch variance (256 rays)
ray_data = torch.randn(256, 4000000, device='cuda', dtype=torch.float32)
variances = torch.var(ray_data, dim=1)  # Per-ray variance
```

---

## 🎯 Рекомендации для GPUWorkLib (256×4M)

### Архитектура

**Модуль**: `Statistics` (уже в Planned)

**API дизайн:**
```cpp
class StatisticsProcessor {
public:
    StatisticsProcessor(DrvGPU::GPUContext& ctx);

    // Single array variance
    float compute_variance(cl_mem data, size_t N, VarianceAlgorithm algo = WELFORD);

    // Batched variance (multiple arrays)
    void compute_variance_batch(
        cl_mem data,           // [num_arrays * points_per_array]
        size_t points_per_array,
        size_t num_arrays,
        cl_mem output_variance, // [num_arrays]
        cl_mem output_mean = nullptr  // optional
    );

    // Combined mean + variance
    struct Stats {
        float mean;
        float variance;
        float std_dev;
    };

    Stats compute_statistics(cl_mem data, size_t N);
};

enum VarianceAlgorithm {
    TWO_PASS,
    WELFORD,
    PAIRWISE,
    AUTO  // Choose based on data size
};
```

---

### Memory Layout для 256 rays × 4M points

**Рекомендуется SoA (Structure of Arrays):**
```cpp
// Вариант 1: Continuous block
float* data = new float[256 * 4000000];
// data layout: [ray0_point0, ray0_point1, ..., ray0_point3999999,
//               ray1_point0, ray1_point1, ..., ray1_point3999999,
//               ...]

// GPU kernel: каждая workgroup обрабатывает один ray
// Coalesced access внутри каждого ray

// Вариант 2: Separate arrays (если rays обрабатываются независимо)
std::vector<cl_mem> ray_buffers(256);
for (int i = 0; i < 256; i++) {
    ray_buffers[i] = clCreateBuffer(..., 4000000 * sizeof(float), ...);
}
```

**Для batched processing:**
```
Memory: [Ray0: 4M points][Ray1: 4M points]...[Ray255: 4M points]

Kernel dispatch:
- 256 workgroups (one per ray)
- 256 threads per workgroup
- Each workgroup computes variance for its ray independently
```

---

### Выбор алгоритма

**Для 256 rays × 4M points:**

1. **Welford's algorithm** — рекомендуется
   - ✅ Численная устойчивость
   - ✅ Single pass
   - ✅ Compute mean + variance одновременно
   - ✅ Хорошо параллелизуется

2. **Pairwise algorithm** — альтернатива
   - ✅ Отличная параллелизация (tree reduction)
   - ✅ Numerical stability
   - ⚠️ Чуть сложнее реализация

3. **НЕ использовать:**
   - ❌ One-pass naïve — риск catastrophic cancellation
   - ❌ Kahan compensated — плохая параллелизация

---

### Workgroup Configuration

**Рекомендации:**
```cpp
// OpenCL configuration
const size_t WORKGROUP_SIZE = 256;  // Good for both NVIDIA and AMD
const size_t NUM_WORKGROUPS = 256;  // One per ray

// For single ray (4M points):
// - Launch 256 workgroups
// - Each thread processes ~16K points (4M / 256 / 64 threads)
// - Reduction in shared memory

// For 256 rays (batched):
// - Launch 256 workgroups (one per ray)
// - Each workgroup: 256 threads
// - Each thread processes ~15.6K points (4M / 256)
```

---

### Precision Strategy

**Default: float32 с Welford**
- Достаточно для большинства случаев
- Численно устойчиво
- Быстрая производительность

**Optional: float64 mode**
- Для критических расчётов
- Compile-time template или runtime switch
- Проверка доступности FP64 на GPU

```cpp
// Template approach
template<typename T>
class StatisticsProcessor {
    // T = float or double
};

// Runtime approach
void compute_variance(
    cl_mem data,
    size_t N,
    float* result,
    bool use_double_precision = false
);
```

---

### Integration с DrvGPU

**Используем существующую инфраструктуру:**
- ✅ `DrvGPU::GPUContext` — управление устройством
- ✅ `CommandQueue` — выполнение kernels
- ✅ `BatchManager` — для больших данных
- ✅ `plog` logger — per-GPU логи
- ✅ `console_output` — вывод на консоль
- ✅ `GPUProfiler` — профилирование

```cpp
class StatisticsProcessor {
private:
    DrvGPU::GPUContext& gpu_ctx_;
    cl_kernel kernel_welford_;
    cl_kernel kernel_pairwise_;

public:
    StatisticsProcessor(DrvGPU::GPUContext& ctx)
        : gpu_ctx_(ctx) {
        // Load kernels from file
        std::string kernel_source = load_kernel_file("statistics_kernels.cl");

        // Build program
        cl_program program = gpu_ctx_.build_program(kernel_source);

        // Create kernels
        kernel_welford_ = clCreateKernel(program, "welford_variance", nullptr);
        kernel_pairwise_ = clCreateKernel(program, "pairwise_variance", nullptr);

        // Log
        gpu_ctx_.log_info("StatisticsProcessor initialized");
    }
};
```

---

### Performance Estimates

**Для 256 rays × 4M points (float32):**

Hardware: AMD Radeon RX 6800 XT (wavefront 64, ~20 TFLOPS FP32)

**Theoretical:**
```
Data size: 256 × 4M × 4 bytes = 4 GB
Memory bandwidth: ~512 GB/s

Welford algorithm:
- FLOPs: 5N × 256 rays = 5 × 4M × 256 = 5.12 GFLOP
- Memory: Read 4 GB + Write 2 KB (results) ≈ 4 GB
- Time (bandwidth bound): 4 GB / 512 GB/s ≈ 7.8 ms
- Time (compute bound): 5.12 GFLOP / 20 TFLOPS ≈ 0.26 ms

Expected: ~10-15 ms (bandwidth bound + overhead)
```

**Comparison:**
```
CPU (single thread, AVX2):
- Time: ~200-300 ms

CPU (16 threads):
- Time: ~15-20 ms

GPU (OpenCL):
- Time: ~10-15 ms

Speedup vs single thread: 15-30×
```

---

## 🔍 Testing Strategy

### Numerical Accuracy Tests

**Test 1: Small variance (catastrophic cancellation test)**
```python
import numpy as np

# Data with very small variance
data = np.array([1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3], dtype=np.float32)

# Reference (NumPy, float64)
ref_var = np.var(data.astype(np.float64))

# GPU result
gpu_var = compute_variance_gpu(data)

# Check relative error
rel_error = abs(gpu_var - ref_var) / ref_var
assert rel_error < 1e-5  # 0.001% error
```

**Test 2: Large N accumulation**
```python
# 4M random points
data = np.random.randn(4_000_000).astype(np.float32)

ref_var = np.var(data.astype(np.float64))
gpu_var = compute_variance_gpu(data)

rel_error = abs(gpu_var - ref_var) / ref_var
assert rel_error < 1e-4  # 0.01% error
```

**Test 3: Known distributions**
```python
# Normal distribution N(μ=5, σ=2)
data = np.random.normal(5.0, 2.0, 1_000_000).astype(np.float32)

gpu_mean, gpu_var = compute_mean_variance_gpu(data)

assert abs(gpu_mean - 5.0) < 0.01
assert abs(np.sqrt(gpu_var) - 2.0) < 0.01
```

---

### Performance Benchmarks

```python
import time

sizes = [1000, 10000, 100000, 1_000_000, 4_000_000, 10_000_000]

for N in sizes:
    data = np.random.randn(N).astype(np.float32)

    # CPU
    start = time.time()
    cpu_var = np.var(data)
    cpu_time = time.time() - start

    # GPU
    d_data = upload_to_gpu(data)
    start = time.time()
    gpu_var = compute_variance_gpu(d_data)
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"N={N}: CPU={cpu_time*1000:.2f}ms, GPU={gpu_time*1000:.2f}ms, Speedup={speedup:.1f}x")
```

---

## 📖 References & Sources

### Academic Papers
- [Chan, Golub, LeVeque (1979) - "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
- [Pébay (2008) - "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments"](https://www.osti.gov/servlets/purl/1028931/)
- [Algorithms for calculating variance - Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)

### GPU Programming
- [NVIDIA - Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [NVIDIA - Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [AMD GPUOpen - Understanding Memory Coalescing on GCN](https://gpuopen.com/learn/gcn-memory-coalescing/)
- [GPU Pattern: Reduction](https://ajdillhoff.github.io/notes/gpu_pattern_reduction/)

### Numerical Stability
- [Catastrophic cancellation - Wikipedia](https://en.wikipedia.org/wiki/Catastrophic_cancellation)
- [Welford algorithm for updating variance](https://changyaochen.github.io/welford/)
- [Kahan summation algorithm - Wikipedia](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

### OpenCL & ROCm
- [rocPRIM documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/what-is-rocm.html)
- [NVIDIA OpenCL Examples - oclReduction](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/)

### Libraries
- [CuPy: NumPy & SciPy for GPU](https://cupy.dev/)
- [Thrust Library](https://nvidia.github.io/cccl/thrust/)
- [rocThrust (AMD)](https://github.com/ROCmSoftwarePlatform/rocThrust)

### Performance & Optimization
- [AoS and SoA - Wikipedia](https://en.wikipedia.org/wiki/AoS_and_SoA)
- [Memory Coalescing | GPU Glossary](https://modal.com/gpu-glossary/perf/memory-coalescing)
- [How to Access Global Memory Efficiently in CUDA](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

---

## ✅ Выводы и рекомендации

### Для GPUWorkLib Statistics Module

1. **Алгоритм по умолчанию: Welford**
   - Численная устойчивость ⭐
   - Single pass
   - Combined mean + variance
   - Хорошая параллелизация

2. **Alternative: Pairwise**
   - Для случаев когда нужна максимальная параллелизация
   - Tree reduction pattern

3. **Memory layout: SoA**
   - Structure of Arrays для 256 rays
   - Coalesced memory access

4. **Precision: float32 + optional float64**
   - Default float32 с Welford достаточно
   - Runtime switch для критических случаев

5. **Batched processing**
   - 256 workgroups (one per ray)
   - Efficient для 256×4M данных

6. **Integration с DrvGPU**
   - Использовать существующую инфраструктуру
   - Logger, Profiler, BatchManager

### Next Steps

1. ✅ Прочитать спецификацию Statistics module
2. ⏳ Реализовать Welford kernel (OpenCL)
3. ⏳ Реализовать batched variance
4. ⏳ Написать unit tests (numerical accuracy)
5. ⏳ Benchmark на реальных данных
6. ⏳ Python bindings
7. ⏳ Документация

---

*Исследование завершено: 2026-02-14*
*Кодо — AI Assistant для GPUWorkLib*
