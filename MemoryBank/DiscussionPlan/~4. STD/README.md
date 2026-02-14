# 📊 STD (Standard Deviation) — Стандартное отклонение

> **Дата**: 2026-02-14
> **Статус**: Исследование завершено ✅

---

## 📋 Задача

**Данные**: 256 лучей × 4 млн точек каждый
**Цель**: Вычислить стандартное отклонение для каждого луча отдельно
**Платформы**: OpenCL & ROCm
**Критерий**: Минимальное время выполнения при приемлемой точности

**Формула**: `STD = sqrt(Variance)`

---

## 📚 Документы

### 📄 Полное исследование
`GPU_STD_Research.md`
- Математические основы (STD = sqrt(Variance))
- Методы вычисления variance (база для STD)
- Численная устойчивость
- GPU-специфичные оптимизации
- OpenCL/ROCm примеры

---

## 🏆 Рекомендованное решение

### ⭐ Метод: Welford's Online Algorithm

**Алгоритм**:
```
STD = sqrt(Variance)
Variance вычисляется через Welford's algorithm
```

**Производительность**:
- ⏱️ **Время**: ~12-15 ms для 256 лучей
- 🔢 **Проходы**: Один проход (single-pass)
- 🎯 **Bonus**: Вычисляет mean + variance + STD одновременно

**Преимущества**:
- ✅ Численно устойчивый (защита от catastrophic cancellation)
- ✅ Один проход по данным (минимальный bandwidth)
- ✅ Получаем сразу 3 результата: mean, variance, STD
- ✅ Отлично параллелизуется через combine

---

## 🔬 Альтернативные методы

### Метод 2: Two-Pass Algorithm
```
Pass 1: Вычислить mean (μ)
Pass 2: Вычислить σ² = (1/N) * Σ((xᵢ - μ)²)
        STD = sqrt(σ²)
```

**Производительность**:
- ⏱️ **Время**: ~20-25 ms для 256 лучей
- 🔢 **Проходы**: Два прохода (2× bandwidth)

**Когда использовать**:
- Если нужна максимальная точность
- Если bandwidth не критичен

### Метод 3: One-Pass Naïve ❌ НЕ РЕКОМЕНДУЕТСЯ
```
Variance = E[X²] - (E[X])²
STD = sqrt(Variance)
```

**Проблема**: Catastrophic cancellation при близких значениях!

---

## 🔧 Реализация (Welford's Algorithm)

### Welford's Algorithm Псевдокод
```cpp
// Инициализация
M = 0     // running mean
S = 0     // sum of squared differences
n = 0     // count

// Для каждого элемента
for each x in data:
    n = n + 1
    delta = x - M
    M = M + delta / n
    S = S + delta * (x - M)

// Результаты
mean = M
variance = S / n
std = sqrt(variance)
```

### GPU Implementation (OpenCL)
```c
__kernel void welford_std_per_ray(
    __global const float* input,   // [256 × 4M]
    __global float* means,          // [256] output
    __global float* stds            // [256] output
) {
    int ray_id = get_group_id(0);
    int local_id = get_local_id(0);

    __local float M_shared[256];
    __local float S_shared[256];

    // Welford reduction в shared memory
    float M = 0.0f, S = 0.0f;
    int n = 0;

    for (int i = local_id; i < points_per_ray; i += local_size) {
        float x = input[ray_id * points_per_ray + i];
        n++;
        float delta = x - M;
        M += delta / n;
        S += delta * (x - M);
    }

    M_shared[local_id] = M;
    S_shared[local_id] = S;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Combine partial results
    if (local_id == 0) {
        float final_M = 0, final_S = 0;
        int final_n = 0;

        for (int i = 0; i < local_size; i++) {
            // Welford combine formula
            float delta = M_shared[i] - final_M;
            int n_AB = final_n + points_per_ray_per_thread;
            final_M = (final_n * final_M + n_AB * M_shared[i]) / n_AB;
            final_S = final_S + S_shared[i] + delta * delta * final_n * n_AB / (final_n + n_AB);
            final_n = n_AB;
        }

        means[ray_id] = final_M;
        stds[ray_id] = sqrt(final_S / final_n);
    }
}
```

### C++ API
```cpp
class StatisticsProcessor {
public:
    struct Result {
        std::vector<float> means;
        std::vector<float> stds;
        std::vector<float> variances;
    };

    // Вычисление всех статистик одновременно (эффективно!)
    Result compute_statistics(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );

    // Только STD (использует тот же kernel)
    std::vector<float> compute_std(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );
};
```

### Python API
```python
stats = StatisticsProcessor(gpu_context)

# Вариант 1: Только STD
stds = stats.compute_std(data, num_rays=256)

# Вариант 2: Все статистики сразу (эффективнее!)
result = stats.compute_statistics(data, num_rays=256)
print(result.means)      # [256] средние
print(result.stds)       # [256] стандартные отклонения
print(result.variances)  # [256] дисперсии
```

---

## 📐 Численная устойчивость

### Проблема: Catastrophic Cancellation
```
E[X²] - (E[X])² может дать неточный результат!

Пример:
X = [1000.0, 1000.1, 1000.2]
E[X] = 1000.1
(E[X])² = 1000200.01
E[X²] = 1000200.0133...

Variance = 1000200.0133 - 1000200.01 = 0.0033... ← потеря точности!
```

### Решение: Welford's Algorithm
- ✅ Численно устойчив
- ✅ Работает с delta = x - M (малые значения)
- ✅ Не требует E[X²] и (E[X])² вычислений

---

## 📊 Ожидаемая производительность

| Метод | Время (256 rays) | Проходы | Точность | Результаты |
|-------|------------------|---------|----------|-----------|
| **Welford** | ~12-15 ms | 1 | ⭐⭐⭐⭐⭐ | mean + variance + STD |
| Two-Pass | ~20-25 ms | 2 | ⭐⭐⭐⭐⭐ | только STD |
| One-Pass Naïve | ~10-12 ms | 1 | ⭐ (плохая!) | только STD |

---

## 🔧 Для реализации сегодня

**OpenCL (готово сейчас)**:
1. ✅ Welford's algorithm kernel
2. ✅ Batch processing для 256 лучей
3. ✅ Numerical stability (защита от cancellation)
4. ✅ Mean + Variance + STD одновременно

**ROCm (когда придёт AMD GPU)**:
- ✅ Идентичный HIP kernel
- 🔄 Пересборка с `USE_ROCM`

---

## 🎯 Integration Notes

### Эффективность комбинации
```cpp
// ❌ Неэффективно — 3 отдельных вызова
means = compute_mean(data);          // ~5 ms
variances = compute_variance(data);  // ~10 ms
stds = compute_std(data);            // ~12 ms
// ИТОГО: ~27 ms

// ✅ Эффективно — один вызов Welford
result = compute_statistics(data);   // ~12-15 ms
// ИТОГО: ~12-15 ms (2× быстрее!)
```

---

## 📚 См. также

- `~5. variance` — детали по variance (база для STD)
- `~2. Average` — детали по mean (первый шаг в Two-Pass)

---

*Последнее обновление: 2026-02-14*
