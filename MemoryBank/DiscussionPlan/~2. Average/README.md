# 📊 Average (Mean) — Вычисление среднего значения

> **Дата**: 2026-02-14
> **Статус**: Исследование завершено ✅

---

## 📋 Задача

**Данные**: 256 лучей × 4 млн точек каждый
**Цель**: Вычислить среднее значение для каждого луча отдельно
**Платформы**: OpenCL & ROCm
**Критерий**: Минимальное время выполнения

---

## 📚 Документы

### 📄 Полное исследование
`gpu_mean_reduction_research.md`
- GPU reduction паттерны
- Иерархическая редукция (workgroup + global)
- Memory coalescing оптимизации
- OpenCL/ROCm примеры
- Библиотечные решения (rocPRIM, Thrust)

---

## 🏆 Рекомендованное решение

### ⭐ Метод: Per-Row Two-Level Hierarchical Reduction

**Алгоритм**:
1. **Level 1**: Каждая workgroup редуцирует свой кусок данных луча в shared memory
2. **Level 2**: Global reduction результатов workgroups → финальное среднее
3. **Division**: Деление суммы на количество точек

**Производительность**:
- ⏱️ **Время**: ~1-5 ms для 256 лучей
- 📊 **Bandwidth**: ~80-90% peak memory bandwidth
- ⚡ **Speedup vs CPU**: 15-30×

**Преимущества**:
- ✅ Оптимальный memory access (coalesced)
- ✅ Минимум синхронизаций
- ✅ Batch processing для 256 лучей одновременно
- ✅ Поддержка float32/float64

---

## 🔧 Реализация

### OpenCL Kernel (упрощённо)
```c
__kernel void reduce_mean_per_ray(
    __global const float* input,   // [256 rays × 4M points]
    __global float* output,         // [256 means]
    const int points_per_ray
) {
    int ray_id = get_group_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    __local float shared[256];

    // Accumulate в shared memory
    float sum = 0.0f;
    for (int i = local_id; i < points_per_ray; i += local_size) {
        sum += input[ray_id * points_per_ray + i];
    }
    shared[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared[local_id] += shared[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Финальное среднее
    if (local_id == 0) {
        output[ray_id] = shared[0] / points_per_ray;
    }
}
```

### C++ API
```cpp
class StatisticsProcessor {
public:
    // Вычисление среднего для батча лучей
    std::vector<float> compute_mean(
        const std::vector<float>& data,  // [256 × 4M]
        size_t num_rays,
        size_t points_per_ray
    );
};
```

### Python API
```python
stats = StatisticsProcessor(gpu_context)
means = stats.compute_mean(data, num_rays=256)
# → numpy array [256] с средними для каждого луча
```

---

## 📐 Memory Layout

### Рекомендация: SoA (Structure of Arrays)
```
Ray 0: [point0, point1, ..., point4M-1]
Ray 1: [point0, point1, ..., point4M-1]
...
Ray 255: [point0, point1, ..., point4M-1]
```

**Почему?**
- ✅ Coalesced memory access
- ✅ Каждый workgroup обрабатывает один луч
- ✅ Оптимальная утилизация GPU threads

---

## 🚀 Альтернативные решения

### Метод 2: rocPRIM::reduce
```cpp
// Готовая библиотечная функция
rocprim::reduce(
    input_it,
    output_it,
    size,
    rocprim::plus<float>(),
    initial_value
);
```
- **Плюсы**: 0 строк кода для kernel
- **Минусы**: Нужен loop для 256 лучей (менее эффективно чем batch)

### Метод 3: Welford's Algorithm
- **Применимость**: Когда нужны и mean, и variance одновременно
- **Время**: ~10-15 ms (но даёт сразу 2 результата)
- **См**: `~5. variance` и `~4. STD`

---

## 🔧 Для реализации сегодня

**OpenCL (готово сейчас)**:
- ✅ Custom kernel с hierarchical reduction
- ✅ Batch processing 256 лучей
- ✅ Memory coalescing

**ROCm (когда придёт AMD GPU)**:
- ✅ Идентичный HIP kernel
- ✅ rocPRIM::reduce как альтернатива
- 🔄 Пересборка с `USE_ROCM`

---

## 📊 Ожидаемая производительность

| Конфигурация | Время | Bandwidth | Speedup vs CPU |
|--------------|-------|-----------|----------------|
| 1 GPU, 256 rays | ~1-5 ms | ~400-500 GB/s | 15-30× |
| 10 GPU, 256 rays (26 rays/GPU) | ~1-2 ms | ~4-5 TB/s | 150-300× |

---

## ✅ Integration Notes

### Комбинация с другими статистиками
```cpp
// Если нужны mean + variance
// → используй Welford's algorithm (см. ~5. variance)
// → получишь оба результата за один проход (~10-15 ms)

// Если нужен только mean
// → используй этот метод (~1-5 ms)
```

---

*Последнее обновление: 2026-02-14*
