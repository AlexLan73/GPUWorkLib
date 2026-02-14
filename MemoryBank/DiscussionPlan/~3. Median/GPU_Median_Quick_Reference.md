# GPU Median — Quick Reference

**Задача**: 256 лучей × 4M точек → 256 медиан
**Платформы**: OpenCL, ROCm

---

## TL;DR — Быстрый выбор

### Для быстрого старта (прямо сейчас)
```cpp
// ROCm/rocThrust
rocprim::radix_sort_keys(d_data, d_sorted, 4000000);
float median = d_sorted[2000000];  // середина
```
**Скорость**: ⭐⭐⭐ (хорошо)
**Сложность**: ⭐⭐⭐⭐⭐ (готовые библиотеки)

### Для максимальной скорости (после портации)
**RadiK Algorithm** — портировать с CUDA на OpenCL/HIP
**Скорость**: ⭐⭐⭐⭐⭐ (в 2× быстрее сортировки)
**Сложность**: ⭐⭐⭐ (требует работы)

---

## Методы (краткая таблица)

| Метод | Скорость | Реализация | Рекомендация |
|-------|----------|------------|--------------|
| **RadiK** | Fastest (2.5×) | Портация CUDA | ✅ Best choice |
| **Radix Sort** | Good | rocPRIM ready | ✅ Simplest |
| **SampleSelect** | Fast | Портация CUDA | 🔶 Alternative |
| **Approximate** | Very Fast (4×) | Custom | 🔶 If error OK |
| **Bitonic Sort** | Slow | Доступно | ❌ Too slow |

---

## Производительность (оценка для 256×4M)

| Метод | Время (256 лучей) | Примечания |
|-------|-------------------|------------|
| RadiK (batch) | ~10-30ms | Лучший, batch-оптимизация |
| Radix Sort | ~20-60ms | Простейший, готовый |
| SampleSelect | ~15-40ms | Устойчив к данным |
| Approximate | ~5-20ms | Если ε=0.01 допустимо |

*GPU: современная AMD/NVIDIA карта, ~500 GB/s bandwidth*

---

## Код-примеры

### ROCm (rocPRIM)
```cpp
#include <rocprim/rocprim.hpp>

// Sort + median
void* d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;

// Get temp storage size
rocprim::radix_sort_keys(
    d_temp_storage, temp_storage_bytes,
    d_input, d_sorted, 4000000
);

// Allocate
hipMalloc(&d_temp_storage, temp_storage_bytes);

// Sort
rocprim::radix_sort_keys(
    d_temp_storage, temp_storage_bytes,
    d_input, d_sorted, 4000000
);

// Get median
float median;
hipMemcpy(&median, d_sorted + 2000000, sizeof(float), hipMemcpyDeviceToHost);
```

### OpenCL (custom radix or use library)
```cpp
// Нужна библиотека типа VexCL или custom kernel
// Рекомендация: использовать готовую реализацию radix sort
```

---

## Ссылки

- **Детальное исследование**: `2026-02-14_GPU_Median_Algorithms_Research.md`
- **RadiK paper**: https://arxiv.org/html/2501.14336v1
- **SampleSelect (CUDA)**: https://github.com/upsj/gpu_selection
- **rocPRIM docs**: https://rocm.docs.amd.com/projects/rocPRIM/

---

**Дата**: 2026-02-14
**Статус**: ✅ Quick Reference
