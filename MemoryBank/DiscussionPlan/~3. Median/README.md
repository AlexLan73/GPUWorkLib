# 📈 Median — Вычисление медианы

> **Дата**: 2026-02-14
> **Статус**: Исследование завершено ✅

---

## 📋 Задача

**Данные**: 256 лучей × 4 млн точек каждый
**Цель**: Вычислить медиану для каждого луча отдельно
**Платформы**: OpenCL & ROCm
**Критерий**: Минимальное время выполнения

**Особенность**: Медиана требует частичной сортировки или статистики порядка

---

## 📚 Документы

### 1️⃣ Полное исследование
📄 `2026-02-14_GPU_Median_Algorithms_Research.md` (24 KB)
- 7 методов вычисления медианы на GPU
- Детальный анализ каждого метода
- Сложность (теоретическая и практическая)
- Требования по памяти
- Точность (exact vs approximate)
- Примеры реализаций
- Рекомендации для 4M элементов

### 2️⃣ Quick Reference
📄 `GPU_Median_Quick_Reference.md` (3 KB)
- TL;DR с кодом
- Краткая таблица сравнения
- Готовые примеры

---

## 🏆 Рекомендованное решение

### ⭐ Метод сейчас: Radix Sort + Middle Element

**Алгоритм**:
1. **Сортировка**: rocPRIM radix sort для каждого луча
2. **Выбор**: Взять средний элемент `sorted[N/2]`

**Производительность**:
- ⏱️ **Время**: ~20-60 ms для 256 лучей
- 📦 **Библиотека**: rocPRIM (готова!)
- 📝 **Код**: 5-10 строк

**Пример**:
```cpp
// ROCm/rocPRIM — готовая библиотека
rocprim::radix_sort_keys(d_data, d_sorted, 4000000);
float median = d_sorted[2000000];  // средний элемент
```

**Преимущества**:
- ✅ Простота реализации (готовые библиотеки)
- ✅ Работает прямо сейчас (OpenCL/ROCm)
- ✅ Exact median (точная медиана)
- ✅ Batch processing для 256 лучей

---

## 🚀 Метод будущего: RadiK Algorithm

**Статус**: ⚠️ Требует портации с CUDA на OpenCL/ROCm

**Производительность**:
- ⏱️ **Время**: ~10-30 ms для 256 лучей (2× быстрее!)
- 🏆 **Speedup**: 2.5× vs предыдущие top-k методы
- 📊 **Batch**: Оптимизирован для параллельной обработки лучей

**Когда использовать**:
- Когда нужна максимальная производительность
- После портации с CUDA
- Когда придёт AMD GPU для тестирования

**Источник**: [RadiK: Scalable GPU-Parallel Radix Top-K Selection (2025)](https://arxiv.org/html/2501.14336v1)

---

## 📊 Сравнительная таблица методов

| Метод | Скорость (4M) | Память | Точность | Доступность OpenCL/ROCm | Рекомендация |
|-------|---------------|---------|----------|-------------------------|--------------|
| **Radix Sort** | ⭐⭐⭐ Good | 2n | Exact | ✅ Готово (rocPRIM) | ✅ **Сейчас** |
| **RadiK** | ⭐⭐⭐⭐⭐ Fastest | O(n) | Exact | ⚠️ Требует портации | 🚀 **Будущее** |
| **SampleSelect** | ⭐⭐⭐⭐ Fast | O(n) | Exact | ⚠️ Требует портации | 🔶 Alternative |
| **Approximate** | ⭐⭐⭐⭐⭐ Very Fast | O(1/ε) | ε-approx | ❌ Custom | 🔶 If ε OK |
| **Bitonic** | ⭐ Very Slow | In-place | Exact | ✅ Доступно | ❌ Too slow |

---

## 🔧 Реализация для сегодня

### C++ API
```cpp
class StatisticsProcessor {
public:
    // Вычисление медианы для батча лучей
    std::vector<float> compute_median(
        const std::vector<float>& data,  // [256 × 4M]
        size_t num_rays,
        size_t points_per_ray
    );

private:
    // rocPRIM sort backend
    rocprim::radix_sort_keys(...);
};
```

### Python API
```python
stats = StatisticsProcessor(gpu_context)
medians = stats.compute_median(data, num_rays=256)
# → numpy array [256] с медианами для каждого луча
```

---

## 🎯 Approximate Median (если допустима погрешность)

### Когда использовать
- Допустима погрешность 0.1-1% (ε = 0.001-0.01)
- Нужна максимальная скорость
- Минимальная память

### Производительность
- ⏱️ **Время**: ~5-20 ms для 256 лучей (4× быстрее!)
- 📦 **Память**: Минимальная (~KB vs GB)

### Статус
- ❌ Мало GPU-реализаций
- 🔧 Требует custom разработки
- 📚 См. [Quantile Sketch Algorithms](https://arxiv.org/abs/1603.05346)

---

## 📐 Memory & Performance Notes

### Optimal Configuration для 256 лучей × 4M
```
Layout: SoA (Structure of Arrays)
Processing: Parallel (один луч = одна задача сортировки)
Memory: Temporary buffers 2× (input + sorted) = 2 × 256 × 4M × 4 bytes = 8 GB
```

### Trade-offs
| Подход | Время | Память | Точность |
|--------|-------|--------|----------|
| Radix Sort | 20-60 ms | 8 GB | 100% |
| RadiK (future) | 10-30 ms | 4 GB | 100% |
| Approximate | 5-20 ms | <100 MB | 99-99.9% |

---

## 🔧 Для реализации

**Сегодня (OpenCL)**:
1. ✅ Использовать rocPRIM radix sort
2. ✅ Batch processing 256 лучей
3. ✅ Python bindings
4. ✅ Unit tests (сравнение с NumPy `median`)

**Когда придёт AMD GPU**:
1. 🔄 Тестирование на ROCm
2. 🔄 Benchmark производительности
3. 🚀 Опционально: портация RadiK для 2× speedup

---

## 📊 Ожидаемая производительность

| Конфигурация | Метод | Время |
|--------------|-------|-------|
| 1 GPU, 256 rays | Radix Sort | ~20-60 ms |
| 10 GPU, 256 rays | Radix Sort | ~5-15 ms |
| 1 GPU, 256 rays | RadiK (future) | ~10-30 ms |
| 10 GPU, 256 rays | RadiK (future) | ~2-8 ms |

---

## 📚 Источники

- [RadiK Algorithm (2025)](https://arxiv.org/html/2501.14336v1)
- [Parallel selection on GPUs (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0167819119301796)
- [Fast K-selection for GPUs](https://blanchard.math.grinnell.edu/Research/ABGS_KSelection.pdf)
- [rocPRIM Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.0.0/device_ops/sort.html)

---

*Последнее обновление: 2026-02-14*
