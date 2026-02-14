# 🔍 FFT FindAllMax — Поиск всех максимумов после FFT

> **Дата**: 2026-02-14
> **Статус**: Исследование завершено ✅

---

## 📋 Задача

**Данные**: 256 лучей × 4 млн точек каждый
**Цель**: Найти все локальные максимумы спектра после FFT
**Платформы**: OpenCL & ROCm
**Критерий**: Минимальное время выполнения

---

## 📚 Документы

### 1️⃣ Полное исследование
📄 `GPU_Maxima_Detection_Methods_2026-02-14.md`
- Все методы детектирования максимумов на GPU
- Сравнительный анализ подходов
- Численные методы и алгоритмы
- Примеры реализаций OpenCL/ROCm

### 2️⃣ Quick Reference
📄 `GPU_Maxima_Quick_Reference.md`
- Краткая сводка методов
- Готовые code snippets
- Таблица сравнения производительности

---

## 🏆 Рекомендованное решение

### ⭐ Метод: Custom Kernel + Scan-based Compaction

**Алгоритм**:
1. **Детекция** — kernel отмечает локальные максимумы (условие: `data[i] > data[i-1] && data[i] > data[i+1]`)
2. **Scan** — prefix sum через rocPRIM для подсчёта позиций
3. **Compaction** — записываем только максимумы в выходной буфер

**Производительность**:
- ⏱️ **Время**: ~2-3 ms на луч
- ⚡ **Для 256 лучей на 10 GPU**: ~50-75 ms
- 📊 **Масштабируемость**: Линейная с количеством лучей

**Преимущества**:
- ✅ Находит ВСЕ максимумы (не top-k)
- ✅ Готовые библиотеки (rocPRIM scan)
- ✅ Поддержка OpenCL & ROCm
- ✅ Batch processing для 256 лучей

---

## 🚀 Альтернативные решения

### Метод 2: RadiK Top-K Selection
- **Статус**: ⚠️ Требует портации с CUDA
- **Применимость**: Если нужны только top-k максимумов (не все)
- **Время**: ~10-30 ms (2× быстрее после портации)

### Метод 3: Threshold-based + Atomic Counter
- **Применимость**: Когда известен минимальный порог максимума
- **Время**: ~1-2 ms (самый быстрый, но требует априорных знаний)

---

## 📝 Интеграция с GPUWorkLib

### Архитектура
```cpp
class SpectrumMaximaFinder {
  // Использует DrvGPU context
  GPUContext& gpu_ctx_;

  // rocPRIM для scan
  rocprim::exclusive_scan(...);

  // Custom kernels для детекции
  cl_kernel detect_kernel_;
  cl_kernel compact_kernel_;
};
```

### API для Python
```python
# После FFT
maxima_finder = SpectrumMaximaFinder(gpu_context)
maxima_positions, maxima_values = maxima_finder.find_all(fft_result)
# → массивы с позициями и значениями всех максимумов
```

---

## 🔧 Для реализации сегодня

**OpenCL (готово сейчас)**:
- ✅ rocPRIM scan доступен
- ✅ Custom kernels (детекция + compaction)
- ✅ Batch processing через DrvGPU

**ROCm (когда придёт AMD GPU)**:
- ✅ Всё готово к запуску (HIP API идентично)
- ✅ rocPRIM нативно поддерживается
- 🔄 Потребуется только пересборка с флагом `USE_ROCM`

---

## 📊 Ожидаемая производительность

| Конфигурация | Время на луч | Время на 256 лучей | Speedup vs CPU |
|--------------|--------------|-------------------|----------------|
| 1 GPU | ~2-3 ms | ~512-768 ms | 15-30× |
| 10 GPU (batch) | ~2-3 ms | ~50-75 ms | 150-300× |

---

## ✅ Next Steps

1. Создать класс `SpectrumMaximaFinder`
2. Реализовать detection kernel (OpenCL)
3. Интегрировать rocPRIM scan
4. Добавить compaction kernel
5. Batch processing для 256 лучей
6. Python bindings
7. Unit tests (сравнение с SciPy `find_peaks`)

---

*Последнее обновление: 2026-02-14*
