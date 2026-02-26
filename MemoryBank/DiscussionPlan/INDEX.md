# 📚 Discussion Plans Index

> **Дата**: 2026-02-24
> **Статус**: Исследования Statistics завершены ✅ + планы модулей

---

## 📋 Планы модулей

- **[PLAN_Vector_Algebra_Cholesky.md](PLAN_Vector_Algebra_Cholesky.md)** — модуль vector_algebra: инверсия матрицы методом Холецкого (ROCm-only). Вход/выход: InputData\<T\> (vector\<complex\<float\>\>, void*).
- **[~!_внимание_PLAN_Discrepancies_ROCm_Tasks.md](~!_внимание_PLAN_Discrepancies_ROCm_Tasks.md)** — разногласия планов ROCm.

---

## 📋 Задачи исследования (Statistics)

Все задачи связаны с обработкой данных:
- **256 лучей** × **4 млн точек** каждый
- **Платформы**: OpenCL & ROCm
- **Критерий**: Минимальное время выполнения

---

## 📁 Структура

### 1️⃣ FFT FindAllMax — Поиск всех максимумов спектра
📂 `~1. FFT_FindAllMax/`
- 📄 `README.md` — краткая сводка
- 📄 `GPU_Maxima_Detection_Methods_2026-02-14.md` — полное исследование
- 📄 `GPU_Maxima_Quick_Reference.md` — quick reference

**Рекомендация**: Custom Kernel + Scan-based Compaction
- ⏱️ **Время**: ~50-75 ms для 256 лучей на 10 GPU
- ✅ **Готово**: OpenCL (rocPRIM scan)

---

### 2️⃣ Average (Mean) — Среднее значение
📂 `~2. Average/`
- 📄 `README.md` — краткая сводка
- 📄 `gpu_mean_reduction_research.md` — полное исследование

**Рекомендация**: Per-Row Two-Level Hierarchical Reduction
- ⏱️ **Время**: ~1-5 ms для 256 лучей
- ✅ **Готово**: OpenCL (custom kernel)

---

### 3️⃣ Median — Медиана
📂 `~3. Median/`
- 📄 `README.md` — краткая сводка
- 📄 `2026-02-14_GPU_Median_Algorithms_Research.md` — полное исследование (24 KB)
- 📄 `GPU_Median_Quick_Reference.md` — quick reference

**Рекомендация сейчас**: Radix Sort + Middle Element (rocPRIM)
- ⏱️ **Время**: ~20-60 ms для 256 лучей
- ✅ **Готово**: OpenCL (rocPRIM radix sort)

**Рекомендация будущего**: RadiK Algorithm (требует портации)
- ⏱️ **Время**: ~10-30 ms (2× быстрее)
- ⚠️ **Статус**: Требует портации с CUDA

---

### 4️⃣ STD (Standard Deviation) — Стандартное отклонение
📂 `~4. STD/`
- 📄 `README.md` — краткая сводка
- 📄 `GPU_STD_Research.md` — полное исследование

**Рекомендация**: Welford's Online Algorithm
- ⏱️ **Время**: ~12-15 ms для 256 лучей
- 🎁 **Bonus**: Вычисляет mean + variance + STD одновременно
- ✅ **Готово**: OpenCL (custom Welford kernel)

---

### 5️⃣ Variance — Дисперсия
📂 `~5. variance/`
- 📄 `README.md` — краткая сводка
- 📄 `GPU_Variance_Research.md` — полное исследование (45 KB)
- 📄 `Quick_Reference.md` — quick reference

**Рекомендация**: Welford's Online Algorithm
- ⏱️ **Время**: ~10-15 ms для 256 лучей
- 🎁 **Bonus**: Вычисляет mean + variance одновременно
- ✅ **Готово**: OpenCL (custom Welford kernel)

---

## ⏱️ Общая производительность (256 лучей на 10 GPU)

| Операция | Метод | Время | Статус |
|----------|-------|-------|--------|
| **FFT FindAllMax** | Scan Compaction | ~50-75 ms | ✅ Готово (OpenCL) |
| **Mean** | Hierarchical Reduction | ~1-5 ms | ✅ Готово (OpenCL) |
| **Median** | Radix Sort | ~20-60 ms | ✅ Готово (rocPRIM) |
| **STD** | Welford | ~12-15 ms | ✅ Готово (OpenCL) |
| **Variance** | Welford | ~10-15 ms | ✅ Готово (OpenCL) |
| **Mean+Variance+STD** | Welford (combo) | ~12-15 ms | ✅ Готово (2× эффективнее!) |

**ИТОГО** (все операции): ~93-170 ms

---

## 🚀 Оптимизации

### Комбинированные вычисления
```
❌ Раздельно:
mean:     ~5 ms
variance: ~10 ms
std:      ~12 ms
────────────────
ИТОГО:    ~27 ms

✅ Welford (combo):
mean + variance + std: ~12-15 ms (2× быстрее!)
```

### Будущие улучшения (когда придёт AMD GPU)
- 🚀 **Median**: RadiK algorithm → ~10-30 ms (2× быстрее)
- 🔄 **ROCm testing**: Оптимизация для AMD GPU
- 📊 **Profiling**: GPUProfiler для поиска узких мест

---

## 🎯 Для реализации сегодня

### Приоритет 1: Statistics Module (базовые функции)
```cpp
class StatisticsProcessor {
public:
    // Базовые функции
    std::vector<float> compute_mean(data, num_rays);
    std::vector<float> compute_median(data, num_rays);

    // Комбинированная функция (эффективнее!)
    struct Result {
        std::vector<float> means;
        std::vector<float> variances;
        std::vector<float> stds;
    };
    Result compute_statistics(data, num_rays);
};
```

### Приоритет 2: SpectrumMaximaFinder (расширение)
```cpp
class SpectrumMaximaFinder {
public:
    // Найти все максимумы после FFT
    struct MaximaResult {
        std::vector<int> positions;
        std::vector<float> values;
    };
    MaximaResult find_all(fft_result);
};
```

---

## 📝 Integration с GPUWorkLib

### Архитектурные принципы
- ✅ Используем DrvGPU контекст (не плодим сущности)
- ✅ Batch processing через BatchManager
- ✅ Логи через plog (per-GPU)
- ✅ Вывод через console_output (мультиGPU-safe)
- ✅ Профилирование через GPUProfiler

### Python Bindings
```python
# Базовые функции
stats = StatisticsProcessor(gpu_context)
means = stats.compute_mean(data, num_rays=256)
medians = stats.compute_median(data, num_rays=256)

# Комбинированная функция (рекомендуется!)
result = stats.compute_statistics(data, num_rays=256)
print(result.means)      # numpy array [256]
print(result.variances)  # numpy array [256]
print(result.stds)       # numpy array [256]

# Поиск максимумов
maxima_finder = SpectrumMaximaFinder(gpu_context)
maxima = maxima_finder.find_all(fft_result)
print(maxima.positions)  # позиции максимумов
print(maxima.values)     # значения максимумов
```

---

## 🔧 OpenCL → ROCm Migration

**Когда придёт AMD GPU**:
1. ✅ Все kernels готовы к портации (OpenCL → HIP)
2. ✅ rocPRIM доступен для sort/scan
3. 🔄 Пересборка с флагом `USE_ROCM`
4. 📊 Benchmark на AMD GPU
5. 🚀 Опционально: RadiK портация для median

---

## 📚 Источники и ссылки

Все источники, научные статьи, GitHub репозитории и документация указаны в соответствующих полных исследованиях:
- `~1. FFT_FindAllMax/GPU_Maxima_Detection_Methods_2026-02-14.md`
- `~2. Average/gpu_mean_reduction_research.md`
- `~3. Median/2026-02-14_GPU_Median_Algorithms_Research.md`
- `~4. STD/GPU_STD_Research.md`
- `~5. variance/GPU_Variance_Research.md`

---

## ✅ Следующие шаги

### Сегодня (2026-02-14)
1. Создать спецификацию `MemoryBank/specs/statistics_module.md`
2. Создать план задач в `MemoryBank/tasks/`
3. Начать реализацию через feature-dev или вручную

### Эта неделя
1. Реализовать Welford kernel (mean + variance + std)
2. Реализовать Mean reduction kernel
3. Интегрировать rocPRIM median
4. Python bindings
5. Unit tests

### Когда придёт AMD GPU
1. ROCm testing и profiling
2. Оптимизация kernels для AMD
3. RadiK портация (опционально)

---

*Последнее обновление: 2026-02-14*
*Все исследования завершены ✅*
*Готово к реализации!*
