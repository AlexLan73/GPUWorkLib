# GPU Median Computation Algorithms - Research Report

**Дата**: 2026-02-14
**Контекст**: Поиск оптимального метода вычисления медианы на GPU
**Данные**: 256 лучей × 4 млн точек каждый
**Платформы**: OpenCL и ROCm

---

## Executive Summary

Для вычисления медианы на массивах размером 4M элементов существует несколько подходов с различными trade-off между точностью, скоростью и памятью:

**Рекомендации для 256×4M массива:**
1. **RadiK (Radix Select)** — лучший выбор для точной медианы, масштабируется до k=n/2
2. **SampleSelect** — альтернатива, устойчива к неравномерным распределениям
3. **Полная сортировка** — проще в реализации, но медленнее (~2× overhead)
4. **Approximate Median** — если допустима погрешность, можно ускорить в 2-4×

---

## 1. Radix Select (RadiK Algorithm)

### Описание
Итеративный алгоритм, использующий побитовое разделение данных на корзины (bins) для быстрого поиска k-го элемента.

### Как работает
1. **Phase 1 - Radix Select**: Итерируется по битам (старшие → младшие), создавая гистограмму на каждой итерации
2. Определяет корзину с k-ым элементом и отбрасывает остальные
3. **Phase 2 - Filter**: Использует найденный pivot для извлечения всех топ-k элементов

### Ключевые инновации (RadiK 2025)
- **Hierarchical Atomics**: Распределение атомарных операций по памяти (registers → shared → global)
- **Flush-Efficient Write Buffer**: Агрегация разрозненных записей в shared memory
- **Task Rescheduling**: Оптимизация для batch-запросов (все 256 лучей одновременно)
- **Adaptive Scaling**: Защита от adversarial distributions

### Производительность
- **Non-batch**: до 2.5× быстрее Bitonic Select
- **Batch (256 лучей)**: до 4.8× быстрее priority queue методов
- **Медиана (k=n/2)**: всего +50% latency при n=2²² (~4M) vs k=512

### Сложность
- **Теоретическая**: O(n × log(max_value) / d), где d — ширина radix-бита
- **Практическая на GPU**: ~O(n × iterations), iterations обычно 4-8 для float32
- **Память**: O(2^d) для гистограммы + O(n) для промежуточных данных

### Требования по памяти
- **On-chip**: Константная — только гистограмма (2^d bins) + write buffer (2×BLOCK)
- **Off-chip (global)**: O(n) для промежуточного массива кандидатов
- **Преимущество**: Не зависит от k, масштабируется до k=n

### Точность
**Exact** — находит точную медиану

### Реализации
- **GitHub**: [RadiK paper (Jan 2025)](https://arxiv.org/html/2501.14336v1) — пока нет публичного кода
- **Платформа**: CUDA (возможна портация на ROCm/HIP)

### Рекомендации для 4M элементов
✅ **Лучший выбор** для точной медианы на больших массивах
✅ Отлично подходит для batch-обработки (256 лучей)
✅ Масштабируется на медиану без ухудшения производительности
⚠️ Требует портации с CUDA на OpenCL/ROCm

---

## 2. SampleSelect Algorithm

### Описание
Bucket-based selection с использованием сэмплирования для построения search tree.

### Как работает
1. **Sampling**: Берется выборка из входных данных
2. **Tree Building**: Строится search tree на основе выборки
3. **Counting**: Подсчет элементов в каждом bucket
4. **Filtering**: Фокусировка на bucket с k-ым элементом

### Преимущества
- Не требует предположений о распределении данных
- Малая глубина рекурсии (vs QuickSelect)
- Работает с рангами, а не значениями → устойчивость к adversarial data

### Сложность
- **Теоретическая**: O(n) в среднем, O(n log n) worst case
- **Практическая на GPU**: зависит от распределения данных

### Требования по памяти
- O(n) для основных данных
- O(sample_size) для выборки и search tree
- Обычно sample_size << n (например, √n)

### Точность
**Exact** — находит точную медиану

### Реализации
- **GitHub**: [upsj/gpu_selection](https://github.com/upsj/gpu_selection)
- **Платформа**: CUDA только (нет OpenCL)
- **Статья**: "Parallel selection on GPUs" (2019)

### Рекомендации для 4M элементов
✅ Хорошая альтернатива RadiK
✅ Устойчивость к неравномерным распределениям
❌ Только CUDA (требует портации)
⚠️ Может быть медленнее RadiK на batch-запросах

---

## 3. QuickSelect на GPU

### Описание
Классический QuickSelect с параллельным partitioning.

### Как работает
1. Выбор pivot (обычно случайный или медиана-из-трех)
2. Параллельное разделение массива на две части (< pivot и ≥ pivot)
3. Рекурсия только в нужной части

### Преимущества
- Простая концепция
- Хорошо известен на CPU

### Недостатки на GPU
- **Высокая рекурсия**: плохо для GPU
- **Load imbalance**: разные потоки делают разный объем работы
- **Memory overhead**: нужно 2n элементов на каждой итерации (read + write)

### Сложность
- **Теоретическая**: O(n) в среднем, O(n²) worst case
- **Практическая на GPU**: Часто хуже из-за рекурсии

### Требования по памяти
- O(n/2) auxiliary storage если нельзя перезаписать вход

### Точность
**Exact**

### Реализации
- Нет современных оптимизированных GPU-реализаций
- Упоминается в литературе как baseline для сравнения

### Рекомендации для 4M элементов
❌ **Не рекомендуется** для GPU
❌ Рекурсия и load imbalance делают его медленным

---

## 4. Sorting + Middle Element

### Описание
Полная сортировка массива + выбор среднего элемента.

### Алгоритмы сортировки на GPU
1. **Radix Sort** — самый быстрый для большинства случаев
2. **Merge Sort** — стабильная, предсказуемая память
3. **Bitonic Sort** — только для малых массивов (&lt;10⁵ элементов)

### Производительность
- **Radix Sort**: самый быстрый параллельный алгоритм
- **vs Bitonic**: Radix &gt;2× быстрее на больших массивах
- **Overhead для медианы**: ~2× медленнее чем RadiK (сортируем все, а нужна только медиана)

### Сложность
- **Radix Sort**: O(n × k_bits) где k_bits = log₂(max_value)
- **Практическая**: O(n) для фиксированного типа данных (float32/int32)

### Требования по памяти
- O(n) для output массива
- O(n) для temporary buffer (double buffering в radix sort)
- **Итого**: 2n + оригинал

### Точность
**Exact**

### Реализации
- **CUDA**: CUB DeviceRadixSort, Thrust sort
- **ROCm**: rocPRIM sort, rocThrust sort
- **OpenCL**: Множество реализаций на GitHub

### Рекомендации для 4M элементов
✅ **Простейший в реализации** — готовые библиотеки
✅ Доступно для OpenCL и ROCm
⚠️ Медленнее RadiK на ~2×, но проще в реализации
✅ Подходит если нужно отсортированные данные для других целей

---

## 5. Histogram-Based Median

### Описание
Построение гистограммы значений → поиск bin с медианой.

### Как работает
1. **Pass 1**: Построение гистограммы (обычно на MSB битах)
2. **Find Median Bin**: Накопление counts до достижения n/2
3. **Pass 2**: Уточнение на LSB битах в выбранном bin
4. Опционально: Hierarchical histogram для меньшей памяти

### Преимущества
- Фиксированная память под гистограмму (обычно 256-4096 bins)
- Высокая параллельность

### Недостатки
- **Multi-pass**: требует несколько проходов по данным
- **Atomic contention**: тысячи потоков обновляют малое число bins
- **Approximate**: может быть только приближенной (зависит от числа bins)

### Сложность
- **Теоретическая**: O(n × passes), обычно passes = 2-4
- **Практическая**: Сильно зависит от atomic contention

### Требования по памяти
- O(num_bins) для гистограммы (обычно 256-4096)
- **Hierarchical**: можно уменьшить на порядок, но +instructions

### Точность
- **Exact**: если достаточно bins и есть refinement pass
- **Approximate**: если грубая гистограмма

### Реализации
- Упоминается в академических работах
- Нет готовых библиотек для median (только для median filter в обработке изображений)

### Рекомендации для 4M элементов
⚠️ **Специфичный метод** — подходит для ограниченного диапазона значений
⚠️ Atomic contention может быть проблемой
❓ Может работать для целых чисел с известным диапазоном
❌ Не подходит для float с широким диапазоном значений

---

## 6. Approximate Median (Streaming Quantile Sketches)

### Описание
Приближенное вычисление медианы с гарантированными границами ошибки.

### Алгоритмы
1. **DDSketch**: relative-error guarantees
2. **P² Quantile Estimator**: пять маркеров для оценки квантиля
3. **Stratified Reservoir Sampling**: для stream processing

### Производительность
- Потенциально **2-4× быстрее** точных методов
- Один проход по данным (vs multi-pass)

### Сложность
- **Теоретическая**: O(n) — один проход
- **Практическая**: Зависит от алгоритма

### Требования по памяти
- **Минимальная**: O((1/ε) log log(1/δ)) где ε — точность, δ — вероятность ошибки
- **Практическая**: Обычно несколько КБ независимо от n

### Точность
- **Approximate** с гарантиями: ε-approximate quantile
- Ранг медианы: rank ± εn с вероятностью ≥ 1-δ
- Типичные значения: ε = 0.001-0.01

### Реализации
- **CPU**: множество библиотек (quantiles, DDSketch)
- **GPU**: редко (упоминается MedianSketch для Paint.NET с P² алгоритмом)

### Рекомендации для 4M элементов
✅ Если допустима погрешность 0.1-1%
✅ Минимальные требования по памяти
✅ Может быть в 2-4× быстрее
❌ Мало готовых GPU-реализаций
❓ Нужно оценить применимость для вашей задачи

---

## 7. Bitonic Sort

### Описание
Sorting network с фиксированной последовательностью сравнений.

### Преимущества
- **Lockstep execution**: все потоки выполняют одинаковые инструкции
- **In-place**: не требует дополнительной памяти
- **Простота**: легко реализуется на GPU

### Недостатки
- **O(n log² n)** vs O(n) для radix sort
- Эффективен только для **малых массивов** (&lt;10⁵ элементов)

### Производительность на 4M элементах
❌ **Не рекомендуется** для 4M элементов
- Radix sort &gt;2× быстрее на больших массивах
- Bitonic конкурентен только на массивах &lt;100K элементов

### Сложность
- **Теоретическая**: O(n log² n)
- **Практическая на GPU**: 2-4× медленнее radix sort для n &gt; 1M

### Требования по памяти
- **In-place**: можно сортировать на месте
- Меньше памяти чем radix sort

### Точность
**Exact** (если используется для полной сортировки)

### Реализации
- Множество OpenCL реализаций на GitHub
- Часто используется как учебный пример

### Рекомендации для 4M элементов
❌ **Слишком медленно** для 4M элементов
✅ Подходит только для малых подмассивов (&lt;10K)

---

## Сравнительная таблица

| Метод | Сложность | Память | Точность | Скорость (4M) | OpenCL/ROCm | Рекомендация |
|-------|-----------|---------|----------|---------------|-------------|--------------|
| **RadiK (Radix Select)** | O(n × log(max)/d) | O(n) | Exact | ⭐⭐⭐⭐⭐ Fastest | ⚠️ Портация | ✅ **Лучший выбор** |
| **SampleSelect** | O(n) avg | O(n) | Exact | ⭐⭐⭐⭐ Fast | ⚠️ Портация | ✅ Альтернатива |
| **Radix Sort + Select** | O(n) | 2n | Exact | ⭐⭐⭐ Good | ✅ Доступно | ✅ Простейший |
| **QuickSelect** | O(n) avg, O(n²) worst | O(n/2) | Exact | ⭐⭐ Slow | ❌ Нет | ❌ Не для GPU |
| **Histogram-based** | O(n × passes) | O(bins) | Configurable | ⭐⭐⭐ Depends | 🔶 Custom | 🔶 Специфика |
| **Approximate Median** | O(n) | O(1/ε) | ε-approx | ⭐⭐⭐⭐ Very Fast | ❌ Редко | 🔶 Если допустима ошибка |
| **Bitonic Sort** | O(n log² n) | In-place | Exact | ⭐ Very Slow | ✅ Доступно | ❌ Слишком медленно |

**Легенда:**
- ✅ Рекомендуется
- 🔶 Зависит от контекста
- ⚠️ Требует работы
- ❌ Не рекомендуется

---

## Рекомендации для вашей задачи (256×4M)

### Вариант 1: RadiK (Best Performance)
**Производительность**: ⭐⭐⭐⭐⭐
**Сложность реализации**: ⭐⭐⭐ (требует портации)

**Действия:**
1. Портировать RadiK алгоритм с CUDA на OpenCL/HIP
2. Использовать Task Rescheduling для batch из 256 лучей
3. Оптимизировать под вашу архитектуру GPU

**Ожидаемая производительность:**
- 256 лучей × 4M точек: ~10-50ms на современном GPU
- Batch обработка дает 4-5× speedup vs 256 отдельных вызовов

---

### Вариант 2: Radix Sort + Middle Element (Simplest)
**Производительность**: ⭐⭐⭐
**Сложность реализации**: ⭐⭐⭐⭐⭐ (готовые библиотеки)

**Действия:**
1. Использовать rocPRIM для ROCm или clFFT sort для OpenCL
2. После сортировки взять элемент с индексом n/2
3. Запустить 256 параллельных сортировок (по одной на луч)

**Ожидаемая производительность:**
- 256 лучей × 4M точек: ~20-100ms
- Примерно в 2× медленнее RadiK, но проще в реализации

**Код (псевдокод ROCm/rocThrust):**
```cpp
// Для каждого луча
for (int beam = 0; beam < 256; beam++) {
    // Сортировка на GPU
    rocprim::radix_sort_keys(
        d_temp_storage, temp_storage_bytes,
        d_beam_data[beam], d_sorted[beam],
        4000000  // 4M elements
    );

    // Медиана = средний элемент
    float median = d_sorted[beam][2000000];  // n/2
}
```

---

### Вариант 3: SampleSelect (Good Balance)
**Производительность**: ⭐⭐⭐⭐
**Сложность реализации**: ⭐⭐⭐ (требует портации)

**Действия:**
1. Портировать SampleSelect с CUDA на OpenCL
2. Возможна модификация под batch-обработку

**Ожидаемая производительность:**
- Близко к RadiK на равномерных распределениях
- Лучше RadiK на adversarial distributions

---

### Вариант 4: Approximate Median (If Acceptable)
**Производительность**: ⭐⭐⭐⭐⭐
**Сложность реализации**: ⭐⭐⭐⭐ (нужна custom реализация)

**Когда подходит:**
- Допустима погрешность 0.1-1% в определении медианы
- Критична максимальная скорость
- Ограничена память

**Ожидаемая производительность:**
- Потенциально 2-4× быстрее точных методов
- Минимальные требования по памяти

---

## Доступность библиотек

### CUDA
✅ RadiK (upcoming, Jan 2025)
✅ SampleSelect ([upsj/gpu_selection](https://github.com/upsj/gpu_selection))
✅ CUB DeviceRadixSort
✅ Thrust sort

### ROCm
✅ rocPRIM sort (radix, merge)
✅ rocThrust sort
❌ Нет готовых selection algorithms
⚠️ Можно портировать CUDA-код через HIP

### OpenCL
✅ Множество реализаций radix/bitonic sort на GitHub
❌ Нет готовых selection algorithms
⚠️ Требуется custom реализация или портация

---

## Итоговые рекомендации

### Для немедленной реализации (сейчас)
👉 **Radix Sort + Middle Element** (rocPRIM/rocThrust)
- Готовые библиотеки
- Простая интеграция
- Приемлемая производительность (~2× медленнее лучшего)

### Для максимальной производительности (после портации)
👉 **RadiK Algorithm**
- Портировать с CUDA на HIP/OpenCL
- Batch-оптимизация для 256 лучей
- Лучшая производительность на рынке

### Для устойчивости к данным
👉 **SampleSelect**
- Если данные могут иметь adversarial distributions
- Хорошая альтернатива RadiK

### Для минимальной памяти
👉 **Approximate Median (Quantile Sketch)**
- Если допустима погрешность 0.1-1%
- Минимальная память (~KB vs GB)

---

## Ссылки и источники

### Академические статьи
- [Parallel selection on GPUs (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0167819119301796) — SampleSelect
- [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection (2025)](https://arxiv.org/html/2501.14336v1) — RadiK
- [Fast K-selection Algorithms for Graphics Processing Units](https://blanchard.math.grinnell.edu/Research/ABGS_KSelection.pdf)
- [Parallel calculation of the median and order statistics](https://arxiv.org/pdf/1104.2732)
- [Optimal Quantile Approximation in Streams](https://arxiv.org/abs/1603.05346)

### GitHub реализации
- [upsj/gpu_selection](https://github.com/upsj/gpu_selection) — SampleSelect (CUDA)
- [detel/Median-Filtering-GPU](https://github.com/detel/Median-Filtering-GPU) — Median Filter
- [gyatskov/radix-sort](https://github.com/gyatskov/radix-sort) — OpenCL Radix Sort
- [ROCm/rocThrust](https://github.com/rocm/rocthrust) — ROCm Thrust library
- [ROCm/rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/) — ROCm Primitives

### Документация
- [rocPRIM Sort Documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.0.0/device_ops/sort.html)
- [OpenCL Sorting - Bealto](https://www.bealto.com/gpu-sorting_parallel-selection.html)
- [GPU Gems 2: Chapter 46 - Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)

### Форумы и обсуждения
- [NVIDIA Forums: CUDA calculate median of 4096 elements array](https://forums.developer.nvidia.com/t/cuda-calculate-median-of-4096-elements-array/305372)
- [NVIDIA Forums: Order Statistics (median, etc.)](https://forums.developer.nvidia.com/t/order-statistics-median-etc/36282)

---

## Следующие шаги

1. **Прототип на Radix Sort**
   - Использовать rocPRIM/rocThrust для быстрого прототипа
   - Измерить реальную производительность на ваших данных
   - Оценить допустимость времени выполнения

2. **Оценка необходимости оптимизации**
   - Если Radix Sort достаточно быстр → использовать его
   - Если нужно быстрее → планировать портацию RadiK

3. **Портация RadiK (если нужна максимальная производительность)**
   - Изучить [исходник RadiK](https://arxiv.org/html/2501.14336v1) детально
   - Портировать на HIP/OpenCL
   - Адаптировать под batch из 256 лучей

4. **Тестирование на реальных данных**
   - Сравнить производительность разных методов
   - Профилировать через GPUProfiler
   - Выбрать оптимальный для вашего случая

---

**Дата**: 2026-02-14
**Исследователь**: Кодо (AI Assistant)
**Статус**: ✅ Completed
