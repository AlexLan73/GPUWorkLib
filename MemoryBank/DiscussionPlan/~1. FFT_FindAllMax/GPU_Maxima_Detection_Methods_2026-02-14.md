# Оптимальные методы поиска всех максимумов после FFT на GPU (OpenCL и ROCm)

**Дата исследования**: 2026-02-14
**Контекст**: 256 лучей × 4 млн точек каждый
**Платформы**: OpenCL и ROCm
**Критерий**: минимальное время выполнения

---

## 📊 Executive Summary

Для задачи поиска всех локальных максимумов в спектре FFT на GPU существует несколько подходов с различными характеристиками производительности:

1. **Reduction-based** — эффективен для поиска одного глобального максимума
2. **Top-K Selection** — оптимален для поиска K наибольших максимумов
3. **Scan-based (Prefix Sum)** — универсален для маркировки локальных максимумов
4. **Sort-based** — наименее эффективен, используется как fallback
5. **Custom Kernels** — максимальная производительность при правильной оптимизации

**Рекомендация для 4M элементов**: Комбинация Custom Kernel (локальная детекция) + Top-K Selection (финальный отбор).

---

## 🔍 Детальный анализ методов

### 1. Reduction-based (Параллельная редукция)

#### Описание алгоритма
Parallel reduction — фундаментальная операция GPU, использующая древовидную структуру для агрегации данных. Для поиска максимума применяется оператор `max()` вместо суммы.

**Этапы**:
1. Загрузка данных из глобальной памяти в локальную (shared memory)
2. Итеративное сравнение пар элементов с барьерной синхронизацией
3. Возврат финального максимума

**Оптимизированные варианты** (из NVIDIA OpenCL examples):
- `reduce0`: Базовая версия с modulo операторами (медленно)
- `reduce1`: Смежные потоки (bank conflicts)
- `reduce2`: Sequential addressing (устранение конфликтов)
- `reduce3`: Половина потоков + начальная редукция при чтении
- `reduce4`: Разворачивание финального warp (без синхронизации)
- `reduce5`: Полное разворачивание для степеней 2
- `reduce6`: Несколько элементов на поток (Brent's Theorem)

#### Сложность
- **Время**: O(log₂ n) для n элементов
- **Память**: O(n) глобальная + O(workgroup_size) локальная
- **Work complexity**: O(n) в reduce6 (work-efficient)

#### Преимущества
- Простая реализация
- Высокая производительность для одного максимума
- Хорошая масштабируемость
- Доступны готовые оптимизированные реализации

#### Недостатки
- Находит только **один глобальный максимум**, не все локальные
- Для поиска всех максимумов требуется многократный запуск
- Не подходит для задачи "найти ВСЕ локальные максимумы"

#### Примеры реализаций
- [NVIDIA OpenCL Reduction Examples](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction_kernel.cl)
- PyOpenCL `ReductionKernel`

#### Оценка производительности (4M элементов)
- ~1-5 ms на современном GPU для одного глобального максимума
- Для всех локальных максимумов — **не применим напрямую**

---

### 2. Top-K Selection (Выбор K наибольших)

#### Описание алгоритма
Специализированные алгоритмы для поиска K наибольших элементов без полной сортировки.

**Основные подходы**:

##### 2.1. Radix Select (RadiK)
Двухфазный алгоритм на основе поразрядной сортировки:
- **Фаза 1**: Итеративное разбиение по битам (MSB → LSB) для нахождения k-го элемента (pivot)
- **Фаза 2**: Фильтрация исходных данных с использованием pivot

**Преимущества RadiK**:
- Константная память on-chip (только гистограммы)
- Поддержка больших K (до 100k+)
- Устойчивость к распределению данных (с adaptive scaling)
- До 2.5× быстрее Bitonic Select на больших данных
- До 4.8× быстрее PQ-block в batch режиме

##### 2.2. Bitonic Select
Merge-based алгоритм на основе Bitonic Sort:
- Каждый поток поддерживает отсортированную очередь
- Все потоки в warp совместно поддерживают warp queue
- Слияние через битоническую сортировку

**Ограничения**:
- Обычно k ≤ 512-2048 (из-за shared memory)
- Производительность падает с ростом k

##### 2.3. Block Select
Улучшение Warp Select:
- Использует 2-4 warps в блоке
- Shared memory для координации
- Выше параллелизм, чем в Warp Select

#### Сложность
- **RadiK время**: O(n × d), где d — число итераций (зависит от распределения)
- **Bitonic Select время**: O(n log k)
- **Память**: O(k) локальная + O(n) глобальная

#### Преимущества
- Намного быстрее полной сортировки
- RadiK поддерживает произвольно большие K
- Оптимизирован для batch операций
- Стабильная производительность при росте k (RadiK)

#### Недостатки
- Требует заранее знать K (число максимумов)
- Bitonic Select ограничен малыми K
- RadiK может деградировать на сильно скошенных распределениях (без scaling)
- Только CUDA реализации (RadiK), OpenCL/ROCm требует портирования

#### Примеры реализаций
- [gpu-topk (CUDA)](https://github.com/anilshanbhag/gpu-topk) — Bitonic, Radix Select, Sort
- [RadiK paper](https://arxiv.org/html/2501.14336v1) — новейший radix-based метод

#### Оценка производительности (536M элементов, k=32)
По данным gpu-topk:
- **Bitonic TopK**: 28.7 ms (min), 135 ms (avg)
- **Radix Select**: 63.7 ms (min), 132 ms (avg)
- **Full Sort**: 215.5 ms (min), 219 ms (avg)

**Экстраполяция на 4M элементов**: ~0.2-2 ms для k=32-256

---

### 3. Scan-based (Prefix Sum + Marking)

#### Описание алгоритма
Использует параллельный prefix sum (scan) для маркировки и подсчёта локальных максимумов.

**Этапы**:
1. **Детекция**: Кернел сравнивает каждый элемент с соседями, создаёт binary mask (1 = локальный максимум)
2. **Scan**: Exclusive prefix sum по маске для подсчёта позиций
3. **Компактификация**: Stream compaction — копирование только максимумов в выходной массив

**Work-Efficient Scan** (из GPU Gems 3):
- **Up-sweep** (reduce): Построение дерева частичных сумм — O(n)
- **Down-sweep**: Обратный проход для вычисления scan — O(n)
- Общая сложность: O(n) работы, O(log n) шагов

#### Сложность
- **Время**: O(log n) для scan + O(1) для детекции
- **Память**: O(n) для всех буферов
- **Work complexity**: O(n) (work-efficient)

#### Преимущества
- Находит **ВСЕ локальные максимумы**
- Work-efficient реализация (O(n) работы)
- Универсальность: легко адаптируется под разные критерии максимума
- Хорошая производительность на GPU (до 20× быстрее CPU)
- Доступны библиотеки: rocPRIM/rocThrust (scan), PyOpenCL (scan)

#### Недостатки
- Требует 3 прохода по данным (детекция, scan, компактификация)
- Дополнительная память для промежуточных буферов
- Bank conflicts в shared memory (требует оптимизации)
- Не оптимален, если нужно только K наибольших

#### Примеры реализаций
- [NVIDIA GPU Gems 3 Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- PyOpenCL: `InclusiveScanKernel`, `ExclusiveScanKernel`
- rocPRIM: `rocprim::exclusive_scan`

#### Оценка производительности (4M элементов)
- Детекция: ~0.5-1 ms
- Scan: ~1-2 ms
- Компактификация: ~0.5-1 ms
- **Итого**: ~2-4 ms для всех максимумов

---

### 4. Sort-based (Сортировка + Выбор)

#### Описание алгоритма
Полная сортировка массива с последующим выбором top-K элементов.

**Алгоритмы сортировки на GPU**:
- **Bitonic Sort**: O(n log² n), регулярная структура, идеальна для GPU
- **Radix Sort**: O(dn), где d — число битов
- **Merge Sort**: O(n log n), но сложнее на GPU

#### Сложность
- **Bitonic Sort**: O(n log² n)
- **Radix Sort**: O(dn) ≈ O(32n) для float
- **Память**: O(n)

#### Преимущества
- Простота реализации
- Доступны готовые библиотеки (clFFT, rocPRIM)
- Гарантированно корректный результат

#### Недостатки
- **Самый медленный** метод для top-k задач
- Избыточная работа: сортирует весь массив, когда нужно только k элементов
- Занимает до 28.9% времени в LLM inference (по данным RadiK paper)

#### Примеры реализаций
- [OpenCL Bitonic Sort](https://www.bealto.com/gpu-sorting_parallel-bitonic-1.html)
- [GPU Sorting examples](https://github.com/Gram21/GPUSorting)
- rocPRIM: `rocprim::radix_sort_keys`

#### Оценка производительности (4M элементов)
- Bitonic Sort: ~10-20 ms
- Radix Sort: ~5-10 ms
- **Не рекомендуется** для задачи поиска максимумов

---

### 5. Custom Kernels (Специализированные ядра)

#### Описание алгоритма
Написание специализированных OpenCL/HIP кернелов под конкретную задачу с максимальной оптимизацией.

**Стратегии оптимизации**:

##### 5.1. Локальная детекция максимумов
```c
// Псевдокод
__kernel void detect_local_maxima(
    __global float* spectrum,
    __global int* maxima_flags,
    int window_size
) {
    int gid = get_global_id(0);

    // Загрузка в local memory (shared memory)
    __local float local_data[WORK_GROUP_SIZE + 2*WINDOW];
    // ... cooperative loading ...
    barrier(CLK_LOCAL_MEM_FENCE);

    // Проверка локальности максимума
    float center = local_data[lid + WINDOW];
    bool is_max = true;
    for (int i = -window_size; i <= window_size; i++) {
        if (i != 0 && center <= local_data[lid + WINDOW + i]) {
            is_max = false;
            break;
        }
    }

    maxima_flags[gid] = is_max ? 1 : 0;
}
```

##### 5.2. Оптимизации
- **Coalesced memory access**: Выравнивание доступа к глобальной памяти
- **Local memory**: Кеширование данных в shared memory
- **Bank conflict avoidance**: Padding индексов для избежания конфликтов
- **Loop unrolling**: Разворачивание циклов для малых window_size
- **Multiple elements per thread**: Обработка нескольких элементов одним потоком
- **Vectorized loads**: Использование float4/float8 для загрузки данных

##### 5.3. Двухуровневый подход
Для 4M элементов:
1. **Грубая фильтрация**: Кернел с большим window (например, 64-128) — выбрасывает явно не-максимумы
2. **Точная детекция**: Кернел с малым window (например, 3-16) — точное определение локальных максимумов

#### Сложность
- **Время**: O(n × w), где w — размер окна
- **Память**: O(n) глобальная + O(workgroup_size + 2w) локальная
- **Практически**: O(n) при малых w (w << workgroup_size)

#### Преимущества
- **Максимальная производительность** при правильной оптимизации
- Полный контроль над алгоритмом
- Адаптация под специфику задачи (окно, пороги, критерии)
- Минимальное число проходов (1-2)
- Портируемость OpenCL → ROCm (HIP)

#### Недостатки
- Требует глубоких знаний GPU-программирования
- Сложность отладки
- Требует профилирования и тюнинга
- Поддержка кода

#### Примеры реализаций
- [HeCBench - Find local maxima](https://github.com/zjin-lcf/HeCBench)
- [Canny Edge Detection (non-maximum suppression)](https://github.com/smskelley/canny-opencl)
- [Non-Maximum Suppression GPU](https://github.com/hertasecurity/gpu-nms)

#### Оценка производительности (4M элементов)
При оптимизации:
- **Детекция с window=8**: ~0.5-1.5 ms
- **Компактификация** (если нужна): +1 ms
- **Итого**: ~1.5-2.5 ms для всех максимумов

---

## 📈 Сравнительная таблица

| Метод | Время (4M) | Память | Все максимумы | Top-K | OpenCL | ROCm | Сложность |
|-------|------------|--------|---------------|-------|--------|------|-----------|
| **Reduction** | 1-5 ms | O(n) | ❌ | ❌ | ✅ | ✅ | Низкая |
| **Radix Select** | 0.2-2 ms | O(n) | ❌ | ✅ | ⚠️ | ⚠️ | Средняя |
| **Bitonic Select** | 0.2-2 ms | O(n+k) | ❌ | ✅ (k≤512) | ⚠️ | ⚠️ | Средняя |
| **Scan-based** | 2-4 ms | O(n) | ✅ | ⚠️ | ✅ | ✅ | Средняя |
| **Sort** | 5-20 ms | O(n) | ✅ | ✅ | ✅ | ✅ | Низкая |
| **Custom Kernel** | 1.5-2.5 ms | O(n) | ✅ | ✅ | ✅ | ✅ | Высокая |

**Легенда**:
- ✅ Полная поддержка
- ⚠️ Требует портирования/адаптации
- ❌ Не поддерживается / Не эффективно

---

## 🎯 Рекомендации для задачи: 256 лучей × 4M точек

### Сценарий 1: Нужно K наибольших максимумов (K известно)

**Оптимальное решение**: Custom Kernel + Top-K Selection

1. **Custom kernel** для детекции локальных максимумов с флагами
2. **Stream compaction** (scan-based) для извлечения кандидатов
3. **Radix Select / Bitonic Select** для финального выбора top-K

**Ожидаемое время** (на луч): 2-3 ms
**Для 256 лучей** (параллельно на нескольких GPU): Зависит от числа GPU

---

### Сценарий 2: Нужны ВСЕ локальные максимумы

**Оптимальное решение**: Custom Kernel + Scan-based Compaction

1. **Custom kernel** детектирует локальные максимумы (сравнение с соседями в local memory)
2. **Scan** для подсчёта позиций
3. **Compaction kernel** для копирования в выходной массив

**Ожидаемое время** (на луч): 2-4 ms
**Для 256 лучей**: ~500-1000 ms на одном GPU, ~50-100 ms на 10 GPU

**Альтернатива**: Scan-based с готовыми примитивами (rocPRIM)
- Проще в реализации
- +10-20% времени выполнения
- Меньше кода для поддержки

---

### Сценарий 3: Только глобальный максимум на луч

**Оптимальное решение**: Reduction-based

Использовать готовые реализации:
- PyOpenCL `ReductionKernel`
- rocPRIM `rocprim::reduce`
- Собственный optimized reduce6 kernel

**Ожидаемое время** (на луч): 1-2 ms
**Для 256 лучей**: ~250-500 ms на одном GPU, ~25-50 ms на 10 GPU

---

## 🔧 Практические рекомендации по реализации

### OpenCL vs ROCm

**OpenCL (clFFT, PyOpenCL)**:
- ✅ Кроссплатформенность (NVIDIA, AMD, Intel)
- ✅ Стабильные библиотеки
- ⚠️ Немного медленнее ROCm на AMD GPU
- ⚠️ Меньше современных примитивов

**ROCm (rocPRIM, rocThrust, hipCUB)**:
- ✅ Максимальная производительность на AMD GPU
- ✅ Современные примитивы (scan, reduce, sort)
- ✅ Совместимость с CUDA кодом через HIP
- ❌ Только AMD GPU

**Рекомендация**:
- Для продакшена на AMD: ROCm/HIP + rocPRIM
- Для прототипирования: OpenCL + PyOpenCL
- Для переносимости: Abstraction layer поверх обоих

---

### Оптимизация производительности

#### 1. Параметры work-group
- **Оптимальный размер**: 64-256 (зависит от GPU)
- Использовать `clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)`
- Профилировать разные размеры

#### 2. Локальная память
- AMD GPU: 64KB local memory per compute unit
- Максимизировать использование для кеширования
- Избегать bank conflicts через padding

#### 3. Глобальная память
- Coalesced access pattern критичен
- Для 4M float: 16MB — помещается в кеш GPU
- Использовать async копирование (clEnqueueMapBuffer)

#### 4. Batch обработка (256 лучей)
- Обрабатывать несколько лучей параллельно на разных Compute Units
- Использовать `clEnqueueNDRangeKernel` с 2D/3D grid
- На 10 GPU: ~26 лучей на GPU

#### 5. Профилирование
- Использовать `GPUProfiler` из DrvGPU
- Измерять: kernel time, memory transfer, occupancy
- Целевая производительность: >100 GFLOPs для данной задачи

---

## 📚 Ключевые источники и библиотеки

### Научные статьи
- [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](https://arxiv.org/html/2501.14336v1) (2025)
- [Efficient Top-K Query Processing on Massively Parallel Hardware](https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf)
- [GPU Gems 3: Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [Work-Efficient Parallel Non-Maximum Suppression for Embedded GPU](https://hertasecurity.com/wp-content/uploads/work-efficient-parallel-non-maximum-suppression.pdf)

### Библиотеки

#### OpenCL
- [clFFT](https://clmathlibraries.github.io/clFFT/) — FFT на OpenCL
- [PyOpenCL](https://documen.tician.de/pyopencl/) — Python bindings + Reduction/Scan kernels
- [NVIDIA OpenCL Examples](https://github.com/sschaetz/nvidia-opencl-examples) — Оптимизированные reduction kernels

#### ROCm
- [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/) — FFT для AMD GPU
- [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/) — Параллельные примитивы (reduce, scan, sort)
- [rocThrust](https://rocm.docs.amd.com/projects/rocThrust/) — Высокоуровневые алгоритмы
- [hipCUB](https://rocm.docs.amd.com/projects/hipCUB/) — CUB-совместимый интерфейс

#### Примеры кода
- [gpu-topk](https://github.com/anilshanbhag/gpu-topk) — Top-K selection (CUDA, требует портирования)
- [HeCBench](https://github.com/zjin-lcf/HeCBench) — Набор benchmark'ов включая local maxima
- [GPU Sorting](https://github.com/Gram21/GPUSorting) — Bitonic/Radix sort на OpenCL
- [OpenCL Reduction](https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclReduction/oclReduction_kernel.cl)

### Дополнительные ресурсы
- [OpenCL Bitonic Sorting](https://www.bealto.com/gpu-sorting_parallel-bitonic-1.html)
- [Parallel Sum Reduction - GPU/OpenCL vs CPU](https://dournac.org/info/gpu_sum_reduction)
- [cuSignal](https://developer.nvidia.com/blog/accelerated-signal-processing-with-cusignal/) — GPU signal processing (NVIDIA, reference)

---

## 🧪 План дальнейших действий

### Этап 1: Прототипирование (1-2 дня)
1. Реализовать Custom Kernel для детекции локальных максимумов (OpenCL)
2. Интегрировать scan из PyOpenCL или написать собственный
3. Протестировать на реальных данных FFT (4M точек)
4. Сравнить с baseline (CPU NumPy/SciPy)

### Этап 2: Оптимизация (2-3 дня)
1. Профилировать через `GPUProfiler`
2. Оптимизировать local memory usage
3. Тюнинг work-group размеров
4. Batch обработка для 256 лучей

### Этап 3: Портирование на ROCm (1-2 дня)
1. Адаптация кода на HIP
2. Использование rocPRIM для scan/reduce
3. Сравнение производительности OpenCL vs ROCm

### Этап 4: Интеграция с FFTProcessor (1 день)
1. API для поиска максимумов после FFT
2. Python bindings
3. Unit-тесты

### Этап 5: Документация (1 день)
1. Обновить спецификацию в `MemoryBank/specs/`
2. Примеры использования в Python
3. Benchmark результаты

---

## 💡 Выводы

1. **Для задачи "найти ВСЕ локальные максимумы"**: Custom Kernel + Scan-based Compaction оптимален
2. **Для задачи "найти K наибольших"**: RadiK или Bitonic Select (требует портирования на ROCm)
3. **Для простоты**: Использовать готовые rocPRIM/PyOpenCL примитивы
4. **Производительность**: Custom kernel даёт ~1.5-2.5 ms на луч (4M точек)
5. **Масштабирование**: На 10 GPU обработка 256 лучей займёт ~50-100 ms

**Next Steps**: Начать с прототипа Custom Kernel + PyOpenCL scan для проверки концепции.

---

**Автор**: Кодо (AI Assistant)
**Дата**: 2026-02-14
**Связано с**: FFTProcessor, SpectrumMaximaFinder, DrvGPU
**Статус**: Research Complete → Ready for Implementation
